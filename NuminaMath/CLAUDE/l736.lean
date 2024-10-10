import Mathlib

namespace smallest_N_is_110_l736_73632

/-- Represents a point in the rectangular array -/
structure Point where
  row : Fin 6
  col : ℕ

/-- The x-coordinate of a point after initial numbering -/
def x (p : Point) (N : ℕ) : ℕ := p.row.val * N + p.col

/-- The y-coordinate of a point after renumbering -/
def y (p : Point) : ℕ := (p.col - 1) * 6 + p.row.val + 1

/-- Predicate that checks if the given conditions are satisfied -/
def satisfiesConditions (N : ℕ) (p₁ p₂ p₃ p₄ p₅ p₆ : Point) : Prop :=
  x p₁ N = y p₂ ∧
  x p₂ N = y p₁ ∧
  x p₃ N = y p₄ ∧
  x p₄ N = y p₅ ∧
  x p₅ N = y p₆ ∧
  x p₆ N = y p₃

/-- The main theorem stating that 110 is the smallest N satisfying the conditions -/
theorem smallest_N_is_110 :
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : Point),
    satisfiesConditions 110 p₁ p₂ p₃ p₄ p₅ p₆ ∧
    ∀ (N : ℕ), N < 110 → ¬∃ (q₁ q₂ q₃ q₄ q₅ q₆ : Point),
      satisfiesConditions N q₁ q₂ q₃ q₄ q₅ q₆ :=
by sorry

end smallest_N_is_110_l736_73632


namespace digit_sum_property_l736_73676

/-- A function that returns true if a number has no zero digits -/
def has_no_zero_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0

/-- A function that returns all digit permutations of a number -/
def digit_permutations (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that checks if a number is composed entirely of ones -/
def all_ones (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1

/-- A function that checks if a number has a digit 5 or greater -/
def has_digit_ge_5 (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d ≥ 5

theorem digit_sum_property (n : ℕ) (h1 : has_no_zero_digits n) 
  (h2 : all_ones (n + (Finset.sum (digit_permutations n) id))) :
  has_digit_ge_5 n :=
sorry

end digit_sum_property_l736_73676


namespace train_speed_l736_73601

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length time : ℝ) (h1 : length = 700) (h2 : time = 40) :
  length / time = 17.5 := by
  sorry

end train_speed_l736_73601


namespace sum_of_bases_is_nineteen_l736_73672

/-- Represents a repeating decimal in a given base -/
def RepeatingDecimal (numerator : ℕ) (denominator : ℕ) (base : ℕ) : Prop :=
  ∃ (k : ℕ), (base ^ k * numerator) % denominator = numerator

/-- The main theorem -/
theorem sum_of_bases_is_nineteen (R₁ R₂ : ℕ) :
  R₁ > 1 ∧ R₂ > 1 ∧
  RepeatingDecimal 5 11 R₁ ∧
  RepeatingDecimal 6 11 R₁ ∧
  RepeatingDecimal 5 13 R₂ ∧
  RepeatingDecimal 8 13 R₂ →
  R₁ + R₂ = 19 := by sorry

end sum_of_bases_is_nineteen_l736_73672


namespace triangle_side_range_l736_73667

theorem triangle_side_range (a b c : ℝ) : 
  -- Triangle ABC is acute
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  -- Side lengths form an arithmetic sequence
  (b - a = c - b) ∧
  -- Sum of squares of sides equals 21
  a^2 + b^2 + c^2 = 21 →
  -- Range of b
  2 * Real.sqrt 42 / 5 < b ∧ b ≤ Real.sqrt 7 := by
sorry

end triangle_side_range_l736_73667


namespace triangle_cosine_theorem_l736_73677

theorem triangle_cosine_theorem (A B C : ℝ) (h1 : A + C = 2 * B) 
  (h2 : 1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B) :
  Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 := by
  sorry

end triangle_cosine_theorem_l736_73677


namespace cube_surface_area_increase_l736_73682

theorem cube_surface_area_increase :
  ∀ (s : ℝ), s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.4 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 :=
by sorry

end cube_surface_area_increase_l736_73682


namespace triangle_angle_identity_l736_73627

theorem triangle_angle_identity (α β γ : Real) (h : α + β + γ = π) :
  (Real.sin β)^2 + (Real.sin γ)^2 - 2 * (Real.sin β) * (Real.sin γ) * (Real.cos α) = (Real.sin α)^2 := by
  sorry

end triangle_angle_identity_l736_73627


namespace retailer_profit_calculation_l736_73691

/-- Calculates the actual profit percentage for a retailer who marks up goods
    by a certain percentage and then offers a discount. -/
theorem retailer_profit_calculation 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (markup_percentage_is_40 : markup_percentage = 40)
  (discount_percentage_is_25 : discount_percentage = 25)
  : let marked_price := cost_price * (1 + markup_percentage / 100)
    let selling_price := marked_price * (1 - discount_percentage / 100)
    let profit := selling_price - cost_price
    let profit_percentage := (profit / cost_price) * 100
    profit_percentage = 5 := by
  sorry

end retailer_profit_calculation_l736_73691


namespace sperm_genotypes_l736_73619

-- Define the possible alleles
inductive Allele
| A
| a
| Xb
| Y

-- Define a genotype as a list of alleles
def Genotype := List Allele

-- Define the initial spermatogonial cell genotype
def initialGenotype : Genotype := [Allele.A, Allele.a, Allele.Xb, Allele.Y]

-- Define the genotype of the abnormal sperm
def abnormalSperm : Genotype := [Allele.A, Allele.A, Allele.a, Allele.Xb]

-- Define the function to check if a list of genotypes is valid
def isValidResult (sperm1 sperm2 sperm3 : Genotype) : Prop :=
  sperm1 = [Allele.a, Allele.Xb] ∧
  sperm2 = [Allele.Y] ∧
  sperm3 = [Allele.Y]

-- State the theorem
theorem sperm_genotypes (initialCell : Genotype) (abnormalSperm : Genotype) :
  initialCell = initialGenotype →
  abnormalSperm = abnormalSperm →
  ∃ (sperm1 sperm2 sperm3 : Genotype), isValidResult sperm1 sperm2 sperm3 :=
sorry

end sperm_genotypes_l736_73619


namespace circle_diameter_endpoint_l736_73600

/-- Given a circle with center (4, 6) and one endpoint of a diameter at (2, 1),
    prove that the other endpoint of the diameter is at (6, 11). -/
theorem circle_diameter_endpoint (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : 
  P = (4, 6) →  -- Center of the circle
  A = (2, 1) →  -- One endpoint of the diameter
  (P.1 - A.1 = B.1 - P.1 ∧ P.2 - A.2 = B.2 - P.2) →  -- B is symmetric to A with respect to P
  B = (6, 11) :=  -- The other endpoint of the diameter
by sorry

end circle_diameter_endpoint_l736_73600


namespace q_work_time_l736_73636

-- Define the work rates and total work
variable (W : ℝ) -- Total work
variable (Wp Wq Wr : ℝ) -- Work rates of p, q, and r

-- Define the conditions
axiom condition1 : Wp = Wq + Wr
axiom condition2 : Wp + Wq = W / 10
axiom condition3 : Wr = W / 60

-- Theorem to prove
theorem q_work_time : Wq = W / 24 := by
  sorry


end q_work_time_l736_73636


namespace largest_fraction_l736_73635

theorem largest_fraction :
  let fractions := [2/5, 3/7, 4/9, 7/15, 9/20, 11/25]
  ∀ x ∈ fractions, (7:ℚ)/15 ≥ x := by
  sorry

end largest_fraction_l736_73635


namespace sector_area_l736_73679

/-- The area of a circular sector with central angle 120° and radius 3/2 is 3/4 π. -/
theorem sector_area (angle : Real) (radius : Real) : 
  angle = 120 * π / 180 → radius = 3 / 2 → 
  (angle / (2 * π)) * π * radius^2 = 3 / 4 * π := by
  sorry

#check sector_area

end sector_area_l736_73679


namespace sum_units_digits_734_99_347_83_l736_73690

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of the units digits of 734^99 and 347^83 is 7 -/
theorem sum_units_digits_734_99_347_83 : 
  (unitsDigit (734^99) + unitsDigit (347^83)) = 7 := by
  sorry

end sum_units_digits_734_99_347_83_l736_73690


namespace max_distance_line_ellipse_intersection_l736_73664

/-- The maximum distance between two intersection points of a line and an ellipse -/
theorem max_distance_line_ellipse_intersection :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + 4*y^2 = 4}
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | y = x + m}
  let intersection (m : ℝ) := {p : ℝ × ℝ | p ∈ ellipse ∧ p ∈ line m}
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (m : ℝ), ∀ (p q : ℝ × ℝ), p ∈ intersection m → q ∈ intersection m → p ≠ q →
    distance p q ≤ (4/5) * Real.sqrt 10 ∧
    ∃ (m' : ℝ) (p' q' : ℝ × ℝ), p' ∈ intersection m' ∧ q' ∈ intersection m' ∧ p' ≠ q' ∧
      distance p' q' = (4/5) * Real.sqrt 10 :=
sorry

end max_distance_line_ellipse_intersection_l736_73664


namespace koala_fiber_consumption_l736_73644

theorem koala_fiber_consumption (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_consumed : ℝ) :
  absorption_rate = 0.25 →
  absorbed_amount = 10.5 →
  absorbed_amount = absorption_rate * total_consumed →
  total_consumed = 42 := by
  sorry

end koala_fiber_consumption_l736_73644


namespace black_rhinos_count_l736_73611

/-- The number of white rhinos -/
def num_white_rhinos : ℕ := 7

/-- The weight of each white rhino in pounds -/
def weight_white_rhino : ℕ := 5100

/-- The weight of each black rhino in pounds -/
def weight_black_rhino : ℕ := 2000

/-- The total weight of all rhinos in pounds -/
def total_weight : ℕ := 51700

/-- The number of black rhinos -/
def num_black_rhinos : ℕ := (total_weight - num_white_rhinos * weight_white_rhino) / weight_black_rhino

theorem black_rhinos_count : num_black_rhinos = 8 := by sorry

end black_rhinos_count_l736_73611


namespace least_sum_of_bases_l736_73637

/-- Represents a number in a given base -/
def BaseRepresentation (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

/-- The problem statement -/
theorem least_sum_of_bases :
  ∃ (c d : Nat),
    c > 0 ∧ d > 0 ∧
    BaseRepresentation [5, 8] c = BaseRepresentation [8, 5] d ∧
    (∀ (c' d' : Nat),
      c' > 0 → d' > 0 →
      BaseRepresentation [5, 8] c' = BaseRepresentation [8, 5] d' →
      c + d ≤ c' + d') ∧
    c + d = 15 :=
  sorry

end least_sum_of_bases_l736_73637


namespace problem_solution_l736_73630

theorem problem_solution (a b : ℝ) (h1 : a + b = 4) (h2 : a * b = 1) : 
  (a - b)^2 = 12 ∧ a^5*b - 2*a^4*b^4 + a*b^5 = 192 := by
  sorry

end problem_solution_l736_73630


namespace final_number_calculation_l736_73615

theorem final_number_calculation : 
  let initial_number : ℕ := 9
  let doubled := initial_number * 2
  let added_13 := doubled + 13
  let final_number := added_13 * 3
  final_number = 93 := by sorry

end final_number_calculation_l736_73615


namespace ninth_term_of_arithmetic_sequence_l736_73609

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with the given properties, the ninth term is 35. -/
theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 23)
  (h_sixth_term : a 6 = 29) :
  a 9 = 35 := by
  sorry

end ninth_term_of_arithmetic_sequence_l736_73609


namespace cat_arrangement_count_l736_73665

/-- Represents the number of cat cages -/
def num_cages : ℕ := 5

/-- Represents the number of golden tabby cats -/
def num_golden : ℕ := 3

/-- Represents the number of silver tabby cats -/
def num_silver : ℕ := 4

/-- Represents the number of ragdoll cats -/
def num_ragdoll : ℕ := 1

/-- Represents the number of ways to arrange silver tabby cats in pairs -/
def silver_arrangements : ℕ := 3

/-- Represents the total number of units to arrange (golden group, 2 silver pairs, ragdoll) -/
def total_units : ℕ := 4

/-- Theorem stating the number of possible arrangements -/
theorem cat_arrangement_count :
  (Nat.choose num_cages total_units) * Nat.factorial total_units * silver_arrangements = 360 := by
  sorry

end cat_arrangement_count_l736_73665


namespace negation_of_universal_proposition_l736_73607

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 1) ↔ (∃ x : ℝ, x < 1) := by
  sorry

end negation_of_universal_proposition_l736_73607


namespace parallel_planes_normal_vectors_l736_73675

/-- Given two planes α and β with normal vectors (x, 1, -2) and (-1, y, 1/2) respectively,
    if α is parallel to β, then x + y = 15/4 -/
theorem parallel_planes_normal_vectors (x y : ℝ) :
  let n1 : ℝ × ℝ × ℝ := (x, 1, -2)
  let n2 : ℝ × ℝ × ℝ := (-1, y, 1/2)
  (∃ (k : ℝ), n1 = k • n2) →
  x + y = 15/4 := by
sorry

end parallel_planes_normal_vectors_l736_73675


namespace erroneous_product_l736_73653

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a and multiplying by b, then adding 2, results in 240,
    then the actual product of a and b is 301. -/
theorem erroneous_product (a b : ℕ) : 
  a > 9 ∧ a < 100 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (((a % 10) * 10 + (a / 10)) * b + 2 = 240) →  -- erroneous calculation
  a * b = 301 := by
sorry

end erroneous_product_l736_73653


namespace union_of_A_and_B_l736_73638

-- Define the sets A and B
def A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
def B : Set ℝ := { x | 3 * x - 7 ≥ 8 - 2 * x }

-- State the theorem
theorem union_of_A_and_B : A ∪ B = { x | x ≥ 2 } := by
  sorry

end union_of_A_and_B_l736_73638


namespace cost_price_calculation_l736_73674

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 600)
  (h2 : profit_percentage = 25) :
  let cost_price := selling_price / (1 + profit_percentage / 100)
  cost_price = 480 := by
sorry

end cost_price_calculation_l736_73674


namespace last_eight_digits_of_product_l736_73629

theorem last_eight_digits_of_product : ∃ n : ℕ, 
  11 * 101 * 1001 * 10001 * 100001 * 1000001 * 111 ≡ 19754321 [MOD 100000000] :=
by sorry

end last_eight_digits_of_product_l736_73629


namespace binomial_sum_theorem_l736_73641

theorem binomial_sum_theorem (A B C D : ℚ) :
  (∀ n : ℕ, n ≥ 4 → n^4 = A * Nat.choose n 4 + B * Nat.choose n 3 + C * Nat.choose n 2 + D * Nat.choose n 1) →
  A + B + C + D = 75 := by
sorry

end binomial_sum_theorem_l736_73641


namespace absolute_value_inequality_implies_a_bound_l736_73603

theorem absolute_value_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, |x + 2| - |x - 1| ≥ a^3 - 4*a^2 - 3) →
  a ≤ 4 := by
sorry

end absolute_value_inequality_implies_a_bound_l736_73603


namespace value_of_expression_l736_73642

theorem value_of_expression (a b : ℤ) (A B : ℤ) 
  (h1 : A = 3 * b^2 - 2 * a^2)
  (h2 : B = a * b - 2 * b^2 - a^2)
  (h3 : a = 2)
  (h4 : b = -1) :
  A - 2 * B = 11 := by
  sorry

end value_of_expression_l736_73642


namespace expression_value_l736_73670

theorem expression_value (a b : ℤ) (ha : a = -3) (hb : b = 2) :
  -a - b^3 + a*b = -11 := by sorry

end expression_value_l736_73670


namespace expression_value_l736_73608

theorem expression_value : 
  (150^2 - 13^2) / (90^2 - 17^2) * ((90-17)*(90+17)) / ((150-13)*(150+13)) = 1 := by
  sorry

end expression_value_l736_73608


namespace orange_preference_percentage_l736_73633

def survey_results : List (String × Nat) :=
  [("Red", 70), ("Orange", 50), ("Green", 60), ("Yellow", 80), ("Blue", 40), ("Purple", 50)]

def total_responses : Nat :=
  (survey_results.map (λ (_, count) => count)).sum

def orange_preference : Nat :=
  match survey_results.find? (λ (color, _) => color = "Orange") with
  | some (_, count) => count
  | none => 0

theorem orange_preference_percentage :
  (orange_preference : ℚ) / (total_responses : ℚ) * 100 = 14 := by sorry

end orange_preference_percentage_l736_73633


namespace patty_weeks_without_chores_l736_73647

/-- Calculates the number of weeks Patty can go without doing chores --/
def weeks_without_chores (
  cookies_per_chore : ℕ) 
  (chores_per_kid_per_week : ℕ) 
  (money_available : ℕ) 
  (cookies_per_pack : ℕ) 
  (cost_per_pack : ℕ) 
  (num_siblings : ℕ) : ℕ :=
  let packs_bought := money_available / cost_per_pack
  let total_cookies := packs_bought * cookies_per_pack
  let cookies_per_sibling_per_week := chores_per_kid_per_week * cookies_per_chore
  let cookies_needed_per_week := cookies_per_sibling_per_week * num_siblings
  total_cookies / cookies_needed_per_week

theorem patty_weeks_without_chores :
  weeks_without_chores 3 4 15 24 3 2 = 5 := by
  sorry

end patty_weeks_without_chores_l736_73647


namespace egg_problem_l736_73694

/-- The initial number of eggs in the basket -/
def initial_eggs : ℕ := 120

/-- The number of broken eggs -/
def broken_eggs : ℕ := 20

/-- The total price in fillérs -/
def total_price : ℕ := 600

/-- Proves that the initial number of eggs was 120 -/
theorem egg_problem :
  initial_eggs = 120 ∧
  broken_eggs = 20 ∧
  total_price = 600 ∧
  (total_price : ℚ) / initial_eggs + 1 = total_price / (initial_eggs - broken_eggs) :=
by sorry

end egg_problem_l736_73694


namespace units_digit_of_factorial_sum_plus_three_l736_73613

-- Define a function to calculate factorial
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the sum of factorials from 1 to 10
def factorialSum : ℕ := 
  List.sum (List.map factorial (List.range 10))

-- Theorem to prove
theorem units_digit_of_factorial_sum_plus_three : 
  unitsDigit (factorialSum + 3) = 6 := by sorry

end units_digit_of_factorial_sum_plus_three_l736_73613


namespace total_distance_is_66_l736_73648

def first_museum_distance : ℕ := 5
def second_museum_distance : ℕ := 15
def cultural_center_distance : ℕ := 10
def detour_distance : ℕ := 3

def total_distance : ℕ :=
  2 * (first_museum_distance + detour_distance) +
  2 * second_museum_distance +
  2 * cultural_center_distance

theorem total_distance_is_66 : total_distance = 66 := by
  sorry

end total_distance_is_66_l736_73648


namespace greatest_value_quadratic_inequality_l736_73695

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, -a^2 + 9*a - 14 ≥ 0 → a ≤ 7 :=
by sorry

end greatest_value_quadratic_inequality_l736_73695


namespace inequalities_given_sum_positive_l736_73604

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end inequalities_given_sum_positive_l736_73604


namespace bert_toy_phones_l736_73645

/-- 
Proves that Bert sold 8 toy phones given the conditions of the problem.
-/
theorem bert_toy_phones :
  ∀ (bert_phones : ℕ),
  (18 * bert_phones = 20 * 7 + 4) →
  bert_phones = 8 := by
  sorry

end bert_toy_phones_l736_73645


namespace evaluate_expression_l736_73678

theorem evaluate_expression (x z : ℤ) (hx : x = 4) (hz : z = -2) :
  z * (z - 4 * x) = 36 := by
  sorry

end evaluate_expression_l736_73678


namespace geometric_sequence_sum_l736_73661

/-- 
Given a geometric sequence with positive terms, if the sum of the first n terms is 3
and the sum of the first 3n terms is 21, then the sum of the first 2n terms is 9.
-/
theorem geometric_sequence_sum (n : ℕ) (a : ℝ) (r : ℝ) 
  (h_positive : ∀ k, a * r ^ k > 0)
  (h_sum_n : (a * (1 - r^n)) / (1 - r) = 3)
  (h_sum_3n : (a * (1 - r^(3*n))) / (1 - r) = 21) :
  (a * (1 - r^(2*n))) / (1 - r) = 9 := by
  sorry

end geometric_sequence_sum_l736_73661


namespace min_value_exponential_sum_l736_73699

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2 * y = 1) :
  2^x + 4^y ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_exponential_sum_l736_73699


namespace platform_length_l736_73623

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 8 seconds to cross a signal pole, prove that the length of the platform is 1162.5 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 8) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 1162.5 := by
  sorry

end platform_length_l736_73623


namespace power_of_six_with_nine_tens_digit_l736_73696

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem power_of_six_with_nine_tens_digit :
  ∃ (k : ℕ), k > 0 ∧ tens_digit (6^k) = 9 ∧ ∀ (m : ℕ), m > 0 ∧ m < k → tens_digit (6^m) ≠ 9 :=
sorry

end power_of_six_with_nine_tens_digit_l736_73696


namespace zeros_of_f_with_fixed_points_range_of_b_with_no_fixed_points_l736_73618

-- Define the function f(x)
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Theorem 1
theorem zeros_of_f_with_fixed_points (b c : ℝ) :
  (f b c (-3) = -3) ∧ (f b c 2 = 2) →
  (∃ x : ℝ, f b c x = 0) ∧ 
  (∀ x : ℝ, f b c x = 0 ↔ (x = -1 + Real.sqrt 7 ∨ x = -1 - Real.sqrt 7)) :=
sorry

-- Theorem 2
theorem range_of_b_with_no_fixed_points :
  (∀ b : ℝ, ∀ x : ℝ, f b (b^2/4) x ≠ x) →
  (∀ b : ℝ, (b < -1 ∨ b > 1/3) ↔ (∀ x : ℝ, f b (b^2/4) x ≠ x)) :=
sorry

end zeros_of_f_with_fixed_points_range_of_b_with_no_fixed_points_l736_73618


namespace max_value_a_squared_b_l736_73692

theorem max_value_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) :
  a^2 * b ≤ 54 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ * (a₀ + b₀) = 27 ∧ a₀^2 * b₀ = 54 := by
  sorry

end max_value_a_squared_b_l736_73692


namespace vector_equation_solution_l736_73621

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) 
  (h : 3 • a + (3/5 : ℝ) • (b - x) = b) : 
  x = 5 • a - (2/3 : ℝ) • b :=
sorry

end vector_equation_solution_l736_73621


namespace mixed_committee_probability_l736_73684

def total_members : ℕ := 30
def num_boys : ℕ := 13
def num_girls : ℕ := 17
def committee_size : ℕ := 6

def probability_mixed_committee : ℚ :=
  1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size) / Nat.choose total_members committee_size

theorem mixed_committee_probability :
  probability_mixed_committee = 579683 / 593775 :=
by sorry

end mixed_committee_probability_l736_73684


namespace chosen_number_proof_l736_73655

theorem chosen_number_proof :
  ∃! (x : ℝ), x > 0 ∧ (Real.sqrt (x^2) / 6) - 189 = 3 :=
by
  -- The proof goes here
  sorry

end chosen_number_proof_l736_73655


namespace initial_players_l736_73620

theorem initial_players (initial_players new_players lives_per_player total_lives : ℕ) :
  new_players = 5 →
  lives_per_player = 3 →
  total_lives = 27 →
  (initial_players + new_players) * lives_per_player = total_lives →
  initial_players = 4 := by
sorry

end initial_players_l736_73620


namespace seating_theorem_l736_73673

/-- The number of ways to arrange n objects from m choices --/
def permutation (n m : ℕ) : ℕ := 
  if n > m then 0
  else Nat.factorial m / Nat.factorial (m - n)

/-- The number of ways four people can sit in a row of five chairs --/
def seating_arrangements : ℕ := permutation 4 5

theorem seating_theorem : seating_arrangements = 120 := by
  sorry

end seating_theorem_l736_73673


namespace upstream_distance_l736_73643

/-- Proves that given a man who swims downstream 30 km in 6 hours and upstream for 6 hours, 
    with a speed of 4 km/h in still water, the distance he swims upstream is 18 km. -/
theorem upstream_distance 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : downstream_distance = 30)
  (h2 : downstream_time = 6)
  (h3 : upstream_time = 6)
  (h4 : still_water_speed = 4) : 
  ∃ upstream_distance : ℝ, upstream_distance = 18 := by
  sorry


end upstream_distance_l736_73643


namespace external_circle_radius_l736_73680

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at C
  (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0 ∧
  -- Angle A is 45°
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) / Real.sqrt 2 ∧
  -- AC = 12
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 144

-- Define the external circle
def ExternalCircle (center : ℝ × ℝ) (radius : ℝ) (A B C : ℝ × ℝ) : Prop :=
  -- Circle is tangent to AB
  ((center.1 - A.1) * (B.2 - A.2) - (center.2 - A.2) * (B.1 - A.1))^2 = 
    radius^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2) ∧
  -- Center lies on line AB
  (center.2 - A.2) * (B.1 - A.1) = (center.1 - A.1) * (B.2 - A.2)

-- Theorem statement
theorem external_circle_radius (A B C : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) :
  Triangle A B C → ExternalCircle center radius A B C → radius = 6 * Real.sqrt 2 := by sorry

end external_circle_radius_l736_73680


namespace max_white_pieces_correct_l736_73686

/-- Represents a game board with m rows and n columns -/
structure Board (m n : ℕ) where
  white_pieces : Finset (ℕ × ℕ)
  no_same_row_col : ∀ (i j k l : ℕ), (i, j) ∈ white_pieces → (k, l) ∈ white_pieces → i = k ∨ j = l → (i, j) = (k, l)

/-- The maximum number of white pieces that can be placed on the board -/
def max_white_pieces (m n : ℕ) : ℕ := m + n - 1

/-- Theorem stating that the maximum number of white pieces is m + n - 1 -/
theorem max_white_pieces_correct (m n : ℕ) :
  ∀ (b : Board m n), b.white_pieces.card ≤ max_white_pieces m n :=
by sorry

end max_white_pieces_correct_l736_73686


namespace intersection_point_l736_73628

/-- A parabola in the xy-plane defined by y^2 - 4y + x = 6 -/
def parabola (x y : ℝ) : Prop := y^2 - 4*y + x = 6

/-- A vertical line in the xy-plane defined by x = k -/
def vertical_line (k x : ℝ) : Prop := x = k

/-- The condition for a quadratic equation ay^2 + by + c = 0 to have exactly one solution -/
def has_unique_solution (a b c : ℝ) : Prop := b^2 - 4*a*c = 0

theorem intersection_point (k : ℝ) : 
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ vertical_line k p.1) ↔ k = 10 := by
  sorry

end intersection_point_l736_73628


namespace percentage_equation_solution_l736_73689

theorem percentage_equation_solution :
  ∃ x : ℝ, (12.4 * 350) + (9.9 * 275) = (8.6 * x) + (5.3 * (2250 - x)) := by
  sorry

end percentage_equation_solution_l736_73689


namespace percentage_of_female_officers_on_duty_l736_73617

def total_officers_on_duty : ℕ := 160
def total_female_officers : ℕ := 500

def female_officers_on_duty : ℕ := total_officers_on_duty / 2

def percentage_on_duty : ℚ := (female_officers_on_duty : ℚ) / total_female_officers * 100

theorem percentage_of_female_officers_on_duty :
  percentage_on_duty = 16 := by sorry

end percentage_of_female_officers_on_duty_l736_73617


namespace product_equals_243_l736_73652

theorem product_equals_243 : 
  (1 / 3 : ℚ) * 9 * (1 / 27 : ℚ) * 81 * (1 / 243 : ℚ) * 729 * (1 / 2187 : ℚ) * 6561 * (1 / 19683 : ℚ) * 59049 = 243 := by
  sorry

end product_equals_243_l736_73652


namespace sum_interior_angles_formula_l736_73657

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: For a polygon with n sides (n > 2), the sum of interior angles is 180° × (n-2) -/
theorem sum_interior_angles_formula {n : ℕ} (h : n > 2) :
  sum_interior_angles n = 180 * (n - 2) := by
  sorry

end sum_interior_angles_formula_l736_73657


namespace daily_water_intake_l736_73669

-- Define the given conditions
def daily_soda_cans : ℕ := 5
def ounces_per_can : ℕ := 12
def weekly_total_fluid : ℕ := 868

-- Define the daily soda intake in ounces
def daily_soda_ounces : ℕ := daily_soda_cans * ounces_per_can

-- Define the weekly soda intake in ounces
def weekly_soda_ounces : ℕ := daily_soda_ounces * 7

-- Define the weekly water intake in ounces
def weekly_water_ounces : ℕ := weekly_total_fluid - weekly_soda_ounces

-- Theorem to prove
theorem daily_water_intake : weekly_water_ounces / 7 = 64 := by
  sorry

end daily_water_intake_l736_73669


namespace flowerbed_length_difference_l736_73656

theorem flowerbed_length_difference (width length : ℝ) : 
  width = 4 →
  2 * length + 2 * width = 22 →
  2 * width - length = 1 :=
by
  sorry

end flowerbed_length_difference_l736_73656


namespace largest_two_digit_multiple_of_seven_l736_73602

def digits : Set Nat := {3, 5, 6, 7}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def formed_from_digits (n : Nat) : Prop :=
  ∃ (d1 d2 : Nat), d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2

theorem largest_two_digit_multiple_of_seven :
  ∀ n : Nat, is_two_digit n → formed_from_digits n → n % 7 = 0 →
  n ≤ 63 :=
sorry

end largest_two_digit_multiple_of_seven_l736_73602


namespace g_1003_fixed_point_l736_73688

def g₁ (x : ℚ) : ℚ := 1/2 - 4/(4*x+2)

def g (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => g₁ x
  | n+1 => g₁ (g n x)

theorem g_1003_fixed_point :
  g 1003 (11/2) = 11/2 - 4 := by sorry

end g_1003_fixed_point_l736_73688


namespace toys_sold_is_eighteen_l736_73626

/-- The number of toys sold by a man, given the selling price, gain, and cost price per toy. -/
def number_of_toys_sold (selling_price gain cost_per_toy : ℕ) : ℕ :=
  (selling_price - gain) / cost_per_toy

/-- Theorem stating that the number of toys sold is 18 under the given conditions. -/
theorem toys_sold_is_eighteen :
  let selling_price : ℕ := 25200
  let cost_per_toy : ℕ := 1200
  let gain : ℕ := 3 * cost_per_toy
  number_of_toys_sold selling_price gain cost_per_toy = 18 := by
sorry

#eval number_of_toys_sold 25200 (3 * 1200) 1200

end toys_sold_is_eighteen_l736_73626


namespace birth_rate_calculation_l736_73697

/-- Represents the annual birth rate per 1000 people in a country. -/
def birth_rate : ℝ := sorry

/-- Represents the annual death rate per 1000 people in a country. -/
def death_rate : ℝ := 19.4

/-- Represents the number of years it takes for the population to double. -/
def doubling_time : ℝ := 35

/-- The Rule of 70 for population growth. -/
axiom rule_of_70 (growth_rate : ℝ) : 
  doubling_time = 70 / growth_rate

/-- The net growth rate is the difference between birth rate and death rate. -/
def net_growth_rate : ℝ := birth_rate - death_rate

theorem birth_rate_calculation : birth_rate = 21.4 := by sorry

end birth_rate_calculation_l736_73697


namespace stamp_collection_problem_l736_73639

/-- Represents the number of stamps Simon received from each friend -/
structure FriendStamps where
  x1 : ℕ
  x2 : ℕ
  x3 : ℕ
  x4 : ℕ
  x5 : ℕ

/-- Theorem representing the stamp collection problem -/
theorem stamp_collection_problem 
  (initial_stamps final_stamps : ℕ) 
  (friend_stamps : FriendStamps) : 
  initial_stamps = 34 →
  final_stamps = 61 →
  friend_stamps.x1 = 12 →
  friend_stamps.x3 = 21 →
  friend_stamps.x5 = 10 →
  friend_stamps.x1 + friend_stamps.x2 + friend_stamps.x3 + 
  friend_stamps.x4 + friend_stamps.x5 = final_stamps - initial_stamps :=
by
  sorry

#check stamp_collection_problem

end stamp_collection_problem_l736_73639


namespace sadies_daily_burger_spending_l736_73612

/-- Sadie's daily burger spending in June -/
def daily_burger_spending (total_spending : ℚ) (days : ℕ) : ℚ :=
  total_spending / days

theorem sadies_daily_burger_spending :
  let total_spending : ℚ := 372
  let days : ℕ := 30
  daily_burger_spending total_spending days = 12.4 := by
  sorry

end sadies_daily_burger_spending_l736_73612


namespace integer_division_l736_73625

theorem integer_division (x : ℤ) :
  (∃ k : ℤ, (5 * x + 2) = 17 * k) ↔ (∃ m : ℤ, x = 17 * m + 3) :=
by sorry

end integer_division_l736_73625


namespace no_rational_roots_l736_73610

def polynomial (x : ℚ) : ℚ :=
  3 * x^5 + 4 * x^4 - 5 * x^3 - 15 * x^2 + 7 * x + 3

theorem no_rational_roots :
  ∀ (x : ℚ), polynomial x ≠ 0 := by
sorry

end no_rational_roots_l736_73610


namespace complex_fraction_equality_l736_73646

theorem complex_fraction_equality (a b : ℝ) :
  (a / (1 - Complex.I)) + (b / (2 - Complex.I)) = 1 / (3 - Complex.I) →
  a = -1/5 ∧ b = 1 := by
  sorry

end complex_fraction_equality_l736_73646


namespace ellianna_fat_served_l736_73622

/-- The amount of fat in ounces for a herring -/
def herring_fat : ℕ := 40

/-- The amount of fat in ounces for an eel -/
def eel_fat : ℕ := 20

/-- The amount of fat in ounces for a pike -/
def pike_fat : ℕ := eel_fat + 10

/-- The number of fish of each type that Ellianna cooked and served -/
def fish_count : ℕ := 40

/-- The total amount of fat in ounces served by Ellianna -/
def total_fat : ℕ := fish_count * (herring_fat + eel_fat + pike_fat)

theorem ellianna_fat_served : total_fat = 3600 := by
  sorry

end ellianna_fat_served_l736_73622


namespace triangle_similarity_equivalence_l736_73654

theorem triangle_similarity_equivalence 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ k : ℝ, k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) ↔ 
  (Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
   Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁))) :=
by sorry

end triangle_similarity_equivalence_l736_73654


namespace two_digit_number_puzzle_l736_73614

theorem two_digit_number_puzzle : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 + n % 10 = 13) ∧
  (10 * (n % 10) + (n / 10) = n - 27) :=
by sorry

end two_digit_number_puzzle_l736_73614


namespace prob_three_green_in_seven_trials_l736_73693

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The number of purple marbles -/
def purple_marbles : ℕ := 4

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The number of successful trials (picking green marbles) -/
def num_success : ℕ := 3

/-- The probability of picking a green marble in a single trial -/
def prob_green : ℚ := green_marbles / total_marbles

/-- The probability of picking a purple marble in a single trial -/
def prob_purple : ℚ := purple_marbles / total_marbles

/-- The probability of picking exactly three green marbles in seven trials -/
theorem prob_three_green_in_seven_trials :
  (Nat.choose num_trials num_success : ℚ) * prob_green ^ num_success * prob_purple ^ (num_trials - num_success) = 280 / 729 := by
  sorry

end prob_three_green_in_seven_trials_l736_73693


namespace min_distance_four_points_l736_73668

/-- Given four points A, B, C, and D on a line, where the distances between consecutive
    points are AB = 10, BC = 4, and CD = 3, the minimum possible distance between A and D is 3. -/
theorem min_distance_four_points (A B C D : ℝ) : 
  |B - A| = 10 → |C - B| = 4 → |D - C| = 3 → 
  (∃ (A' B' C' D' : ℝ), |B' - A'| = 10 ∧ |C' - B'| = 4 ∧ |D' - C'| = 3 ∧ 
    ∀ (X Y Z W : ℝ), |Y - X| = 10 → |Z - Y| = 4 → |W - Z| = 3 → |W - X| ≥ |D' - A'|) →
  (∃ (A₀ B₀ C₀ D₀ : ℝ), |B₀ - A₀| = 10 ∧ |C₀ - B₀| = 4 ∧ |D₀ - C₀| = 3 ∧ |D₀ - A₀| = 3) :=
by sorry

end min_distance_four_points_l736_73668


namespace car_journey_time_proof_l736_73687

/-- Represents the speed and distance of a car's journey -/
structure CarJourney where
  speed : ℝ
  distance : ℝ
  time : ℝ

/-- Given two car journeys, proves that the time taken by the second car
    is 4/3 hours under specific conditions -/
theorem car_journey_time_proof
  (m n : CarJourney)
  (h1 : m.time = 4)
  (h2 : n.speed = 3 * m.speed)
  (h3 : n.distance = 3 * m.distance)
  (h4 : m.distance = m.speed * m.time)
  (h5 : n.distance = n.speed * n.time) :
  n.time = 4 / 3 := by
sorry

end car_journey_time_proof_l736_73687


namespace suki_bag_weight_l736_73698

-- Define the given quantities
def suki_bags : ℝ := 6.5
def jimmy_bags : ℝ := 4.5
def jimmy_bag_weight : ℝ := 18
def container_weight : ℝ := 8
def total_containers : ℕ := 28

-- Define the theorem
theorem suki_bag_weight :
  let total_weight := container_weight * total_containers
  let jimmy_total_weight := jimmy_bags * jimmy_bag_weight
  let suki_total_weight := total_weight - jimmy_total_weight
  suki_total_weight / suki_bags = 22 := by
sorry


end suki_bag_weight_l736_73698


namespace function_property_l736_73624

/-- Given a function f: ℕ → ℕ satisfying the property that
    for all positive integers a, b, n such that a + b = 3^n,
    f(a) + f(b) = 2n^2, prove that f(3003) = 44 -/
theorem function_property (f : ℕ → ℕ) 
  (h : ∀ (a b n : ℕ), 0 < a → 0 < b → 0 < n → a + b = 3^n → f a + f b = 2*n^2) :
  f 3003 = 44 := by
  sorry

end function_property_l736_73624


namespace ginger_garden_work_hours_l736_73685

/-- Calculates the number of hours Ginger worked in her garden --/
def hours_worked (bottle_capacity : ℕ) (bottles_for_plants : ℕ) (total_water_used : ℕ) : ℕ :=
  (total_water_used - bottles_for_plants * bottle_capacity) / bottle_capacity

/-- Proves that Ginger worked 8 hours in her garden given the problem conditions --/
theorem ginger_garden_work_hours :
  let bottle_capacity : ℕ := 2
  let bottles_for_plants : ℕ := 5
  let total_water_used : ℕ := 26
  hours_worked bottle_capacity bottles_for_plants total_water_used = 8 := by
  sorry


end ginger_garden_work_hours_l736_73685


namespace equation_equivalence_l736_73631

theorem equation_equivalence (x y : ℝ) :
  (3 * x^2 + 9 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 5 = 0) →
  4 * y^2 + 23 * y - 14 = 0 := by
sorry

end equation_equivalence_l736_73631


namespace sqrt_equation_solution_l736_73666

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_solution_l736_73666


namespace train_speed_proof_l736_73649

/-- Proves that the speed of each train is 54 km/hr given the problem conditions -/
theorem train_speed_proof (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 120) (h2 : crossing_time = 8) : 
  let relative_speed := (2 * train_length) / crossing_time
  let train_speed_ms := relative_speed / 2
  let train_speed_kmh := train_speed_ms * 3.6
  train_speed_kmh = 54 := by
sorry


end train_speed_proof_l736_73649


namespace inequalities_and_minimum_value_l736_73659

theorem inequalities_and_minimum_value :
  (∀ a b, a > b ∧ b > 0 → (1 / a : ℝ) < (1 / b)) ∧
  (∀ a b, a > b ∧ b > 0 → a - 1 / a > b - 1 / b) ∧
  (∀ a b, a > b ∧ b > 0 → (2 * a + b) / (a + 2 * b) < a / b) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → 2 / a + 1 / b ≥ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ 2 / a + 1 / b = 9) :=
by sorry


end inequalities_and_minimum_value_l736_73659


namespace bill_calculation_l736_73634

theorem bill_calculation (a b c d : ℝ) 
  (h1 : (a - b) + c - d = 19) 
  (h2 : a - b - c - d = 9) : 
  a - b = 14 := by
sorry

end bill_calculation_l736_73634


namespace max_value_of_expression_l736_73671

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 := by
  sorry

end max_value_of_expression_l736_73671


namespace smallest_possible_median_l736_73616

def number_set (x : ℤ) : Finset ℤ := {x, 3*x, 4, 1, 6}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ y => y ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ y => y ≥ m)).card ≥ s.card

theorem smallest_possible_median :
  ∃ (x : ℤ), is_median 1 (number_set x) ∧
  ∀ (y : ℤ) (m : ℤ), is_median m (number_set y) → m ≥ 1 :=
sorry

end smallest_possible_median_l736_73616


namespace coin_sequences_ten_l736_73640

/-- The number of distinct sequences when flipping a coin n times -/
def coin_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem coin_sequences_ten : coin_sequences 10 = 1024 := by
  sorry

end coin_sequences_ten_l736_73640


namespace decimal_41_to_binary_l736_73663

-- Define a function to convert decimal to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

-- Theorem statement
theorem decimal_41_to_binary :
  decimalToBinary 41 = [1, 0, 1, 0, 0, 1] := by
  sorry

end decimal_41_to_binary_l736_73663


namespace vertical_line_angle_is_90_degrees_l736_73662

/-- The angle of inclination of a vertical line -/
def angle_of_vertical_line : ℝ := 90

/-- A vertical line is defined by the equation x = 0 -/
def is_vertical_line (f : ℝ → ℝ) : Prop := ∀ y, f y = 0

theorem vertical_line_angle_is_90_degrees (f : ℝ → ℝ) (h : is_vertical_line f) :
  angle_of_vertical_line = 90 := by
  sorry

end vertical_line_angle_is_90_degrees_l736_73662


namespace symmetric_lines_coefficient_sum_l736_73683

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def line2 (a b x y : ℝ) : Prop := a*x + 4*y + b = 0

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Define symmetry with respect to a point
def symmetric_wrt (p : ℝ × ℝ) (l1 l2 : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ l2 (2*p.1 - x) (2*p.2 - y)

-- Theorem statement
theorem symmetric_lines_coefficient_sum (a b : ℝ) :
  symmetric_wrt point_A (line1) (line2 a b) →
  a + b = 0 :=
sorry

end symmetric_lines_coefficient_sum_l736_73683


namespace min_distance_sum_l736_73605

theorem min_distance_sum (x : ℝ) : 
  ∃ (min : ℝ) (x_min : ℝ), 
    min = |x_min - 2| + |x_min - 4| + |x_min - 10| ∧
    min = 8 ∧ 
    x_min = 4 ∧
    ∀ y : ℝ, |y - 2| + |y - 4| + |y - 10| ≥ min :=
by sorry

end min_distance_sum_l736_73605


namespace line_slope_is_two_l736_73651

/-- Given a line with y-intercept 2 and passing through the point (498, 998), its slope is 2 -/
theorem line_slope_is_two (f : ℝ → ℝ) (h1 : f 0 = 2) (h2 : f 498 = 998) :
  (f 498 - f 0) / (498 - 0) = 2 := by
sorry

end line_slope_is_two_l736_73651


namespace special_triangle_sides_l736_73681

/-- A triangle with consecutive integer side lengths and a median perpendicular to an angle bisector -/
structure SpecialTriangle where
  a : ℕ
  has_consecutive_sides : a > 0
  median_perpendicular_to_bisector : Bool

/-- The sides of a special triangle are 2, 3, and 4 -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 2 := by
  sorry

#check special_triangle_sides

end special_triangle_sides_l736_73681


namespace qin_jiushao_triangle_area_l736_73658

theorem qin_jiushao_triangle_area : 
  let a : ℝ := 5
  let b : ℝ := 6
  let c : ℝ := 7
  let S := Real.sqrt ((1/4) * (a^2 * b^2 - ((a^2 + b^2 - c^2)/2)^2))
  S = 6 * Real.sqrt 6 := by
sorry

end qin_jiushao_triangle_area_l736_73658


namespace arcsin_neg_one_l736_73660

theorem arcsin_neg_one : Real.arcsin (-1) = -π / 2 := by
  sorry

end arcsin_neg_one_l736_73660


namespace root_equation_solution_l736_73606

theorem root_equation_solution (a b c : ℕ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (∀ N : ℝ, N ≠ 1 → (N^(1/a) * (N^(1/b) * N^(3/c))^(1/a))^a = N^(15/24)) →
  c = 6 := by
  sorry

end root_equation_solution_l736_73606


namespace first_quartile_of_data_set_l736_73650

def data_set : List ℕ := [296, 301, 305, 293, 293, 305, 302, 303, 306, 294]

def first_quartile (l : List ℕ) : ℕ := sorry

theorem first_quartile_of_data_set :
  first_quartile data_set = 294 := by sorry

end first_quartile_of_data_set_l736_73650
