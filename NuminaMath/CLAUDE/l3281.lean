import Mathlib

namespace ice_cream_volume_l3281_328109

/-- The volume of ice cream in a cone with hemisphere and cylinder topping -/
theorem ice_cream_volume (cone_height : Real) (cone_radius : Real) (cylinder_height : Real) :
  cone_height = 12 →
  cone_radius = 3 →
  cylinder_height = 2 →
  (1/3 * Real.pi * cone_radius^2 * cone_height) + 
  (2/3 * Real.pi * cone_radius^3) + 
  (Real.pi * cone_radius^2 * cylinder_height) = 72 * Real.pi := by
  sorry

end ice_cream_volume_l3281_328109


namespace quadratic_equation_solution_l3281_328153

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end quadratic_equation_solution_l3281_328153


namespace constant_function_theorem_l3281_328148

/-- The set of all points in the plane -/
def S : Type := ℝ × ℝ

/-- A function from the plane to real numbers -/
def PlaneFunction : Type := S → ℝ

/-- Predicate for a nondegenerate triangle -/
def NonDegenerateTriangle (A B C : S) : Prop := sorry

/-- The orthocenter of a triangle -/
def Orthocenter (A B C : S) : S := sorry

/-- The property that the function satisfies for all nondegenerate triangles -/
def SatisfiesTriangleProperty (f : PlaneFunction) : Prop :=
  ∀ A B C : S, NonDegenerateTriangle A B C →
    let H := Orthocenter A B C
    (f A ≤ f B ∧ f B ≤ f C) → f A + f C = f B + f H

/-- The main theorem: if a function satisfies the triangle property, it must be constant -/
theorem constant_function_theorem (f : PlaneFunction) 
  (h : SatisfiesTriangleProperty f) : 
  ∀ x y : S, f x = f y := sorry

end constant_function_theorem_l3281_328148


namespace lemonade_mixture_l3281_328189

theorem lemonade_mixture (L : ℝ) : 
  -- First solution composition
  let first_lemonade : ℝ := 20
  let first_carbonated : ℝ := 80
  -- Second solution composition
  let second_lemonade : ℝ := L
  let second_carbonated : ℝ := 55
  -- Mixture composition
  let mixture_carbonated : ℝ := 60
  let mixture_first_solution : ℝ := 20
  -- Conditions
  first_lemonade + first_carbonated = 100 →
  second_lemonade + second_carbonated = 100 →
  mixture_first_solution * first_carbonated / 100 + 
    (100 - mixture_first_solution) * second_carbonated / 100 = mixture_carbonated →
  -- Conclusion
  L = 45 := by
sorry

end lemonade_mixture_l3281_328189


namespace hexagon_shape_partition_ways_l3281_328119

/-- A shape formed by gluing together congruent regular hexagons -/
structure HexagonShape where
  num_hexagons : ℕ
  num_quadrilaterals : ℕ

/-- The number of ways to partition a HexagonShape -/
def partition_ways (shape : HexagonShape) : ℕ :=
  2 ^ shape.num_hexagons

/-- The theorem to prove -/
theorem hexagon_shape_partition_ways :
  ∀ (shape : HexagonShape),
    shape.num_hexagons = 7 →
    shape.num_quadrilaterals = 21 →
    partition_ways shape = 128 := by
  sorry

end hexagon_shape_partition_ways_l3281_328119


namespace simple_interest_problem_l3281_328100

/-- Given a sum at simple interest for 10 years, if increasing the interest rate by 5%
    results in Rs. 200 more interest, then the original sum is Rs. 2000. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 200 → P = 2000 := by
  sorry

end simple_interest_problem_l3281_328100


namespace rectangle_center_line_slope_l3281_328193

/-- The slope of a line passing through the origin and the center of a rectangle with given vertices is 1/5 -/
theorem rectangle_center_line_slope :
  let vertices : List (ℝ × ℝ) := [(1, 0), (9, 0), (1, 2), (9, 2)]
  let center_x : ℝ := (vertices.map Prod.fst).sum / vertices.length
  let center_y : ℝ := (vertices.map Prod.snd).sum / vertices.length
  let slope : ℝ := center_y / center_x
  slope = 1 / 5 := by
  sorry

end rectangle_center_line_slope_l3281_328193


namespace heptagon_diagonals_l3281_328170

-- Define a heptagon
def Heptagon : Nat := 7

-- Define the formula for the number of diagonals in a polygon
def numDiagonals (n : Nat) : Nat := n * (n - 3) / 2

-- Theorem: The number of diagonals in a heptagon is 14
theorem heptagon_diagonals : numDiagonals Heptagon = 14 := by
  sorry

end heptagon_diagonals_l3281_328170


namespace base_five_digits_of_3125_l3281_328143

theorem base_five_digits_of_3125 : ∃ n : ℕ, n = 6 ∧ 
  (∀ k : ℕ, 5^k ≤ 3125 → k + 1 ≤ n) ∧
  (∀ m : ℕ, (∀ k : ℕ, 5^k ≤ 3125 → k + 1 ≤ m) → n ≤ m) :=
by sorry

end base_five_digits_of_3125_l3281_328143


namespace elder_son_toys_l3281_328116

theorem elder_son_toys (total : ℕ) (younger_ratio : ℕ) : 
  total = 240 → younger_ratio = 3 → 
  ∃ (elder : ℕ), elder * (1 + younger_ratio) = total ∧ elder = 60 := by
sorry

end elder_son_toys_l3281_328116


namespace student_selection_l3281_328164

theorem student_selection (boys girls : ℕ) (ways : ℕ) : 
  boys = 15 → 
  girls = 10 → 
  ways = 1050 → 
  ways = (girls.choose 1) * (boys.choose 2) →
  1 + 2 = 3 := by
  sorry

end student_selection_l3281_328164


namespace sum_of_four_consecutive_squares_not_divisible_by_5_or_13_l3281_328135

theorem sum_of_four_consecutive_squares_not_divisible_by_5_or_13 (n : ℤ) :
  ∃ (k : ℤ), k ≠ 0 ∧ ((n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 5 = k ∧
              ((n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 13 = k :=
by sorry

end sum_of_four_consecutive_squares_not_divisible_by_5_or_13_l3281_328135


namespace gcd_lcm_8951_4267_l3281_328158

theorem gcd_lcm_8951_4267 : 
  (Nat.gcd 8951 4267 = 1) ∧ 
  (Nat.lcm 8951 4267 = 38212917) := by
sorry

end gcd_lcm_8951_4267_l3281_328158


namespace probability_seven_tails_l3281_328160

/-- The probability of flipping exactly k tails in n flips of an unfair coin -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 7 tails in 10 flips of an unfair coin with 2/3 probability of tails -/
theorem probability_seven_tails : 
  binomial_probability 10 7 (2/3) = 5120/19683 := by
  sorry

end probability_seven_tails_l3281_328160


namespace simplify_fraction_l3281_328184

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) :
  (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) := by
sorry

end simplify_fraction_l3281_328184


namespace intersection_points_on_circle_l3281_328154

/-- The parabolas y = (x + 1)² and x + 4 = (y - 3)² intersect at four points that lie on a circle --/
theorem intersection_points_on_circle (x y : ℝ) : 
  (y = (x + 1)^2 ∧ x + 4 = (y - 3)^2) →
  ∃ (center : ℝ × ℝ), (x - center.1)^2 + (y - center.2)^2 = 13/2 :=
sorry

end intersection_points_on_circle_l3281_328154


namespace pepik_problem_l3281_328150

def letter_sum (M A T R D E I K U : Nat) : Nat :=
  4*M + 4*A + R + D + 2*T + E + I + K + U

theorem pepik_problem :
  (∀ M A T R D E I K U : Nat,
    M ≤ 9 ∧ A ≤ 9 ∧ T ≤ 9 ∧ R ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ I ≤ 9 ∧ K ≤ 9 ∧ U ≤ 9 ∧
    M ≠ 0 ∧ A ≠ 0 ∧ T ≠ 0 ∧ R ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ I ≠ 0 ∧ K ≠ 0 ∧ U ≠ 0 ∧
    M ≠ A ∧ M ≠ T ∧ M ≠ R ∧ M ≠ D ∧ M ≠ E ∧ M ≠ I ∧ M ≠ K ∧ M ≠ U ∧
    A ≠ T ∧ A ≠ R ∧ A ≠ D ∧ A ≠ E ∧ A ≠ I ∧ A ≠ K ∧ A ≠ U ∧
    T ≠ R ∧ T ≠ D ∧ T ≠ E ∧ T ≠ I ∧ T ≠ K ∧ T ≠ U ∧
    R ≠ D ∧ R ≠ E ∧ R ≠ I ∧ R ≠ K ∧ R ≠ U ∧
    D ≠ E ∧ D ≠ I ∧ D ≠ K ∧ D ≠ U ∧
    E ≠ I ∧ E ≠ K ∧ E ≠ U ∧
    I ≠ K ∧ I ≠ U ∧
    K ≠ U →
    (∀ x : Nat, letter_sum M A T R D E I K U ≤ 103) ∧
    (letter_sum M A T R D E I K U ≠ 50) ∧
    (letter_sum M A T R D E I K U = 59 → (T = 5 ∨ T = 2))) :=
by sorry

end pepik_problem_l3281_328150


namespace sally_lemonade_sales_l3281_328165

/-- Calculates the total number of lemonade cups sold over two weeks -/
def total_lemonade_sales (last_week : ℕ) (increase_percentage : ℕ) : ℕ :=
  let this_week := last_week + last_week * increase_percentage / 100
  last_week + this_week

/-- Proves that given the conditions, Sally sold 46 cups of lemonade in total -/
theorem sally_lemonade_sales : total_lemonade_sales 20 30 = 46 := by
  sorry

end sally_lemonade_sales_l3281_328165


namespace teresas_age_at_birth_l3281_328190

/-- Given the current ages of Teresa and Morio, and Morio's age when their daughter Michiko was born,
    prove that Teresa's age when Michiko was born is 26. -/
theorem teresas_age_at_birth (teresa_current_age morio_current_age morio_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59)
  (h2 : morio_current_age = 71)
  (h3 : morio_age_at_birth = 38) :
  teresa_current_age - (morio_current_age - morio_age_at_birth) = 26 := by
  sorry

end teresas_age_at_birth_l3281_328190


namespace chip_cost_is_correct_l3281_328117

/-- The cost of a bag of chips, given Amber's spending scenario -/
def chip_cost (total_money : ℚ) (candy_cost : ℚ) (candy_ounces : ℚ) (chip_ounces : ℚ) (max_ounces : ℚ) : ℚ :=
  total_money / (max_ounces / chip_ounces)

/-- Theorem stating that the cost of a bag of chips is $1.40 in Amber's scenario -/
theorem chip_cost_is_correct :
  chip_cost 7 1 12 17 85 = (14 : ℚ) / 10 := by
  sorry

end chip_cost_is_correct_l3281_328117


namespace fraction_value_l3281_328118

theorem fraction_value (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) 
  (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end fraction_value_l3281_328118


namespace parabola_equation_l3281_328108

theorem parabola_equation (a : ℝ) (x₀ : ℝ) : 
  (∃ (x : ℝ → ℝ) (y : ℝ → ℝ), 
    (∀ t, x t ^ 2 = a * y t) ∧ 
    (y x₀ = 2) ∧ 
    ((x x₀ - 0) ^ 2 + (y x₀ - a / 4) ^ 2 = 3 ^ 2)) → 
  a = 4 := by
sorry

end parabola_equation_l3281_328108


namespace quadratic_function_unique_form_l3281_328161

/-- A quadratic function f(x) = x^2 + ax + b that intersects the x-axis at (1,0) 
    and has an axis of symmetry at x = 2 is equal to x^2 - 4x + 3. -/
theorem quadratic_function_unique_form 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = x^2 + a*x + b) 
  (h2 : f 1 = 0) 
  (h3 : ∀ x, f (2 + x) = f (2 - x)) : 
  ∀ x, f x = x^2 - 4*x + 3 := by 
sorry

end quadratic_function_unique_form_l3281_328161


namespace digit_206788_is_7_l3281_328144

/-- The sequence of digits formed by concatenating all natural numbers from 1 onwards -/
def digit_sequence : ℕ → ℕ :=
  sorry

/-- The number of digits used to represent all natural numbers up to n -/
def digits_used_up_to (n : ℕ) : ℕ :=
  sorry

/-- The function that returns the digit at a given position in the sequence -/
def digit_at_position (pos : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the 206788th digit in the sequence is 7 -/
theorem digit_206788_is_7 : digit_at_position 206788 = 7 :=
  sorry

end digit_206788_is_7_l3281_328144


namespace responses_always_match_l3281_328121

-- Define the types of inhabitants
inductive Inhabitant : Type
| Knight : Inhabitant
| Liar : Inhabitant

-- Define a function to represent an inhabitant's response
def responds (a b : Inhabitant) : Prop :=
  match a, b with
  | Inhabitant.Knight, Inhabitant.Knight => true
  | Inhabitant.Knight, Inhabitant.Liar => false
  | Inhabitant.Liar, Inhabitant.Knight => true
  | Inhabitant.Liar, Inhabitant.Liar => true

-- Theorem: The responses of two inhabitants about each other are always the same
theorem responses_always_match (a b : Inhabitant) :
  responds a b = responds b a :=
sorry

end responses_always_match_l3281_328121


namespace age_of_other_man_l3281_328126

theorem age_of_other_man (n : ℕ) (avg_increase : ℝ) (age_one_man : ℕ) (avg_age_women : ℝ) :
  n = 8 ∧ 
  avg_increase = 2 ∧ 
  age_one_man = 20 ∧ 
  avg_age_women = 29 →
  ∃ (original_avg : ℝ) (age_other_man : ℕ),
    n * (original_avg + avg_increase) = n * original_avg + 2 * avg_age_women - (age_one_man + age_other_man) ∧
    age_other_man = 22 :=
by sorry

end age_of_other_man_l3281_328126


namespace banana_proportion_after_adding_l3281_328168

/-- Represents a fruit basket with apples and bananas -/
structure FruitBasket where
  apples : ℕ
  bananas : ℕ

/-- Calculates the fraction of bananas in the basket -/
def bananaProportion (basket : FruitBasket) : ℚ :=
  basket.bananas / (basket.apples + basket.bananas)

/-- The initial basket -/
def initialBasket : FruitBasket := ⟨12, 15⟩

/-- The basket after adding 3 bananas -/
def finalBasket : FruitBasket := ⟨initialBasket.apples, initialBasket.bananas + 3⟩

/-- Theorem stating that the proportion of bananas in the final basket is 3/5 -/
theorem banana_proportion_after_adding : bananaProportion finalBasket = 3/5 := by
  sorry


end banana_proportion_after_adding_l3281_328168


namespace journey_distance_l3281_328192

/-- Given a constant speed, if a journey of 120 miles takes 3 hours, 
    then a journey of 5 hours at the same speed covers a distance of 200 miles. -/
theorem journey_distance (speed : ℝ) 
  (h1 : speed * 3 = 120) 
  (h2 : speed > 0) : 
  speed * 5 = 200 := by
  sorry

end journey_distance_l3281_328192


namespace circular_arrangement_pairs_l3281_328197

/-- Represents the number of adjacent pairs of children of the same gender -/
def adjacentPairs (total : Nat) (groups : Nat) : Nat :=
  total - groups

/-- The problem statement -/
theorem circular_arrangement_pairs (boys girls groups : Nat) 
  (h1 : boys = 15)
  (h2 : girls = 20)
  (h3 : adjacentPairs boys groups = (2 : Nat) / (3 : Nat) * adjacentPairs girls groups) :
  boys + girls - (adjacentPairs boys groups + adjacentPairs girls groups) = 10 := by
  sorry

end circular_arrangement_pairs_l3281_328197


namespace probability_two_red_balls_l3281_328174

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def drawn_balls : ℕ := 5

def probability_two_red : ℚ := 10 / 21

theorem probability_two_red_balls :
  (Nat.choose red_balls 2 * Nat.choose white_balls 3) / Nat.choose total_balls drawn_balls = probability_two_red :=
sorry

end probability_two_red_balls_l3281_328174


namespace rectangle_area_18_pairs_l3281_328120

def rectangle_area_18 : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 18}

theorem rectangle_area_18_pairs : 
  rectangle_area_18 = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end rectangle_area_18_pairs_l3281_328120


namespace absolute_difference_of_numbers_l3281_328182

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (product_eq : x * y = 396) : 
  |x - y| = 4 := by
sorry

end absolute_difference_of_numbers_l3281_328182


namespace new_library_capacity_l3281_328167

theorem new_library_capacity 
  (M : ℚ) -- Millicent's books
  (H : ℚ) -- Harold's books
  (G : ℚ) -- Gertrude's books
  (h1 : H = (1/2) * M) -- Harold has 1/2 as many books as Millicent
  (h2 : G = 3 * H) -- Gertrude has 3 times more books than Harold
  : (1/3) * H + (2/5) * G + (1/2) * M = (29/30) * M := by
  sorry

end new_library_capacity_l3281_328167


namespace arithmetic_progression_equality_l3281_328159

theorem arithmetic_progression_equality (n : ℕ) 
  (a b : Fin n → ℕ) 
  (h_n : n ≥ 2018) 
  (h_distinct_a : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_distinct_b : ∀ i j : Fin n, i ≠ j → b i ≠ b j)
  (h_bound_a : ∀ i : Fin n, a i ≤ 5*n)
  (h_bound_b : ∀ i : Fin n, b i ≤ 5*n)
  (h_positive_a : ∀ i : Fin n, a i > 0)
  (h_positive_b : ∀ i : Fin n, b i > 0)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = (i.val - j.val : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) := by
sorry

end arithmetic_progression_equality_l3281_328159


namespace fishing_theorem_l3281_328146

def fishing_problem (caleb_catch dad_catch : ℕ) : Prop :=
  caleb_catch = 2 ∧ 
  dad_catch = 3 * caleb_catch ∧ 
  dad_catch - caleb_catch = 4

theorem fishing_theorem : ∃ caleb_catch dad_catch, fishing_problem caleb_catch dad_catch :=
  sorry

end fishing_theorem_l3281_328146


namespace function_properties_l3281_328136

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

theorem function_properties (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  (∃ a₀ b₀ : ℝ, ∀ x, f a b x = -3 * x^2 - 3 * x + 18) ∧
  (∀ c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) ↔ c ≤ -25/12) ∧
  (∃ M : ℝ, M = -3 ∧ ∀ x > -1, (f a b x - 21) / (x + 1) ≤ M) := by
  sorry

end function_properties_l3281_328136


namespace wills_hourly_rate_l3281_328132

/-- Proof of Will's hourly rate given his work hours and total earnings -/
theorem wills_hourly_rate (monday_hours tuesday_hours total_earnings : ℕ) 
  (h1 : monday_hours = 8)
  (h2 : tuesday_hours = 2)
  (h3 : total_earnings = 80) :
  total_earnings / (monday_hours + tuesday_hours) = 8 := by
  sorry

#check wills_hourly_rate

end wills_hourly_rate_l3281_328132


namespace log_identity_l3281_328149

theorem log_identity : Real.log 2 ^ 3 + 3 * Real.log 2 * Real.log 5 + Real.log 5 ^ 3 = 1 := by
  sorry

end log_identity_l3281_328149


namespace intersection_point_l3281_328176

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = x + 3

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_point :
  ∃ (x y : ℝ), line_equation x y ∧ y_axis x ∧ x = 0 ∧ y = 3 := by sorry

end intersection_point_l3281_328176


namespace intersection_sum_l3281_328188

theorem intersection_sum (a b : ℚ) : 
  (3 = (1/3) * 4 + a) → 
  (4 = (1/3) * 3 + b) → 
  a + b = 14/3 := by
sorry

end intersection_sum_l3281_328188


namespace original_number_is_75_l3281_328111

theorem original_number_is_75 (x : ℝ) : ((x / 2.5) - 10.5) * 0.3 = 5.85 → x = 75 := by
  sorry

end original_number_is_75_l3281_328111


namespace dog_food_total_l3281_328162

/-- Theorem: Given an initial amount of dog food and two additional purchases,
    prove the total amount of dog food. -/
theorem dog_food_total (initial : ℕ) (bag1 : ℕ) (bag2 : ℕ) :
  initial = 15 → bag1 = 15 → bag2 = 10 → initial + bag1 + bag2 = 40 := by
  sorry

end dog_food_total_l3281_328162


namespace factor_sum_l3281_328107

theorem factor_sum (a b : ℤ) : 
  (∀ x, 25 * x^2 - 130 * x - 120 = (5 * x + a) * (5 * x + b)) →
  a + 3 * b = -86 := by
sorry

end factor_sum_l3281_328107


namespace digital_earth_implies_science_technology_expression_l3281_328163

-- Define the concept of Digital Earth
def DigitalEarth : Prop := sorry

-- Define the concept of technological innovation paradigm
def TechnologicalInnovationParadigm : Prop := sorry

-- Define the concept of science and technology as expression of advanced productive forces
def ScienceTechnologyExpression : Prop := sorry

-- Theorem statement
theorem digital_earth_implies_science_technology_expression :
  (DigitalEarth → TechnologicalInnovationParadigm) →
  (TechnologicalInnovationParadigm → ScienceTechnologyExpression) :=
by
  sorry

end digital_earth_implies_science_technology_expression_l3281_328163


namespace vector_triangle_l3281_328183

/-- Given a triangle ABC, a point D on BC such that BD = 3DC, and E the midpoint of AC,
    prove that ED = 1/4 AB + 1/4 AC -/
theorem vector_triangle (A B C D E : ℝ × ℝ) : 
  (∃ (t : ℝ), D = B + t • (C - B) ∧ t = 3/4) →  -- D is on BC with BD = 3DC
  E = A + (1/2 : ℝ) • (C - A) →                 -- E is midpoint of AC
  E - D = (1/4 : ℝ) • (B - A) + (1/4 : ℝ) • (C - A) := by
sorry

end vector_triangle_l3281_328183


namespace decagon_diagonals_l3281_328166

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by sorry

end decagon_diagonals_l3281_328166


namespace bijective_function_decomposition_l3281_328169

theorem bijective_function_decomposition
  (f : ℤ → ℤ) (hf : Function.Bijective f) :
  ∃ (u v : ℤ → ℤ), Function.Bijective u ∧ Function.Bijective v ∧ (∀ x, f x = u x + v x) := by
  sorry

end bijective_function_decomposition_l3281_328169


namespace hall_length_l3281_328105

/-- Given a rectangular hall where the length is 5 meters more than the breadth
    and the area is 750 square meters, prove that the length is 30 meters. -/
theorem hall_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = breadth + 5 →
  area = length * breadth →
  area = 750 →
  length = 30 := by
sorry

end hall_length_l3281_328105


namespace angle_XZY_is_50_l3281_328155

/-- Given a diagram where AB and CD are straight lines -/
structure Diagram where
  /-- The angle AXB is 180 degrees -/
  angle_AXB : ℝ
  /-- The angle CYX is 120 degrees -/
  angle_CYX : ℝ
  /-- The angle YXB is 60 degrees -/
  angle_YXB : ℝ
  /-- The angle AXY is 50 degrees -/
  angle_AXY : ℝ
  /-- AB is a straight line -/
  h_AB_straight : angle_AXB = 180
  /-- CD is a straight line (not directly used but implied) -/
  h_CYX : angle_CYX = 120
  h_YXB : angle_YXB = 60
  h_AXY : angle_AXY = 50

/-- The theorem stating that the angle XZY is 50 degrees -/
theorem angle_XZY_is_50 (d : Diagram) : ∃ x, x = 50 ∧ x = d.angle_AXB - d.angle_CYX + d.angle_YXB - d.angle_AXY :=
  sorry

end angle_XZY_is_50_l3281_328155


namespace equation_solutions_l3281_328180

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x + 6) = 3 * (x - 1) ∧ x = 15) ∧
  (∃ x : ℝ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29) := by
  sorry

end equation_solutions_l3281_328180


namespace definite_integral_f_l3281_328141

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 2

-- State the theorem
theorem definite_integral_f : ∫ x in (0:ℝ)..(1:ℝ), f x = 11/6 := by sorry

end definite_integral_f_l3281_328141


namespace sqrt_nine_subtraction_l3281_328172

theorem sqrt_nine_subtraction : 1 - Real.sqrt 9 = -2 := by sorry

end sqrt_nine_subtraction_l3281_328172


namespace distance_between_buildings_eight_trees_nine_meters_l3281_328151

/-- Given two buildings with trees planted between them, calculate the distance between the buildings. -/
theorem distance_between_buildings (num_trees : ℕ) (tree_spacing : ℕ) : ℕ :=
  (num_trees + 1) * tree_spacing

/-- Prove that with 8 trees planted 1 meter apart, the distance between buildings is 9 meters. -/
theorem eight_trees_nine_meters :
  distance_between_buildings 8 1 = 9 := by
  sorry

end distance_between_buildings_eight_trees_nine_meters_l3281_328151


namespace modified_mindmaster_codes_l3281_328137

/-- The number of possible secret codes in a modified Mindmaster game -/
def secret_codes (num_colors : ℕ) (num_slots : ℕ) : ℕ :=
  num_colors ^ num_slots

/-- Theorem: The number of secret codes in a game with 5 colors and 6 slots is 15625 -/
theorem modified_mindmaster_codes :
  secret_codes 5 6 = 15625 := by
  sorry

end modified_mindmaster_codes_l3281_328137


namespace circle_area_from_polar_equation_l3281_328106

/-- The area of the circle represented by the polar equation r = 3 cos θ - 4 sin θ -/
theorem circle_area_from_polar_equation : 
  let r : ℝ → ℝ := λ θ => 3 * Real.cos θ - 4 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    Real.pi * radius^2 = 25 * Real.pi / 4 :=
by sorry

end circle_area_from_polar_equation_l3281_328106


namespace sum_in_base7_l3281_328122

/-- Converts a base 7 number to base 10 --/
def base7_to_base10 (x : ℕ) : ℕ :=
  (x / 10) * 7 + (x % 10)

/-- Converts a base 10 number to base 7 --/
def base10_to_base7 (x : ℕ) : ℕ :=
  if x < 7 then x
  else (base10_to_base7 (x / 7)) * 10 + (x % 7)

theorem sum_in_base7 :
  base10_to_base7 (base7_to_base10 15 + base7_to_base10 26) = 44 :=
by sorry

end sum_in_base7_l3281_328122


namespace gcd_180_480_l3281_328142

theorem gcd_180_480 : Nat.gcd 180 480 = 60 := by
  sorry

end gcd_180_480_l3281_328142


namespace smallest_five_digit_divisible_by_smallest_primes_l3281_328191

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A function to check if a number is five-digit -/
def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function to check if a number is divisible by all numbers in a list -/
def divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ m ∈ list, n % m = 0

theorem smallest_five_digit_divisible_by_smallest_primes :
  (is_five_digit 11550) ∧ 
  (divisible_by_all 11550 smallest_primes) ∧ 
  (∀ n : Nat, is_five_digit n ∧ divisible_by_all n smallest_primes → 11550 ≤ n) := by
  sorry

#check smallest_five_digit_divisible_by_smallest_primes

end smallest_five_digit_divisible_by_smallest_primes_l3281_328191


namespace student_calculation_mistake_l3281_328171

theorem student_calculation_mistake (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by
sorry

end student_calculation_mistake_l3281_328171


namespace triangle_inequality_l3281_328140

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * Real.sqrt (s * (s - a) * (s - b) * (s - c)) := by
  sorry

end triangle_inequality_l3281_328140


namespace arccos_sqrt3_div2_l3281_328145

theorem arccos_sqrt3_div2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end arccos_sqrt3_div2_l3281_328145


namespace tan_sum_zero_implies_tan_sqrt_three_l3281_328177

open Real

theorem tan_sum_zero_implies_tan_sqrt_three (θ : ℝ) :
  π/4 < θ ∧ θ < π/2 →
  tan θ + tan (2*θ) + tan (3*θ) + tan (4*θ) = 0 →
  tan θ = sqrt 3 := by
sorry

end tan_sum_zero_implies_tan_sqrt_three_l3281_328177


namespace max_sum_of_factors_l3281_328185

theorem max_sum_of_factors (diamond heart : ℕ) : 
  diamond * heart = 48 → (∀ x y : ℕ, x * y = 48 → x + y ≤ diamond + heart) → diamond + heart = 49 :=
by
  sorry

end max_sum_of_factors_l3281_328185


namespace cross_flag_center_area_ratio_l3281_328114

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  centerArea : ℝ
  crossSymmetric : Bool
  crossUniformWidth : Bool
  crossAreaRatio : crossArea = 0.49 * side * side

/-- Theorem: If the cross occupies 49% of the flag's area, then the center square occupies 25.14% of the flag's area -/
theorem cross_flag_center_area_ratio (flag : CrossFlag) :
  flag.crossSymmetric ∧ flag.crossUniformWidth →
  flag.centerArea / (flag.side * flag.side) = 0.2514 := by
  sorry

end cross_flag_center_area_ratio_l3281_328114


namespace algebraic_expression_proof_l3281_328102

-- Define the condition
theorem algebraic_expression_proof (a b : ℝ) (h : a - b + 3 = Real.sqrt 2) :
  (2*a - 2*b + 6)^4 = 64 := by
  sorry

end algebraic_expression_proof_l3281_328102


namespace catrionas_aquarium_l3281_328178

/-- The number of goldfish in Catriona's aquarium -/
def num_goldfish : ℕ := 8

/-- The number of angelfish in Catriona's aquarium -/
def num_angelfish : ℕ := num_goldfish + 4

/-- The number of guppies in Catriona's aquarium -/
def num_guppies : ℕ := 2 * num_angelfish

/-- The total number of fish in Catriona's aquarium -/
def total_fish : ℕ := num_goldfish + num_angelfish + num_guppies

theorem catrionas_aquarium : total_fish = 44 := by
  sorry

end catrionas_aquarium_l3281_328178


namespace feb_7_is_saturday_l3281_328131

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that February 14 is a Saturday, February 7 is also a Saturday -/
theorem feb_7_is_saturday (feb14 : FebruaryDate) 
    (h14 : feb14.day = 14 ∧ feb14.dayOfWeek = DayOfWeek.Saturday) :
    ∃ (feb7 : FebruaryDate), feb7.day = 7 ∧ feb7.dayOfWeek = DayOfWeek.Saturday := by
  sorry

end feb_7_is_saturday_l3281_328131


namespace chord_length_of_intersecting_curves_l3281_328134

/-- The length of the chord formed by the intersection of two curves in polar coordinates -/
theorem chord_length_of_intersecting_curves (C₁ C₂ : ℝ → ℝ → Prop) :
  (∀ ρ θ, C₁ ρ θ ↔ ρ = 2 * Real.sin θ) →
  (∀ ρ θ, C₂ ρ θ ↔ θ = Real.pi / 3) →
  ∃ M N : ℝ × ℝ,
    (∃ ρ₁ θ₁, C₁ ρ₁ θ₁ ∧ M = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁)) ∧
    (∃ ρ₂ θ₂, C₂ ρ₂ θ₂ ∧ M = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂)) ∧
    (∃ ρ₃ θ₃, C₁ ρ₃ θ₃ ∧ N = (ρ₃ * Real.cos θ₃, ρ₃ * Real.sin θ₃)) ∧
    (∃ ρ₄ θ₄, C₂ ρ₄ θ₄ ∧ N = (ρ₄ * Real.cos θ₄, ρ₄ * Real.sin θ₄)) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 3 :=
by sorry

end chord_length_of_intersecting_curves_l3281_328134


namespace harrietts_pennies_l3281_328194

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- The problem statement -/
theorem harrietts_pennies :
  let quarters := 10
  let dimes := 3
  let nickels := 3
  let total_cents := 300  -- $3 in cents
  let other_coins_value := 
    quarters * coin_value "quarter" + 
    dimes * coin_value "dime" + 
    nickels * coin_value "nickel"
  let pennies := total_cents - other_coins_value
  pennies = 5 := by sorry

end harrietts_pennies_l3281_328194


namespace jungkook_has_largest_number_l3281_328152

/-- Given the numbers collected by Yoongi, Yuna, and Jungkook, prove that Jungkook has the largest number. -/
theorem jungkook_has_largest_number (yoongi_number yuna_number : ℕ) : 
  yoongi_number = 4 → 
  yuna_number = 5 → 
  6 + 3 > yoongi_number ∧ 6 + 3 > yuna_number := by
  sorry

#check jungkook_has_largest_number

end jungkook_has_largest_number_l3281_328152


namespace new_observation_count_l3281_328147

theorem new_observation_count (initial_count : ℕ) (initial_avg : ℚ) (new_obs : ℚ) (avg_decrease : ℚ) : 
  initial_count = 6 → 
  initial_avg = 13 → 
  new_obs = 6 → 
  avg_decrease = 1 → 
  (initial_count * initial_avg + new_obs) / (initial_count + 1) = initial_avg - avg_decrease → 
  initial_count + 1 = 7 := by
sorry

end new_observation_count_l3281_328147


namespace female_turtle_percentage_is_60_l3281_328124

/-- Represents the number of turtles in the lake -/
def total_turtles : ℕ := 100

/-- Represents the fraction of male turtles that have stripes -/
def male_stripe_ratio : ℚ := 1 / 4

/-- Represents the number of baby striped male turtles -/
def baby_striped_males : ℕ := 4

/-- Represents the percentage of adult striped male turtles -/
def adult_striped_male_percentage : ℚ := 60 / 100

/-- Calculates the percentage of female turtles in the lake -/
def female_turtle_percentage : ℚ :=
  let total_striped_males : ℚ := baby_striped_males / (1 - adult_striped_male_percentage)
  let total_males : ℚ := total_striped_males / male_stripe_ratio
  let total_females : ℚ := total_turtles - total_males
  (total_females / total_turtles) * 100

theorem female_turtle_percentage_is_60 :
  female_turtle_percentage = 60 := by sorry

end female_turtle_percentage_is_60_l3281_328124


namespace book_pages_calculation_l3281_328113

theorem book_pages_calculation (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_day = 8 → days_to_finish = 72 → pages_per_day * days_to_finish = 576 := by
  sorry

end book_pages_calculation_l3281_328113


namespace fraction_equality_l3281_328173

theorem fraction_equality : (1877^2 - 1862^2) / (1880^2 - 1859^2) = 5/7 := by
  sorry

end fraction_equality_l3281_328173


namespace car_price_increase_l3281_328186

/-- Proves that given a discount and profit on the original price, 
    we can calculate the percentage increase on the discounted price. -/
theorem car_price_increase 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (h1 : discount_rate = 0.40) 
  (h2 : profit_rate = 0.08000000000000007) : 
  let discounted_price := original_price * (1 - discount_rate)
  let selling_price := original_price * (1 + profit_rate)
  let increase_rate := (selling_price - discounted_price) / discounted_price
  increase_rate = 0.8000000000000001 := by
  sorry

end car_price_increase_l3281_328186


namespace place_mat_side_length_l3281_328199

theorem place_mat_side_length (r : ℝ) (n : ℕ) (x : ℝ) : 
  r = 5 →
  n = 8 →
  x = 2 * r * Real.sin (π / (2 * n)) →
  x = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end place_mat_side_length_l3281_328199


namespace runner_ends_in_quadrant_A_l3281_328179

/-- Represents the quadrants of the circular track -/
inductive Quadrant
  | A
  | B
  | C
  | D

/-- Represents a point on the circular track -/
structure Point where
  angle : ℝ  -- Angle in radians from the starting point S

/-- The circular track -/
structure Track where
  circumference : ℝ
  start : Point

/-- A runner on the track -/
structure Runner where
  position : Point
  distance_run : ℝ

/-- Function to determine which quadrant a point is in -/
def point_to_quadrant (p : Point) : Quadrant :=
  sorry

/-- Function to update a runner's position after running a certain distance -/
def update_position (r : Runner) (d : ℝ) (t : Track) : Runner :=
  sorry

/-- Main theorem: After running one mile, the runner ends up in quadrant A -/
theorem runner_ends_in_quadrant_A (t : Track) (r : Runner) :
  t.circumference = 60 ∧ 
  r.position = t.start ∧
  (update_position r 5280 t).position = t.start →
  point_to_quadrant ((update_position r 5280 t).position) = Quadrant.A :=
  sorry

end runner_ends_in_quadrant_A_l3281_328179


namespace pelicans_remaining_l3281_328139

/-- Represents the number of pelicans in Shark Bite Cove -/
def original_pelicans : ℕ := 30

/-- Represents the number of sharks in Pelican Bay -/
def sharks : ℕ := 60

/-- Represents the fraction of pelicans that moved from Shark Bite Cove to Pelican Bay -/
def moved_fraction : ℚ := 1/3

/-- The theorem stating the number of pelicans remaining in Shark Bite Cove -/
theorem pelicans_remaining : 
  sharks = 2 * original_pelicans ∧ 
  (original_pelicans : ℚ) * (1 - moved_fraction) = 20 :=
sorry

end pelicans_remaining_l3281_328139


namespace circles_intersection_sum_l3281_328125

/-- Given two circles intersecting at points (1,3) and (m,1), with their centers 
    on the line 2x-y+c=0, prove that m + c = 1 -/
theorem circles_intersection_sum (m c : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Centers of circles lie on the line 2x-y+c=0
    (2 * x₁ - y₁ + c = 0) ∧ 
    (2 * x₂ - y₂ + c = 0) ∧ 
    -- Circles intersect at (1,3) and (m,1)
    ((x₁ - 1)^2 + (y₁ - 3)^2 = (x₁ - m)^2 + (y₁ - 1)^2) ∧
    ((x₂ - 1)^2 + (y₂ - 3)^2 = (x₂ - m)^2 + (y₂ - 1)^2)) →
  m + c = 1 := by
sorry

end circles_intersection_sum_l3281_328125


namespace daily_forfeit_is_25_l3281_328112

/-- Calculates the daily forfeit amount for idle days given work conditions --/
def calculate_daily_forfeit (daily_pay : ℕ) (total_days : ℕ) (net_earnings : ℕ) (worked_days : ℕ) : ℕ :=
  let idle_days := total_days - worked_days
  let total_possible_earnings := daily_pay * total_days
  let total_forfeit := total_possible_earnings - net_earnings
  total_forfeit / idle_days

/-- Proves that the daily forfeit amount is 25 dollars given the specific work conditions --/
theorem daily_forfeit_is_25 :
  calculate_daily_forfeit 20 25 450 23 = 25 := by
  sorry

end daily_forfeit_is_25_l3281_328112


namespace pyramid_face_area_l3281_328104

theorem pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8) 
  (h_lateral : lateral_edge = 7) : 
  4 * (1/2 * base_edge * Real.sqrt (lateral_edge^2 - (base_edge/2)^2)) = 16 * Real.sqrt 33 := by
  sorry

end pyramid_face_area_l3281_328104


namespace treehouse_paint_calculation_l3281_328157

/-- The amount of paint needed for a treehouse project, including paint loss. -/
def total_paint_needed (white_paint green_paint brown_paint blue_paint : Real)
  (paint_loss_percentage : Real) (oz_to_liter_conversion : Real) : Real :=
  let total_oz := white_paint + green_paint + brown_paint + blue_paint
  let total_oz_with_loss := total_oz * (1 + paint_loss_percentage)
  total_oz_with_loss * oz_to_liter_conversion

/-- Theorem stating the total amount of paint needed is approximately 2.635 liters. -/
theorem treehouse_paint_calculation :
  let white_paint := 20
  let green_paint := 15
  let brown_paint := 34
  let blue_paint := 12
  let paint_loss_percentage := 0.1
  let oz_to_liter_conversion := 0.0295735
  ∃ ε > 0, |total_paint_needed white_paint green_paint brown_paint blue_paint
    paint_loss_percentage oz_to_liter_conversion - 2.635| < ε :=
by sorry

end treehouse_paint_calculation_l3281_328157


namespace divided_square_area_is_eight_l3281_328110

/-- A square with a diagonal divided into three segments -/
structure DividedSquare where
  side : ℝ
  diagonal_length : ℝ
  de : ℝ
  ef : ℝ
  fb : ℝ
  diagonal_sum : de + ef + fb = diagonal_length
  diagonal_pythagoras : 2 * side * side = diagonal_length * diagonal_length

/-- The area of a square with a divided diagonal is 8 -/
theorem divided_square_area_is_eight (s : DividedSquare) 
  (h1 : s.de = 1) (h2 : s.ef = 2) (h3 : s.fb = 1) : s.side * s.side = 8 := by
  sorry

#check divided_square_area_is_eight

end divided_square_area_is_eight_l3281_328110


namespace total_sides_is_118_l3281_328195

/-- The number of sides for each shape --/
def sides_of_shape (shape : String) : ℕ :=
  match shape with
  | "triangle" => 3
  | "square" => 4
  | "pentagon" => 5
  | "hexagon" => 6
  | "heptagon" => 7
  | "octagon" => 8
  | "nonagon" => 9
  | "hendecagon" => 11
  | "circle" => 0
  | _ => 0

/-- The count of each shape in the top layer --/
def top_layer : List (String × ℕ) :=
  [("triangle", 6), ("nonagon", 1), ("heptagon", 2)]

/-- The count of each shape in the middle layer --/
def middle_layer : List (String × ℕ) :=
  [("square", 4), ("hexagon", 2), ("hendecagon", 1)]

/-- The count of each shape in the bottom layer --/
def bottom_layer : List (String × ℕ) :=
  [("octagon", 3), ("circle", 5), ("pentagon", 1), ("nonagon", 1)]

/-- Calculate the total number of sides for a given layer --/
def total_sides_in_layer (layer : List (String × ℕ)) : ℕ :=
  layer.foldl (fun acc (shape, count) => acc + count * sides_of_shape shape) 0

/-- The main theorem stating that the total number of sides is 118 --/
theorem total_sides_is_118 :
  total_sides_in_layer top_layer +
  total_sides_in_layer middle_layer +
  total_sides_in_layer bottom_layer = 118 := by
  sorry

end total_sides_is_118_l3281_328195


namespace paving_rate_per_square_metre_l3281_328101

/-- Given a room with length 5.5 m and width 3.75 m, and a total paving cost of Rs. 28875,
    the rate of paving per square metre is Rs. 1400. -/
theorem paving_rate_per_square_metre 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : total_cost = 28875) :
  total_cost / (length * width) = 1400 := by
sorry


end paving_rate_per_square_metre_l3281_328101


namespace line_properties_l3281_328133

-- Define the lines
def line (A B C : ℝ) := {(x, y) : ℝ × ℝ | A * x + B * y + C = 0}

-- Define when two lines intersect
def intersect (l1 l2 : Set (ℝ × ℝ)) := ∃ p, p ∈ l1 ∧ p ∈ l2

-- Define when two lines are perpendicular
def perpendicular (A1 B1 A2 B2 : ℝ) := A1 * A2 + B1 * B2 = 0

-- Theorem statement
theorem line_properties (A1 B1 C1 A2 B2 C2 : ℝ) :
  (A1 * B2 - A2 * B1 ≠ 0 → intersect (line A1 B1 C1) (line A2 B2 C2)) ∧
  (perpendicular A1 B1 A2 B2 → 
    ∃ (x1 y1 x2 y2 : ℝ), 
      (x1, y1) ∈ line A1 B1 C1 ∧ 
      (x2, y2) ∈ line A2 B2 C2 ∧ 
      (x2 - x1) * (y2 - y1) = 0) :=
sorry

end line_properties_l3281_328133


namespace problem_statement_l3281_328181

open Real

theorem problem_statement : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 3^x₀ + x₀ = 2016) ∧ 
  ¬(∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, |x| - a*x = |-x| - a*(-x)) := by
  sorry

end problem_statement_l3281_328181


namespace remaining_movies_to_watch_l3281_328127

theorem remaining_movies_to_watch 
  (total_movies : ℕ) 
  (watched_movies : ℕ) 
  (total_books : ℕ) 
  (read_books : ℕ) 
  (h1 : total_movies = 12) 
  (h2 : watched_movies = 6) 
  (h3 : total_books = 21) 
  (h4 : read_books = 7) 
  (h5 : watched_movies ≤ total_movies) : 
  total_movies - watched_movies = 6 := by
sorry

end remaining_movies_to_watch_l3281_328127


namespace quadratic_equation_m_value_l3281_328129

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m - 2) * x^(|m|) + x - 1 = a * x^2 + b * x + c) → m = -2 := by
  sorry

end quadratic_equation_m_value_l3281_328129


namespace largest_modulus_of_cubic_root_l3281_328103

theorem largest_modulus_of_cubic_root (a b c d z : ℂ) :
  (Complex.abs a = Complex.abs b) →
  (Complex.abs b = Complex.abs c) →
  (Complex.abs c = Complex.abs d) →
  (Complex.abs a > 0) →
  (a * z^3 + b * z^2 + c * z + d = 0) →
  ∃ t : ℝ, t^3 - t^2 - t - 1 = 0 ∧ Complex.abs z ≤ t :=
by sorry

end largest_modulus_of_cubic_root_l3281_328103


namespace square_side_length_l3281_328198

theorem square_side_length (overlap1 overlap2 overlap3 non_overlap_total : ℝ) 
  (h1 : overlap1 = 2)
  (h2 : overlap2 = 5)
  (h3 : overlap3 = 8)
  (h4 : non_overlap_total = 117)
  (h5 : overlap1 > 0 ∧ overlap2 > 0 ∧ overlap3 > 0 ∧ non_overlap_total > 0) :
  ∃ (side_length : ℝ), 
    side_length = 7 ∧ 
    3 * side_length ^ 2 = non_overlap_total + 2 * (overlap1 + overlap2 + overlap3) :=
by sorry

end square_side_length_l3281_328198


namespace all_real_roots_condition_l3281_328128

theorem all_real_roots_condition (k : ℝ) : 
  (∀ x : ℂ, x^4 - 4*x^3 + 4*x^2 + k*x - 4 = 0 → x.im = 0) ↔ k = -8 := by
sorry

end all_real_roots_condition_l3281_328128


namespace no_perfect_square_in_sequence_l3281_328130

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_perfect_square_in_sequence :
  ∀ n : ℕ, ¬∃ q : ℚ, sequence_a n = q ^ 2 := by
  sorry

end no_perfect_square_in_sequence_l3281_328130


namespace train_crossing_time_l3281_328138

/-- Time for a train to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 350 → train_speed_kmh = 144 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 8.75 := by
  sorry

end train_crossing_time_l3281_328138


namespace cross_section_area_cross_section_area_is_14_l3281_328123

/-- Regular triangular prism with cross-section --/
structure Prism where
  a : ℝ  -- side length of the base
  S_ABC : ℝ  -- area of the base
  base_area_eq : S_ABC = a^2 * Real.sqrt 3 / 4
  D_midpoint : ℝ  -- D is midpoint of AB
  D_midpoint_eq : D_midpoint = a / 2
  K_on_BC : ℝ  -- distance BK
  K_on_BC_eq : K_on_BC = 3 * a / 4
  M_on_AC1 : ℝ  -- height of the prism
  N_on_A1B1 : ℝ  -- distance BG (projection of N)
  N_on_A1B1_eq : N_on_A1B1 = a / 6

/-- Theorem: The area of the cross-section is 14 --/
theorem cross_section_area (p : Prism) : ℝ :=
  let S_np := p.S_ABC * (3/8 - 1/24)  -- area of projection
  let cos_alpha := 1 / Real.sqrt 3
  S_np / cos_alpha

/-- Main theorem: The area of the cross-section is equal to 14 --/
theorem cross_section_area_is_14 (p : Prism) : cross_section_area p = 14 := by
  sorry

end cross_section_area_cross_section_area_is_14_l3281_328123


namespace binomial_square_constant_l3281_328156

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end binomial_square_constant_l3281_328156


namespace polygon_angle_ratio_l3281_328187

theorem polygon_angle_ratio (n : ℕ) : 
  (((n - 2) * 180) / 360 : ℚ) = 9/2 ↔ n = 11 := by sorry

end polygon_angle_ratio_l3281_328187


namespace unique_fixed_point_l3281_328115

noncomputable def F (a b c : ℝ) (x y z : ℝ) : ℝ × ℝ × ℝ :=
  ((Real.sqrt (c^2 + z^2) - z + Real.sqrt (c^2 + y^2) - y) / 2,
   (Real.sqrt (b^2 + z^2) - z + Real.sqrt (b^2 + x^2) - x) / 2,
   (Real.sqrt (a^2 + x^2) - x + Real.sqrt (a^2 + y^2) - y) / 2)

theorem unique_fixed_point (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! p : ℝ × ℝ × ℝ, F a b c p.1 p.2.1 p.2.2 = p ∧ p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 :=
by sorry

end unique_fixed_point_l3281_328115


namespace exam_scores_l3281_328196

theorem exam_scores (average : ℝ) (difference : ℝ) 
  (h_average : average = 98) 
  (h_difference : difference = 2) : 
  ∃ (chinese math : ℝ), 
    chinese + math = 2 * average ∧ 
    math = chinese + difference ∧ 
    chinese = 97 ∧ 
    math = 99 := by
  sorry

end exam_scores_l3281_328196


namespace trillion_scientific_notation_l3281_328175

theorem trillion_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1 ∧ n = 12 ∧ 1000000000000 = a * (10 : ℝ) ^ n :=
by
  sorry

end trillion_scientific_notation_l3281_328175
