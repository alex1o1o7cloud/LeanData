import Mathlib

namespace NUMINAMATH_CALUDE_integral_one_plus_sin_l3866_386611

theorem integral_one_plus_sin : ∫ x in -π..π, (1 + Real.sin x) = 2 * π := by sorry

end NUMINAMATH_CALUDE_integral_one_plus_sin_l3866_386611


namespace NUMINAMATH_CALUDE_speed_ratio_l3866_386639

/-- Two perpendicular lines intersecting at O with points A and B moving along them -/
structure PointMovement where
  O : ℝ × ℝ
  speedA : ℝ
  speedB : ℝ
  initialDistB : ℝ
  time1 : ℝ
  time2 : ℝ

/-- The conditions of the problem -/
def problem_conditions (pm : PointMovement) : Prop :=
  pm.O = (0, 0) ∧
  pm.speedA > 0 ∧
  pm.speedB > 0 ∧
  pm.initialDistB = 500 ∧
  pm.time1 = 2 ∧
  pm.time2 = 10 ∧
  pm.speedA * pm.time1 = pm.initialDistB - pm.speedB * pm.time1 ∧
  pm.speedA * pm.time2 = pm.speedB * pm.time2 - pm.initialDistB

/-- The theorem to be proved -/
theorem speed_ratio (pm : PointMovement) :
  problem_conditions pm → pm.speedA / pm.speedB = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_l3866_386639


namespace NUMINAMATH_CALUDE_circle_equation_l3866_386620

/-- A circle C in the polar coordinate system -/
structure PolarCircle where
  /-- The point through which the circle passes -/
  passingPoint : (ℝ × ℝ)
  /-- The line equation whose intersection with polar axis determines the circle's center -/
  centerLine : ℝ → ℝ → Prop

/-- The polar equation of a circle -/
def polarEquation (c : PolarCircle) (ρ θ : ℝ) : Prop := sorry

theorem circle_equation (c : PolarCircle) (h1 : c.passingPoint = (Real.sqrt 2, π/4)) 
  (h2 : c.centerLine = fun ρ θ ↦ ρ * Real.sin (θ - π/3) = -Real.sqrt 3/2) :
  polarEquation c = fun ρ θ ↦ ρ = 2 * Real.cos θ := by sorry

end NUMINAMATH_CALUDE_circle_equation_l3866_386620


namespace NUMINAMATH_CALUDE_percentage_problem_l3866_386640

theorem percentage_problem (p : ℝ) : p = 60 ↔ 180 * (1/3) - (p * 180 * (1/3) / 100) = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3866_386640


namespace NUMINAMATH_CALUDE_base12_remainder_is_4_l3866_386627

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of the number --/
def base12Number : List Nat := [1, 5, 3, 4]

/-- The theorem stating that the remainder of the base-12 number divided by 9 is 4 --/
theorem base12_remainder_is_4 : 
  (base12ToBase10 base12Number) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base12_remainder_is_4_l3866_386627


namespace NUMINAMATH_CALUDE_germination_rate_proof_l3866_386642

/-- The relative frequency of germinating seeds -/
def relative_frequency_germinating_seeds (total_seeds : ℕ) (non_germinating_seeds : ℕ) : ℚ :=
  (total_seeds - non_germinating_seeds : ℚ) / total_seeds

/-- Theorem: The relative frequency of germinating seeds in a sample of 1000 seeds, 
    where 90 seeds did not germinate, is equal to 0.91 -/
theorem germination_rate_proof :
  relative_frequency_germinating_seeds 1000 90 = 91 / 100 := by
  sorry

end NUMINAMATH_CALUDE_germination_rate_proof_l3866_386642


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3866_386683

def i : ℂ := Complex.I

theorem point_in_fourth_quadrant :
  let z : ℂ := (5 - i) / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3866_386683


namespace NUMINAMATH_CALUDE_smallest_factorial_divisor_l3866_386660

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_factorial_divisor (n : ℕ) (h1 : n > 1) :
  (∀ k : ℕ, k > 1 ∧ k < 7 → ¬(factorial k % n = 0)) ∧ (factorial 7 % n = 0) →
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorial_divisor_l3866_386660


namespace NUMINAMATH_CALUDE_dunkers_lineup_count_l3866_386632

theorem dunkers_lineup_count (n : ℕ) (k : ℕ) (a : ℕ) (z : ℕ) : 
  n = 15 → k = 5 → a ≠ z → a ≤ n → z ≤ n →
  (Nat.choose (n - 2) (k - 1) * 2 + Nat.choose (n - 2) k) = 2717 :=
by sorry

end NUMINAMATH_CALUDE_dunkers_lineup_count_l3866_386632


namespace NUMINAMATH_CALUDE_no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four_l3866_386616

theorem no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four 
  (p : Nat) (hp : Prime p) (hp_cong : p % 4 = 3) :
  ∀ n : Nat, n > 0 → ¬∃ x y : Nat, x > 0 ∧ y > 0 ∧ x^2 + y^2 = p^n :=
by sorry

end NUMINAMATH_CALUDE_no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four_l3866_386616


namespace NUMINAMATH_CALUDE_bob_sandwich_options_l3866_386665

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 6

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 4

/-- Represents whether Bob orders sandwiches with turkey and Swiss cheese. -/
def orders_turkey_swiss : Bool := false

/-- Represents whether Bob orders sandwiches with multigrain bread and turkey. -/
def orders_multigrain_turkey : Bool := false

/-- Calculates the number of sandwiches Bob can order. -/
def num_bob_sandwiches : ℕ := 
  num_breads * num_meats * num_cheeses - 
  (if orders_turkey_swiss then 0 else num_breads) - 
  (if orders_multigrain_turkey then 0 else num_cheeses)

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : num_bob_sandwiches = 111 := by
  sorry

end NUMINAMATH_CALUDE_bob_sandwich_options_l3866_386665


namespace NUMINAMATH_CALUDE_no_primes_satisfy_equation_l3866_386661

theorem no_primes_satisfy_equation :
  ∀ (p q : ℕ) (n : ℕ+), 
    Prime p → Prime q → p ≠ q → p^(q-1) - q^(p-1) ≠ 4*(n:ℕ)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_primes_satisfy_equation_l3866_386661


namespace NUMINAMATH_CALUDE_linear_function_properties_l3866_386624

def f (x : ℝ) := -2 * x + 1

theorem linear_function_properties :
  (∀ x y, x < y → f x > f y) ∧  -- decreasing
  (∀ x, f x - (-2 * x) = 1) ∧  -- parallel to y = -2x
  (f 0 = 1) ∧  -- intersection with y-axis
  (∃ x y z, x > 0 ∧ y < 0 ∧ z > 0 ∧ f x > 0 ∧ f y < 0 ∧ f z < 0) :=  -- passes through 1st, 2nd, and 4th quadrants
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3866_386624


namespace NUMINAMATH_CALUDE_total_revenue_proof_l3866_386684

def sneakers_price : ℝ := 80
def sandals_price : ℝ := 60
def boots_price : ℝ := 120

def sneakers_discount : ℝ := 0.25
def sandals_discount : ℝ := 0.35
def boots_discount : ℝ := 0.40

def sneakers_quantity : ℕ := 2
def sandals_quantity : ℕ := 4
def boots_quantity : ℕ := 11

def discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

def revenue (price discount quantity : ℝ) : ℝ :=
  discounted_price price discount * quantity

theorem total_revenue_proof :
  revenue sneakers_price sneakers_discount (sneakers_quantity : ℝ) +
  revenue sandals_price sandals_discount (sandals_quantity : ℝ) +
  revenue boots_price boots_discount (boots_quantity : ℝ) = 1068 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_proof_l3866_386684


namespace NUMINAMATH_CALUDE_system_inequality_equivalence_l3866_386601

theorem system_inequality_equivalence (x y m : ℝ) :
  (x - 2*y = 1 ∧ 2*x + y = 4*m) → (x + 3*y < 6 ↔ m < 7/4) := by
  sorry

end NUMINAMATH_CALUDE_system_inequality_equivalence_l3866_386601


namespace NUMINAMATH_CALUDE_division_problem_l3866_386638

theorem division_problem : ∃ x : ℝ, (x / 1.33 = 48) ↔ (x = 63.84) := by sorry

end NUMINAMATH_CALUDE_division_problem_l3866_386638


namespace NUMINAMATH_CALUDE_girl_multiplication_problem_l3866_386687

theorem girl_multiplication_problem (mistake_factor : ℕ) (difference : ℕ) (base : ℕ) (correct_factor : ℕ) : 
  mistake_factor = 34 →
  difference = 1233 →
  base = 137 →
  base * correct_factor = base * mistake_factor + difference →
  correct_factor = 43 := by
sorry

end NUMINAMATH_CALUDE_girl_multiplication_problem_l3866_386687


namespace NUMINAMATH_CALUDE_problem_solution_l3866_386692

/-- The function f(x) defined in the problem -/
noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

/-- The theorem statement -/
theorem problem_solution :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x > 0 → x + 2 * f a b x - 3 = 0 → x = 1) ∧
    (a = 1 ∧ b = 1) ∧
    (∀ k x : ℝ, k ≤ 0 → x > 0 → x ≠ 1 → f a b x > Real.log x / (x - 1) + k / x) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3866_386692


namespace NUMINAMATH_CALUDE_polynomial_difference_simplification_l3866_386669

/-- The difference of two polynomials is equal to a simplified polynomial. -/
theorem polynomial_difference_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 5 * x^3 + x^2 + 7) - 
  (x^6 + 4 * x^5 + 2 * x^4 - x^3 + x^2 + 8) = 
  x^6 - x^5 - x^4 + 6 * x^3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_simplification_l3866_386669


namespace NUMINAMATH_CALUDE_rectangle_area_l3866_386659

theorem rectangle_area (l w : ℕ) : 
  l * l + w * w = 17 * 17 →  -- diagonal is 17 cm
  2 * l + 2 * w = 46 →       -- perimeter is 46 cm
  l * w = 120 :=             -- area is 120 cm²
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3866_386659


namespace NUMINAMATH_CALUDE_second_person_average_pages_per_day_l3866_386613

theorem second_person_average_pages_per_day 
  (summer_days : ℕ) 
  (books_read : ℕ) 
  (avg_pages_per_book : ℕ) 
  (second_person_percentage : ℚ) 
  (h1 : summer_days = 80)
  (h2 : books_read = 60)
  (h3 : avg_pages_per_book = 320)
  (h4 : second_person_percentage = 3/4) : 
  (books_read * avg_pages_per_book * second_person_percentage) / summer_days = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_person_average_pages_per_day_l3866_386613


namespace NUMINAMATH_CALUDE_sum_of_squares_geq_sum_of_products_inequality_of_square_roots_l3866_386629

-- Statement 1
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) : 
  a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by sorry

-- Statement 2
theorem inequality_of_square_roots : 
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_geq_sum_of_products_inequality_of_square_roots_l3866_386629


namespace NUMINAMATH_CALUDE_balloon_count_theorem_l3866_386645

/-- Represents the number of balloons a person has for each color -/
structure BalloonCount where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- The total number of balloons for all people -/
def totalBalloons (people : List BalloonCount) : BalloonCount :=
  { blue := people.foldl (fun acc p => acc + p.blue) 0,
    red := people.foldl (fun acc p => acc + p.red) 0,
    green := people.foldl (fun acc p => acc + p.green) 0,
    yellow := people.foldl (fun acc p => acc + p.yellow) 0 }

theorem balloon_count_theorem (joan melanie eric : BalloonCount)
  (h_joan : joan = { blue := 40, red := 30, green := 0, yellow := 0 })
  (h_melanie : melanie = { blue := 41, red := 0, green := 20, yellow := 0 })
  (h_eric : eric = { blue := 0, red := 25, green := 0, yellow := 15 }) :
  totalBalloons [joan, melanie, eric] = { blue := 81, red := 55, green := 20, yellow := 15 } := by
  sorry

#check balloon_count_theorem

end NUMINAMATH_CALUDE_balloon_count_theorem_l3866_386645


namespace NUMINAMATH_CALUDE_contrapositive_equality_l3866_386602

theorem contrapositive_equality (a b : ℝ) : 
  (¬(a = 0 → a * b = 0)) ↔ (a * b ≠ 0 → a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equality_l3866_386602


namespace NUMINAMATH_CALUDE_volume_equality_l3866_386621

/-- The region R₁ bounded by x² = 4y, x² = -4y, x = 4, and x = -4 -/
def R₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2 ∨ p.1 = 4 ∨ p.1 = -4}

/-- The region R₂ satisfying x² - y² ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def R₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 ≤ 16 ∧ p.1^2 + (p.2 - 2)^2 ≥ 4 ∧ p.1^2 + (p.2 + 2)^2 ≥ 4}

/-- The volume V₁ obtained by rotating R₁ about the y-axis -/
noncomputable def V₁ : ℝ := sorry

/-- The volume V₂ obtained by rotating R₂ about the y-axis -/
noncomputable def V₂ : ℝ := sorry

/-- The theorem stating that V₁ equals V₂ -/
theorem volume_equality : V₁ = V₂ := by sorry

end NUMINAMATH_CALUDE_volume_equality_l3866_386621


namespace NUMINAMATH_CALUDE_quadratic_roots_average_l3866_386694

theorem quadratic_roots_average (d : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ 3 * x^2 - 9 * x + d = 0 ∧ 3 * y^2 - 9 * y + d = 0) :
  (∃ x y : ℝ, x ≠ y ∧ 3 * x^2 - 9 * x + d = 0 ∧ 3 * y^2 - 9 * y + d = 0 ∧ (x + y) / 2 = 1.5) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_average_l3866_386694


namespace NUMINAMATH_CALUDE_intersection_complement_A_and_B_l3866_386604

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define function f
def f : A → ℝ := fun x ↦ 2 - x

-- Define the range of f as B
def B : Set ℝ := Set.range f

-- Theorem statement
theorem intersection_complement_A_and_B :
  (Set.univ \ A) ∩ B = Set.Ioo 1 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_A_and_B_l3866_386604


namespace NUMINAMATH_CALUDE_maria_savings_percentage_l3866_386674

/-- Represents the "sundown deal" discount structure -/
structure SundownDeal where
  regular_price : ℝ
  second_pair_discount : ℝ
  additional_pair_discount : ℝ

/-- Calculates the total cost and savings for a given number of pairs -/
def calculate_deal (deal : SundownDeal) (num_pairs : ℕ) : ℝ × ℝ :=
  let regular_total := deal.regular_price * num_pairs
  let discounted_total := 
    if num_pairs ≥ 1 then deal.regular_price else 0 +
    if num_pairs ≥ 2 then deal.regular_price * (1 - deal.second_pair_discount) else 0 +
    if num_pairs > 2 then deal.regular_price * (1 - deal.additional_pair_discount) * (num_pairs - 2) else 0
  let savings := regular_total - discounted_total
  (discounted_total, savings)

/-- Theorem stating that Maria's savings percentage is 42% -/
theorem maria_savings_percentage (deal : SundownDeal) 
  (h1 : deal.regular_price = 60)
  (h2 : deal.second_pair_discount = 0.3)
  (h3 : deal.additional_pair_discount = 0.6) :
  let (_, savings) := calculate_deal deal 5
  let regular_total := deal.regular_price * 5
  (savings / regular_total) * 100 = 42 := by
  sorry


end NUMINAMATH_CALUDE_maria_savings_percentage_l3866_386674


namespace NUMINAMATH_CALUDE_mary_sugar_amount_l3866_386671

/-- The amount of sugar required by the recipe in cups -/
def total_sugar : ℕ := 14

/-- The amount of sugar Mary still needs to add in cups -/
def sugar_to_add : ℕ := 12

/-- The amount of sugar Mary has already put in -/
def sugar_already_added : ℕ := total_sugar - sugar_to_add

theorem mary_sugar_amount : sugar_already_added = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_amount_l3866_386671


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3866_386623

theorem trigonometric_simplification (α : Real) :
  (1 - 2 * Real.sin α ^ 2) / (2 * Real.tan (5 * Real.pi / 4 + α) * Real.cos (Real.pi / 4 + α) ^ 2) -
  Real.tan α + Real.sin (Real.pi / 2 + α) - Real.cos (α - Real.pi / 2) =
  (2 * Real.sqrt 2 * Real.cos (Real.pi / 4 + α) * Real.cos (α / 2) ^ 2) / Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3866_386623


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_180_l3866_386606

/-- The sum of an arithmetic sequence with first term 30, common difference 10, and 4 terms -/
def arithmeticSum : ℕ := sorry

/-- The first term of the sequence -/
def firstTerm : ℕ := 30

/-- The common difference between consecutive terms -/
def commonDifference : ℕ := 10

/-- The number of terms in the sequence -/
def numberOfTerms : ℕ := 4

/-- Theorem stating that the sum of the arithmetic sequence is 180 -/
theorem arithmetic_sum_equals_180 : arithmeticSum = 180 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_180_l3866_386606


namespace NUMINAMATH_CALUDE_divisibility_problem_l3866_386617

theorem divisibility_problem (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.gcd b c = 36)
  (h3 : Nat.gcd c d = 54)
  (h4 : 70 < Nat.gcd d a ∧ Nat.gcd d a < 100) :
  13 ∣ a.val := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3866_386617


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3866_386657

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -1 ∧ 
  (3 + Real.sqrt 45 - 3)^2 = c^2 ∧ 
  (6 - 3)^2 = a^2 ∧ 
  b^2 = c^2 - a^2 → 
  h + k + a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3866_386657


namespace NUMINAMATH_CALUDE_power_mod_seven_l3866_386678

theorem power_mod_seven : 2^19 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seven_l3866_386678


namespace NUMINAMATH_CALUDE_intersection_P_complement_M_l3866_386679

theorem intersection_P_complement_M (U : Set ℤ) (M P : Set ℤ) : 
  U = Set.univ ∧ 
  M = {1, 2} ∧ 
  P = {-2, -1, 0, 1, 2} →
  P ∩ (U \ M) = {-2, -1, 0} := by
sorry

end NUMINAMATH_CALUDE_intersection_P_complement_M_l3866_386679


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l3866_386685

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l3866_386685


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l3866_386693

/-- Given a rectangular metallic sheet with length 48 meters, 
    from which squares of side 8 meters are cut from each corner to form a box,
    if the resulting box has a volume of 5632 cubic meters,
    then the width of the original sheet is 38 meters. -/
theorem metallic_sheet_width :
  ∀ (w : ℝ), 
    w > 0 →
    (48 - 2 * 8) * (w - 2 * 8) * 8 = 5632 →
    w = 38 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l3866_386693


namespace NUMINAMATH_CALUDE_set_relationship_l3866_386675

-- Define the sets M, P, and S
def M : Set ℤ := {x | ∃ k : ℤ, x = 3*k - 2}
def P : Set ℤ := {y | ∃ n : ℤ, y = 3*n + 1}
def S : Set ℤ := {z | ∃ m : ℤ, z = 6*m + 1}

-- State the theorem
theorem set_relationship : S ⊆ P ∧ P = M := by sorry

end NUMINAMATH_CALUDE_set_relationship_l3866_386675


namespace NUMINAMATH_CALUDE_lisa_weight_l3866_386654

theorem lisa_weight (amy lisa : ℝ) 
  (h1 : amy + lisa = 240)
  (h2 : lisa - amy = lisa / 3) : 
  lisa = 144 := by sorry

end NUMINAMATH_CALUDE_lisa_weight_l3866_386654


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3866_386608

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3866_386608


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3866_386649

/-- Given two adjacent points (1,2) and (4,6) on a square in a Cartesian coordinate plane,
    the area of the square is 25. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3866_386649


namespace NUMINAMATH_CALUDE_five_thirteenths_repeating_decimal_sum_l3866_386609

theorem five_thirteenths_repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d + 
    (0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d) / 999999 →
  c + d = 11 :=
by sorry

end NUMINAMATH_CALUDE_five_thirteenths_repeating_decimal_sum_l3866_386609


namespace NUMINAMATH_CALUDE_shoe_multiple_l3866_386673

theorem shoe_multiple (jacob edward brian : ℕ) : 
  jacob = edward / 2 →
  brian = 22 →
  jacob + edward + brian = 121 →
  edward / brian = 3 :=
by sorry

end NUMINAMATH_CALUDE_shoe_multiple_l3866_386673


namespace NUMINAMATH_CALUDE_rice_mixture_price_l3866_386682

/-- Proves that mixing rice at given prices in a specific ratio results in the desired mixture price -/
theorem rice_mixture_price (price1 price2 mixture_price : ℚ) (ratio1 ratio2 : ℕ) : 
  price1 = 31/10 ∧ price2 = 36/10 ∧ mixture_price = 13/4 ∧ ratio1 = 3 ∧ ratio2 = 7 →
  (ratio1 : ℚ) * price1 + (ratio2 : ℚ) * price2 = (ratio1 + ratio2 : ℚ) * mixture_price :=
by
  sorry

#check rice_mixture_price

end NUMINAMATH_CALUDE_rice_mixture_price_l3866_386682


namespace NUMINAMATH_CALUDE_jerry_debt_payment_l3866_386653

/-- Jerry's debt payment problem -/
theorem jerry_debt_payment (total_debt : ℕ) (remaining_debt : ℕ) (payment_two_months_ago : ℕ) 
  (h1 : total_debt = 50)
  (h2 : remaining_debt = 23)
  (h3 : payment_two_months_ago = 12)
  (h4 : total_debt > remaining_debt)
  (h5 : total_debt - remaining_debt > payment_two_months_ago) :
  ∃ (payment_last_month : ℕ), 
    payment_last_month - payment_two_months_ago = 3 ∧
    payment_last_month > payment_two_months_ago ∧
    payment_last_month + payment_two_months_ago = total_debt - remaining_debt :=
by
  sorry


end NUMINAMATH_CALUDE_jerry_debt_payment_l3866_386653


namespace NUMINAMATH_CALUDE_complex_power_modulus_l3866_386636

theorem complex_power_modulus : Complex.abs ((2 + 2*Complex.I)^6) = 512 := by sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l3866_386636


namespace NUMINAMATH_CALUDE_no_integer_square_root_l3866_386689

theorem no_integer_square_root : 
  ¬ ∃ (x : ℤ), ∃ (y : ℤ), x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_l3866_386689


namespace NUMINAMATH_CALUDE_range_of_a_l3866_386686

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + (1/2 : ℝ) > 0) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3866_386686


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l3866_386634

/-- The number of ways to distribute n teachers to k schools --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n teachers to k schools,
    with each school receiving at least 1 teacher --/
def distributeAtLeastOne (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 150 ways to distribute 5 teachers to 3 schools,
    with each school receiving at least 1 teacher --/
theorem distribute_five_to_three :
  distributeAtLeastOne 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l3866_386634


namespace NUMINAMATH_CALUDE_problem_solution_l3866_386610

theorem problem_solution (a z : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * z) : z = 2205 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3866_386610


namespace NUMINAMATH_CALUDE_product_of_digits_5432_base8_l3866_386612

/-- Convert a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem product_of_digits_5432_base8 :
  productOfList (toBase8 5432) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_5432_base8_l3866_386612


namespace NUMINAMATH_CALUDE_jessica_fraction_proof_l3866_386695

/-- Represents Jessica's collection of quarters -/
structure QuarterCollection where
  total : ℕ
  from_1790s : ℕ

/-- The fraction of quarters from states admitted in 1790-1799 -/
def fraction_from_1790s (c : QuarterCollection) : ℚ :=
  c.from_1790s / c.total

/-- Jessica's actual collection -/
def jessica_collection : QuarterCollection :=
  { total := 30, from_1790s := 16 }

theorem jessica_fraction_proof :
  fraction_from_1790s jessica_collection = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_jessica_fraction_proof_l3866_386695


namespace NUMINAMATH_CALUDE_cookie_ratio_l3866_386668

def cookie_problem (clementine_cookies jake_cookies tory_cookies : ℕ) 
  (price_per_cookie total_money : ℚ) : Prop :=
  clementine_cookies = 72 ∧
  jake_cookies = 2 * clementine_cookies ∧
  price_per_cookie = 2 ∧
  total_money = 648 ∧
  price_per_cookie * (clementine_cookies + jake_cookies + tory_cookies) = total_money ∧
  tory_cookies * 2 = clementine_cookies + jake_cookies

theorem cookie_ratio (clementine_cookies jake_cookies tory_cookies : ℕ) 
  (price_per_cookie total_money : ℚ) :
  cookie_problem clementine_cookies jake_cookies tory_cookies price_per_cookie total_money →
  tory_cookies * 2 = clementine_cookies + jake_cookies :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3866_386668


namespace NUMINAMATH_CALUDE_problem_l3866_386652

def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem problem (n : ℕ) (d : ℕ → ℕ) :
  n > 0 ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 15 → d i < d j) ∧
  (∀ i, 1 ≤ i ∧ i ≤ 15 → is_divisor (d i) n) ∧
  d 1 = 1 ∧
  n = d 13 + d 14 + d 15 ∧
  (d 5 + 1)^3 = d 15 + 1 →
  n = 1998 := by
sorry

end NUMINAMATH_CALUDE_problem_l3866_386652


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_two_l3866_386618

theorem opposite_of_sqrt_two : -(Real.sqrt 2) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_two_l3866_386618


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l3866_386637

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l3866_386637


namespace NUMINAMATH_CALUDE_quadratic_inequality_relation_l3866_386672

theorem quadratic_inequality_relation :
  (∀ x : ℝ, x > 3 → x^2 - 2*x - 3 > 0) ∧
  (∃ x : ℝ, x^2 - 2*x - 3 > 0 ∧ ¬(x > 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relation_l3866_386672


namespace NUMINAMATH_CALUDE_kelly_textbook_weight_difference_l3866_386631

/-- The weight difference between Kelly's chemistry and geometry textbooks -/
theorem kelly_textbook_weight_difference :
  let chemistry_weight : ℚ := 7125 / 1000
  let geometry_weight : ℚ := 625 / 1000
  chemistry_weight - geometry_weight = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_kelly_textbook_weight_difference_l3866_386631


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3866_386630

theorem inequality_equivalence (x : ℝ) : x * (x^2 + 1) > (x + 1) * (x^2 - x + 1) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3866_386630


namespace NUMINAMATH_CALUDE_framed_painting_ratio_approaches_one_l3866_386600

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framedDimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.side_frame_width,
   fp.painting_height + 6 * fp.side_frame_width)

/-- Calculates the area of the framed painting -/
def framedArea (fp : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions fp
  w * h

/-- Theorem: The ratio of dimensions of the framed painting approaches 1:1 -/
theorem framed_painting_ratio_approaches_one (ε : ℝ) (hε : ε > 0) :
  ∃ (fp : FramedPainting),
    fp.painting_width = 30 ∧
    fp.painting_height = 20 ∧
    framedArea fp = fp.painting_width * fp.painting_height ∧
    let (w, h) := framedDimensions fp
    |w / h - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_approaches_one_l3866_386600


namespace NUMINAMATH_CALUDE_max_students_distribution_l3866_386622

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : 
  Nat.gcd pens pencils = 91 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3866_386622


namespace NUMINAMATH_CALUDE_complex_modulus_l3866_386680

theorem complex_modulus (Z : ℂ) (h : Z * Complex.I = 2 + Complex.I) : Complex.abs Z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3866_386680


namespace NUMINAMATH_CALUDE_population_average_age_l3866_386633

theorem population_average_age 
  (ratio_women_men : ℚ) 
  (avg_age_women : ℝ) 
  (avg_age_men : ℝ) 
  (h1 : ratio_women_men = 7 / 5) 
  (h2 : avg_age_women = 38) 
  (h3 : avg_age_men = 36) : 
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 37 + 1/6 := by
sorry

end NUMINAMATH_CALUDE_population_average_age_l3866_386633


namespace NUMINAMATH_CALUDE_sum_of_factorization_coefficients_l3866_386690

theorem sum_of_factorization_coefficients (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 20 = (x + a) * (x + b)) →
  (∀ x : ℝ, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factorization_coefficients_l3866_386690


namespace NUMINAMATH_CALUDE_convention_handshakes_l3866_386635

theorem convention_handshakes (twin_sets triplet_sets : ℕ) 
  (h1 : twin_sets = 10)
  (h2 : triplet_sets = 7)
  (h3 : ∀ t : ℕ, t ≤ twin_sets → (t * 2 - 2) * 2 = t * 2 * (t * 2 - 2))
  (h4 : ∀ t : ℕ, t ≤ triplet_sets → (t * 3 - 3) * 3 = t * 3 * (t * 3 - 3))
  (h5 : ∀ t : ℕ, t ≤ twin_sets → (t * 2) * (2 * triplet_sets) = 3 * (t * 2) * triplet_sets)
  (h6 : ∀ t : ℕ, t ≤ triplet_sets → (t * 3) * (2 * twin_sets) = 3 * (t * 3) * twin_sets) :
  ((twin_sets * 2) * ((twin_sets * 2) - 2)) / 2 +
  ((triplet_sets * 3) * ((triplet_sets * 3) - 3)) / 2 +
  (twin_sets * 2) * (2 * triplet_sets) / 3 +
  (triplet_sets * 3) * (2 * twin_sets) / 3 = 922 := by
sorry

end NUMINAMATH_CALUDE_convention_handshakes_l3866_386635


namespace NUMINAMATH_CALUDE_f_min_value_l3866_386607

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |5 - x|

-- State the theorem
theorem f_min_value :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 3) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l3866_386607


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l3866_386614

/-- Given a point A(3,1) in a Cartesian coordinate system, 
    its symmetric point with respect to the y-axis has coordinates (-3,1). -/
theorem symmetric_point_wrt_y_axis : 
  let A : ℝ × ℝ := (3, 1)
  let symmetric_point := (-A.1, A.2)
  symmetric_point = (-3, 1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l3866_386614


namespace NUMINAMATH_CALUDE_shared_vertex_angle_is_84_l3866_386697

/-- The angle between an edge of an equilateral triangle and an edge of a regular pentagon,
    when both shapes are inscribed in a circle and share a common vertex. -/
def shared_vertex_angle : ℝ := 84

/-- An equilateral triangle inscribed in a circle -/
structure EquilateralTriangleInCircle :=
  (vertices : Fin 3 → ℝ × ℝ)
  (is_equilateral : ∀ i j : Fin 3, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
  (on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin 3, dist (vertices i) center = radius)

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle :=
  (vertices : Fin 5 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 5, dist (vertices i) (vertices ((i + 1) % 5)) = dist (vertices j) (vertices ((j + 1) % 5)))
  (on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin 5, dist (vertices i) center = radius)

theorem shared_vertex_angle_is_84 
  (triangle : EquilateralTriangleInCircle) 
  (pentagon : RegularPentagonInCircle) 
  (shared_vertex : ∃ i j, triangle.vertices i = pentagon.vertices j) :
  shared_vertex_angle = 84 := by
  sorry

end NUMINAMATH_CALUDE_shared_vertex_angle_is_84_l3866_386697


namespace NUMINAMATH_CALUDE_min_distance_point_l3866_386641

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2 * Real.log x + a

def g (x : ℝ) : ℝ := -x^2 + 3*x - 4

def h (x : ℝ) : ℝ := f 0 x - g x

theorem min_distance_point (t : ℝ) :
  t > 0 →
  (∀ x > 0, |h x| ≥ |h t|) →
  t = (3 + Real.sqrt 33) / 6 :=
sorry

end

end NUMINAMATH_CALUDE_min_distance_point_l3866_386641


namespace NUMINAMATH_CALUDE_basketball_league_games_l3866_386646

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180. -/
theorem basketball_league_games :
  total_games 10 4 = 180 := by sorry

end NUMINAMATH_CALUDE_basketball_league_games_l3866_386646


namespace NUMINAMATH_CALUDE_product_mod_600_l3866_386647

theorem product_mod_600 : (1853 * 2101) % 600 = 553 := by sorry

end NUMINAMATH_CALUDE_product_mod_600_l3866_386647


namespace NUMINAMATH_CALUDE_sara_quarters_count_l3866_386655

theorem sara_quarters_count (initial : Nat) (from_dad : Nat) (total : Nat) : 
  initial = 21 → from_dad = 49 → total = initial + from_dad → total = 70 :=
by sorry

end NUMINAMATH_CALUDE_sara_quarters_count_l3866_386655


namespace NUMINAMATH_CALUDE_geometric_sequence_max_value_l3866_386663

/-- Given a geometric sequence {a_n} with common ratio √2, 
    T_n = (17S_n - S_{2n}) / a_{n+1} attains its maximum value when n = 4 -/
theorem geometric_sequence_max_value (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →
  (∀ n, S n = a 1 * (1 - (Real.sqrt 2)^n) / (1 - Real.sqrt 2)) →
  (∀ n, T n = (17 * S n - S (2 * n)) / a (n + 1)) →
  (∃ B : ℝ, ∀ n, T n ≤ B ∧ T 4 = B) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_value_l3866_386663


namespace NUMINAMATH_CALUDE_egg_count_proof_l3866_386666

def initial_eggs : ℕ := 47
def eggs_added : ℝ := 5.0
def final_eggs : ℕ := 52

theorem egg_count_proof : 
  (initial_eggs : ℝ) + eggs_added = final_eggs := by sorry

end NUMINAMATH_CALUDE_egg_count_proof_l3866_386666


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l3866_386658

theorem trigonometric_inequalities :
  (Real.tan (3 * π / 5) < Real.tan (π / 5)) ∧
  (Real.cos (-17 * π / 4) > Real.cos (-23 * π / 5)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l3866_386658


namespace NUMINAMATH_CALUDE_daily_allowance_calculation_l3866_386625

/-- Proves that if a person saves half of their daily allowance for 6 days
    and a quarter of their daily allowance for 1 day, and the total saved is $39,
    then their daily allowance is $12. -/
theorem daily_allowance_calculation (allowance : ℚ) : 
  (6 * (allowance / 2) + 1 * (allowance / 4) = 39) → allowance = 12 := by
  sorry

end NUMINAMATH_CALUDE_daily_allowance_calculation_l3866_386625


namespace NUMINAMATH_CALUDE_expression_value_l3866_386650

theorem expression_value (x y : ℝ) (h : x - 2*y + 2 = 0) :
  (2*y - x)^2 - 2*x + 4*y - 1 = 7 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3866_386650


namespace NUMINAMATH_CALUDE_scalene_triangle_ratio_bounds_l3866_386626

theorem scalene_triangle_ratio_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_scalene : a > b ∧ b > c) (h_avg : a + c = 2 * b) : 1/3 < c/a ∧ c/a < 1 := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_ratio_bounds_l3866_386626


namespace NUMINAMATH_CALUDE_jacob_lunch_calories_l3866_386662

theorem jacob_lunch_calories (planned : ℕ) (breakfast dinner extra : ℕ) 
  (h1 : planned < 1800)
  (h2 : breakfast = 400)
  (h3 : dinner = 1100)
  (h4 : extra = 600) :
  planned + extra - (breakfast + dinner) = 900 :=
by sorry

end NUMINAMATH_CALUDE_jacob_lunch_calories_l3866_386662


namespace NUMINAMATH_CALUDE_distance_between_points_l3866_386677

-- Define the initial meeting time in hours
def initial_meeting_time : ℝ := 4

-- Define the new meeting time in hours after speed increase
def new_meeting_time : ℝ := 3.5

-- Define the speed increase in km/h
def speed_increase : ℝ := 3

-- Define the function to calculate the distance
def calculate_distance (v_A v_B : ℝ) : ℝ := initial_meeting_time * (v_A + v_B)

-- Theorem statement
theorem distance_between_points : 
  ∃ (v_A v_B : ℝ), 
    v_A > 0 ∧ v_B > 0 ∧
    calculate_distance v_A v_B = 
    new_meeting_time * ((v_A + speed_increase) + (v_B + speed_increase)) ∧
    calculate_distance v_A v_B = 168 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3866_386677


namespace NUMINAMATH_CALUDE_odd_number_induction_l3866_386667

theorem odd_number_induction (P : ℕ → Prop) 
  (base : P 1)
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) :
  ∀ n : ℕ, n ≥ 1 → Odd n → P n :=
sorry

end NUMINAMATH_CALUDE_odd_number_induction_l3866_386667


namespace NUMINAMATH_CALUDE_product_of_sums_l3866_386699

theorem product_of_sums (x y : ℝ) (h1 : x + y = -3) (h2 : x * y = 1) :
  (x + 5) * (y + 5) = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l3866_386699


namespace NUMINAMATH_CALUDE_xyz_range_l3866_386651

theorem xyz_range (x y z : ℝ) 
  (sum_condition : x + y + z = 1) 
  (square_sum_condition : x^2 + y^2 + z^2 = 3) : 
  -1 ≤ x * y * z ∧ x * y * z ≤ 5/27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_range_l3866_386651


namespace NUMINAMATH_CALUDE_subtraction_rule_rational_l3866_386603

theorem subtraction_rule_rational (x : ℚ) : ∀ y : ℚ, y - x = y + (-x) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_rule_rational_l3866_386603


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_complex_expression_evaluation_l3866_386691

-- Part 1
theorem log_inequality_solution_set (x : ℝ) :
  (Real.log (x + 2) / Real.log (1/2) > -3) ↔ (-2 < x ∧ x < 6) :=
sorry

-- Part 2
theorem complex_expression_evaluation :
  (1/8)^(1/3) * (-7/6)^0 + 8^0.25 * 2^(1/4) + (2^(1/3) * 3^(1/2))^6 = 221/2 :=
sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_complex_expression_evaluation_l3866_386691


namespace NUMINAMATH_CALUDE_razorback_shop_tshirt_revenue_l3866_386664

/-- The amount of money made from each t-shirt -/
def tshirt_price : ℕ := 62

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 183

/-- The total money made from selling t-shirts -/
def total_tshirt_money : ℕ := tshirt_price * tshirts_sold

theorem razorback_shop_tshirt_revenue : total_tshirt_money = 11346 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_tshirt_revenue_l3866_386664


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3866_386681

/-- The distance between the vertices of the hyperbola (x²/16) - (y²/25) = 1 is 8 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y => (x^2 / 16) - (y^2 / 25) = 1
  ∃ x₁ x₂ : ℝ, (h x₁ 0 ∧ h x₂ 0) ∧ |x₁ - x₂| = 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3866_386681


namespace NUMINAMATH_CALUDE_contradiction_proof_l3866_386605

theorem contradiction_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) : x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l3866_386605


namespace NUMINAMATH_CALUDE_triangle_equilateral_iff_sum_squares_eq_sum_products_l3866_386688

/-- A triangle with sides a, b, and c is equilateral if and only if a² + b² + c² = ab + bc + ca -/
theorem triangle_equilateral_iff_sum_squares_eq_sum_products {a b c : ℝ} (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (a = b ∧ b = c) ↔ a^2 + b^2 + c^2 = a*b + b*c + c*a := by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_iff_sum_squares_eq_sum_products_l3866_386688


namespace NUMINAMATH_CALUDE_songs_per_album_l3866_386644

/-- Given that Rachel bought 8 albums and a total of 16 songs, prove that each album has 2 songs. -/
theorem songs_per_album (num_albums : ℕ) (total_songs : ℕ) (h1 : num_albums = 8) (h2 : total_songs = 16) :
  total_songs / num_albums = 2 := by
  sorry

end NUMINAMATH_CALUDE_songs_per_album_l3866_386644


namespace NUMINAMATH_CALUDE_checkers_game_possibilities_l3866_386696

/-- Represents the number of games played by each friend in a checkers game. -/
structure CheckersGames where
  friend1 : ℕ
  friend2 : ℕ
  friend3 : ℕ

/-- Checks if the given number of games for three friends is valid. -/
def isValidGameCount (games : CheckersGames) : Prop :=
  ∃ (a b c : ℕ), 
    a + b + c = (games.friend1 + games.friend2 + games.friend3) / 2 ∧
    a + c = games.friend1 ∧
    b + c = games.friend2 ∧
    a + b = games.friend3

/-- Theorem stating the validity of different game counts for the third friend. -/
theorem checkers_game_possibilities : 
  let games1 := CheckersGames.mk 25 17 34
  let games2 := CheckersGames.mk 25 17 35
  let games3 := CheckersGames.mk 25 17 56
  isValidGameCount games1 ∧ 
  ¬isValidGameCount games2 ∧ 
  ¬isValidGameCount games3 := by
  sorry

end NUMINAMATH_CALUDE_checkers_game_possibilities_l3866_386696


namespace NUMINAMATH_CALUDE_leadership_assignment_theorem_l3866_386615

def community_size : ℕ := 12
def chief_count : ℕ := 1
def supporting_chief_count : ℕ := 2
def senior_officer_count : ℕ := 2
def inferior_officer_count : ℕ := 2

def leadership_assignment_count : ℕ :=
  community_size *
  (community_size - chief_count).choose supporting_chief_count *
  (community_size - chief_count - supporting_chief_count).choose senior_officer_count *
  (community_size - chief_count - supporting_chief_count - senior_officer_count).choose inferior_officer_count

theorem leadership_assignment_theorem :
  leadership_assignment_count = 498960 := by
  sorry

end NUMINAMATH_CALUDE_leadership_assignment_theorem_l3866_386615


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3866_386676

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3866_386676


namespace NUMINAMATH_CALUDE_solve_inequality_find_m_range_l3866_386619

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + 2
def g (m : ℝ) (x : ℝ) : ℝ := m * |x|

-- Theorem for part (1)
theorem solve_inequality (x : ℝ) : f x > 5 ↔ x < -1 ∨ x > 5 := by sorry

-- Theorem for part (2)
theorem find_m_range (m : ℝ) : (∀ x : ℝ, f x ≥ g m x) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_find_m_range_l3866_386619


namespace NUMINAMATH_CALUDE_fourth_number_in_row_15_l3866_386670

def pascal_triangle (n k : ℕ) : ℕ := Nat.choose n k

theorem fourth_number_in_row_15 : pascal_triangle 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_row_15_l3866_386670


namespace NUMINAMATH_CALUDE_three_divides_difference_l3866_386656

/-- Represents a three-digit number ABC --/
structure ThreeDigitNumber where
  A : Nat
  B : Nat
  C : Nat
  A_is_digit : A < 10
  B_is_digit : B < 10
  C_is_digit : C < 10

/-- The value of a three-digit number ABC --/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.A + 10 * n.B + n.C

/-- The reversed value of a three-digit number ABC (i.e., CBA) --/
def reversed_value (n : ThreeDigitNumber) : Nat :=
  100 * n.C + 10 * n.B + n.A

theorem three_divides_difference (n : ThreeDigitNumber) (h : n.A ≠ n.C) :
  3 ∣ (value n - reversed_value n) := by
  sorry

end NUMINAMATH_CALUDE_three_divides_difference_l3866_386656


namespace NUMINAMATH_CALUDE_partner_q_invest_time_l3866_386628

/-- Represents the investment and profit data for three partners -/
structure PartnerData where
  investment_ratio : Fin 3 → ℚ
  profit_ratio : Fin 3 → ℚ
  p_invest_time : ℚ
  r_invest_time : ℚ

/-- Calculates the investment time for partner q given the partner data -/
def calculate_q_invest_time (data : PartnerData) : ℚ :=
  (data.investment_ratio 0 * data.p_invest_time * data.profit_ratio 1) /
  (data.investment_ratio 1 * data.profit_ratio 0)

/-- Theorem stating that partner q's investment time is 14 months -/
theorem partner_q_invest_time (data : PartnerData)
  (h1 : data.investment_ratio 0 = 7)
  (h2 : data.investment_ratio 1 = 5)
  (h3 : data.investment_ratio 2 = 3)
  (h4 : data.profit_ratio 0 = 7)
  (h5 : data.profit_ratio 1 = 14)
  (h6 : data.profit_ratio 2 = 9)
  (h7 : data.p_invest_time = 5)
  (h8 : data.r_invest_time = 9) :
  calculate_q_invest_time data = 14 := by
  sorry

end NUMINAMATH_CALUDE_partner_q_invest_time_l3866_386628


namespace NUMINAMATH_CALUDE_boys_in_class_l3866_386648

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 32) (h2 : ratio_girls = 3) (h3 : ratio_boys = 5) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l3866_386648


namespace NUMINAMATH_CALUDE_complex_magnitude_l3866_386643

theorem complex_magnitude (z : ℂ) : z = (1 + Complex.I) / (2 - 2 * Complex.I) → Complex.abs z = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3866_386643


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3866_386698

theorem quadratic_solution_property (a b : ℝ) :
  (a * 1^2 + b * 1 - 1 = 0) → (2023 - a - b = 2022) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3866_386698
