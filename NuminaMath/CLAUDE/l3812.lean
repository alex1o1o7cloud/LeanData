import Mathlib

namespace NUMINAMATH_CALUDE_glee_club_female_members_l3812_381242

theorem glee_club_female_members :
  ∀ (male female : ℕ),
  female = 2 * male →
  male + female = 18 →
  female = 12 := by
sorry

end NUMINAMATH_CALUDE_glee_club_female_members_l3812_381242


namespace NUMINAMATH_CALUDE_remainder_theorem_l3812_381213

theorem remainder_theorem (x : ℤ) (h : (x + 2) % 45 = 7) : 
  ((x + 2) % 20 = 7) ∧ (x % 19 = 5) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3812_381213


namespace NUMINAMATH_CALUDE_equation_solution_pairs_l3812_381221

theorem equation_solution_pairs : 
  ∀ x y : ℕ+, x^(y : ℕ) - y^(x : ℕ) = 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_pairs_l3812_381221


namespace NUMINAMATH_CALUDE_locus_of_tangent_circles_l3812_381235

/-- The locus of centers of circles externally tangent to a given circle and a line -/
theorem locus_of_tangent_circles (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    ((x - 0)^2 + (y - 3)^2)^(1/2) = r + 1 ∧
    y = r) → 
  ∃ (a b c : ℝ), a ≠ 0 ∧ (y - b)^2 = 4 * a * (x - c) :=
sorry

end NUMINAMATH_CALUDE_locus_of_tangent_circles_l3812_381235


namespace NUMINAMATH_CALUDE_intersection_condition_l3812_381226

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1

/-- The statement that f(x) intersects y = 3 at only one point -/
def intersects_once (a : ℝ) : Prop :=
  ∃! x : ℝ, f a x = 3

/-- The main theorem to be proved -/
theorem intersection_condition :
  ∀ a : ℝ, intersects_once a ↔ -1 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l3812_381226


namespace NUMINAMATH_CALUDE_fraction_equality_l3812_381273

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3812_381273


namespace NUMINAMATH_CALUDE_union_equals_A_l3812_381291

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}

theorem union_equals_A (a : ℝ) : (A ∪ B a = A) → (a = 2 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l3812_381291


namespace NUMINAMATH_CALUDE_existence_of_digit_in_power_of_two_l3812_381239

theorem existence_of_digit_in_power_of_two (k d : ℕ) (h1 : k > 1) (h2 : d < 9) :
  ∃ n : ℕ, (2^n : ℕ) % 10^k = d := by
  sorry

end NUMINAMATH_CALUDE_existence_of_digit_in_power_of_two_l3812_381239


namespace NUMINAMATH_CALUDE_clock_time_after_hours_l3812_381288

theorem clock_time_after_hours (current_time hours_passed : ℕ) : 
  current_time = 2 → 
  hours_passed = 3467 → 
  (current_time + hours_passed) % 12 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_clock_time_after_hours_l3812_381288


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l3812_381249

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r ^ n

theorem sum_of_fourth_and_fifth_terms :
  ∀ (a₀ r : ℝ),
    geometric_sequence a₀ r 0 = 4096 →
    geometric_sequence a₀ r 1 = 1024 →
    geometric_sequence a₀ r 2 = 256 →
    geometric_sequence a₀ r 5 = 4 →
    geometric_sequence a₀ r 6 = 1 →
    geometric_sequence a₀ r 3 + geometric_sequence a₀ r 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l3812_381249


namespace NUMINAMATH_CALUDE_odd_prime_divisor_property_l3812_381245

theorem odd_prime_divisor_property (n : ℕ+) : 
  (∀ d : ℕ+, d ∣ n → (d + 1) ∣ (n + 1)) ↔ Nat.Prime n.val ∧ n.val % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_divisor_property_l3812_381245


namespace NUMINAMATH_CALUDE_six_people_circular_table_l3812_381207

/-- The number of distinct circular permutations of n elements -/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- Two seating arrangements are considered the same if one is a rotation of the other -/
axiom rotation_equivalence : ∀ n : ℕ, n > 0 → circularPermutations n = (n - 1).factorial

theorem six_people_circular_table : circularPermutations 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_six_people_circular_table_l3812_381207


namespace NUMINAMATH_CALUDE_inner_triangle_side_length_l3812_381261

theorem inner_triangle_side_length 
  (outer_side : ℝ) 
  (inner_side : ℝ) 
  (small_side : ℝ) 
  (h_outer : outer_side = 6) 
  (h_small : small_side = 1) 
  (h_parallel : inner_triangles_parallel_to_outer)
  (h_vertex_outer : inner_triangles_vertex_on_outer_side)
  (h_vertex_inner : inner_triangles_vertex_on_other_inner)
  (h_congruent : inner_triangles_congruent)
  : inner_side = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_side_length_l3812_381261


namespace NUMINAMATH_CALUDE_trig_identity_l3812_381201

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3812_381201


namespace NUMINAMATH_CALUDE_partnership_contribution_l3812_381200

theorem partnership_contribution 
  (a_capital : ℕ) 
  (a_time : ℕ) 
  (b_time : ℕ) 
  (total_profit : ℕ) 
  (a_profit : ℕ) 
  (h1 : a_capital = 5000)
  (h2 : a_time = 8)
  (h3 : b_time = 5)
  (h4 : total_profit = 8400)
  (h5 : a_profit = 4800) :
  ∃ b_capital : ℕ, 
    (a_capital * a_time : ℚ) / ((a_capital * a_time + b_capital * b_time) : ℚ) = 
    (a_profit : ℚ) / (total_profit : ℚ) ∧ 
    b_capital = 6000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_contribution_l3812_381200


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3812_381263

theorem pie_eating_contest (student1 student2 student3 : ℚ) 
  (h1 : student1 = 5/6)
  (h2 : student2 = 7/8)
  (h3 : student3 = 1/2) :
  student1 + student2 - student3 = 29/24 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3812_381263


namespace NUMINAMATH_CALUDE_mushroom_collection_problem_l3812_381258

theorem mushroom_collection_problem :
  ∃ (x₁ x₂ x₃ x₄ : ℕ),
    x₁ + x₂ = 7 ∧
    x₁ + x₃ = 9 ∧
    x₁ + x₄ = 10 ∧
    x₂ + x₃ = 10 ∧
    x₂ + x₄ = 11 ∧
    x₃ + x₄ = 13 ∧
    x₁ ≤ x₂ ∧ x₂ ≤ x₃ ∧ x₃ ≤ x₄ :=
by sorry

end NUMINAMATH_CALUDE_mushroom_collection_problem_l3812_381258


namespace NUMINAMATH_CALUDE_probability_power_of_two_four_digit_l3812_381228

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is a power of 2 if its base-2 logarithm is an integer. -/
def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- The count of four-digit numbers that are powers of 2. -/
def CountPowersOfTwoFourDigit : ℕ := 4

/-- The total count of four-digit numbers. -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The probability of a randomly chosen four-digit number being a power of 2. -/
def ProbabilityPowerOfTwo : ℚ := CountPowersOfTwoFourDigit / TotalFourDigitNumbers

theorem probability_power_of_two_four_digit :
  ProbabilityPowerOfTwo = 1 / 2250 := by sorry

end NUMINAMATH_CALUDE_probability_power_of_two_four_digit_l3812_381228


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3812_381219

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3812_381219


namespace NUMINAMATH_CALUDE_jacoby_trip_savings_l3812_381210

/-- The amount Jacoby needs for his trip to Brickville --/
def trip_cost : ℝ := 8000

/-- Jacoby's hourly wage --/
def hourly_wage : ℝ := 25

/-- Hours Jacoby worked --/
def hours_worked : ℝ := 15

/-- Tax rate on Jacoby's salary --/
def tax_rate : ℝ := 0.1

/-- Price of each cookie --/
def cookie_price : ℝ := 5

/-- Number of cookies sold --/
def cookies_sold : ℕ := 30

/-- Weekly tutoring earnings --/
def tutoring_weekly : ℝ := 100

/-- Weeks of tutoring --/
def tutoring_weeks : ℕ := 4

/-- Cost of lottery ticket --/
def lottery_ticket_cost : ℝ := 20

/-- Lottery winnings --/
def lottery_winnings : ℝ := 700

/-- Percentage of lottery winnings given to friend --/
def lottery_share : ℝ := 0.3

/-- Gift amount from each sister --/
def sister_gift : ℝ := 700

/-- Number of sisters --/
def number_of_sisters : ℕ := 2

/-- Cost of keychain --/
def keychain_cost : ℝ := 3

/-- Cost of backpack --/
def backpack_cost : ℝ := 47

/-- The amount Jacoby still needs for his trip --/
def amount_needed : ℝ := 5286.50

theorem jacoby_trip_savings : 
  trip_cost - (
    (hourly_wage * hours_worked * (1 - tax_rate)) +
    (cookie_price * cookies_sold) +
    (tutoring_weekly * tutoring_weeks) +
    ((lottery_winnings - lottery_ticket_cost) * (1 - lottery_share)) +
    (sister_gift * number_of_sisters) -
    (keychain_cost + backpack_cost)
  ) = amount_needed := by sorry

end NUMINAMATH_CALUDE_jacoby_trip_savings_l3812_381210


namespace NUMINAMATH_CALUDE_expected_mass_of_disks_l3812_381233

/-- The expected mass of 100 metal disks with manufacturing errors -/
theorem expected_mass_of_disks (
  perfect_diameter : ℝ) 
  (perfect_mass : ℝ) 
  (radius_std_dev : ℝ) 
  (num_disks : ℕ) 
  (h1 : perfect_diameter = 1) 
  (h2 : perfect_mass = 100) 
  (h3 : radius_std_dev = 0.01) 
  (h4 : num_disks = 100) : 
  ∃ (expected_mass : ℝ), expected_mass = 10004 := by
  sorry

end NUMINAMATH_CALUDE_expected_mass_of_disks_l3812_381233


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l3812_381236

theorem percentage_of_male_employees 
  (total_employees : ℕ)
  (males_below_50 : ℕ)
  (h1 : total_employees = 800)
  (h2 : males_below_50 = 120)
  (h3 : (males_below_50 : ℝ) = 0.6 * (total_employees * (percentage_males / 100))) :
  percentage_males = 25 := by
  sorry

#check percentage_of_male_employees

end NUMINAMATH_CALUDE_percentage_of_male_employees_l3812_381236


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l3812_381248

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem fourth_term_of_sequence (y : ℝ) :
  let a₁ := 8
  let a₂ := 32 * y^2
  let a₃ := 128 * y^4
  let r := a₂ / a₁
  geometric_sequence a₁ r 4 = 512 * y^6 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l3812_381248


namespace NUMINAMATH_CALUDE_total_interest_calculation_total_interest_is_805_l3812_381296

/-- Calculate the total interest after 10 years given the conditions -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 700) 
  (h2 : P * R = 700) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 805 := by
  sorry

/-- Prove that the total interest is 805 -/
theorem total_interest_is_805 (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 700) 
  (h2 : P * R = 700) : 
  ∃ (total_interest : ℝ), total_interest = 805 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_total_interest_is_805_l3812_381296


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l3812_381252

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length calculation -/
theorem bridge_length_proof :
  bridge_length 80 45 30 = 295 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l3812_381252


namespace NUMINAMATH_CALUDE_g_form_l3812_381205

-- Define polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_g_prod : ∀ x, f (g x) = f x * g x
axiom g_3_eq_50 : g 3 = 50

-- Define the theorem
theorem g_form : g = fun x ↦ x^2 + 20*x - 20 :=
sorry

end NUMINAMATH_CALUDE_g_form_l3812_381205


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_4_and_5_l3812_381251

theorem largest_three_digit_multiple_of_4_and_5 : 
  ∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 4 ∣ n ∧ 5 ∣ n → n ≤ 980 :=
by
  sorry

#check largest_three_digit_multiple_of_4_and_5

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_4_and_5_l3812_381251


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_144_l3812_381237

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to choose k objects from n objects. -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / (factorial k * factorial (n - k))

/-- The number of seating arrangements for 4 students and 2 teachers,
    where teachers cannot sit at either end and must not sit next to each other. -/
def seating_arrangements : ℕ := 
  let student_arrangements := factorial 4
  let teacher_positions := choose 3 2
  let teacher_arrangements := factorial 2
  student_arrangements * teacher_positions * teacher_arrangements

theorem seating_arrangements_eq_144 : seating_arrangements = 144 := by
  sorry

#eval seating_arrangements

end NUMINAMATH_CALUDE_seating_arrangements_eq_144_l3812_381237


namespace NUMINAMATH_CALUDE_simplify_expression_l3812_381206

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  (10 * x * y^3) / (15 * x^2 * y^2) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3812_381206


namespace NUMINAMATH_CALUDE_three_double_derivative_l3812_381292

-- Define the derivative operation
noncomputable def derive (f : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the given equation as a property
axiom equation (q : ℝ) : derive (λ x => x) q = 3 * q - 3

-- State the theorem
theorem three_double_derivative : derive (derive (λ x => x)) 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_three_double_derivative_l3812_381292


namespace NUMINAMATH_CALUDE_min_sum_squares_l3812_381232

/-- The polynomial equation we're considering -/
def P (a b x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + a*x + 1

/-- The condition that the polynomial has at least one real root -/
def has_real_root (a b : ℝ) : Prop := ∃ x : ℝ, P a b x = 0

/-- The theorem statement -/
theorem min_sum_squares (a b : ℝ) (h : has_real_root a b) :
  ∃ (a₀ b₀ : ℝ), has_real_root a₀ b₀ ∧ a₀^2 + b₀^2 = 4/5 ∧ 
  ∀ (a' b' : ℝ), has_real_root a' b' → a'^2 + b'^2 ≥ 4/5 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3812_381232


namespace NUMINAMATH_CALUDE_line_slope_l3812_381216

theorem line_slope (x y : ℝ) : 
  (2 * x + Real.sqrt 3 * y - 1 = 0) → 
  (∃ m : ℝ, m = -(2 * Real.sqrt 3) / 3 ∧ 
   ∀ x₁ x₂ y₁ y₂ : ℝ, 
   x₁ ≠ x₂ → 
   (2 * x₁ + Real.sqrt 3 * y₁ - 1 = 0) → 
   (2 * x₂ + Real.sqrt 3 * y₂ - 1 = 0) → 
   m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l3812_381216


namespace NUMINAMATH_CALUDE_quick_response_solution_l3812_381222

def quick_response_problem (x y z : ℕ) : Prop :=
  5 * x + 4 * y + 3 * z = 15 ∧ 
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1)

theorem quick_response_solution :
  ∀ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 → quick_response_problem x y z :=
by
  sorry

#check quick_response_solution

end NUMINAMATH_CALUDE_quick_response_solution_l3812_381222


namespace NUMINAMATH_CALUDE_seating_arrangements_l3812_381285

/-- Represents the number of seats in a row -/
def total_seats : ℕ := 12

/-- Represents the number of people to be seated -/
def num_people : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the number of possible arrangements of A between the other two people -/
def a_between_arrangements : ℕ := 2

/-- Represents the number of empty seats after arranging people and mandatory empty seats -/
def remaining_empty_seats : ℕ := 8

/-- Represents the number of empty seats to be chosen from remaining empty seats -/
def seats_to_choose : ℕ := 5

/-- The main theorem stating the total number of seating arrangements -/
theorem seating_arrangements :
  a_between_arrangements * choose remaining_empty_seats seats_to_choose = 112 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3812_381285


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l3812_381272

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 8 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -8 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l3812_381272


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3812_381290

/-- A regular polygon with side length 6 units and exterior angle 60 degrees has a perimeter of 36 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (h1 : s = 6) (h2 : θ = 60) :
  let n : ℝ := 360 / θ
  s * n = 36 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3812_381290


namespace NUMINAMATH_CALUDE_plant_arrangement_l3812_381286

theorem plant_arrangement (basil_count : Nat) (tomato_count : Nat) :
  basil_count = 6 →
  tomato_count = 3 →
  (Nat.factorial (basil_count + 1)) * (Nat.factorial tomato_count) = 30240 :=
by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_l3812_381286


namespace NUMINAMATH_CALUDE_four_three_three_cuboid_two_face_count_l3812_381247

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with exactly two painted faces in a cuboid -/
def count_two_face_cubes (c : Cuboid) : ℕ :=
  2 * (c.length - 2) + 2 * (c.width - 2) + 2 * (c.height - 2)

/-- Theorem: A 4x3x3 cuboid has 16 cubes with exactly two painted faces -/
theorem four_three_three_cuboid_two_face_count :
  count_two_face_cubes ⟨4, 3, 3⟩ = 16 := by
  sorry

end NUMINAMATH_CALUDE_four_three_three_cuboid_two_face_count_l3812_381247


namespace NUMINAMATH_CALUDE_min_value_theorem_l3812_381214

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (2^x) + Real.log (8^y) = Real.log 2) : 
  1/x + 1/(3*y) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3812_381214


namespace NUMINAMATH_CALUDE_fixed_point_of_line_family_l3812_381275

theorem fixed_point_of_line_family (k : ℝ) : 
  (3 * k - 1) * (2 / 7) + (k + 2) * (1 / 7) - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_family_l3812_381275


namespace NUMINAMATH_CALUDE_find_x_l3812_381224

def numbers : List ℕ := [201, 202, 204, 205, 206, 209, 209, 210]

theorem find_x (x : ℕ) :
  let all_numbers := numbers ++ [x]
  (all_numbers.sum / all_numbers.length : ℚ) = 207 →
  x = 217 := by sorry

end NUMINAMATH_CALUDE_find_x_l3812_381224


namespace NUMINAMATH_CALUDE_dogs_accessible_area_l3812_381257

theorem dogs_accessible_area (s : ℝ) (s_pos : s > 0) :
  let square_area := (2 * s) ^ 2
  let circle_area := π * s ^ 2
  circle_area / square_area = π / 4 := by
  sorry

#check dogs_accessible_area

end NUMINAMATH_CALUDE_dogs_accessible_area_l3812_381257


namespace NUMINAMATH_CALUDE_tech_students_count_l3812_381217

/-- Number of students in subject elective courses -/
def subject_students : ℕ → ℕ := fun m ↦ m

/-- Number of students in physical education and arts elective courses -/
def pe_arts_students : ℕ → ℕ := fun m ↦ m + 9

/-- Number of students in technology elective courses -/
def tech_students : ℕ → ℕ := fun m ↦ (pe_arts_students m) / 3 + 5

theorem tech_students_count (m : ℕ) : 
  tech_students m = m / 3 + 8 := by sorry

end NUMINAMATH_CALUDE_tech_students_count_l3812_381217


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3812_381293

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem f_derivative_at_one : 
  deriv f 1 = 2 * Real.log 2 - 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3812_381293


namespace NUMINAMATH_CALUDE_system_one_solution_l3812_381230

theorem system_one_solution (x : ℝ) : 
  (2 * x > 1 - x ∧ x + 2 < 4 * x - 1) ↔ x > 1 := by
sorry

end NUMINAMATH_CALUDE_system_one_solution_l3812_381230


namespace NUMINAMATH_CALUDE_AC_length_l3812_381298

/-- A right triangle with a circle passing through its altitude --/
structure RightTriangleWithCircle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- ABC is a right triangle with right angle at A
  right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- AH is perpendicular to BC
  AH_perpendicular_BC : (H.1 - A.1) * (C.1 - B.1) + (H.2 - A.2) * (C.2 - B.2) = 0
  -- A circle passes through A, H, X, and Y
  circle_passes : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (X.1 - center.1)^2 + (X.2 - center.2)^2 = radius^2 ∧
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2
  -- X is on AB
  X_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  Y_on_AC : ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- Given lengths
  AX_length : ((X.1 - A.1)^2 + (X.2 - A.2)^2)^(1/2 : ℝ) = 5
  AY_length : ((Y.1 - A.1)^2 + (Y.2 - A.2)^2)^(1/2 : ℝ) = 6
  AB_length : ((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2 : ℝ) = 9

/-- The main theorem --/
theorem AC_length (t : RightTriangleWithCircle) : 
  ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)^(1/2 : ℝ) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_AC_length_l3812_381298


namespace NUMINAMATH_CALUDE_present_age_of_B_present_age_of_B_exists_l3812_381238

/-- Proves that given the conditions in the problem, B's current age is 37 years. -/
theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 10 = 2 * (b - 10)) →  -- A will be twice as old as B was 10 years ago, in 10 years
    (a = b + 7) →              -- A is now 7 years older than B
    (b = 37)                   -- B's current age is 37

/-- The theorem holds for some values of a and b. -/
theorem present_age_of_B_exists : ∃ a b, present_age_of_B a b :=
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_present_age_of_B_exists_l3812_381238


namespace NUMINAMATH_CALUDE_product_of_fractions_l3812_381268

theorem product_of_fractions : (2 : ℚ) / 5 * (3 : ℚ) / 4 = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3812_381268


namespace NUMINAMATH_CALUDE_license_plate_difference_l3812_381271

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible Florida license plates -/
def florida_plates : ℕ := num_letters^6 * num_digits^2

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_letters^3 * num_digits^4

/-- The difference between Florida and Texas license plate possibilities -/
def plate_difference : ℕ := florida_plates - texas_plates

theorem license_plate_difference :
  plate_difference = 54293545536 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3812_381271


namespace NUMINAMATH_CALUDE_find_m_value_l3812_381223

/-- Given functions f and g, and a condition on their values at x = 3, 
    prove that the parameter m in g equals -11/3 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) 
    (hf : ∀ x, f x = 3 * x^2 + 2 / x - 1)
    (hg : ∀ x, g x = 2 * x^2 - m)
    (h_diff : f 3 - g 3 = 5) : 
  m = -11/3 := by sorry

end NUMINAMATH_CALUDE_find_m_value_l3812_381223


namespace NUMINAMATH_CALUDE_equation_solution_l3812_381241

theorem equation_solution (a : ℤ) : 
  (∃ x : ℕ+, (x - 4) / 6 - (a * x - 1) / 3 = 1 / 3) ↔ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3812_381241


namespace NUMINAMATH_CALUDE_min_polyline_distance_circle_line_l3812_381208

/-- Polyline distance between two points -/
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- A point is on the unit circle -/
def on_unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- A point is on the given line -/
def on_line (x y : ℝ) : Prop :=
  2*x + y - 2*Real.sqrt 5 = 0

/-- The minimum polyline distance between the circle and the line -/
theorem min_polyline_distance_circle_line :
  ∃ (min_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), on_unit_circle x₁ y₁ → on_line x₂ y₂ →
      polyline_distance x₁ y₁ x₂ y₂ ≥ min_dist) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ), on_unit_circle x₁ y₁ ∧ on_line x₂ y₂ ∧
      polyline_distance x₁ y₁ x₂ y₂ = min_dist) ∧
    min_dist = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_polyline_distance_circle_line_l3812_381208


namespace NUMINAMATH_CALUDE_min_output_avoids_losses_l3812_381240

/-- The profit function for a company's product -/
def profit_function (x : ℝ) : ℝ := 0.1 * x - 150

/-- The minimum output to avoid losses -/
def min_output : ℝ := 1500

theorem min_output_avoids_losses :
  ∀ x : ℝ, x ≥ min_output → profit_function x ≥ 0 ∧
  ∀ y : ℝ, y < min_output → ∃ z : ℝ, z ≥ y ∧ profit_function z < 0 :=
by sorry

end NUMINAMATH_CALUDE_min_output_avoids_losses_l3812_381240


namespace NUMINAMATH_CALUDE_tournament_games_count_l3812_381209

/-- Calculates the total number of games played in a tournament given the ratio of outcomes and the number of games won. -/
def totalGamesPlayed (ratioWon ratioLost ratioTied : ℕ) (gamesWon : ℕ) : ℕ :=
  let partValue := gamesWon / ratioWon
  let gamesLost := ratioLost * partValue
  let gamesTied := ratioTied * partValue
  gamesWon + gamesLost + gamesTied

/-- Theorem stating that given the specified ratio and number of games won, the total games played is 96. -/
theorem tournament_games_count :
  totalGamesPlayed 7 4 5 42 = 96 := by
  sorry

#eval totalGamesPlayed 7 4 5 42

end NUMINAMATH_CALUDE_tournament_games_count_l3812_381209


namespace NUMINAMATH_CALUDE_average_increment_l3812_381274

theorem average_increment (a b c : ℝ) (h : (a + b + c) / 3 = 8) :
  ((a + 1) + (b + 2) + (c + 3)) / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_increment_l3812_381274


namespace NUMINAMATH_CALUDE_rectangle_height_l3812_381278

/-- Given a rectangle with width 32 cm and area divided by diagonal 576 cm², prove its height is 36 cm. -/
theorem rectangle_height (w h : ℝ) (area_div_diagonal : ℝ) : 
  w = 32 → 
  area_div_diagonal = 576 →
  (w * h) / 2 = area_div_diagonal →
  h = 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_height_l3812_381278


namespace NUMINAMATH_CALUDE_prob_advance_four_shots_value_l3812_381234

/-- The probability of a successful shot -/
def p : ℝ := 0.6

/-- The probability of advancing after exactly four shots in a basketball contest -/
def prob_advance_four_shots : ℝ :=
  (1 : ℝ) * (1 - p) * p * p

/-- Theorem stating the probability of advancing after exactly four shots -/
theorem prob_advance_four_shots_value :
  prob_advance_four_shots = 18 / 125 := by
  sorry

end NUMINAMATH_CALUDE_prob_advance_four_shots_value_l3812_381234


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3812_381297

theorem fraction_equivalence : 
  ∃ (n : ℚ), (3 + n) / (5 + n) = 9 / 11 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3812_381297


namespace NUMINAMATH_CALUDE_count_quads_with_perimeter_36_l3812_381281

/-- A convex cyclic quadrilateral with integer sides --/
structure ConvexCyclicQuad where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  convex : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c
  cyclic : a * c = b * d

/-- The set of all convex cyclic quadrilaterals with perimeter 36 --/
def QuadsWithPerimeter36 : Set ConvexCyclicQuad :=
  {q : ConvexCyclicQuad | q.a + q.b + q.c + q.d = 36}

/-- Counts the number of distinct quadrilaterals in the set --/
def CountDistinctQuads (s : Set ConvexCyclicQuad) : ℕ :=
  sorry

theorem count_quads_with_perimeter_36 :
  CountDistinctQuads QuadsWithPerimeter36 = 1026 := by
  sorry

end NUMINAMATH_CALUDE_count_quads_with_perimeter_36_l3812_381281


namespace NUMINAMATH_CALUDE_orange_juice_mix_l3812_381270

/-- Given the conditions for preparing orange juice, prove that 3 cans of water are needed per can of concentrate. -/
theorem orange_juice_mix (servings : ℕ) (serving_size : ℚ) (concentrate_cans : ℕ) (concentrate_size : ℚ) 
  (h1 : servings = 200)
  (h2 : serving_size = 6)
  (h3 : concentrate_cans = 60)
  (h4 : concentrate_size = 5) : 
  (servings * serving_size - concentrate_cans * concentrate_size) / (concentrate_cans * concentrate_size) = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_mix_l3812_381270


namespace NUMINAMATH_CALUDE_three_valid_floor_dimensions_l3812_381204

/-- 
Represents the number of valid floor dimensions (m, n) satisfying:
1. n > m
2. (m-6)(n-6) = 12
3. m ≥ 7 and n ≥ 7
where m and n are positive integers, and the unpainted border is 2 feet wide on each side.
-/
def validFloorDimensions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let m := p.1
    let n := p.2
    n > m ∧ (m - 6) * (n - 6) = 12 ∧ m ≥ 7 ∧ n ≥ 7
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The main theorem stating that there are exactly 3 valid floor dimensions. -/
theorem three_valid_floor_dimensions : validFloorDimensions = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_valid_floor_dimensions_l3812_381204


namespace NUMINAMATH_CALUDE_initial_seashells_count_l3812_381256

/-- The number of seashells Jason found initially -/
def initial_seashells : ℕ := sorry

/-- The number of starfish Jason found -/
def starfish : ℕ := 48

/-- The number of seashells Jason gave to Tim -/
def seashells_given_away : ℕ := 13

/-- The number of seashells Jason has now -/
def current_seashells : ℕ := 36

/-- Theorem stating that the initial number of seashells is equal to the current number plus the number given away -/
theorem initial_seashells_count : initial_seashells = current_seashells + seashells_given_away := by
  sorry

end NUMINAMATH_CALUDE_initial_seashells_count_l3812_381256


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3812_381243

def U : Set ℤ := {x | |x| < 5}
def A : Set ℤ := {-2, 1, 3, 4}
def B : Set ℤ := {0, 2, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {-2, 1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3812_381243


namespace NUMINAMATH_CALUDE_zhaoqing_population_l3812_381260

theorem zhaoqing_population (total_population : ℝ) 
  (h1 : 3.06 = 0.8 * total_population - 0.18) 
  (h2 : 3.06 = agricultural_population) : 
  total_population = 4.05 := by
sorry

end NUMINAMATH_CALUDE_zhaoqing_population_l3812_381260


namespace NUMINAMATH_CALUDE_number_comparison_l3812_381259

theorem number_comparison : ∃ (a b c : ℝ), 
  a = 7^(0.3 : ℝ) ∧ 
  b = (0.3 : ℝ)^7 ∧ 
  c = Real.log 0.3 ∧ 
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l3812_381259


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3812_381266

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    is_scalene_triangle a b c →
    is_prime a ∧ is_prime b ∧ is_prime c →
    is_prime (a + b + c) →
    a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 →
    triangle_inequality a b c →
    (a + b + c ≥ 23) ∧ (∃ x y z : ℕ, x + y + z = 23 ∧ 
      is_scalene_triangle x y z ∧
      is_prime x ∧ is_prime y ∧ is_prime z ∧
      is_prime (x + y + z) ∧
      x ≥ 5 ∧ y ≥ 5 ∧ z ≥ 5 ∧
      triangle_inequality x y z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3812_381266


namespace NUMINAMATH_CALUDE_bicycle_weight_proof_l3812_381276

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 20

/-- The weight of one scooter in pounds -/
def scooter_weight : ℝ := 40

theorem bicycle_weight_proof :
  (10 * bicycle_weight = 5 * scooter_weight) ∧
  (5 * scooter_weight = 200) →
  bicycle_weight = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_bicycle_weight_proof_l3812_381276


namespace NUMINAMATH_CALUDE_square_equality_solution_l3812_381231

theorem square_equality_solution (x : ℝ) : (2012 + x)^2 = x^2 ↔ x = -1006 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_solution_l3812_381231


namespace NUMINAMATH_CALUDE_min_surface_area_height_l3812_381211

/-- The height that minimizes the surface area of an open-top rectangular box with square base and volume 4 -/
theorem min_surface_area_height : ∃ (h : ℝ), h > 0 ∧ 
  (∀ (x : ℝ), x > 0 → x^2 * h = 4 → 
    ∀ (h' : ℝ), h' > 0 → x^2 * h' = 4 → 
      x^2 + 4*x*h ≤ x^2 + 4*x*h') ∧ 
  h = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_surface_area_height_l3812_381211


namespace NUMINAMATH_CALUDE_tan_240_degrees_l3812_381294

theorem tan_240_degrees : Real.tan (240 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_240_degrees_l3812_381294


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3812_381267

/-- Given a geometric sequence with first term a₁ and common ratio q,
    S₃ represents the sum of the first 3 terms -/
def S₃ (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q^2

/-- Theorem: For a geometric sequence with common ratio q,
    if S₃ = 7a₁, then q = 2 or q = -3 -/
theorem geometric_sequence_ratio (a₁ q : ℝ) (h : a₁ ≠ 0) :
  S₃ a₁ q = 7 * a₁ → q = 2 ∨ q = -3 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3812_381267


namespace NUMINAMATH_CALUDE_diameter_of_figure_F_l3812_381254

/-- A triangle with semicircles constructed outwardly on each side -/
structure TriangleWithSemicircles where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- The figure F composed of the triangle and the three semicircles -/
def FigureF (t : TriangleWithSemicircles) : Set (ℝ × ℝ) :=
  sorry

/-- The diameter of a set in the plane -/
def diameter (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The diameter of figure F is equal to the semi-perimeter of the triangle -/
theorem diameter_of_figure_F (t : TriangleWithSemicircles) :
    diameter (FigureF t) = (t.a + t.b + t.c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_diameter_of_figure_F_l3812_381254


namespace NUMINAMATH_CALUDE_statement_analysis_l3812_381287

theorem statement_analysis (m n : ℝ) : 
  (∀ m n, m + n ≤ 0 → m ≤ 0 ∨ n ≤ 0) ∧ 
  (∀ m n, m > 0 ∧ n > 0 → m + n > 0) ∧ 
  (∃ m n, m + n > 0 ∧ ¬(m > 0 ∧ n > 0)) :=
by sorry

end NUMINAMATH_CALUDE_statement_analysis_l3812_381287


namespace NUMINAMATH_CALUDE_decreasing_number_a312_max_decreasing_number_divisible_by_9_l3812_381255

/-- A four-digit natural number with all digits different and not equal to 0 -/
structure DecreasingNumber :=
  (a b c d : ℕ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0)
  (d_pos : d > 0)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (decreasing_property : 10 * a + b - (10 * b + c) = 10 * c + d)

theorem decreasing_number_a312 :
  ∃ (n : DecreasingNumber), n.a = 4 ∧ n.b = 3 ∧ n.c = 1 ∧ n.d = 2 :=
sorry

theorem max_decreasing_number_divisible_by_9 :
  ∃ (n : DecreasingNumber),
    (100 * n.a + 10 * n.b + n.c + 100 * n.b + 10 * n.c + n.d) % 9 = 0 ∧
    ∀ (m : DecreasingNumber),
      (100 * m.a + 10 * m.b + m.c + 100 * m.b + 10 * m.c + m.d) % 9 = 0 →
      1000 * n.a + 100 * n.b + 10 * n.c + n.d ≥ 1000 * m.a + 100 * m.b + 10 * m.c + m.d ∧
    n.a = 8 ∧ n.b = 1 ∧ n.c = 6 ∧ n.d = 5 :=
sorry

end NUMINAMATH_CALUDE_decreasing_number_a312_max_decreasing_number_divisible_by_9_l3812_381255


namespace NUMINAMATH_CALUDE_f_equals_g_l3812_381227

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l3812_381227


namespace NUMINAMATH_CALUDE_brennan_pepper_usage_l3812_381269

/-- The amount of pepper Brennan used for scrambled eggs -/
def pepper_used (initial : ℝ) (remaining : ℝ) : ℝ := initial - remaining

/-- Theorem: Given Brennan's initial and remaining pepper amounts, prove he used 0.16 grams for scrambled eggs -/
theorem brennan_pepper_usage :
  let initial : ℝ := 0.25
  let remaining : ℝ := 0.09
  pepper_used initial remaining = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_brennan_pepper_usage_l3812_381269


namespace NUMINAMATH_CALUDE_kristoff_sticker_count_l3812_381284

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := 2210

/-- The ratio of Riku's stickers to Kristoff's stickers -/
def sticker_ratio : ℕ := 25

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := riku_stickers / sticker_ratio

theorem kristoff_sticker_count : kristoff_stickers = 88 := by
  sorry

end NUMINAMATH_CALUDE_kristoff_sticker_count_l3812_381284


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l3812_381220

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  a > 0 → b > 0 → a > b → a + b = 5 * (a - b) → a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l3812_381220


namespace NUMINAMATH_CALUDE_factorization_theorem1_factorization_theorem2_l3812_381218

-- For the first expression
theorem factorization_theorem1 (x y : ℝ) :
  3 * (x + y) * (x - y) - (x - y)^2 = 2 * (x - y) * (x + 2*y) := by sorry

-- For the second expression
theorem factorization_theorem2 (x y : ℝ) :
  x^2 * (y^2 - 1) + 2*x * (y^2 - 1) = x * (y + 1) * (y - 1) * (x + 2) := by sorry

end NUMINAMATH_CALUDE_factorization_theorem1_factorization_theorem2_l3812_381218


namespace NUMINAMATH_CALUDE_common_point_in_intervals_l3812_381299

theorem common_point_in_intervals (n : ℕ) (a b : Fin n → ℝ) 
  (h_closed : ∀ i, a i ≤ b i) 
  (h_intersect : ∀ i j, ∃ x, a i ≤ x ∧ x ≤ b i ∧ a j ≤ x ∧ x ≤ b j) : 
  ∃ p, ∀ i, a i ≤ p ∧ p ≤ b i :=
sorry

end NUMINAMATH_CALUDE_common_point_in_intervals_l3812_381299


namespace NUMINAMATH_CALUDE_expression_value_l3812_381202

theorem expression_value (a b c : ℝ) (ha : a ≠ 3) (hb : b ≠ 4) (hc : c ≠ 5) :
  (a - 3) / (5 - c) * (b - 4) / (3 - a) * (c - 5) / (4 - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3812_381202


namespace NUMINAMATH_CALUDE_volunteer_schedule_l3812_381250

theorem volunteer_schedule (ella fiona george harry : ℕ) 
  (h_ella : ella = 5)
  (h_fiona : fiona = 6)
  (h_george : george = 8)
  (h_harry : harry = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm ella fiona) george) harry = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_l3812_381250


namespace NUMINAMATH_CALUDE_decimal_operations_l3812_381295

theorem decimal_operations (x y : ℝ) : 
  (x / 10 = 0.09 → x = 0.9) ∧ 
  (3.24 * y = 3240 → y = 1000) := by
  sorry

end NUMINAMATH_CALUDE_decimal_operations_l3812_381295


namespace NUMINAMATH_CALUDE_value_of_c_l3812_381212

theorem value_of_c (a b c d : ℝ) : 
  12 = 0.04 * (a + d) →
  4 = 0.12 * (b - d) →
  c = (b - d) / (a + d) →
  c = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l3812_381212


namespace NUMINAMATH_CALUDE_min_handshakes_for_35_people_l3812_381282

/-- Represents a handshake graph for a conference. -/
structure ConferenceHandshakes where
  people : ℕ
  min_handshakes_per_person : ℕ
  total_handshakes : ℕ

/-- The minimum number of handshakes for a conference with given parameters. -/
def min_handshakes (c : ConferenceHandshakes) : ℕ := c.total_handshakes

/-- Theorem stating the minimum number of handshakes for the specific conference scenario. -/
theorem min_handshakes_for_35_people : 
  ∀ c : ConferenceHandshakes, 
  c.people = 35 → 
  c.min_handshakes_per_person = 3 → 
  min_handshakes c = 51 := by
  sorry


end NUMINAMATH_CALUDE_min_handshakes_for_35_people_l3812_381282


namespace NUMINAMATH_CALUDE_total_pictures_correct_l3812_381246

/-- The number of pictures Nancy uploaded to Facebook -/
def total_pictures : ℕ := 51

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 11

/-- The number of additional albums -/
def additional_albums : ℕ := 8

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 5

/-- Theorem stating that the total number of pictures is correct -/
theorem total_pictures_correct : 
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end NUMINAMATH_CALUDE_total_pictures_correct_l3812_381246


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l3812_381280

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l3812_381280


namespace NUMINAMATH_CALUDE_segments_AB_CD_parallel_l3812_381283

-- Define points in 2D space
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, -1)
def C : ℝ × ℝ := (0, 4)
def D : ℝ × ℝ := (2, -4)

-- Define a function to check if two segments are parallel
def are_parallel (p1 p2 q1 q2 : ℝ × ℝ) : Prop :=
  let v1 := (p2.1 - p1.1, p2.2 - p1.2)
  let v2 := (q2.1 - q1.1, q2.2 - q1.2)
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Theorem statement
theorem segments_AB_CD_parallel :
  are_parallel A B C D := by
  sorry

end NUMINAMATH_CALUDE_segments_AB_CD_parallel_l3812_381283


namespace NUMINAMATH_CALUDE_circle_equation_l3812_381279

theorem circle_equation (x y θ : ℝ) : 
  (x = 3 + 4 * Real.cos θ ∧ y = -2 + 4 * Real.sin θ) → 
  (x - 3)^2 + (y + 2)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l3812_381279


namespace NUMINAMATH_CALUDE_max_regions_formula_l3812_381265

/-- The maximum number of regions formed by n lines in a plane -/
def max_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- Conditions for the lines in the plane -/
structure PlaneLines where
  n : ℕ
  n_ge_3 : n ≥ 3
  no_parallel : True  -- represents the condition that no two lines are parallel
  no_triple_intersection : True  -- represents the condition that no three lines intersect at the same point

theorem max_regions_formula (p : PlaneLines) :
  max_regions p.n = (p.n^2 + p.n + 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_regions_formula_l3812_381265


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_l3812_381277

theorem other_solution_of_quadratic (x : ℚ) : 
  (48 * (3/4)^2 + 29 = 35 * (3/4) + 12) → 
  (48 * (1/3)^2 + 29 = 35 * (1/3) + 12) := by
  sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_l3812_381277


namespace NUMINAMATH_CALUDE_propositions_proof_l3812_381225

theorem propositions_proof :
  (∀ a b : ℝ, a > b ∧ (1 / a) > (1 / b) → a * b < 0) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → ¬(a^2 < a * b ∧ a * b < b^2)) ∧
  (∀ a b c : ℝ, c > a ∧ a > b ∧ b > 0 → ¬(a / (c - a) < b / (c - b))) ∧
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → a / b > (a + c) / (b + c)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_proof_l3812_381225


namespace NUMINAMATH_CALUDE_sin_max_value_l3812_381229

theorem sin_max_value (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π/3 ∧ 
    (∀ y : ℝ, 0 ≤ y ∧ y ≤ π/3 → 2 * Real.sin (ω * y) ≤ 2 * Real.sin (ω * x)) ∧
    2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_max_value_l3812_381229


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3812_381262

/-- A plane in 3D Cartesian coordinates with intercepts a, b, and c on the x, y, and z axes respectively. -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : a ≠ 0
  h₂ : b ≠ 0
  h₃ : c ≠ 0

/-- The equation of a plane in 3D Cartesian coordinates with given intercepts. -/
def planeEquation (p : Plane3D) (x y z : ℝ) : Prop :=
  x / p.a + y / p.b + z / p.c = 1

/-- Theorem stating that the equation x/a + y/b + z/c = 1 represents a plane
    with intercepts a, b, and c on the x, y, and z axes respectively. -/
theorem plane_equation_correct (p : Plane3D) :
  ∀ x y z : ℝ, planeEquation p x y z ↔ 
    (x = p.a ∧ y = 0 ∧ z = 0) ∨
    (x = 0 ∧ y = p.b ∧ z = 0) ∨
    (x = 0 ∧ y = 0 ∧ z = p.c) :=
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l3812_381262


namespace NUMINAMATH_CALUDE_marathon_calories_burned_l3812_381203

/-- Represents a cycling ride with its distance relative to the base distance -/
structure Ride :=
  (distance : ℝ)

/-- Calculates the adjusted distance for a ride given the actual distance and base distance -/
def adjustedDistance (actualDistance : ℝ) (baseDistance : ℝ) : ℝ :=
  actualDistance - baseDistance

/-- Calculates the total calories burned given a list of rides, base distance, and calorie burn rate -/
def totalCaloriesBurned (rides : List Ride) (baseDistance : ℝ) (caloriesPerKm : ℝ) : ℝ :=
  (rides.map (λ ride => ride.distance + baseDistance)).sum * caloriesPerKm

theorem marathon_calories_burned 
  (rides : List Ride)
  (baseDistance : ℝ)
  (caloriesPerKm : ℝ)
  (h1 : rides.length = 10)
  (h2 : baseDistance = 15)
  (h3 : caloriesPerKm = 20)
  (h4 : rides[3].distance = adjustedDistance 16.5 baseDistance)
  (h5 : rides[6].distance = adjustedDistance 14.1 baseDistance)
  : totalCaloriesBurned rides baseDistance caloriesPerKm = 3040 := by
  sorry

end NUMINAMATH_CALUDE_marathon_calories_burned_l3812_381203


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_geq_two_l3812_381215

theorem sum_and_reciprocal_geq_two (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_geq_two_l3812_381215


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l3812_381264

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 4609 : ℤ) ≡ 2104 [ZMOD 12] ∧
  ∀ (y : ℕ), y > 0 → (y + 4609 : ℤ) ≡ 2104 [ZMOD 12] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l3812_381264


namespace NUMINAMATH_CALUDE_carly_job_applications_l3812_381244

/-- The number of job applications Carly sent to her home state -/
def home_state_apps : ℕ := 200

/-- The number of job applications Carly sent to the neighboring state -/
def neighboring_state_apps : ℕ := 2 * home_state_apps

/-- The number of job applications Carly sent to each of the other 3 states -/
def other_state_apps : ℕ := neighboring_state_apps - 50

/-- The number of other states Carly sent applications to -/
def num_other_states : ℕ := 3

/-- The total number of job applications Carly sent -/
def total_applications : ℕ := home_state_apps + neighboring_state_apps + (num_other_states * other_state_apps)

theorem carly_job_applications : total_applications = 1650 := by
  sorry

end NUMINAMATH_CALUDE_carly_job_applications_l3812_381244


namespace NUMINAMATH_CALUDE_equation_solution_l3812_381289

theorem equation_solution : 
  ∀ x : ℝ, 
    (9 / (Real.sqrt (x - 5) - 10) + 
     2 / (Real.sqrt (x - 5) - 5) + 
     8 / (Real.sqrt (x - 5) + 5) + 
     15 / (Real.sqrt (x - 5) + 10) = 0) ↔ 
    (x = 14 ∨ x = 1335 / 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3812_381289


namespace NUMINAMATH_CALUDE_spade_or_king_probability_l3812_381253

/-- The probability of drawing a spade or a king from a standard deck of cards -/
theorem spade_or_king_probability (total_cards : ℕ) (spades : ℕ) (kings : ℕ) (overlap : ℕ) :
  total_cards = 52 →
  spades = 13 →
  kings = 4 →
  overlap = 1 →
  (spades + kings - overlap : ℚ) / total_cards = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_spade_or_king_probability_l3812_381253
