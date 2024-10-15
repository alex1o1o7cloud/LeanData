import Mathlib

namespace NUMINAMATH_CALUDE_investment_equation_l2323_232350

/-- Proves that the total amount invested satisfies the given equation based on the problem conditions -/
theorem investment_equation (total_interest : ℝ) (higher_rate_fraction : ℝ) 
  (lower_rate : ℝ) (higher_rate : ℝ) :
  total_interest = 1440 →
  higher_rate_fraction = 0.55 →
  lower_rate = 0.06 →
  higher_rate = 0.09 →
  ∃ T : ℝ, 0.0765 * T = 1440 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_equation_l2323_232350


namespace NUMINAMATH_CALUDE_certain_number_proof_l2323_232375

theorem certain_number_proof (D S X : ℤ) : 
  D = 20 → S = 55 → X + (D - S) = 3 * D - 90 → X = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2323_232375


namespace NUMINAMATH_CALUDE_deposit_withdrawal_amount_l2323_232374

/-- Proves that the total amount withdrawn after 4 years of annual deposits
    with compound interest is equal to (a/p) * ((1+p)^5 - (1+p)),
    where a is the annual deposit amount and p is the interest rate. -/
theorem deposit_withdrawal_amount (a p : ℝ) (h₁ : a > 0) (h₂ : p > 0) :
  a * (1 + p)^4 + a * (1 + p)^3 + a * (1 + p)^2 + a * (1 + p) + a = 
  (a / p) * ((1 + p)^5 - (1 + p)) :=
sorry

end NUMINAMATH_CALUDE_deposit_withdrawal_amount_l2323_232374


namespace NUMINAMATH_CALUDE_bread_cost_l2323_232326

def total_cost : ℕ := 42
def banana_cost : ℕ := 12
def milk_cost : ℕ := 7
def apple_cost : ℕ := 14

theorem bread_cost : 
  total_cost - (banana_cost + milk_cost + apple_cost) = 9 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l2323_232326


namespace NUMINAMATH_CALUDE_midpoint_ordinate_l2323_232358

theorem midpoint_ordinate (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  let P : Real × Real := (a, Real.sin a)
  let Q : Real × Real := (a, Real.cos a)
  let distance := |P.2 - Q.2|
  let midpoint_y := (P.2 + Q.2) / 2
  distance = 1/4 → midpoint_y = Real.sqrt 31 / 8 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_ordinate_l2323_232358


namespace NUMINAMATH_CALUDE_chess_team_probability_l2323_232390

def chess_club_size : ℕ := 20
def num_boys : ℕ := 12
def num_girls : ℕ := 8
def team_size : ℕ := 4

theorem chess_team_probability :
  let total_combinations := Nat.choose chess_club_size team_size
  let all_boys_combinations := Nat.choose num_boys team_size
  let all_girls_combinations := Nat.choose num_girls team_size
  let probability_at_least_one_each := 1 - (all_boys_combinations + all_girls_combinations : ℚ) / total_combinations
  probability_at_least_one_each = 4280 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_probability_l2323_232390


namespace NUMINAMATH_CALUDE_exponential_grows_faster_than_quadratic_l2323_232316

theorem exponential_grows_faster_than_quadratic : 
  ∀ ε > 0, ∃ x₀ > 0, ∀ x > x₀, (2:ℝ)^x > x^2 + ε := by
  sorry

end NUMINAMATH_CALUDE_exponential_grows_faster_than_quadratic_l2323_232316


namespace NUMINAMATH_CALUDE_vector_properties_l2323_232334

def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (1, 2)

theorem vector_properties :
  let cos_angle := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  let projection := ((a.1 + b.1) * a.1 + (a.2 + b.2) * a.2) / Real.sqrt (a.1^2 + a.2^2)
  cos_angle = 4/5 ∧ projection = 14 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l2323_232334


namespace NUMINAMATH_CALUDE_crayons_per_pack_l2323_232329

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) :
  total_crayons / num_packs = 15 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_pack_l2323_232329


namespace NUMINAMATH_CALUDE_partnership_profit_l2323_232322

/-- Partnership profit calculation -/
theorem partnership_profit (a b c : ℚ) (b_share : ℚ) : 
  a = 3 * b ∧ b = (2/3) * c ∧ b_share = 600 →
  (11/2) * b_share = 3300 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l2323_232322


namespace NUMINAMATH_CALUDE_max_projection_area_parallelepiped_l2323_232333

/-- The maximum area of the projection of a rectangular parallelepiped with edge lengths √70, √99, and √126 onto any plane is 168. -/
theorem max_projection_area_parallelepiped :
  let a := Real.sqrt 70
  let b := Real.sqrt 99
  let c := Real.sqrt 126
  ∃ (proj : ℝ → ℝ → ℝ → ℝ), 
    (∀ x y z, proj x y z ≤ 168) ∧ 
    (∃ x y z, proj x y z = 168) :=
by sorry

end NUMINAMATH_CALUDE_max_projection_area_parallelepiped_l2323_232333


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2323_232311

/-- Given a quadratic equation (k+2)x^2 + 6x + k^2 + k - 2 = 0 where one of its roots is 0,
    prove that k = 1 -/
theorem quadratic_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0) ∧ 
  ((k + 2) * 0^2 + 6 * 0 + k^2 + k - 2 = 0) →
  k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2323_232311


namespace NUMINAMATH_CALUDE_Z_set_eq_roster_l2323_232341

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the set we want to prove
def Z_set : Set ℂ := {z | ∃ n : ℤ, z = i^n + i^(-n)}

-- The theorem to prove
theorem Z_set_eq_roster : Z_set = {0, 2, -2} := by sorry

end NUMINAMATH_CALUDE_Z_set_eq_roster_l2323_232341


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2323_232373

theorem arithmetic_calculation : 2354 + 240 / 60 - 354 * 2 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2323_232373


namespace NUMINAMATH_CALUDE_remainder_of_2007_pow_2008_mod_10_l2323_232380

theorem remainder_of_2007_pow_2008_mod_10 : 2007^2008 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2007_pow_2008_mod_10_l2323_232380


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_4_and_16_l2323_232368

theorem arithmetic_mean_of_4_and_16 (x : ℝ) :
  x = (4 + 16) / 2 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_4_and_16_l2323_232368


namespace NUMINAMATH_CALUDE_nails_per_plank_l2323_232356

theorem nails_per_plank (total_planks : ℕ) (total_nails : ℕ) 
  (h1 : total_planks = 16) (h2 : total_nails = 32) : 
  total_nails / total_planks = 2 := by
  sorry

end NUMINAMATH_CALUDE_nails_per_plank_l2323_232356


namespace NUMINAMATH_CALUDE_rectangle_division_exists_l2323_232313

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a division of a rectangle into parts -/
structure RectangleDivision where
  parts : List ℝ

/-- Checks if a division is valid for a given rectangle -/
def isValidDivision (r : Rectangle) (d : RectangleDivision) : Prop :=
  d.parts.length = 4 ∧ d.parts.sum = r.width * r.height

/-- The main theorem to be proved -/
theorem rectangle_division_exists : ∃ (d : RectangleDivision), 
  isValidDivision ⟨6, 10⟩ d ∧ 
  d.parts = [8, 12, 16, 24] := by
  sorry


end NUMINAMATH_CALUDE_rectangle_division_exists_l2323_232313


namespace NUMINAMATH_CALUDE_bucket_fill_lcm_l2323_232335

/-- Time to fill bucket A completely -/
def time_A : ℕ := 135

/-- Time to fill bucket B completely -/
def time_B : ℕ := 240

/-- Time to fill bucket C completely -/
def time_C : ℕ := 200

/-- Function to calculate the least common multiple of three natural numbers -/
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem bucket_fill_lcm :
  (2 * time_A = 3 * 90) ∧
  (time_B = 2 * 120) ∧
  (3 * time_C = 4 * 150) →
  lcm_three time_A time_B time_C = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bucket_fill_lcm_l2323_232335


namespace NUMINAMATH_CALUDE_union_complement_equality_l2323_232384

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l2323_232384


namespace NUMINAMATH_CALUDE_willow_play_time_l2323_232319

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Willow played football in minutes -/
def football_time : ℕ := 60

/-- The time Willow played basketball in minutes -/
def basketball_time : ℕ := 60

/-- The total time Willow played in hours -/
def total_time_hours : ℚ := (football_time + basketball_time) / minutes_per_hour

theorem willow_play_time : total_time_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_willow_play_time_l2323_232319


namespace NUMINAMATH_CALUDE_paper_folding_height_l2323_232383

/-- Given a square piece of paper with side length 100 cm, 
    with cuts from each corner starting 8 cm from the corner and meeting at 45°,
    prove that the perpendicular height of the folded shape is 8 cm. -/
theorem paper_folding_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 100 →
  cut_distance = 8 →
  cut_angle = 45 →
  let diagonal_length := side_length * Real.sqrt 2
  let cut_length := cut_distance * Real.sqrt 2
  let height := Real.sqrt (cut_length^2 - (cut_length / 2)^2)
  height = 8 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_height_l2323_232383


namespace NUMINAMATH_CALUDE_monotonic_range_of_a_l2323_232351

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * Real.exp (a * x)

/-- The function f is monotonic on ℝ -/
def is_monotonic (a : ℝ) : Prop :=
  Monotone (f a) ∨ StrictMono (f a)

/-- The theorem stating the range of a for which f is monotonic -/
theorem monotonic_range_of_a :
  ∀ a : ℝ, is_monotonic a ↔ a ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ioo 1 (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_range_of_a_l2323_232351


namespace NUMINAMATH_CALUDE_rectangle_area_y_value_l2323_232391

/-- A rectangle with vertices at (-2, y), (10, y), (-2, 1), and (10, 1) has an area of 108 square units. Prove that y = 10. -/
theorem rectangle_area_y_value (y : ℝ) : 
  y > 0 → -- y is positive
  (10 - (-2)) * (y - 1) = 108 → -- area of the rectangle is 108 square units
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_y_value_l2323_232391


namespace NUMINAMATH_CALUDE_quadratic_roots_and_k_l2323_232340

/-- The quadratic equation x^2 - (k+2)x + 2k - 1 = 0 -/
def quadratic (k x : ℝ) : Prop :=
  x^2 - (k+2)*x + 2*k - 1 = 0

theorem quadratic_roots_and_k :
  (∀ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ quadratic k x ∧ quadratic k y) ∧
  (∃ k : ℝ, quadratic k 3 ∧ quadratic k 1 ∧ k = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_k_l2323_232340


namespace NUMINAMATH_CALUDE_permutation_combination_relation_l2323_232348

-- Define permutation function
def p (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

-- Define combination function
def c (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem permutation_combination_relation :
  ∃ k : ℕ, p 32 6 = k * c 32 6 ∧ k = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_relation_l2323_232348


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_l2323_232398

theorem quadratic_equation_single_solution :
  ∀ b : ℝ, b ≠ 0 →
  (∃! x : ℝ, b * x^2 - 24 * x + 6 = 0) →
  (∃ x : ℝ, b * x^2 - 24 * x + 6 = 0 ∧ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_l2323_232398


namespace NUMINAMATH_CALUDE_triangle_problem_l2323_232325

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  -- Given condition
  2 * b * Real.cos C = a * Real.cos C + c * Real.cos A →
  -- Part 1: Prove C = π/3
  C = π/3 ∧
  -- Part 2: Given additional conditions
  (b = 2 ∧ c = Real.sqrt 7 →
    -- Prove a = 3
    a = 3 ∧
    -- Prove area = 3√3/2
    1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2323_232325


namespace NUMINAMATH_CALUDE_alternate_shading_six_by_six_l2323_232357

theorem alternate_shading_six_by_six (grid_size : Nat) (shaded_squares : Nat) :
  grid_size = 6 → shaded_squares = 18 → (shaded_squares : ℚ) / (grid_size * grid_size) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_alternate_shading_six_by_six_l2323_232357


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2323_232301

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the folding and crease
structure Folding (rect : Rectangle) :=
  (A' : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

-- Define the given dimensions
def given_dimensions (rect : Rectangle) (fold : Folding rect) : Prop :=
  let (ax, ay) := rect.A
  let (ex, ey) := fold.E
  let (fx, fy) := fold.F
  let (cx, cy) := rect.C
  Real.sqrt ((ax - ex)^2 + (ay - ey)^2) = 6 ∧
  Real.sqrt ((ex - rect.B.1)^2 + (ey - rect.B.2)^2) = 15 ∧
  Real.sqrt ((cx - fx)^2 + (cy - fy)^2) = 5

-- Define the theorem
theorem rectangle_perimeter (rect : Rectangle) (fold : Folding rect) :
  given_dimensions rect fold →
  (let perimeter := 2 * (Real.sqrt ((rect.A.1 - rect.B.1)^2 + (rect.A.2 - rect.B.2)^2) +
                         Real.sqrt ((rect.B.1 - rect.C.1)^2 + (rect.B.2 - rect.C.2)^2))
   perimeter = 808) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_l2323_232301


namespace NUMINAMATH_CALUDE_remainder_problem_l2323_232323

theorem remainder_problem (d : ℤ) (r : ℤ) 
  (h1 : d > 1)
  (h2 : 1237 % d = r)
  (h3 : 1694 % d = r)
  (h4 : 2791 % d = r) :
  d - r = 134 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2323_232323


namespace NUMINAMATH_CALUDE_exists_negative_greater_than_neg_half_l2323_232361

theorem exists_negative_greater_than_neg_half : ∃ x : ℚ, -1/2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_negative_greater_than_neg_half_l2323_232361


namespace NUMINAMATH_CALUDE_factor_theorem_application_l2323_232305

theorem factor_theorem_application (c : ℝ) : 
  (∀ x : ℝ, (x + 7) ∣ (c * x^3 + 19 * x^2 - 3 * c * x + 35)) → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l2323_232305


namespace NUMINAMATH_CALUDE_cat_dog_ratio_l2323_232328

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
theorem cat_dog_ratio (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) (num_dogs : ℕ) :
  cat_ratio ≠ 0 ∧ dog_ratio ≠ 0 →
  cat_ratio * num_dogs = dog_ratio * num_cats →
  cat_ratio = 4 ∧ dog_ratio = 5 ∧ num_cats = 24 →
  num_dogs = 30 := by
  sorry

#check cat_dog_ratio

end NUMINAMATH_CALUDE_cat_dog_ratio_l2323_232328


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2323_232352

/-- Given a rhombus with area K and diagonals d and 3d, prove its side length. -/
theorem rhombus_side_length (K d : ℝ) (h1 : K > 0) (h2 : d > 0) : ∃ s : ℝ,
  (K = (3 * d^2) / 2) →  -- Area formula for rhombus
  (s^2 = (d^2 / 4) + ((3 * d)^2 / 4)) →  -- Pythagorean theorem for side length
  s = Real.sqrt ((5 * K) / 3) :=
sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2323_232352


namespace NUMINAMATH_CALUDE_angus_patrick_diff_l2323_232369

/-- The number of fish caught by Ollie -/
def ollie_catch : ℕ := 5

/-- The number of fish caught by Patrick -/
def patrick_catch : ℕ := 8

/-- The difference between Angus and Ollie's catch -/
def angus_ollie_diff : ℕ := 7

/-- The number of fish caught by Angus -/
def angus_catch : ℕ := ollie_catch + angus_ollie_diff

/-- Theorem: The difference between Angus and Patrick's fish catch is 4 -/
theorem angus_patrick_diff : angus_catch - patrick_catch = 4 := by
  sorry

end NUMINAMATH_CALUDE_angus_patrick_diff_l2323_232369


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l2323_232318

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_eq_9 : x + y + z = 9) : 
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l2323_232318


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2323_232310

theorem square_sum_theorem (p q : ℝ) 
  (h1 : p * q = 9)
  (h2 : p^2 * q + q^2 * p + p + q = 70) :
  p^2 + q^2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2323_232310


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_theorem_l2323_232381

-- Define the condition
def condition (x y : ℝ) : Prop := (x + 1) * (y + 2) = 8

-- Define the main theorem
theorem inequality_theorem (x y : ℝ) (h : condition x y) :
  (x * y - 10)^2 ≥ 64 ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) :=
by sorry

-- Define the equality cases
def equality_cases (x y : ℝ) : Prop :=
  (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)

-- Theorem for the equality cases
theorem equality_theorem (x y : ℝ) (h : condition x y) :
  (x * y - 10)^2 = 64 ↔ equality_cases x y :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_theorem_l2323_232381


namespace NUMINAMATH_CALUDE_curry_house_spicy_curries_l2323_232371

/-- Represents the curry house's pepper buying strategy -/
structure CurryHouse where
  very_spicy_peppers : ℕ := 3
  spicy_peppers : ℕ := 2
  mild_peppers : ℕ := 1
  prev_very_spicy : ℕ := 30
  prev_spicy : ℕ := 30
  prev_mild : ℕ := 10
  new_mild : ℕ := 90
  pepper_reduction : ℕ := 40

/-- Calculates the number of spicy curries the curry house now buys peppers for -/
def calculate_new_spicy_curries (ch : CurryHouse) : ℕ :=
  let prev_total := ch.very_spicy_peppers * ch.prev_very_spicy + 
                    ch.spicy_peppers * ch.prev_spicy + 
                    ch.mild_peppers * ch.prev_mild
  let new_total := prev_total - ch.pepper_reduction
  (new_total - ch.mild_peppers * ch.new_mild) / ch.spicy_peppers

/-- Proves that the curry house now buys peppers for 15 spicy curries -/
theorem curry_house_spicy_curries (ch : CurryHouse) : 
  calculate_new_spicy_curries ch = 15 := by
  sorry

end NUMINAMATH_CALUDE_curry_house_spicy_curries_l2323_232371


namespace NUMINAMATH_CALUDE_sector_central_angle_l2323_232342

/-- Given a circular sector with perimeter 8 and area 4, prove that its central angle is 2 radians -/
theorem sector_central_angle (r : ℝ) (α : ℝ) : 
  r + r + r * α = 8 → -- perimeter condition
  (1/2) * r^2 * α = 4 → -- area condition
  α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2323_232342


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_l2323_232343

theorem smallest_k_for_64_power (k : ℕ) (some_exponent : ℕ) : k = 6 → some_exponent < 18 → 64^k > 4^some_exponent := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_l2323_232343


namespace NUMINAMATH_CALUDE_mother_twice_lucy_age_year_l2323_232360

def lucy_age_2006 : ℕ := 10
def mother_age_2006 : ℕ := 5 * lucy_age_2006

def year_mother_twice_lucy (y : ℕ) : Prop :=
  mother_age_2006 + (y - 2006) = 2 * (lucy_age_2006 + (y - 2006))

theorem mother_twice_lucy_age_year :
  ∃ y : ℕ, y = 2036 ∧ year_mother_twice_lucy y := by sorry

end NUMINAMATH_CALUDE_mother_twice_lucy_age_year_l2323_232360


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_equation_l2323_232336

theorem ordered_pairs_satisfying_equation : 
  ∃! (n : ℕ), n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ 
      a * b + 80 = 15 * Nat.lcm a b + 10 * Nat.gcd a b)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_equation_l2323_232336


namespace NUMINAMATH_CALUDE_field_trip_total_l2323_232309

/-- Field trip problem -/
theorem field_trip_total (
  num_vans : ℕ) (num_minibusses : ℕ) (num_coach_buses : ℕ)
  (students_per_van : ℕ) (teachers_per_van : ℕ) (parents_per_van : ℕ)
  (students_per_minibus : ℕ) (teachers_per_minibus : ℕ) (parents_per_minibus : ℕ)
  (students_per_coach : ℕ) (teachers_per_coach : ℕ) (parents_per_coach : ℕ)
  (h1 : num_vans = 6)
  (h2 : num_minibusses = 4)
  (h3 : num_coach_buses = 2)
  (h4 : students_per_van = 10)
  (h5 : teachers_per_van = 2)
  (h6 : parents_per_van = 1)
  (h7 : students_per_minibus = 24)
  (h8 : teachers_per_minibus = 3)
  (h9 : parents_per_minibus = 2)
  (h10 : students_per_coach = 48)
  (h11 : teachers_per_coach = 4)
  (h12 : parents_per_coach = 4) :
  (num_vans * (students_per_van + teachers_per_van + parents_per_van) +
   num_minibusses * (students_per_minibus + teachers_per_minibus + parents_per_minibus) +
   num_coach_buses * (students_per_coach + teachers_per_coach + parents_per_coach)) = 306 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_total_l2323_232309


namespace NUMINAMATH_CALUDE_banana_apple_ratio_l2323_232320

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  oranges : ℕ
  apples : ℕ
  bananas : ℕ
  peaches : ℕ

/-- Checks if the fruit basket satisfies the given conditions -/
def validBasket (basket : FruitBasket) : Prop :=
  basket.oranges = 6 ∧
  basket.apples = basket.oranges - 2 ∧
  basket.peaches * 2 = basket.bananas ∧
  basket.oranges + basket.apples + basket.bananas + basket.peaches = 28

/-- Theorem stating that in a valid fruit basket, the ratio of bananas to apples is 3:1 -/
theorem banana_apple_ratio (basket : FruitBasket) (h : validBasket basket) :
  basket.bananas = 3 * basket.apples := by
  sorry

end NUMINAMATH_CALUDE_banana_apple_ratio_l2323_232320


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2323_232353

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 100 ∧ 
    100 ≤ n ∧ n < 1000 ∧
    75 * n % 450 = 300 % 450 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m < n → 75 * m % 450 ≠ 300 % 450) := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2323_232353


namespace NUMINAMATH_CALUDE_block_running_difference_l2323_232346

theorem block_running_difference (inner_side_length outer_side_length : ℝ) 
  (h1 : inner_side_length = 450)
  (h2 : outer_side_length = inner_side_length + 50) : 
  4 * outer_side_length - 4 * inner_side_length = 200 :=
by sorry

end NUMINAMATH_CALUDE_block_running_difference_l2323_232346


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2323_232345

def U : Set ℕ := {x | x ≥ 3}
def A : Set ℕ := {x | x^2 ≥ 10}

theorem complement_A_in_U : U \ A = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2323_232345


namespace NUMINAMATH_CALUDE_union_equal_iff_x_zero_l2323_232349

def A (x : ℝ) : Set ℝ := {0, Real.exp x}
def B : Set ℝ := {-1, 0, 1}

theorem union_equal_iff_x_zero (x : ℝ) : A x ∪ B = B ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equal_iff_x_zero_l2323_232349


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2323_232379

theorem necessary_not_sufficient (a b c d : ℝ) (h : c > d) :
  (∀ a b, (a - c > b - d) → (a > b)) ∧
  (∃ a b, (a > b) ∧ ¬(a - c > b - d)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2323_232379


namespace NUMINAMATH_CALUDE_units_digit_17_2025_l2323_232399

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property that 17^n and 7^n have the same units digit for all n -/
axiom units_digit_17_7 (n : ℕ) : unitsDigit (17^n) = unitsDigit (7^n)

/-- The main theorem: the units digit of 17^2025 is 7 -/
theorem units_digit_17_2025 : unitsDigit (17^2025) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_2025_l2323_232399


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l2323_232370

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l2323_232370


namespace NUMINAMATH_CALUDE_exists_sum_of_digits_div_13_l2323_232396

def sumOfDigits (n : ℕ) : ℕ := sorry

theorem exists_sum_of_digits_div_13 (n : ℕ) : 
  ∃ k ∈ Finset.range 79, (sumOfDigits (n + k)) % 13 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_sum_of_digits_div_13_l2323_232396


namespace NUMINAMATH_CALUDE_womens_haircut_cost_l2323_232367

theorem womens_haircut_cost :
  let childrens_haircut_cost : ℝ := 36
  let num_children : ℕ := 2
  let tip_percentage : ℝ := 0.20
  let tip_amount : ℝ := 24
  let womens_haircut_cost : ℝ := 48
  tip_amount = tip_percentage * (womens_haircut_cost + num_children * childrens_haircut_cost) :=
by
  sorry

end NUMINAMATH_CALUDE_womens_haircut_cost_l2323_232367


namespace NUMINAMATH_CALUDE_matrix_N_property_l2323_232382

theorem matrix_N_property :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
    (∀ (u : Fin 3 → ℝ), N.mulVec u = (3 : ℝ) • u) ∧
    N = !![3, 0, 0; 0, 3, 0; 0, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_property_l2323_232382


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2323_232327

/-- The volume of a cube given its space diagonal length. -/
theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  (d / Real.sqrt 3) ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2323_232327


namespace NUMINAMATH_CALUDE_employee_relocation_l2323_232393

theorem employee_relocation (E : ℝ) 
  (prefer_Y : ℝ) (prefer_X : ℝ) (max_preferred : ℝ) 
  (h1 : prefer_Y = 0.4 * E)
  (h2 : prefer_X = 0.6 * E)
  (h3 : max_preferred = 140)
  (h4 : prefer_Y + prefer_X = max_preferred) :
  prefer_X / E = 0.6 := by
sorry

end NUMINAMATH_CALUDE_employee_relocation_l2323_232393


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l2323_232388

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (speed : ℝ) (total_time : ℝ) (break_time : ℝ) (distance : ℝ) : 
  speed = 2 / 10 → 
  total_time = 30 → 
  break_time = 5 → 
  distance = speed * (total_time - break_time) → 
  distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_suzanna_bike_ride_l2323_232388


namespace NUMINAMATH_CALUDE_odd_function_zero_l2323_232366

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_l2323_232366


namespace NUMINAMATH_CALUDE_unruly_quadratic_max_sum_of_roots_l2323_232315

/-- A quadratic polynomial of the form q(x) = (x-r)^2 - s -/
def QuadraticPolynomial (r s : ℝ) (x : ℝ) : ℝ := (x - r)^2 - s

/-- The composition of a quadratic polynomial with itself -/
def ComposedQuadratic (r s : ℝ) (x : ℝ) : ℝ :=
  QuadraticPolynomial r s (QuadraticPolynomial r s x)

/-- Predicate for an unruly quadratic polynomial -/
def IsUnruly (r s : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), 
    (ComposedQuadratic r s x₁ = 0 ∧
     ComposedQuadratic r s x₂ = 0 ∧
     ComposedQuadratic r s x₃ = 0) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (∃ (x₄ : ℝ), ComposedQuadratic r s x₄ = 0 ∧
                 (∀ (x : ℝ), ComposedQuadratic r s x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄))

/-- The sum of roots of q(x) = 0 -/
def SumOfRoots (r s : ℝ) : ℝ := 2 * r

theorem unruly_quadratic_max_sum_of_roots :
  ∃ (r s : ℝ), IsUnruly r s ∧
    (∀ (r' s' : ℝ), IsUnruly r' s' → SumOfRoots r s ≥ SumOfRoots r' s') ∧
    QuadraticPolynomial r s 1 = 7/4 :=
sorry

end NUMINAMATH_CALUDE_unruly_quadratic_max_sum_of_roots_l2323_232315


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2323_232394

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

-- Define the translation
def translate_left : ℝ := 3
def translate_up : ℝ := 2

-- Define parabola C after translation
def parabola_C (x : ℝ) : ℝ := original_parabola (x + translate_left) + translate_up

-- Define the symmetric parabola
def symmetric_parabola (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

-- Theorem statement
theorem parabola_symmetry :
  ∀ x : ℝ, parabola_C (-x) = symmetric_parabola x :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2323_232394


namespace NUMINAMATH_CALUDE_marys_age_l2323_232378

theorem marys_age :
  ∃! x : ℕ, 
    (∃ n : ℕ, x - 2 = n^2) ∧ 
    (∃ m : ℕ, x + 2 = m^3) ∧ 
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_marys_age_l2323_232378


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l2323_232376

theorem absolute_value_not_positive (x : ℚ) : 
  |4 * x - 2| ≤ 0 ↔ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l2323_232376


namespace NUMINAMATH_CALUDE_system_solution_implies_m_zero_l2323_232392

theorem system_solution_implies_m_zero (x y m : ℝ) :
  (2 * x + 3 * y = 4) →
  (3 * x + 2 * y = 2 * m - 3) →
  (x + y = 1 / 5) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_m_zero_l2323_232392


namespace NUMINAMATH_CALUDE_susan_age_l2323_232377

theorem susan_age (susan joe billy : ℕ) 
  (h1 : susan = 2 * joe)
  (h2 : susan + joe + billy = 60)
  (h3 : billy = joe + 10) :
  susan = 25 := by sorry

end NUMINAMATH_CALUDE_susan_age_l2323_232377


namespace NUMINAMATH_CALUDE_cityD_highest_increase_l2323_232347

structure City where
  name : String
  population1990 : ℕ
  population2000 : ℕ

def percentageIncrease (city : City) : ℚ :=
  (city.population2000 : ℚ) / (city.population1990 : ℚ)

def cityA : City := ⟨"A", 45, 60⟩
def cityB : City := ⟨"B", 65, 85⟩
def cityC : City := ⟨"C", 90, 120⟩
def cityD : City := ⟨"D", 115, 160⟩
def cityE : City := ⟨"E", 150, 200⟩
def cityF : City := ⟨"F", 130, 180⟩

def cities : List City := [cityA, cityB, cityC, cityD, cityE, cityF]

theorem cityD_highest_increase :
  ∀ city ∈ cities, percentageIncrease cityD ≥ percentageIncrease city :=
by sorry

end NUMINAMATH_CALUDE_cityD_highest_increase_l2323_232347


namespace NUMINAMATH_CALUDE_first_investment_rate_l2323_232312

/-- Represents the interest rate problem --/
structure InterestRateProblem where
  firstInvestment : ℝ
  secondInvestment : ℝ
  totalInterest : ℝ
  knownRate : ℝ
  firstRate : ℝ

/-- The interest rate problem satisfies the given conditions --/
def validProblem (p : InterestRateProblem) : Prop :=
  p.secondInvestment = p.firstInvestment - 100 ∧
  p.secondInvestment = 400 ∧
  p.knownRate = 0.07 ∧
  p.totalInterest = 73 ∧
  p.firstInvestment * p.firstRate + p.secondInvestment * p.knownRate = p.totalInterest

/-- The theorem stating that the first investment's interest rate is 0.15 --/
theorem first_investment_rate (p : InterestRateProblem) 
  (h : validProblem p) : p.firstRate = 0.15 := by
  sorry


end NUMINAMATH_CALUDE_first_investment_rate_l2323_232312


namespace NUMINAMATH_CALUDE_fraction_simplification_l2323_232337

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2323_232337


namespace NUMINAMATH_CALUDE_range_of_m_for_union_equality_l2323_232300

/-- The set A of solutions to x^2 - 3x + 2 = 0 -/
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}

/-- The set B of solutions to x^2 - 2x + m = 0, parameterized by m -/
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + m = 0}

/-- The theorem stating the range of m for which A ∪ B = A -/
theorem range_of_m_for_union_equality :
  {m : ℝ | A ∪ B m = A} = {m : ℝ | m ≥ 1} := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_union_equality_l2323_232300


namespace NUMINAMATH_CALUDE_john_soap_cost_l2323_232354

/-- The amount of money spent on soap given the number of bars, weight per bar, and price per pound -/
def soap_cost (num_bars : ℕ) (weight_per_bar : ℚ) (price_per_pound : ℚ) : ℚ :=
  num_bars * weight_per_bar * price_per_pound

/-- Theorem stating that John spent $15 on soap -/
theorem john_soap_cost : soap_cost 20 (3/2) (1/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_soap_cost_l2323_232354


namespace NUMINAMATH_CALUDE_parking_lot_motorcycles_l2323_232385

theorem parking_lot_motorcycles :
  let total_vehicles : ℕ := 24
  let total_wheels : ℕ := 86
  let car_wheels : ℕ := 4
  let motorcycle_wheels : ℕ := 3
  ∃ (cars motorcycles : ℕ),
    cars + motorcycles = total_vehicles ∧
    car_wheels * cars + motorcycle_wheels * motorcycles = total_wheels ∧
    motorcycles = 10 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_motorcycles_l2323_232385


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_for_quadratic_l2323_232359

theorem one_nonnegative_solution_for_quadratic :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -5*x := by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_for_quadratic_l2323_232359


namespace NUMINAMATH_CALUDE_son_work_time_l2323_232307

-- Define the work rates
def man_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 3

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time : (1 : ℚ) / son_rate = 6 := by sorry

end NUMINAMATH_CALUDE_son_work_time_l2323_232307


namespace NUMINAMATH_CALUDE_equation_system_solutions_l2323_232387

/-- A system of two equations with two unknowns x and y -/
def equation_system (x y : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3) = 0 ∧
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|) = 0

/-- The theorem stating that the equation system has only three specific solutions -/
theorem equation_system_solutions :
  ∀ x y : ℝ, equation_system x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l2323_232387


namespace NUMINAMATH_CALUDE_equation_solution_l2323_232397

theorem equation_solution : ∃ x : ℚ, (3/4 : ℚ) + 1/x = (7/8 : ℚ) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2323_232397


namespace NUMINAMATH_CALUDE_polynomial_division_l2323_232344

/-- The dividend polynomial -/
def dividend (x : ℚ) : ℚ := 9*x^4 + 27*x^3 - 8*x^2 + 8*x + 5

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3*x + 4

/-- The quotient polynomial -/
def quotient (x : ℚ) : ℚ := 3*x^3 + 5*x^2 - (28/3)*x + 136/9

/-- The remainder -/
def remainder : ℚ := 5 - 544/9

theorem polynomial_division :
  ∀ x, dividend x = divisor x * quotient x + remainder := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2323_232344


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_integers_l2323_232395

theorem largest_divisor_of_consecutive_odd_integers (n : ℕ) :
  ∃ (Q : ℕ), Q = (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) ∧
  15 ∣ Q ∧
  ∀ (k : ℕ), k > 15 → ¬(∀ (m : ℕ), k ∣ ((2*m - 3) * (2*m - 1) * (2*m + 1) * (2*m + 3))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_integers_l2323_232395


namespace NUMINAMATH_CALUDE_initial_birds_on_fence_l2323_232317

theorem initial_birds_on_fence :
  ∀ (initial_birds additional_birds total_birds : ℕ),
    additional_birds = 4 →
    total_birds = 6 →
    total_birds = initial_birds + additional_birds →
    initial_birds = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_on_fence_l2323_232317


namespace NUMINAMATH_CALUDE_max_digit_sum_l2323_232389

/-- A_n is an n-digit integer with all digits equal to a -/
def A_n (a : ℕ) (n : ℕ) : ℕ := a * (10^n - 1) / 9

/-- B_n is a 2n-digit integer with all digits equal to b -/
def B_n (b : ℕ) (n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9

/-- C_n is a 3n-digit integer with all digits equal to c -/
def C_n (c : ℕ) (n : ℕ) : ℕ := c * (10^(3*n) - 1) / 9

/-- The theorem statement -/
theorem max_digit_sum (a b c : ℕ) (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10) (hc : 0 < c ∧ c < 10) :
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ 0 < n₁ ∧ 0 < n₂ ∧ 
    C_n c n₁ - A_n a n₁ = (B_n b n₁)^2 ∧
    C_n c n₂ - A_n a n₂ = (B_n b n₂)^2) →
  a + b + c ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l2323_232389


namespace NUMINAMATH_CALUDE_wade_drink_cost_l2323_232314

/-- The cost of each drink given Wade's purchases -/
theorem wade_drink_cost (total_spent : ℝ) (sandwich_cost : ℝ) (num_sandwiches : ℕ) (num_drinks : ℕ) 
  (h1 : total_spent = 26)
  (h2 : sandwich_cost = 6)
  (h3 : num_sandwiches = 3)
  (h4 : num_drinks = 2) :
  (total_spent - num_sandwiches * sandwich_cost) / num_drinks = 4 := by
  sorry

end NUMINAMATH_CALUDE_wade_drink_cost_l2323_232314


namespace NUMINAMATH_CALUDE_football_yards_lost_l2323_232386

theorem football_yards_lost (yards_gained yards_progress : ℤ) 
  (h1 : yards_gained = 8)
  (h2 : yards_progress = 3) :
  ∃ yards_lost : ℤ, yards_lost + yards_gained = yards_progress ∧ yards_lost = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_football_yards_lost_l2323_232386


namespace NUMINAMATH_CALUDE_cube_edge_length_l2323_232308

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 16 * Real.pi) :
  ∃ (a : ℝ), a > 0 ∧ a = (4 * Real.sqrt 3) / 3 ∧ 
  surface_area = 4 * Real.pi * ((Real.sqrt 3 * a) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2323_232308


namespace NUMINAMATH_CALUDE_contractor_absent_days_l2323_232355

/-- Proves that given the specified contract conditions, the number of days absent is 10 -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (payment_per_day : ℚ) 
  (fine_per_day : ℚ) 
  (total_amount : ℚ) : 
  total_days = 30 ∧ 
  payment_per_day = 25 ∧ 
  fine_per_day = 7.5 ∧ 
  total_amount = 425 → 
  ∃ (days_worked : ℕ) (days_absent : ℕ), 
    days_worked + days_absent = total_days ∧ 
    days_absent = 10 ∧
    (payment_per_day * days_worked : ℚ) - (fine_per_day * days_absent : ℚ) = total_amount :=
by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l2323_232355


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2323_232321

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x, 729 * x^3 + 64 = (a*x^2 + b*x + c) * (d*x^2 + e*x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2323_232321


namespace NUMINAMATH_CALUDE_eighth_term_is_25_5_l2323_232304

/-- An arithmetic sequence with 15 terms, first term 3, and last term 48 -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  a₁₅ : ℚ
  h_n : n = 15
  h_a₁ : a₁ = 3
  h_a₁₅ : a₁₅ = 48

/-- The 8th term of the arithmetic sequence is 25.5 -/
theorem eighth_term_is_25_5 (seq : ArithmeticSequence) : 
  let d := (seq.a₁₅ - seq.a₁) / (seq.n - 1)
  seq.a₁ + 7 * d = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_25_5_l2323_232304


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2323_232324

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (3 - 5*I) / (1 - I)
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2323_232324


namespace NUMINAMATH_CALUDE_geometric_mean_of_3_and_12_l2323_232364

theorem geometric_mean_of_3_and_12 :
  let b : ℝ := 3
  let c : ℝ := 12
  Real.sqrt (b * c) = 6 := by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_3_and_12_l2323_232364


namespace NUMINAMATH_CALUDE_total_nails_calculation_l2323_232332

/-- The number of nails left at each station -/
def nails_per_station : ℕ := 7

/-- The number of stations visited -/
def stations_visited : ℕ := 20

/-- The total number of nails brought -/
def total_nails : ℕ := nails_per_station * stations_visited

theorem total_nails_calculation : total_nails = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_calculation_l2323_232332


namespace NUMINAMATH_CALUDE_regular_pyramid_cross_section_l2323_232362

/-- Regular pyramid with inscribed cross-section --/
structure RegularPyramid where
  -- Base side length
  base_side : ℝ
  -- Ratio of edge division by plane
  edge_ratio : ℝ × ℝ
  -- Ratio of volumes divided by plane
  volume_ratio : ℝ × ℝ
  -- Distance from sphere center to plane
  sphere_center_distance : ℝ
  -- Perimeter of cross-section
  cross_section_perimeter : ℝ

/-- Theorem about regular pyramid with specific cross-section --/
theorem regular_pyramid_cross_section 
  (p : RegularPyramid) 
  (h_base : p.base_side = 2) 
  (h_perimeter : p.cross_section_perimeter = 32/5) :
  p.edge_ratio = (2, 3) ∧ 
  p.volume_ratio = (26, 9) ∧ 
  p.sphere_center_distance = (22 * Real.sqrt 14) / (35 * Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_regular_pyramid_cross_section_l2323_232362


namespace NUMINAMATH_CALUDE_solution_set_f_plus_x_squared_range_of_m_l2323_232363

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 3| + m

-- Theorem 1: Solution set of |x-1| + x^2 - 1 > 0
theorem solution_set_f_plus_x_squared (x : ℝ) : 
  (|x - 1| + x^2 - 1 > 0) ↔ (x > 1 ∨ x < 0) := by sorry

-- Theorem 2: If f(x) < g(x) has a non-empty solution set, then m > 4
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < g m x) → m > 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_plus_x_squared_range_of_m_l2323_232363


namespace NUMINAMATH_CALUDE_joan_pinball_spending_l2323_232302

/-- The amount of money in dollars represented by a half-dollar -/
def half_dollar_value : ℚ := 0.5

/-- The total amount spent in dollars given the number of half-dollars spent each day -/
def total_spent (wed thur fri : ℕ) : ℚ :=
  half_dollar_value * (wed + thur + fri : ℚ)

/-- Theorem stating that if Joan spent 4 half-dollars on Wednesday, 14 on Thursday,
    and 8 on Friday, then the total amount she spent playing pinball is $13.00 -/
theorem joan_pinball_spending :
  total_spent 4 14 8 = 13 := by sorry

end NUMINAMATH_CALUDE_joan_pinball_spending_l2323_232302


namespace NUMINAMATH_CALUDE_remainder_x_50_divided_by_x2_minus_4x_plus_3_l2323_232365

theorem remainder_x_50_divided_by_x2_minus_4x_plus_3 (x : ℝ) :
  ∃ (Q : ℝ → ℝ), x^50 = (x^2 - 4*x + 3) * Q x + ((3^50 - 1)/2 * x + (5 - 3^50)/2) :=
by sorry

end NUMINAMATH_CALUDE_remainder_x_50_divided_by_x2_minus_4x_plus_3_l2323_232365


namespace NUMINAMATH_CALUDE_operation_result_l2323_232330

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def op : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem operation_result :
  op (op Element.three Element.one) (op Element.four Element.two) = Element.three := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l2323_232330


namespace NUMINAMATH_CALUDE_modular_inverse_31_mod_45_l2323_232372

theorem modular_inverse_31_mod_45 : ∃ x : ℤ, 0 ≤ x ∧ x < 45 ∧ (31 * x) % 45 = 1 := by
  use 15
  sorry

end NUMINAMATH_CALUDE_modular_inverse_31_mod_45_l2323_232372


namespace NUMINAMATH_CALUDE_homework_difference_is_two_l2323_232339

/-- The number of pages of reading homework Rachel has to complete -/
def reading_pages : ℕ := 2

/-- The number of pages of math homework Rachel has to complete -/
def math_pages : ℕ := 4

/-- The difference in pages between math and reading homework -/
def homework_difference : ℕ := math_pages - reading_pages

theorem homework_difference_is_two : homework_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_is_two_l2323_232339


namespace NUMINAMATH_CALUDE_t_plus_inverse_t_l2323_232303

theorem t_plus_inverse_t (t : ℝ) (h1 : t^2 - 3*t + 1 = 0) (h2 : t ≠ 0) : 
  t + 1/t = 3 := by
  sorry

end NUMINAMATH_CALUDE_t_plus_inverse_t_l2323_232303


namespace NUMINAMATH_CALUDE_isosceles_triangle_fold_crease_length_l2323_232306

theorem isosceles_triangle_fold_crease_length 
  (a b c : ℝ) (h_isosceles : a = b) (h_sides : a = 5 ∧ c = 6) :
  let m := c / 2
  let crease_length := Real.sqrt (a^2 + m^2)
  crease_length = Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_fold_crease_length_l2323_232306


namespace NUMINAMATH_CALUDE_quadrilateral_prism_properties_l2323_232331

structure QuadrilateralPrism where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

theorem quadrilateral_prism_properties :
  ∃ (qp : QuadrilateralPrism), qp.vertices = 8 ∧ qp.edges = 12 ∧ qp.faces = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_prism_properties_l2323_232331


namespace NUMINAMATH_CALUDE_dana_hourly_wage_l2323_232338

/-- Given a person who worked for a certain number of hours and earned a total amount,
    calculate their hourly wage. -/
def hourly_wage (hours_worked : ℕ) (total_earned : ℕ) : ℚ :=
  total_earned / hours_worked

theorem dana_hourly_wage :
  hourly_wage 22 286 = 13 := by sorry

end NUMINAMATH_CALUDE_dana_hourly_wage_l2323_232338
