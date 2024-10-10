import Mathlib

namespace limit_at_negative_one_l1810_181023

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem limit_at_negative_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |((f (-1 + Δx) - f (-1)) / Δx) - (-2)| < ε :=
sorry

end limit_at_negative_one_l1810_181023


namespace circle_equation_proof_l1810_181051

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - x + 2*y - 3 = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := 5*x + 2*y + 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the sought circle
def sought_circle (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4 = 0

-- Theorem statement
theorem circle_equation_proof :
  ∃ (D E F : ℝ),
    (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 ↔ sought_circle x y) ∧
    sought_circle point_P.1 point_P.2 ∧
    (∀ x y : ℝ, given_circle x y ∧ sought_circle x y → given_line x y) :=
sorry

end circle_equation_proof_l1810_181051


namespace oil_remaining_l1810_181054

theorem oil_remaining (x₁ x₂ x₃ : ℕ) : 
  x₁ > 0 → x₂ > 0 → x₃ > 0 →
  3 * x₁ = 2 * x₂ →
  5 * x₁ = 3 * x₃ →
  30 - (x₁ + x₂ + x₃) = 5 :=
by sorry

end oil_remaining_l1810_181054


namespace fixed_point_of_line_l1810_181092

theorem fixed_point_of_line (m : ℝ) :
  (∀ m : ℝ, ∃! p : ℝ × ℝ, m * p.1 + p.2 - 1 + 2 * m = 0) →
  (∃ p : ℝ × ℝ, p = (-2, 1) ∧ ∀ m : ℝ, m * p.1 + p.2 - 1 + 2 * m = 0) :=
by sorry

end fixed_point_of_line_l1810_181092


namespace circle_slope_range_l1810_181026

/-- The range of y/x for points on the circle x^2 + y^2 - 4x - 6y + 12 = 0 -/
theorem circle_slope_range :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 6*p.2 + 12 = 0}
  ∀ (x y : ℝ), (x, y) ∈ circle → x ≠ 0 →
    (6 - 2*Real.sqrt 3) / 3 ≤ y / x ∧ y / x ≤ (6 + 2*Real.sqrt 3) / 3 :=
by sorry


end circle_slope_range_l1810_181026


namespace cube_sum_inverse_l1810_181084

theorem cube_sum_inverse (x R S : ℝ) (hx : x ≠ 0) : 
  x + 1 / x = R → x^3 + 1 / x^3 = S → S = R^3 - 3 * R :=
by
  sorry

end cube_sum_inverse_l1810_181084


namespace absolute_value_inequality_solution_set_l1810_181067

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≤ 2} = Set.Icc (-1) 3 := by
  sorry

end absolute_value_inequality_solution_set_l1810_181067


namespace equation_solution_l1810_181062

theorem equation_solution : ∃ (x₁ x₂ : ℚ), 
  (x₁ = 1 ∧ x₂ = 2/3) ∧ 
  (∀ x : ℚ, 3*x*(x-1) = 2*x-2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l1810_181062


namespace new_energy_vehicle_sales_growth_rate_l1810_181020

theorem new_energy_vehicle_sales_growth_rate 
  (january_sales : ℕ) 
  (march_sales : ℕ) 
  (growth_rate : ℝ) : 
  january_sales = 25 → 
  march_sales = 36 → 
  (1 + growth_rate)^2 = march_sales / january_sales → 
  growth_rate = 0.2 := by
sorry

end new_energy_vehicle_sales_growth_rate_l1810_181020


namespace symmetric_points_sum_l1810_181032

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if point A(a, 1) and point B(-2, b) are symmetric with respect to the origin,
    then a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin a 1 (-2) b → a + b = 1 := by
  sorry

end symmetric_points_sum_l1810_181032


namespace imaginary_part_of_complex_fraction_l1810_181058

theorem imaginary_part_of_complex_fraction (i : ℂ) : i * i = -1 → Complex.im ((1 + i) / (1 - i)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1810_181058


namespace correct_weighted_mean_l1810_181005

theorem correct_weighted_mean (n : ℕ) (incorrect_mean : ℚ) 
  (error1 error2 error3 : ℚ) (w1 w2 w3 : ℕ) (n1 n2 n3 : ℕ) :
  n = 40 →
  incorrect_mean = 150 →
  error1 = 165 - 135 →
  error2 = 200 - 170 →
  error3 = 185 - 155 →
  w1 = 2 →
  w2 = 3 →
  w3 = 4 →
  n1 = 10 →
  n2 = 20 →
  n3 = 10 →
  n = n1 + n2 + n3 →
  let total_error := error1 + error2 + error3
  let correct_sum := n * incorrect_mean + total_error
  let total_weight := n1 * w1 + n2 * w2 + n3 * w3
  let weighted_mean := correct_sum / total_weight
  weighted_mean = 50.75 := by
sorry

end correct_weighted_mean_l1810_181005


namespace loan_years_is_eight_l1810_181010

/-- Given a loan scenario, calculate the number of years for the first part. -/
def calculate_years (total_sum interest_rate1 interest_rate2 second_part_sum second_part_years : ℚ) : ℚ :=
  let first_part_sum := total_sum - second_part_sum
  let second_part_interest := second_part_sum * interest_rate2 * second_part_years / 100
  second_part_interest * 100 / (first_part_sum * interest_rate1)

/-- Prove that the number of years for the first part of the loan is 8. -/
theorem loan_years_is_eight :
  let total_sum : ℚ := 2769
  let interest_rate1 : ℚ := 3
  let interest_rate2 : ℚ := 5
  let second_part_sum : ℚ := 1704
  let second_part_years : ℚ := 3
  calculate_years total_sum interest_rate1 interest_rate2 second_part_sum second_part_years = 8 := by
  sorry


end loan_years_is_eight_l1810_181010


namespace park_conditions_l1810_181059

-- Define the conditions
def temperature_at_least_75 : Prop := sorry
def sunny : Prop := sorry
def park_clean : Prop := sorry
def park_crowded : Prop := sorry

-- Define the main theorem
theorem park_conditions :
  (temperature_at_least_75 ∧ sunny ∧ park_clean → park_crowded) →
  (¬park_crowded → ¬temperature_at_least_75 ∨ ¬sunny ∨ ¬park_clean) :=
by sorry

end park_conditions_l1810_181059


namespace freshman_psych_liberal_arts_percentage_l1810_181082

def college_population (total : ℝ) : Prop :=
  let freshmen := 0.5 * total
  let int_freshmen := 0.3 * freshmen
  let dom_freshmen := 0.7 * freshmen
  let int_lib_arts := 0.4 * int_freshmen
  let dom_lib_arts := 0.35 * dom_freshmen
  let int_psych_lib_arts := 0.2 * int_lib_arts
  let dom_psych_lib_arts := 0.25 * dom_lib_arts
  let total_psych_lib_arts := int_psych_lib_arts + dom_psych_lib_arts
  total_psych_lib_arts / total = 0.04

theorem freshman_psych_liberal_arts_percentage :
  ∀ total : ℝ, total > 0 → college_population total :=
sorry

end freshman_psych_liberal_arts_percentage_l1810_181082


namespace farm_field_correct_l1810_181015

/-- Represents the farm field ploughing problem -/
structure FarmField where
  total_area : ℕ  -- Total area of the farm field in hectares
  planned_days : ℕ  -- Initially planned number of days
  daily_plan : ℕ  -- Hectares planned to be ploughed per day
  actual_daily : ℕ  -- Hectares actually ploughed per day
  extra_days : ℕ  -- Additional days worked
  remaining : ℕ  -- Hectares remaining to be ploughed

/-- The farm field problem solution -/
def farm_field_solution : FarmField :=
  { total_area := 720
  , planned_days := 6
  , daily_plan := 120
  , actual_daily := 85
  , extra_days := 2
  , remaining := 40 }

/-- Theorem stating the correctness of the farm field problem solution -/
theorem farm_field_correct (f : FarmField) : 
  f.daily_plan * f.planned_days = f.total_area ∧
  f.actual_daily * (f.planned_days + f.extra_days) + f.remaining = f.total_area ∧
  f = farm_field_solution := by
  sorry

#check farm_field_correct

end farm_field_correct_l1810_181015


namespace smallest_undefined_inverse_l1810_181013

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b < 14, (Nat.gcd b 70 = 1 ∨ Nat.gcd b 84 = 1)) ∧ 
  (Nat.gcd 14 70 > 1 ∧ Nat.gcd 14 84 > 1) := by
  sorry

end smallest_undefined_inverse_l1810_181013


namespace circle_tangent_range_l1810_181095

/-- The range of k values for which a circle x²+y²+2x-4y+k-2=0 allows
    two tangents from the point (1, 2) -/
theorem circle_tangent_range : 
  ∀ k : ℝ, 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + k - 2 = 0 ∧ 
   ∃ (t₁ t₂ : ℝ × ℝ), t₁ ≠ t₂ ∧ 
   (t₁.1 - 1)^2 + (t₁.2 - 2)^2 = (t₂.1 - 1)^2 + (t₂.2 - 2)^2 ∧
   (t₁.1^2 + t₁.2^2 + 2*t₁.1 - 4*t₁.2 + k - 2 = 0) ∧
   (t₂.1^2 + t₂.2^2 + 2*t₂.1 - 4*t₂.2 + k - 2 = 0)) ↔ 
  (3 < k ∧ k < 7) :=
by sorry

end circle_tangent_range_l1810_181095


namespace andy_coat_production_l1810_181022

/-- Given the conditions about minks and coat production, prove that Andy can make 7 coats. -/
theorem andy_coat_production (
  minks_per_coat : ℕ := 15
  ) (
  initial_minks : ℕ := 30
  ) (
  babies_per_mink : ℕ := 6
  ) (
  freed_fraction : ℚ := 1/2
  ) : ℕ := by
  sorry

end andy_coat_production_l1810_181022


namespace vector_projection_l1810_181037

/-- Given two 2D vectors a and b, prove that the projection of a onto b is √13/13 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (-2, 1) → b = (-2, -3) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 13 / 13 := by
  sorry

end vector_projection_l1810_181037


namespace seventh_root_unity_product_l1810_181006

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end seventh_root_unity_product_l1810_181006


namespace product_from_lcm_gcd_l1810_181042

theorem product_from_lcm_gcd : 
  ∀ a b : ℤ, (Nat.lcm a.natAbs b.natAbs = 72) → (Int.gcd a b = 8) → a * b = 576 := by
  sorry

end product_from_lcm_gcd_l1810_181042


namespace remainder_product_l1810_181030

theorem remainder_product (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (rem_a : a % 8 = 3) (rem_b : b % 6 = 5) : (a * b) % 48 = 15 := by
  sorry

end remainder_product_l1810_181030


namespace sesame_seed_weight_scientific_notation_l1810_181064

theorem sesame_seed_weight_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.00000201 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.01 ∧ n = -6 := by
  sorry

end sesame_seed_weight_scientific_notation_l1810_181064


namespace tiger_count_l1810_181071

/-- Given a zoo where the ratio of lions to tigers is 3:4 and there are 21 lions, 
    prove that the number of tigers is 28. -/
theorem tiger_count (lion_count : ℕ) (tiger_count : ℕ) : 
  (lion_count : ℚ) / tiger_count = 3 / 4 → 
  lion_count = 21 → 
  tiger_count = 28 := by
  sorry

end tiger_count_l1810_181071


namespace sqrt_comparison_l1810_181075

theorem sqrt_comparison : Real.sqrt 10 - Real.sqrt 6 < Real.sqrt 7 - Real.sqrt 3 := by
  sorry

end sqrt_comparison_l1810_181075


namespace square_perimeter_problem_l1810_181061

/-- Given squares I, II, and III, prove that the perimeter of III is 16√2 + 32 -/
theorem square_perimeter_problem (I II III : ℝ → ℝ → Prop) : 
  (∀ x, I x x → 4 * x = 16) →  -- Square I has perimeter 16
  (∀ y, II y y → 4 * y = 32) →  -- Square II has perimeter 32
  (∀ x y z, I x x → II y y → III z z → z = x * Real.sqrt 2 + y) →  -- Side of III is diagonal of I plus side of II
  (∃ z, III z z ∧ 4 * z = 16 * Real.sqrt 2 + 32) :=  -- Perimeter of III is 16√2 + 32
by sorry

end square_perimeter_problem_l1810_181061


namespace negative_half_less_than_negative_third_l1810_181050

theorem negative_half_less_than_negative_third : -1/2 < -1/3 := by
  sorry

end negative_half_less_than_negative_third_l1810_181050


namespace rational_additive_function_is_linear_l1810_181034

theorem rational_additive_function_is_linear 
  (f : ℚ → ℚ) 
  (h : ∀ (x y : ℚ), f (x + y) = f x + f y) : 
  ∃ (c : ℚ), ∀ (x : ℚ), f x = c * x := by
sorry

end rational_additive_function_is_linear_l1810_181034


namespace elvis_song_writing_time_l1810_181073

/-- Given Elvis's album production parameters, prove the time to write each song. -/
theorem elvis_song_writing_time
  (total_songs : ℕ)
  (studio_time_hours : ℕ)
  (recording_time_per_song : ℕ)
  (total_editing_time : ℕ)
  (h1 : total_songs = 15)
  (h2 : studio_time_hours = 7)
  (h3 : recording_time_per_song = 18)
  (h4 : total_editing_time = 45) :
  (studio_time_hours * 60 - recording_time_per_song * total_songs - total_editing_time) / total_songs = 7 :=
by sorry

end elvis_song_writing_time_l1810_181073


namespace quadratic_equal_roots_l1810_181017

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + 6*y + m = 0 → y = x) ↔ 
  m = 9 := by sorry

end quadratic_equal_roots_l1810_181017


namespace power_equality_l1810_181038

theorem power_equality : 32^3 * 8^4 = 2^27 := by
  sorry

end power_equality_l1810_181038


namespace max_angle_C_in_triangle_l1810_181074

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² + b² = 2c², then the maximum value of angle C is π/3 -/
theorem max_angle_C_in_triangle (a b c : ℝ) (h : a^2 + b^2 = 2*c^2) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧ 
    A + B + C = π ∧
    c = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) ∧
    C ≤ π/3 ∧
    (C = π/3 → a = b) := by
  sorry

end max_angle_C_in_triangle_l1810_181074


namespace phoenix_airport_on_time_rate_l1810_181098

def total_flights : ℕ := 8
def late_flights : ℕ := 1
def initial_on_time_flights : ℕ := 3
def subsequent_on_time_flights : ℕ := 4
def target_rate : ℚ := 4/5

def on_time_rate (total : ℕ) (on_time : ℕ) : ℚ :=
  (on_time : ℚ) / (total : ℚ)

theorem phoenix_airport_on_time_rate :
  on_time_rate total_flights (initial_on_time_flights + subsequent_on_time_flights) > target_rate := by
  sorry

end phoenix_airport_on_time_rate_l1810_181098


namespace zero_of_f_l1810_181055

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- The zero of f(x) is -1 -/
theorem zero_of_f : f (-1) = 0 := by sorry

end zero_of_f_l1810_181055


namespace max_a_proof_l1810_181088

/-- The coefficient of x^4 in the expansion of (1 - 2x + ax^2)^8 -/
def coeff_x4 (a : ℝ) : ℝ := 28 * a^2 + 336 * a + 1120

/-- The maximum value of a that satisfies the condition -/
def max_a : ℝ := -5

theorem max_a_proof :
  (∀ a : ℝ, coeff_x4 a = -1540 → a ≤ max_a) ∧
  coeff_x4 max_a = -1540 := by sorry

end max_a_proof_l1810_181088


namespace inverse_proportion_l1810_181016

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 10 * 3 = k) :
  -15 * -2 = k := by
  sorry

end inverse_proportion_l1810_181016


namespace age_ratio_problem_l1810_181009

/-- Given two people p and q, where 8 years ago p was half of q's age,
    and the total of their present ages is 28,
    prove that the ratio of their present ages is 3:4 -/
theorem age_ratio_problem (p q : ℕ) : 
  (p - 8 = (q - 8) / 2) → 
  (p + q = 28) → 
  (p : ℚ) / q = 3 / 4 := by
  sorry

end age_ratio_problem_l1810_181009


namespace hedgehog_strawberries_l1810_181049

theorem hedgehog_strawberries : 
  ∀ (num_hedgehogs num_baskets strawberries_per_basket : ℕ) 
    (remaining_fraction : ℚ),
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  remaining_fraction = 2 / 9 →
  ∃ (strawberries_eaten_per_hedgehog : ℕ),
    strawberries_eaten_per_hedgehog = 1050 ∧
    (num_baskets * strawberries_per_basket) * (1 - remaining_fraction) = 
      num_hedgehogs * strawberries_eaten_per_hedgehog :=
by sorry

end hedgehog_strawberries_l1810_181049


namespace polynomial_factorization_l1810_181035

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 3) * (x^2 + 7*x + 10) + (x^2 + 7*x + 10) =
  (x^2 + 7*x + 20) * (x^2 + 7*x + 6) := by
  sorry

end polynomial_factorization_l1810_181035


namespace line_parallel_to_plane_l1810_181003

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel, perpendicular, and subset relations
variable (parallel : Line → Plane → Prop)
variable (perp : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n)
  (h2 : perp m n) 
  (h3 : perp_plane n α) 
  (h4 : ¬ subset m α) : 
  parallel m α :=
sorry

end line_parallel_to_plane_l1810_181003


namespace always_judge_available_l1810_181053

/-- Represents a tennis tournament in a sports club -/
structure TennisTournament where
  n : ℕ  -- number of matches played so far
  eliminated : ℕ  -- number of eliminated players
  judges : ℕ  -- number of players who have judged a match

/-- The state of the tournament after n matches -/
def tournamentState (n : ℕ) : TennisTournament :=
  { n := n
  , eliminated := n  -- each match eliminates one player
  , judges := if n = 0 then 0 else n - 1 }  -- judges needed for all but the first match

/-- There is always someone available to judge the next match -/
theorem always_judge_available (n : ℕ) :
  let t := tournamentState n
  t.eliminated > t.judges :=
by sorry

end always_judge_available_l1810_181053


namespace sqrt_72_simplification_l1810_181029

theorem sqrt_72_simplification : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_simplification_l1810_181029


namespace runner_speed_problem_l1810_181048

theorem runner_speed_problem (total_distance : ℝ) (total_time : ℝ) (first_segment_distance : ℝ) (first_segment_speed : ℝ) (last_segment_distance : ℝ) :
  total_distance = 16 →
  total_time = 1.5 →
  first_segment_distance = 10 →
  first_segment_speed = 12 →
  last_segment_distance = 6 →
  (last_segment_distance / (total_time - (first_segment_distance / first_segment_speed))) = 9 := by
  sorry

end runner_speed_problem_l1810_181048


namespace num_lines_in_4x4_grid_l1810_181043

/-- Represents a 4-by-4 grid of lattice points -/
structure Grid :=
  (size : Nat)
  (h_size : size = 4)

/-- Represents a line in the grid -/
structure Line :=
  (points : Finset (Nat × Nat))
  (h_distinct : points.card ≥ 2)
  (h_in_grid : ∀ p ∈ points, p.1 < 4 ∧ p.2 < 4)

/-- The set of all lines in the grid -/
def allLines (g : Grid) : Finset Line := sorry

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid -/
def numLines (g : Grid) : Nat :=
  (allLines g).card

/-- Theorem stating that the number of distinct lines in a 4-by-4 grid is 70 -/
theorem num_lines_in_4x4_grid (g : Grid) : numLines g = 70 := by
  sorry

end num_lines_in_4x4_grid_l1810_181043


namespace remainder_3_pow_123_plus_4_mod_8_l1810_181040

theorem remainder_3_pow_123_plus_4_mod_8 : 3^123 + 4 ≡ 7 [MOD 8] := by
  sorry

end remainder_3_pow_123_plus_4_mod_8_l1810_181040


namespace dragon_boat_festival_probability_l1810_181078

theorem dragon_boat_festival_probability (pA pB pC : ℝ) 
  (hA : pA = 2/3) (hB : pB = 1/4) (hC : pC = 3/5) : 
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 9/10 := by
  sorry

end dragon_boat_festival_probability_l1810_181078


namespace quadratic_distinct_roots_l1810_181066

theorem quadratic_distinct_roots (p q : ℚ) : 
  (∃ x y : ℚ, x ≠ y ∧ 
    x^2 + p*x + q = 0 ∧ 
    y^2 + p*y + q = 0 ∧ 
    x = 2*p ∧ 
    y = p + q) → 
  (p = 2/3 ∧ q = -8/3) :=
sorry

end quadratic_distinct_roots_l1810_181066


namespace troys_home_distance_l1810_181083

/-- The distance between Troy's home and school -/
def troys_distance : ℝ := 75

/-- The distance between Emily's home and school -/
def emilys_distance : ℝ := 98

/-- The additional distance Emily walks compared to Troy in five days -/
def additional_distance : ℝ := 230

/-- The number of days -/
def days : ℕ := 5

theorem troys_home_distance :
  troys_distance = 75 ∧
  emilys_distance = 98 ∧
  additional_distance = 230 ∧
  days = 5 →
  days * (2 * emilys_distance) - days * (2 * troys_distance) = additional_distance :=
by sorry

end troys_home_distance_l1810_181083


namespace S_inter_T_finite_l1810_181077

/-- Set S defined as {y | y = 3^x, x ∈ ℝ} -/
def S : Set ℝ := {y | ∃ x, y = Real.exp (Real.log 3 * x)}

/-- Set T defined as {y | y = x^2 - 1, x ∈ ℝ} -/
def T : Set ℝ := {y | ∃ x, y = x^2 - 1}

/-- The intersection of S and T is a finite set -/
theorem S_inter_T_finite : Set.Finite (S ∩ T) := by sorry

end S_inter_T_finite_l1810_181077


namespace bicycle_helmet_cost_l1810_181033

theorem bicycle_helmet_cost (helmet_cost bicycle_cost total_cost : ℕ) : 
  helmet_cost = 40 →
  bicycle_cost = 5 * helmet_cost →
  total_cost = bicycle_cost + helmet_cost →
  total_cost = 240 :=
by
  sorry

end bicycle_helmet_cost_l1810_181033


namespace sqrt_equation_solutions_l1810_181028

theorem sqrt_equation_solutions (x : ℝ) :
  x ≥ 2 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 3) ↔
  (x = 32.25 ∨ x = 8.25) :=
by sorry

end sqrt_equation_solutions_l1810_181028


namespace unique_solution_cube_sum_l1810_181001

theorem unique_solution_cube_sum (n : ℕ+) : 
  (∃ (x y z : ℕ+), x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) ↔ n = 3 := by
  sorry

end unique_solution_cube_sum_l1810_181001


namespace floor_negative_seven_fourths_l1810_181000

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_negative_seven_fourths_l1810_181000


namespace trees_in_garden_l1810_181068

/-- The number of trees in a yard with given length and spacing -/
def number_of_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem: There are 26 trees in a 300-meter yard with 12-meter spacing -/
theorem trees_in_garden : number_of_trees 300 12 = 26 := by
  sorry

end trees_in_garden_l1810_181068


namespace sum_of_squares_of_G_digits_l1810_181014

/-- Represents a fraction m/n -/
structure Fraction where
  m : ℕ+
  n : ℕ+
  m_lt_n : m < n
  lowest_terms : Nat.gcd m n = 1
  no_square_divisor : ∀ k > 1, ¬(k * k ∣ n)
  repeating_length_6 : ∃ k : ℕ+, m * 10^6 = k * n + m

/-- Count of valid fractions -/
def F : ℕ := 1109700

/-- Number of digits in F -/
def p : ℕ := 7

/-- G is defined as F + p -/
def G : ℕ := F + p

/-- Sum of squares of digits of a natural number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem to prove -/
theorem sum_of_squares_of_G_digits :
  sum_of_squares_of_digits G = 181 := by sorry

end sum_of_squares_of_G_digits_l1810_181014


namespace two_tails_one_head_probability_l1810_181069

def coin_toss_probability : ℚ := 3/8

theorem two_tails_one_head_probability :
  let n_coins := 3
  let n_tails := 2
  let n_heads := 1
  let total_outcomes := 2^n_coins
  let favorable_outcomes := n_coins.choose n_tails
  coin_toss_probability = (favorable_outcomes : ℚ) / total_outcomes :=
by sorry

end two_tails_one_head_probability_l1810_181069


namespace unique_solution_cubic_equation_l1810_181024

theorem unique_solution_cubic_equation :
  ∀ x : ℝ, (1 + x^2) * (1 + x^4) = 4 * x^3 → x = 1 :=
by sorry

end unique_solution_cubic_equation_l1810_181024


namespace fraction_equality_implies_constants_l1810_181081

theorem fraction_equality_implies_constants (a b : ℝ) :
  (∀ x : ℝ, x ≠ -b → x ≠ -36 → x ≠ -30 → 
    (x - a) / (x + b) = (x^2 - 45*x + 504) / (x^2 + 66*x - 1080)) →
  a = 18 ∧ b = 30 ∧ a + b = 48 := by
sorry

end fraction_equality_implies_constants_l1810_181081


namespace sum_and_reciprocal_sum_l1810_181079

theorem sum_and_reciprocal_sum (x : ℝ) (h : x > 0) (h_sum_squares : x^2 + (1/x)^2 = 23) : 
  x + (1/x) = 5 := by sorry

end sum_and_reciprocal_sum_l1810_181079


namespace orthogonal_vectors_magnitude_l1810_181089

def vector_a : ℝ × ℝ := (1, -3)
def vector_b (m : ℝ) : ℝ × ℝ := (6, m)

theorem orthogonal_vectors_magnitude (m : ℝ) 
  (h : vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2 = 0) : 
  ‖(2 • vector_a - vector_b m)‖ = 4 * Real.sqrt 5 := by
  sorry

end orthogonal_vectors_magnitude_l1810_181089


namespace tabitha_current_age_l1810_181041

/- Define the problem parameters -/
def start_age : ℕ := 15
def start_colors : ℕ := 2
def future_colors : ℕ := 8
def years_to_future : ℕ := 3

/- Define Tabitha's age as a function of the number of colors -/
def tabitha_age (colors : ℕ) : ℕ := start_age + (colors - start_colors)

/- Define the number of colors Tabitha has now -/
def current_colors : ℕ := future_colors - years_to_future

/- The theorem to prove -/
theorem tabitha_current_age :
  tabitha_age current_colors = 18 := by
  sorry


end tabitha_current_age_l1810_181041


namespace hexagon_diagonals_l1810_181065

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: A hexagon has 9 internal diagonals -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end hexagon_diagonals_l1810_181065


namespace base_two_rep_of_125_l1810_181047

theorem base_two_rep_of_125 : 
  (125 : ℕ).digits 2 = [1, 0, 1, 1, 1, 1, 1] :=
by sorry

end base_two_rep_of_125_l1810_181047


namespace arithmetic_calculation_l1810_181090

theorem arithmetic_calculation : -8 + (-10) - 3 - (-6) = -15 := by
  sorry

end arithmetic_calculation_l1810_181090


namespace green_blue_difference_l1810_181011

/-- Represents the number of parts for each color in the ratio --/
structure ColorRatio :=
  (blue : ℕ)
  (yellow : ℕ)
  (green : ℕ)
  (red : ℕ)

/-- Calculates the total number of parts in the ratio --/
def totalParts (ratio : ColorRatio) : ℕ :=
  ratio.blue + ratio.yellow + ratio.green + ratio.red

/-- Calculates the number of disks for a given color based on the ratio and total disks --/
def disksPerColor (ratio : ColorRatio) (color : ℕ) (totalDisks : ℕ) : ℕ :=
  color * (totalDisks / totalParts ratio)

theorem green_blue_difference (totalDisks : ℕ) (ratio : ColorRatio) :
  totalDisks = 180 →
  ratio = ⟨3, 7, 8, 9⟩ →
  disksPerColor ratio ratio.green totalDisks - disksPerColor ratio ratio.blue totalDisks = 35 :=
by sorry

end green_blue_difference_l1810_181011


namespace simplify_expression_l1810_181086

theorem simplify_expression (x : ℝ) : 2*x - 3*(2-x) + 4*(2+x) - 5*(1-3*x) = 24*x - 3 := by
  sorry

end simplify_expression_l1810_181086


namespace imaginary_part_of_z_l1810_181091

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.abs (1 + Complex.I)) : 
  z.im = -Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_l1810_181091


namespace problem_solution_l1810_181018

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem problem_solution (a : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence (fun n => a (n + 1) - a n) →
  (fun n => a (n + 1) - a n) 1 = 2 →
  (∀ n : ℕ, (fun n => a (n + 1) - a n) (n + 1) - (fun n => a (n + 1) - a n) n = 2) →
  a 1 = 1 →
  43 < a m →
  a m < 73 →
  m = 8 := by
sorry

end problem_solution_l1810_181018


namespace star_equation_solution_l1810_181052

/-- The star operation -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a - 2

/-- Theorem: If 3 ★ y = 25, then y = 5 -/
theorem star_equation_solution (y : ℝ) (h : star 3 y = 25) : y = 5 := by
  sorry

end star_equation_solution_l1810_181052


namespace mike_total_spent_l1810_181085

def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52
def toy_car_cost : ℝ := 3.75
def puzzle_cost : ℝ := 8.99
def stickers_cost : ℝ := 1.25
def puzzle_discount : ℝ := 0.15
def toy_car_discount : ℝ := 0.10
def coupon_value : ℝ := 5.00

def discounted_puzzle_cost : ℝ := puzzle_cost * (1 - puzzle_discount)
def discounted_toy_car_cost : ℝ := toy_car_cost * (1 - toy_car_discount)

def total_cost : ℝ := marbles_cost + football_cost + baseball_cost + 
                       discounted_toy_car_cost + discounted_puzzle_cost + 
                       stickers_cost - coupon_value

theorem mike_total_spent :
  total_cost = 27.7865 :=
by sorry

end mike_total_spent_l1810_181085


namespace books_found_equals_26_l1810_181063

/-- The number of books Joan initially gathered -/
def initial_books : ℕ := 33

/-- The total number of books Joan has now -/
def total_books : ℕ := 59

/-- The number of additional books Joan found -/
def additional_books : ℕ := total_books - initial_books

theorem books_found_equals_26 : additional_books = 26 := by
  sorry

end books_found_equals_26_l1810_181063


namespace perpendicular_bisector_value_l1810_181087

/-- The perpendicular bisector of a line segment passes through its midpoint and is perpendicular to the segment. -/
structure PerpendicularBisector (p₁ p₂ : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop where
  passes_through_midpoint : l ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  is_perpendicular : True  -- We don't need to express this condition for this problem

/-- The line equation x + y = b -/
def line_equation (b : ℝ) : ℝ × ℝ → Prop :=
  fun p => p.1 + p.2 = b

/-- The main theorem: if x + y = b is the perpendicular bisector of the line segment
    from (2,5) to (8,11), then b = 13 -/
theorem perpendicular_bisector_value :
  PerpendicularBisector (2, 5) (8, 11) (line_equation b) → b = 13 :=
by
  sorry


end perpendicular_bisector_value_l1810_181087


namespace alice_apples_l1810_181044

theorem alice_apples (A : ℕ) : 
  A > 2 →
  A % 9 = 2 →
  A % 10 = 2 →
  A % 11 = 2 →
  (∀ B : ℕ, B > 2 → B % 9 = 2 → B % 10 = 2 → B % 11 = 2 → A ≤ B) →
  A = 992 := by
sorry

end alice_apples_l1810_181044


namespace unknown_score_is_66_l1810_181093

def scores : List ℕ := [65, 70, 78, 85, 92]

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

theorem unknown_score_is_66 (x : ℕ) 
  (h1 : is_integer ((scores.sum + x) / 6))
  (h2 : x % 6 = 0)
  (h3 : x ≥ 60 ∧ x ≤ 100) :
  x = 66 := by sorry

end unknown_score_is_66_l1810_181093


namespace problem_1_l1810_181070

theorem problem_1 : 
  Real.sqrt ((-2)^2) + Real.sqrt 2 * (1 - Real.sqrt (1/2)) + |(-Real.sqrt 8)| = 1 + 3 * Real.sqrt 2 := by
  sorry

end problem_1_l1810_181070


namespace sufficient_not_necessary_l1810_181046

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 0 → |x| > 0) ∧ (∃ x : ℝ, |x| > 0 ∧ x ≤ 0) := by sorry

end sufficient_not_necessary_l1810_181046


namespace triangle_area_l1810_181027

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (2 * a * b * Real.sin C = Real.sqrt 3 * (b^2 + c^2 - a^2)) →
  (a = Real.sqrt 13) →
  (c = 3) →
  (1/2 * b * c * Real.sin A = 3 * Real.sqrt 3) :=
by sorry

end triangle_area_l1810_181027


namespace remaining_legos_l1810_181012

theorem remaining_legos (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  initial = 2080 → lost = 17 → remaining = initial - lost → remaining = 2063 := by
sorry

end remaining_legos_l1810_181012


namespace shifted_function_point_l1810_181036

/-- A function whose graph passes through (1, -1) -/
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f 1 = -1

/-- The horizontally shifted function -/
def shift_function (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (x - 3)

theorem shifted_function_point (f : ℝ → ℝ) :
  passes_through_point f → passes_through_point (shift_function f) :=
by
  sorry

#check shifted_function_point

end shifted_function_point_l1810_181036


namespace exists_four_mutually_acquainted_l1810_181072

/-- Represents the acquaintance relation between people --/
def Acquainted (n : ℕ) := Fin n → Fin n → Prop

/-- The property that among every 3 people, at least 2 are acquainted --/
def AtLeastTwoAcquainted (n : ℕ) (acq : Acquainted n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    acq a b ∨ acq b c ∨ acq a c

/-- A subset of 4 mutually acquainted people --/
def FourMutuallyAcquainted (n : ℕ) (acq : Acquainted n) : Prop :=
  ∃ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    acq a b ∧ acq a c ∧ acq a d ∧ acq b c ∧ acq b d ∧ acq c d

/-- The main theorem --/
theorem exists_four_mutually_acquainted :
  ∀ (acq : Acquainted 9),
    AtLeastTwoAcquainted 9 acq →
    FourMutuallyAcquainted 9 acq :=
by
  sorry


end exists_four_mutually_acquainted_l1810_181072


namespace valid_numbers_l1810_181056

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a < b ∧ 
    b ≤ 9 ∧
    n = 10 * a + b ∧ 
    n = (b - a + 1) * (a + b) / 2

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = {14, 26, 37, 48, 59} := by sorry

end valid_numbers_l1810_181056


namespace two_A_minus_three_B_two_A_minus_three_B_equals_seven_l1810_181021

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 + 2 * y^2 - 2 * x * y
def B (x y : ℝ) : ℝ := y^2 - x * y + 2 * x^2

-- Theorem 1: 2A - 3B = y² - xy
theorem two_A_minus_three_B (x y : ℝ) : 2 * A x y - 3 * B x y = y^2 - x * y := by
  sorry

-- Theorem 2: 2A - 3B = 7 under the given condition
theorem two_A_minus_three_B_equals_seven (x y : ℝ) 
  (h : |2 * x - 3| + (y + 2)^2 = 0) : 2 * A x y - 3 * B x y = 7 := by
  sorry

end two_A_minus_three_B_two_A_minus_three_B_equals_seven_l1810_181021


namespace certain_number_problem_l1810_181025

theorem certain_number_problem (N : ℚ) : 
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 300 → N = 576 := by
  sorry

end certain_number_problem_l1810_181025


namespace smallest_valid_number_l1810_181045

def is_valid_number (n : ℕ) : Prop :=
  ∃ (chosen : Finset ℕ) (unchosen : Finset ℕ),
    chosen.card = 5 ∧
    unchosen.card = 4 ∧
    chosen ∪ unchosen = Finset.range 9 ∧
    chosen ∩ unchosen = ∅ ∧
    (∀ d ∈ chosen, n % d = 0) ∧
    (∀ d ∈ unchosen, n % d ≠ 0) ∧
    n ≥ 10000 ∧ n < 100000

theorem smallest_valid_number :
  ∃ (n : ℕ), is_valid_number n ∧ 
  (∀ m, is_valid_number m → n ≤ m) ∧
  n = 14728 := by
  sorry

end smallest_valid_number_l1810_181045


namespace probability_red_from_B_probability_red_from_B_is_correct_l1810_181060

/-- Represents the number of red balls in Box A -/
def red_balls_A : ℕ := 5

/-- Represents the number of white balls in Box A -/
def white_balls_A : ℕ := 2

/-- Represents the number of red balls in Box B -/
def red_balls_B : ℕ := 4

/-- Represents the number of white balls in Box B -/
def white_balls_B : ℕ := 3

/-- Represents the total number of balls in Box A -/
def total_balls_A : ℕ := red_balls_A + white_balls_A

/-- Represents the total number of balls in Box B -/
def total_balls_B : ℕ := red_balls_B + white_balls_B

/-- The probability of drawing a red ball from Box B after the process -/
theorem probability_red_from_B : ℚ :=
  33 / 56

theorem probability_red_from_B_is_correct :
  probability_red_from_B = 33 / 56 := by sorry

end probability_red_from_B_probability_red_from_B_is_correct_l1810_181060


namespace curve_tangent_sum_l1810_181031

/-- The curve C defined by y = x^3 - x^2 - ax + b -/
def C (x y a b : ℝ) : Prop := y = x^3 - x^2 - a*x + b

/-- The derivative of C with respect to x -/
def C_derivative (x a : ℝ) : ℝ := 3*x^2 - 2*x - a

theorem curve_tangent_sum (a b : ℝ) : 
  C 0 1 a b ∧ C_derivative 0 a = 2 → a + b = -1 := by sorry

end curve_tangent_sum_l1810_181031


namespace condition_sufficient_not_necessary_l1810_181097

theorem condition_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end condition_sufficient_not_necessary_l1810_181097


namespace degree_to_radian_conversion_l1810_181057

theorem degree_to_radian_conversion (π : Real) :
  (60 : Real) * (π / 180) = π / 3 := by
  sorry

end degree_to_radian_conversion_l1810_181057


namespace solution_is_correct_l1810_181007

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ → ℝ) : ℝ × ℝ :=
  (1, 3)  -- We define this based on the given lines, without solving the system

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, l1 x y = k * l2 x y

/-- The first given line -/
def line1 (x y : ℝ) : ℝ := 3*x - 2*y + 3

/-- The second given line -/
def line2 (x y : ℝ) : ℝ := x + y - 4

/-- The line parallel to which we need to find our solution -/
def parallel_line (x y : ℝ) : ℝ := 2*x + y - 1

/-- The proposed solution line -/
def solution_line (x y : ℝ) : ℝ := 2*x + y - 5

theorem solution_is_correct : 
  let (ix, iy) := intersection_point line1 line2
  solution_line ix iy = 0 ∧ 
  are_parallel solution_line parallel_line :=
by sorry

end solution_is_correct_l1810_181007


namespace triangle_inequality_cube_l1810_181080

theorem triangle_inequality_cube (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) : a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end triangle_inequality_cube_l1810_181080


namespace power_of_two_equation_l1810_181002

theorem power_of_two_equation (x : ℕ) : 16^3 + 16^3 + 16^3 = 2^x ↔ x = 13 := by
  sorry

end power_of_two_equation_l1810_181002


namespace roots_sum_reciprocal_l1810_181004

theorem roots_sum_reciprocal (α β : ℝ) : 
  α^2 - 10*α + 20 = 0 → β^2 - 10*β + 20 = 0 → 1/α + 1/β = 1/2 := by sorry

end roots_sum_reciprocal_l1810_181004


namespace binary_11011_to_decimal_l1810_181019

def binary_to_decimal (b₄ b₃ b₂ b₁ b₀ : Nat) : Nat :=
  b₄ * 2^4 + b₃ * 2^3 + b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_11011_to_decimal :
  binary_to_decimal 1 1 0 1 1 = 27 := by
  sorry

end binary_11011_to_decimal_l1810_181019


namespace root_in_interval_l1810_181039

theorem root_in_interval : ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x = 2 - x := by sorry

end root_in_interval_l1810_181039


namespace triangle_ratio_l1810_181099

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.cos B + b * Real.cos A = 3 * a →
  c / a = 3 := by sorry

end triangle_ratio_l1810_181099


namespace optimal_selling_price_l1810_181096

def initial_purchase_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 500
def price_increase : ℝ → ℝ := λ x => x
def sales_volume : ℝ → ℝ := λ x => initial_sales_volume - 10 * price_increase x
def selling_price : ℝ → ℝ := λ x => initial_selling_price + price_increase x
def profit : ℝ → ℝ := λ x => (selling_price x * sales_volume x) - (initial_purchase_price * sales_volume x)

theorem optimal_selling_price :
  ∃ x : ℝ, (∀ y : ℝ, profit y ≤ profit x) ∧ selling_price x = 70 := by
  sorry

end optimal_selling_price_l1810_181096


namespace expected_red_pairs_in_standard_deck_l1810_181008

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardCount : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def redAdjacentProbability : ℚ := 25 / 51

/-- The expected number of pairs of adjacent red cards in a standard deck dealt in a circle -/
def expectedRedPairs : ℚ := redCardCount * redAdjacentProbability

theorem expected_red_pairs_in_standard_deck :
  expectedRedPairs = 650 / 51 := by
  sorry

end expected_red_pairs_in_standard_deck_l1810_181008


namespace dough_perimeter_l1810_181094

theorem dough_perimeter (dough_width : ℕ) (mold_side : ℕ) (unused_width : ℕ) (total_cookies : ℕ) :
  dough_width = 34 →
  mold_side = 4 →
  unused_width = 2 →
  total_cookies = 24 →
  let used_width := dough_width - unused_width
  let molds_across := used_width / mold_side
  let molds_along := total_cookies / molds_across
  let dough_length := molds_along * mold_side
  2 * dough_width + 2 * dough_length = 92 := by
  sorry

end dough_perimeter_l1810_181094


namespace stating_bryans_books_l1810_181076

/-- 
Given the number of books per bookshelf and the number of bookshelves,
calculates the total number of books.
-/
def total_books (books_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  books_per_shelf * num_shelves

/-- 
Theorem stating that with 2 books per shelf and 21 shelves,
the total number of books is 42.
-/
theorem bryans_books : 
  total_books 2 21 = 42 := by
  sorry

end stating_bryans_books_l1810_181076
