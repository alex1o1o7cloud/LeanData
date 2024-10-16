import Mathlib

namespace NUMINAMATH_CALUDE_sports_club_players_l3151_315183

/-- The number of players in a sports club with three games: kabaddi, kho kho, and badminton -/
theorem sports_club_players (kabaddi kho_kho_only badminton both_kabaddi_kho_kho both_kabaddi_badminton both_kho_kho_badminton all_three : ℕ) 
  (h1 : kabaddi = 20)
  (h2 : kho_kho_only = 50)
  (h3 : badminton = 25)
  (h4 : both_kabaddi_kho_kho = 15)
  (h5 : both_kabaddi_badminton = 10)
  (h6 : both_kho_kho_badminton = 5)
  (h7 : all_three = 3) :
  kabaddi + kho_kho_only + badminton - both_kabaddi_kho_kho - both_kabaddi_badminton - both_kho_kho_badminton + all_three = 68 := by
  sorry


end NUMINAMATH_CALUDE_sports_club_players_l3151_315183


namespace NUMINAMATH_CALUDE_jerrys_shelf_difference_l3151_315125

/-- Given Jerry's initial book and action figure counts, and the number of action figures added,
    prove that there are 2 more books than action figures on the shelf. -/
theorem jerrys_shelf_difference (initial_books : ℕ) (initial_figures : ℕ) (added_figures : ℕ)
    (h1 : initial_books = 7)
    (h2 : initial_figures = 3)
    (h3 : added_figures = 2) :
    initial_books - (initial_figures + added_figures) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_shelf_difference_l3151_315125


namespace NUMINAMATH_CALUDE_pear_sales_ratio_l3151_315146

theorem pear_sales_ratio (total : ℕ) (afternoon : ℕ) 
  (h1 : total = 390)
  (h2 : afternoon = 260) :
  (afternoon : ℚ) / (total - afternoon : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pear_sales_ratio_l3151_315146


namespace NUMINAMATH_CALUDE_square_of_sum_l3151_315158

theorem square_of_sum (x y : ℝ) 
  (h1 : 3 * x * (2 * x + y) = 14) 
  (h2 : y * (2 * x + y) = 35) : 
  (2 * x + y)^2 = 49 := by sorry

end NUMINAMATH_CALUDE_square_of_sum_l3151_315158


namespace NUMINAMATH_CALUDE_geese_count_l3151_315131

theorem geese_count (initial : ℕ) (flew_away : ℕ) (joined : ℕ) 
  (h1 : initial = 372) 
  (h2 : flew_away = 178) 
  (h3 : joined = 57) : 
  initial - flew_away + joined = 251 := by
sorry

end NUMINAMATH_CALUDE_geese_count_l3151_315131


namespace NUMINAMATH_CALUDE_larger_number_proof_l3151_315132

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 50) (h2 : Nat.lcm a b = 50 * 13 * 23 * 31) :
  max a b = 463450 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3151_315132


namespace NUMINAMATH_CALUDE_parabola_focus_l3151_315104

/-- A parabola is defined by the equation x = -1/16 * y^2 + 2 -/
def parabola (x y : ℝ) : Prop := x = -1/16 * y^2 + 2

/-- The focus of a parabola is a point -/
def focus : ℝ × ℝ := (-2, 0)

/-- Theorem: The focus of the parabola defined by x = -1/16 * y^2 + 2 is at (-2, 0) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola x y → focus = (-2, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l3151_315104


namespace NUMINAMATH_CALUDE_point_distance_from_origin_l3151_315140

theorem point_distance_from_origin (x : ℚ) : 
  |x| = (5 : ℚ) / 2 → x = (5 : ℚ) / 2 ∨ x = -(5 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_from_origin_l3151_315140


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3151_315198

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 10) → x = 25 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3151_315198


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3151_315139

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a-1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3151_315139


namespace NUMINAMATH_CALUDE_heath_carrot_planting_l3151_315193

/-- Calculates the number of carrots planted per hour given the number of rows, plants per row, and total planting time. -/
def carrots_per_hour (rows : ℕ) (plants_per_row : ℕ) (total_hours : ℕ) : ℕ :=
  (rows * plants_per_row) / total_hours

/-- Proves that given 400 rows of carrots, 300 plants per row, and 20 hours of planting time, the number of carrots planted per hour is 6,000. -/
theorem heath_carrot_planting :
  carrots_per_hour 400 300 20 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_heath_carrot_planting_l3151_315193


namespace NUMINAMATH_CALUDE_fold_coincide_points_l3151_315174

/-- Given a number line where folding causes -2 to coincide with 8, and two points A and B
    with a distance of 2024 between them (A to the left of B) that coincide after folding,
    the coordinate of point A is -1009. -/
theorem fold_coincide_points (A B : ℝ) : 
  (A < B) →  -- A is to the left of B
  (B - A = 2024) →  -- Distance between A and B is 2024
  (A + B) / 2 = (-2 + 8) / 2 →  -- Midpoint of A and B is the same as midpoint of -2 and 8
  A = -1009 := by
sorry

end NUMINAMATH_CALUDE_fold_coincide_points_l3151_315174


namespace NUMINAMATH_CALUDE_cylinder_height_ratio_l3151_315129

/-- Given a cylinder whose radius is tripled and whose new volume is 18 times the original,
    prove that the ratio of the new height to the original height is 2:1. -/
theorem cylinder_height_ratio 
  (r : ℝ) -- original radius
  (h : ℝ) -- original height
  (h' : ℝ) -- new height
  (volume_ratio : ℝ) -- ratio of new volume to old volume
  (h_pos : 0 < h) -- ensure original height is positive
  (r_pos : 0 < r) -- ensure original radius is positive
  (volume_eq : π * (3 * r)^2 * h' = volume_ratio * (π * r^2 * h)) -- volume equation
  (volume_ratio_eq : volume_ratio = 18) -- new volume is 18 times the old one
  : h' / h = 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_ratio_l3151_315129


namespace NUMINAMATH_CALUDE_triangle_pqr_rotation_l3151_315110

/-- Triangle PQR with given properties and rotation of PQ --/
theorem triangle_pqr_rotation (P Q R : ℝ × ℝ) (h1 : P = (0, 0)) (h2 : R = (8, 0))
  (h3 : Q.1 ≥ 0 ∧ Q.2 ≥ 0) -- Q in first quadrant
  (h4 : (Q.1 - R.1) * (Q.2 - R.2) = 0) -- ∠QRP = 90°
  (h5 : (Q.2 - P.2) = (Q.1 - P.1)) -- ∠QPR = 45°
  : (- Q.2, Q.1) = (-8, 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_pqr_rotation_l3151_315110


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3151_315109

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + x^2 + 15) - (x^6 + x^5 - 2 * x^4 + 3 * x^2 + 20) =
  x^6 + 2 * x^5 + 3 * x^4 - 2 * x^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3151_315109


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3151_315142

theorem perfect_square_condition (k : ℕ) : 
  (∃ m : ℤ, k^2 - 101*k = m^2) ↔ (k = 101 ∨ k = 2601) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3151_315142


namespace NUMINAMATH_CALUDE_marathon_speed_ratio_l3151_315185

/-- The ratio of average speeds of two marathon runners -/
theorem marathon_speed_ratio (distance : ℝ) (jack_time jill_time : ℝ) 
  (h1 : distance = 41)
  (h2 : jack_time = 4.5)
  (h3 : jill_time = 4.1) : 
  (distance / jack_time) / (distance / jill_time) = 82 / 90 := by
  sorry

end NUMINAMATH_CALUDE_marathon_speed_ratio_l3151_315185


namespace NUMINAMATH_CALUDE_election_votes_l3151_315182

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (excess_percent : ℚ) : 
  total_votes = 5720 →
  invalid_percent = 1/5 →
  excess_percent = 3/20 →
  ∃ (a_votes b_votes : ℕ),
    (a_votes : ℚ) + b_votes = total_votes * (1 - invalid_percent) ∧
    (a_votes : ℚ) = b_votes + total_votes * excess_percent ∧
    b_votes = 1859 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l3151_315182


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_72_l3151_315196

theorem sqrt_nine_factorial_over_72 : Real.sqrt (Nat.factorial 9 / 72) = 12 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_72_l3151_315196


namespace NUMINAMATH_CALUDE_sequence_with_differences_two_or_five_l3151_315180

theorem sequence_with_differences_two_or_five :
  ∃ (p : Fin 101 → Fin 101), Function.Bijective p ∧
    (∀ i : Fin 100, (p (i + 1) : ℕ) - (p i : ℕ) = 2 ∨ (p (i + 1) : ℕ) - (p i : ℕ) = 5 ∨
                    (p i : ℕ) - (p (i + 1) : ℕ) = 2 ∨ (p i : ℕ) - (p (i + 1) : ℕ) = 5) :=
by sorry


end NUMINAMATH_CALUDE_sequence_with_differences_two_or_five_l3151_315180


namespace NUMINAMATH_CALUDE_envelope_count_l3151_315176

/-- Proves that the number of envelopes sent is 850, given the weight of one envelope and the total weight. -/
theorem envelope_count (envelope_weight : ℝ) (total_weight_kg : ℝ) : 
  envelope_weight = 8.5 →
  total_weight_kg = 7.225 →
  (total_weight_kg * 1000) / envelope_weight = 850 := by
  sorry

end NUMINAMATH_CALUDE_envelope_count_l3151_315176


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2007_l3151_315189

/-- Calculates the total rainfall in Mathborough for the year 2007 given the rainfall data from 2005 to 2007. -/
theorem mathborough_rainfall_2007 (rainfall_2005 : ℝ) (increase_2006 : ℝ) (increase_2007 : ℝ) :
  rainfall_2005 = 40.5 →
  increase_2006 = 3 →
  increase_2007 = 4 →
  (rainfall_2005 + increase_2006 + increase_2007) * 12 = 570 := by
  sorry

end NUMINAMATH_CALUDE_mathborough_rainfall_2007_l3151_315189


namespace NUMINAMATH_CALUDE_distance_equality_implies_m_equals_negative_one_l3151_315184

theorem distance_equality_implies_m_equals_negative_one : 
  ∀ m : ℝ, (|m| = |m + 2|) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_distance_equality_implies_m_equals_negative_one_l3151_315184


namespace NUMINAMATH_CALUDE_percentage_markup_approx_l3151_315133

def selling_price : ℝ := 8337
def cost_price : ℝ := 6947.5

theorem percentage_markup_approx (ε : ℝ) (h : ε > 0) :
  ∃ (markup_percentage : ℝ),
    abs (markup_percentage - 19.99) < ε ∧
    markup_percentage = (selling_price - cost_price) / cost_price * 100 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_markup_approx_l3151_315133


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3151_315167

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3151_315167


namespace NUMINAMATH_CALUDE_unique_correct_ranking_l3151_315177

/-- Represents the participants in the long jump competition -/
inductive Participant
| Decimals
| Elementary
| Xiaohua
| Xiaoyuan
| Exploration

/-- Represents the ranking of participants -/
def Ranking := Participant → Fin 5

/-- Checks if a ranking satisfies all the given conditions -/
def satisfies_conditions (r : Ranking) : Prop :=
  (r Participant.Decimals < r Participant.Elementary) ∧
  (r Participant.Xiaohua > r Participant.Xiaoyuan) ∧
  (r Participant.Exploration > r Participant.Elementary) ∧
  (r Participant.Elementary < r Participant.Xiaohua) ∧
  (r Participant.Xiaoyuan > r Participant.Exploration)

/-- The correct ranking of participants -/
def correct_ranking : Ranking :=
  fun p => match p with
    | Participant.Decimals => 0
    | Participant.Elementary => 1
    | Participant.Exploration => 2
    | Participant.Xiaoyuan => 3
    | Participant.Xiaohua => 4

/-- Theorem stating that the correct_ranking is the unique ranking that satisfies all conditions -/
theorem unique_correct_ranking :
  satisfies_conditions correct_ranking ∧
  ∀ (r : Ranking), satisfies_conditions r → r = correct_ranking :=
sorry

end NUMINAMATH_CALUDE_unique_correct_ranking_l3151_315177


namespace NUMINAMATH_CALUDE_problem_statement_l3151_315152

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/y + y/x = 8) :
  (x + y)/(x - y) = Real.sqrt (5/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3151_315152


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3151_315103

theorem min_value_quadratic (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3151_315103


namespace NUMINAMATH_CALUDE_i_power_2016_l3151_315138

-- Define i as a complex number
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem i_power_2016 : i^2016 = 1 :=
  -- Assume the given conditions
  have h1 : i^1 = i := by sorry
  have h2 : i^2 = -1 := by sorry
  have h3 : i^3 = -i := by sorry
  have h4 : i^4 = 1 := by sorry
  have h5 : i^5 = i := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_i_power_2016_l3151_315138


namespace NUMINAMATH_CALUDE_function_properties_l3151_315163

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_negative (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : periodic_negative f)
  (h3 : increasing_on f (-1) 0) :
  (f 2 = f 0) ∧ (symmetric_about f 1) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3151_315163


namespace NUMINAMATH_CALUDE_fraction_equals_point_eight_seven_five_l3151_315164

theorem fraction_equals_point_eight_seven_five (a : ℕ+) :
  (a : ℚ) / (a + 35 : ℚ) = 7/8 → a = 245 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_point_eight_seven_five_l3151_315164


namespace NUMINAMATH_CALUDE_candy_jar_problem_l3151_315190

theorem candy_jar_problem (total : ℕ) (red : ℕ) (blue : ℕ) : 
  total = 3409 → red = 145 → blue = total - red → blue = 3264 := by
  sorry

end NUMINAMATH_CALUDE_candy_jar_problem_l3151_315190


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l3151_315149

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4761 → 
  min a b = 53 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l3151_315149


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l3151_315161

/-- A quadratic function of the form y = mx^2 - 8x + m(m-1) that passes through the origin -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 8 * x + m * (m - 1)

/-- The quadratic function passes through the origin -/
def passes_through_origin (m : ℝ) : Prop := quadratic_function m 0 = 0

/-- The theorem stating that m = 1 for the given quadratic function passing through the origin -/
theorem quadratic_function_m_value :
  ∃ m : ℝ, passes_through_origin m ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l3151_315161


namespace NUMINAMATH_CALUDE_triangle_right_angled_l3151_315128

theorem triangle_right_angled (α β γ : Real) : 
  0 < α ∧ 0 < β ∧ 0 < γ →  -- Angles are positive
  α + β + γ = Real.pi →    -- Sum of angles in a triangle
  (Real.sin α + Real.sin β) / (Real.cos α + Real.cos β) = Real.sin γ →
  γ = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l3151_315128


namespace NUMINAMATH_CALUDE_smallest_repeating_block_8_13_l3151_315194

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def repeating_block (l : List ℕ) : List ℕ := sorry

theorem smallest_repeating_block_8_13 :
  (repeating_block (decimal_expansion 8 13)).length = 6 := by sorry

end NUMINAMATH_CALUDE_smallest_repeating_block_8_13_l3151_315194


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l3151_315166

/-- The series defined by the nth term: 2^n / (3^(2^n) + 1) -/
def series (n : ℕ) : ℚ := (2^n : ℚ) / ((3^(2^n) : ℚ) + 1)

/-- The sum of the series from 0 to infinity -/
noncomputable def seriesSum : ℚ := ∑' n, series n

/-- Theorem stating that the sum of the series equals 1/2 -/
theorem series_sum_equals_half : seriesSum = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l3151_315166


namespace NUMINAMATH_CALUDE_vector_dot_product_inequality_l3151_315102

theorem vector_dot_product_inequality (α β γ α₁ β₁ γ₁ : ℝ) 
  (h1 : α^2 + β^2 + γ^2 = 1) 
  (h2 : α₁^2 + β₁^2 + γ₁^2 = 1) : 
  α * α₁ + β * β₁ + γ * γ₁ ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_inequality_l3151_315102


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3151_315160

def A : Set Int := {-1, 0, 1, 3, 5}
def B : Set Int := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3151_315160


namespace NUMINAMATH_CALUDE_original_pay_before_tax_l3151_315187

/-- Given a 10% tax rate and a take-home pay of $585, prove that the original pay before tax deduction is $650. -/
theorem original_pay_before_tax (tax_rate : ℝ) (take_home_pay : ℝ) (original_pay : ℝ) :
  tax_rate = 0.1 →
  take_home_pay = 585 →
  original_pay * (1 - tax_rate) = take_home_pay →
  original_pay = 650 :=
by sorry

end NUMINAMATH_CALUDE_original_pay_before_tax_l3151_315187


namespace NUMINAMATH_CALUDE_complex_sum_example_l3151_315155

theorem complex_sum_example (z₁ z₂ : ℂ) : 
  z₁ = 2 + 3*I ∧ z₂ = -4 - 5*I → z₁ + z₂ = -2 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_example_l3151_315155


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3151_315106

theorem cricket_team_age_difference :
  let team_size : ℕ := 11
  let captain_age : ℕ := 25
  let team_average_age : ℕ := 22
  let remaining_average_age : ℕ := team_average_age - 1
  let wicket_keeper_age := captain_age + x

  (team_size * team_average_age = 
   (team_size - 2) * remaining_average_age + captain_age + wicket_keeper_age) →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l3151_315106


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3151_315151

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the foci
def focus1 : ℝ × ℝ := (0, 2)
def focus2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3151_315151


namespace NUMINAMATH_CALUDE_sequence_inequality_l3151_315165

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) = 2 * b n

theorem sequence_inequality (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (h1 : a 1 + b 1 > 0)
  (h2 : a 2 + b 2 < 0) :
  let m := a 4 + b 3
  m < 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3151_315165


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l3151_315170

/-- Given that Shannon has 48 heart-shaped stones and wants to make 6 bracelets
    with an equal number of stones in each, prove that the number of
    heart-shaped stones per bracelet is 8. -/
theorem stones_per_bracelet (total_stones : ℕ) (num_bracelets : ℕ) 
  (h1 : total_stones = 48) (h2 : num_bracelets = 6) :
  total_stones / num_bracelets = 8 := by
  sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l3151_315170


namespace NUMINAMATH_CALUDE_sqrt_increasing_l3151_315114

/-- The square root function is increasing on the non-negative real numbers. -/
theorem sqrt_increasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → Real.sqrt x₁ < Real.sqrt x₂ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_increasing_l3151_315114


namespace NUMINAMATH_CALUDE_fraction_product_square_l3151_315169

theorem fraction_product_square : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_square_l3151_315169


namespace NUMINAMATH_CALUDE_geometric_figures_sequence_l3151_315111

/-- The number of nonoverlapping unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 4

theorem geometric_figures_sequence :
  f 0 = 4 ∧ f 1 = 10 ∧ f 2 = 20 ∧ f 3 = 34 → f 150 = 45604 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_figures_sequence_l3151_315111


namespace NUMINAMATH_CALUDE_max_value_theorem_l3151_315145

theorem max_value_theorem (x y : ℝ) (h : x^2 + y^2 = 18*x + 8*y + 10) :
  ∀ z : ℝ, 4*x + 3*y ≤ z → z ≤ 63 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3151_315145


namespace NUMINAMATH_CALUDE_anthony_transaction_percentage_l3151_315153

/-- Proves that Anthony handled 10% more transactions than Mabel -/
theorem anthony_transaction_percentage (mabel cal jade anthony : ℕ) : 
  mabel = 90 →
  cal = (2 : ℚ) / 3 * anthony →
  jade = cal + 17 →
  jade = 83 →
  (anthony - mabel : ℚ) / mabel * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_anthony_transaction_percentage_l3151_315153


namespace NUMINAMATH_CALUDE_smallest_triangle_side_l3151_315120

theorem smallest_triangle_side : ∃ (s : ℕ), 
  (s : ℝ) ≥ 4 ∧ 
  (∀ (t : ℕ), (t : ℝ) ≥ 4 → 
    (8.5 + (t : ℝ) > 11.5) ∧
    (8.5 + 11.5 > (t : ℝ)) ∧
    (11.5 + (t : ℝ) > 8.5)) ∧
  (∀ (u : ℕ), (u : ℝ) < 4 → 
    ¬((8.5 + (u : ℝ) > 11.5) ∧
      (8.5 + 11.5 > (u : ℝ)) ∧
      (11.5 + (u : ℝ) > 8.5))) :=
by
  sorry

#check smallest_triangle_side

end NUMINAMATH_CALUDE_smallest_triangle_side_l3151_315120


namespace NUMINAMATH_CALUDE_triangle_properties_max_area_l3151_315124

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition √2 sin A = √3 cos A -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 2 * Real.sin t.A = Real.sqrt 3 * Real.cos t.A

/-- The equation a² - c² = b² - mbc -/
def equation (t : Triangle) (m : ℝ) : Prop :=
  t.a^2 - t.c^2 = t.b^2 - m * t.b * t.c

theorem triangle_properties (t : Triangle) (m : ℝ) 
    (h1 : condition t) 
    (h2 : equation t m) : 
    m = 1 := by sorry

theorem max_area (t : Triangle) 
    (h1 : condition t) 
    (h2 : t.a = 2) : 
    (t.b * t.c * Real.sin t.A / 2) ≤ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_max_area_l3151_315124


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3151_315162

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  |a₀| + |a₁| + |a₃| = 41 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3151_315162


namespace NUMINAMATH_CALUDE_same_day_after_313_weeks_l3151_315195

/-- The day of the week is represented as an integer from 0 to 6 -/
def DayOfWeek := Fin 7

/-- The number of weeks that have passed -/
def weeks : ℕ := 313

/-- Given an initial day of the week, returns the day of the week after a specified number of weeks -/
def day_after_weeks (initial_day : DayOfWeek) (n : ℕ) : DayOfWeek :=
  ⟨(initial_day.val + 7 * n) % 7, by sorry⟩

/-- Theorem: The day of the week remains the same after exactly 313 weeks -/
theorem same_day_after_313_weeks (d : DayOfWeek) : 
  day_after_weeks d weeks = d := by sorry

end NUMINAMATH_CALUDE_same_day_after_313_weeks_l3151_315195


namespace NUMINAMATH_CALUDE_smallest_positive_resolvable_debt_is_40_l3151_315159

/-- The value of a pig in dollars -/
def pig_value : ℕ := 280

/-- The value of a goat in dollars -/
def goat_value : ℕ := 200

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℤ) : Prop :=
  ∃ (p g : ℤ), debt = p * pig_value + g * goat_value

/-- The smallest positive resolvable debt -/
def smallest_positive_resolvable_debt : ℕ := 40

theorem smallest_positive_resolvable_debt_is_40 :
  (∀ d : ℕ, d < smallest_positive_resolvable_debt → ¬is_resolvable d) ∧
  is_resolvable smallest_positive_resolvable_debt :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_resolvable_debt_is_40_l3151_315159


namespace NUMINAMATH_CALUDE_letter_R_in_13th_space_l3151_315108

/-- The space number where the letter R should be placed on a sign -/
def letter_R_position (total_spaces : ℕ) (word_length : ℕ) : ℕ :=
  (total_spaces - word_length) / 2 + 1

/-- Theorem stating that the letter R should be in the 13th space -/
theorem letter_R_in_13th_space :
  letter_R_position 31 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_letter_R_in_13th_space_l3151_315108


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l3151_315121

theorem opposite_of_fraction : 
  -(11 : ℚ) / 2022 = -(11 / 2022) := by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l3151_315121


namespace NUMINAMATH_CALUDE_initial_friends_correct_l3151_315181

/-- The number of friends initially playing the game -/
def initial_friends : ℕ := 8

/-- The number of additional players who joined -/
def additional_players : ℕ := 2

/-- The number of lives each player has -/
def lives_per_player : ℕ := 6

/-- The total number of lives after new players joined -/
def total_lives : ℕ := 60

/-- Theorem stating that the initial number of friends is correct -/
theorem initial_friends_correct :
  initial_friends * lives_per_player + additional_players * lives_per_player = total_lives :=
by sorry

end NUMINAMATH_CALUDE_initial_friends_correct_l3151_315181


namespace NUMINAMATH_CALUDE_hexagon_segment_length_l3151_315188

/-- A regular hexagon with side length 2 inscribed in a circle -/
structure RegularHexagon :=
  (side_length : ℝ)
  (inscribed_in_circle : Bool)
  (h_side_length : side_length = 2)
  (h_inscribed : inscribed_in_circle = true)

/-- A segment connecting a vertex to the midpoint of the opposite side -/
def opposite_midpoint_segment (h : RegularHexagon) : ℝ → ℝ := sorry

/-- The total length of all segments connecting vertices to opposite midpoints -/
def total_segment_length (h : RegularHexagon) : ℝ :=
  6 * opposite_midpoint_segment h 1

theorem hexagon_segment_length (h : RegularHexagon) :
  total_segment_length h = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_segment_length_l3151_315188


namespace NUMINAMATH_CALUDE_function_value_order_l3151_315156

/-- A quadratic function with symmetry about x = 5 -/
structure SymmetricQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetric : ∀ x, a * (5 - x)^2 + b * (5 - x) + c = a * (5 + x)^2 + b * (5 + x) + c

/-- The quadratic function -/
def f (q : SymmetricQuadratic) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Theorem stating the order of function values -/
theorem function_value_order (q : SymmetricQuadratic) :
  f q (2 * Real.pi) < f q (Real.sqrt 40) ∧ f q (Real.sqrt 40) < f q (5 * Real.sin (π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_function_value_order_l3151_315156


namespace NUMINAMATH_CALUDE_seminar_attendees_l3151_315148

/-- The total number of attendees at a seminar -/
def total_attendees (company_a company_b company_c company_d other : ℕ) : ℕ :=
  company_a + company_b + company_c + company_d + other

/-- Theorem: Given the conditions, the total number of attendees is 185 -/
theorem seminar_attendees : 
  ∀ (company_a company_b company_c company_d other : ℕ),
    company_a = 30 →
    company_b = 2 * company_a →
    company_c = company_a + 10 →
    company_d = company_c - 5 →
    other = 20 →
    total_attendees company_a company_b company_c company_d other = 185 :=
by
  sorry

#eval total_attendees 30 60 40 35 20

end NUMINAMATH_CALUDE_seminar_attendees_l3151_315148


namespace NUMINAMATH_CALUDE_pencil_count_l3151_315123

/-- The number of pencils Reeta has -/
def reeta_pencils : ℕ := 30

/-- The number of pencils Anika has -/
def anika_pencils : ℕ := 2 * reeta_pencils + 4

/-- The number of pencils Kamal has -/
def kamal_pencils : ℕ := 3 * reeta_pencils - 2

/-- The total number of pencils all three have together -/
def total_pencils : ℕ := reeta_pencils + anika_pencils + kamal_pencils

theorem pencil_count : total_pencils = 182 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3151_315123


namespace NUMINAMATH_CALUDE_circumradius_ge_twice_inradius_l3151_315101

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- The inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- 
For any triangle, the radius of its circumcircle is greater than or equal to 
twice the radius of its incircle, with equality if and only if the triangle is equilateral 
-/
theorem circumradius_ge_twice_inradius (t : Triangle) : 
  circumradius t ≥ 2 * inradius t ∧ 
  (circumradius t = 2 * inradius t ↔ is_equilateral t) := by sorry

end NUMINAMATH_CALUDE_circumradius_ge_twice_inradius_l3151_315101


namespace NUMINAMATH_CALUDE_correct_sum_after_mistake_l3151_315118

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem correct_sum_after_mistake (original : ℕ) (mistaken : ℕ) :
  is_three_digit original →
  original % 10 = 9 →
  mistaken = original - 3 →
  mistaken + 57 = 823 →
  original + 57 = 826 := by
sorry

end NUMINAMATH_CALUDE_correct_sum_after_mistake_l3151_315118


namespace NUMINAMATH_CALUDE_johnson_yield_l3151_315178

/-- Represents the yield of corn per hectare every two months -/
structure CornYield where
  amount : ℕ

/-- Represents a cornfield -/
structure Cornfield where
  hectares : ℕ
  yield : CornYield

def total_yield (field : Cornfield) (periods : ℕ) : ℕ :=
  field.hectares * field.yield.amount * periods

theorem johnson_yield (johnson : Cornfield) (neighbor : Cornfield) 
    (h1 : johnson.hectares = 1)
    (h2 : neighbor.hectares = 2)
    (h3 : neighbor.yield.amount = 2 * johnson.yield.amount)
    (h4 : total_yield johnson 3 + total_yield neighbor 3 = 1200) :
  johnson.yield.amount = 80 := by
  sorry

#check johnson_yield

end NUMINAMATH_CALUDE_johnson_yield_l3151_315178


namespace NUMINAMATH_CALUDE_fifth_roots_sum_l3151_315150

theorem fifth_roots_sum (x y : ℂ) : 
  x = Complex.exp (2 * Real.pi * Complex.I / 5) →
  y = Complex.exp (-2 * Real.pi * Complex.I / 5) →
  x^5 = 1 →
  y^5 = 1 →
  x^5 + y^5 = 2 :=
by sorry

end NUMINAMATH_CALUDE_fifth_roots_sum_l3151_315150


namespace NUMINAMATH_CALUDE_distribute_toys_count_l3151_315141

/-- The number of ways to distribute 4 toys out of 6 distinct toys to 4 distinct people -/
def distribute_toys : ℕ :=
  Nat.factorial 6 / Nat.factorial 2

/-- Theorem stating that distributing 4 toys out of 6 distinct toys to 4 distinct people results in 360 different arrangements -/
theorem distribute_toys_count : distribute_toys = 360 := by
  sorry

end NUMINAMATH_CALUDE_distribute_toys_count_l3151_315141


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_condition_l3151_315113

/-- A polynomial ax^2 + by^2 + cz^2 + dxy + exz + fyz is a perfect square of a trinomial
    if and only if d = 2√(ab), e = 2√(ac), and f = 2√(bc) -/
theorem polynomial_perfect_square_condition
  (a b c d e f : ℝ) :
  (∃ (p q r : ℝ), ∀ (x y z : ℝ),
    a * x^2 + b * y^2 + c * z^2 + d * x * y + e * x * z + f * y * z = (p * x + q * y + r * z)^2)
  ↔
  (d^2 = 4 * a * b ∧ e^2 = 4 * a * c ∧ f^2 = 4 * b * c) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_condition_l3151_315113


namespace NUMINAMATH_CALUDE_bears_in_stock_before_shipment_l3151_315144

/-- The number of bears in a new shipment -/
def new_shipment : ℕ := 10

/-- The number of bears on each shelf -/
def bears_per_shelf : ℕ := 9

/-- The number of shelves used -/
def shelves_used : ℕ := 3

/-- The number of bears in stock before the new shipment -/
def bears_before_shipment : ℕ := shelves_used * bears_per_shelf - new_shipment

theorem bears_in_stock_before_shipment :
  bears_before_shipment = 17 := by
  sorry

end NUMINAMATH_CALUDE_bears_in_stock_before_shipment_l3151_315144


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3151_315135

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 15 * x + c = 0) → 
  (a + c = 24) → 
  (a < c) → 
  (a = (24 - Real.sqrt 351) / 2 ∧ c = (24 + Real.sqrt 351) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3151_315135


namespace NUMINAMATH_CALUDE_april_coffee_cost_l3151_315143

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the cost of coffee for a given day -/
def coffeeCost (day: DayOfWeek) (isEarthDay: Bool) : ℚ :=
  match day with
  | DayOfWeek.Monday => 3.5
  | DayOfWeek.Friday => 3
  | _ => if isEarthDay then 3 else 4

/-- Calculates the total cost of coffee for April -/
def aprilCoffeeCost (startDay: DayOfWeek) : ℚ :=
  sorry

/-- Theorem stating that Jon's total spending on coffee in April is $112 -/
theorem april_coffee_cost :
  aprilCoffeeCost DayOfWeek.Thursday = 112 := by
  sorry

end NUMINAMATH_CALUDE_april_coffee_cost_l3151_315143


namespace NUMINAMATH_CALUDE_four_distinct_positive_roots_l3151_315199

/-- The polynomial f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 + 8*a*x^2 - a*x + a^2

/-- Theorem stating the condition for f(x) to have four distinct positive roots -/
theorem four_distinct_positive_roots (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) ↔
  (1/25 < a ∧ a < 1/24) :=
sorry

end NUMINAMATH_CALUDE_four_distinct_positive_roots_l3151_315199


namespace NUMINAMATH_CALUDE_marble_distribution_l3151_315134

theorem marble_distribution (T : ℝ) (C B O : ℝ) : 
  T > 0 →
  C = 0.40 * T →
  O = (2/5) * T →
  C + B + O = T →
  B = 0.20 * T :=
by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3151_315134


namespace NUMINAMATH_CALUDE_prob_two_same_school_l3151_315137

/-- Represents the number of schools --/
def num_schools : ℕ := 5

/-- Represents the number of students per school --/
def students_per_school : ℕ := 2

/-- Represents the total number of students --/
def total_students : ℕ := num_schools * students_per_school

/-- Represents the number of students chosen for the game --/
def chosen_students : ℕ := 4

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly two students coming from the same school --/
theorem prob_two_same_school : 
  (choose num_schools 1 * choose students_per_school 2 * choose (total_students - students_per_school) (chosen_students - 2)) / 
  (choose total_students chosen_students) = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_prob_two_same_school_l3151_315137


namespace NUMINAMATH_CALUDE_steve_height_after_growth_l3151_315168

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Represents a person's height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Calculates the new height after growth -/
def new_height_after_growth (initial_height : Height) (growth_inches : ℕ) : ℕ :=
  feet_inches_to_inches initial_height.feet initial_height.inches + growth_inches

theorem steve_height_after_growth :
  let initial_height : Height := ⟨5, 6⟩
  let growth_inches : ℕ := 6
  new_height_after_growth initial_height growth_inches = 72 := by
  sorry

end NUMINAMATH_CALUDE_steve_height_after_growth_l3151_315168


namespace NUMINAMATH_CALUDE_smallest_multiple_greater_than_30_l3151_315115

theorem smallest_multiple_greater_than_30 : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 5 → k > 0 → n % k = 0) ∧ 
  (n > 30) ∧
  (∀ m : ℕ, m < n → (∃ k : ℕ, k ≤ 5 ∧ k > 0 ∧ m % k ≠ 0) ∨ m ≤ 30) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_greater_than_30_l3151_315115


namespace NUMINAMATH_CALUDE_quadratic_real_equal_roots_l3151_315157

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 6 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 6 = 0 → y = x) ↔ 
  (m = 2 - 6 * Real.sqrt 2 ∨ m = 2 + 6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_equal_roots_l3151_315157


namespace NUMINAMATH_CALUDE_complex_roots_imaginary_condition_l3151_315116

theorem complex_roots_imaginary_condition (k : ℝ) (hk : k > 0) :
  (∃ z₁ z₂ : ℂ, z₁ ≠ z₂ ∧
    12 * z₁^2 - 4 * I * z₁ - k = 0 ∧
    12 * z₂^2 - 4 * I * z₂ - k = 0 ∧
    (z₁.im = 0 ∧ z₂.re = 0) ∨ (z₁.re = 0 ∧ z₂.im = 0)) ↔
  k = (1 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_complex_roots_imaginary_condition_l3151_315116


namespace NUMINAMATH_CALUDE_continuity_properties_l3151_315119

theorem continuity_properties :
  (¬ ∀ a b : ℤ, a < b → ∃ c : ℤ, a < c ∧ c < b) ∧
  (¬ ∀ S : Set ℤ, S.Nonempty → (∃ x : ℤ, ∀ y ∈ S, y ≤ x) → ∃ z : ℤ, ∀ y ∈ S, y ≤ z ∧ ∀ w : ℤ, (∀ y ∈ S, y ≤ w) → z ≤ w) ∧
  (∀ a b : ℚ, a < b → ∃ c : ℚ, a < c ∧ c < b) ∧
  (¬ ∀ S : Set ℚ, S.Nonempty → (∃ x : ℚ, ∀ y ∈ S, y ≤ x) → ∃ z : ℚ, ∀ y ∈ S, y ≤ z ∧ ∀ w : ℚ, (∀ y ∈ S, y ≤ w) → z ≤ w) :=
by sorry

end NUMINAMATH_CALUDE_continuity_properties_l3151_315119


namespace NUMINAMATH_CALUDE_investment_percentage_l3151_315186

theorem investment_percentage (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (additional_rate : ℝ) 
  (h1 : initial_investment = 1400)
  (h2 : initial_rate = 0.05)
  (h3 : additional_investment = 700)
  (h4 : additional_rate = 0.08) : 
  (initial_investment * initial_rate + additional_investment * additional_rate) / 
  (initial_investment + additional_investment) = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_l3151_315186


namespace NUMINAMATH_CALUDE_divisibility_condition_l3151_315191

theorem divisibility_condition (a b : ℕ+) :
  (a.val^2 + b.val^2 - a.val - b.val + 1) % (a.val * b.val) = 0 ↔ a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3151_315191


namespace NUMINAMATH_CALUDE_equal_projections_l3151_315147

/-- A circle divided into 42 equal arcs -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (points : Fin 42 → ℝ × ℝ)

/-- Projection of a point onto a line segment -/
def project (p : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem equal_projections (c : Circle) :
  let A₀ := c.points 0
  let A₃ := c.points 3
  let A₆ := c.points 6
  let A₇ := c.points 7
  let A₉ := c.points 9
  let A₂₁ := c.points 21
  let A'₃ := project A₃ A₀ A₂₁
  let A'₆ := project A₆ A₀ A₂₁
  let A'₇ := project A₇ A₀ A₂₁
  let A'₉ := project A₉ A₀ A₂₁
  distance A'₃ A'₆ = distance A'₇ A'₉ := by
    sorry

end NUMINAMATH_CALUDE_equal_projections_l3151_315147


namespace NUMINAMATH_CALUDE_projection_matrix_values_l3151_315136

def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

def special_matrix (x y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![x, 21/49; y, 35/49]

theorem projection_matrix_values :
  ∀ x y : ℚ,
  is_projection_matrix (special_matrix x y) →
  x = 666/2401 ∧ y = (49 * 2401)/1891 := by
sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l3151_315136


namespace NUMINAMATH_CALUDE_candy_difference_l3151_315100

def frankie_candy : ℕ := 74
def max_candy : ℕ := 92

theorem candy_difference : max_candy - frankie_candy = 18 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l3151_315100


namespace NUMINAMATH_CALUDE_weight_of_b_l3151_315117

theorem weight_of_b (wa wb wc : ℝ) (ha hb hc : ℝ) : 
  (wa + wb + wc) / 3 = 45 →
  hb = 2 * ha →
  hc = ha + 20 →
  (wa + wb) / 2 = 40 →
  (wb + wc) / 2 = 43 →
  (ha + hc) / 2 = 155 →
  wb = 31 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l3151_315117


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_both_zero_l3151_315175

theorem sum_of_squares_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_both_zero_l3151_315175


namespace NUMINAMATH_CALUDE_chocolate_bars_left_l3151_315179

theorem chocolate_bars_left (initial_bars : ℕ) (people : ℕ) (given_to_mother : ℕ) (eaten : ℕ) : 
  initial_bars = 20 →
  people = 5 →
  given_to_mother = 3 →
  eaten = 2 →
  (initial_bars / people / 2 * people) - given_to_mother - eaten = 5 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bars_left_l3151_315179


namespace NUMINAMATH_CALUDE_shaded_area_octagon_semicircles_l3151_315126

/-- The area of the shaded region inside a regular octagon but outside eight semicircles -/
theorem shaded_area_octagon_semicircles : 
  let s : ℝ := 4  -- side length of the octagon
  let octagon_area : ℝ := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area : ℝ := π * (s/2)^2 / 2
  let total_semicircle_area : ℝ := 8 * semicircle_area
  octagon_area - total_semicircle_area = 32 * (1 + Real.sqrt 2) - 16 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_octagon_semicircles_l3151_315126


namespace NUMINAMATH_CALUDE_consecutive_five_digit_numbers_l3151_315154

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def abccc (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * c + c

def abbbb (a b : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * b + 10 * b + b

theorem consecutive_five_digit_numbers :
  ∀ a b c : ℕ,
    a < 10 → b < 10 → c < 10 →
    is_five_digit (abccc a b c) →
    is_five_digit (abbbb a b) →
    (abccc a b c).succ = abbbb a b ∨ (abbbb a b).succ = abccc a b c →
    ((a = 1 ∧ b = 0 ∧ c = 9) ∨ (a = 8 ∧ b = 9 ∧ c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_five_digit_numbers_l3151_315154


namespace NUMINAMATH_CALUDE_machine_work_time_l3151_315192

theorem machine_work_time (x : ℝ) : x > 0 → 
  (1 / (x + 8) + 2 / (3 * x) + 1 / (2 * x) = 1 / (2 * x)) → 
  x = (-1 + Real.sqrt 97) / 6 :=
by sorry

end NUMINAMATH_CALUDE_machine_work_time_l3151_315192


namespace NUMINAMATH_CALUDE_series_sum_equals_ln2_minus_half_l3151_315127

open Real

/-- The sum of the series Σ(1/((2n-1) * 2n * (2n+1))) for n from 1 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, 1 / ((2*n - 1) * (2*n) * (2*n + 1))

/-- Theorem stating that the sum of the series equals ln 2 - 1/2 -/
theorem series_sum_equals_ln2_minus_half : seriesSum = log 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_ln2_minus_half_l3151_315127


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3151_315173

/-- Represents a two-digit number with units digit x -/
def two_digit_number (x : ℕ) : ℕ := 10 * (x + 4) + x

/-- The equation that needs to be proven -/
def equation_holds (x : ℕ) : Prop :=
  x^2 + (x + 4)^2 = 10 * (x + 4) + x - 4

/-- The sum of squares of digits is 4 less than the number -/
def sum_squares_property (x : ℕ) : Prop :=
  x^2 + (x + 4)^2 = two_digit_number x - 4

theorem two_digit_number_property (x : ℕ) :
  equation_holds x ↔ sum_squares_property x :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3151_315173


namespace NUMINAMATH_CALUDE_birds_in_tree_l3151_315112

theorem birds_in_tree (initial_birds : Real) (birds_flew_away : Real) 
  (h1 : initial_birds = 42.5)
  (h2 : birds_flew_away = 27.3) : 
  initial_birds - birds_flew_away = 15.2 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3151_315112


namespace NUMINAMATH_CALUDE_positive_polynomial_fraction_representation_l3151_315130

/-- A polynomial with real coefficients. -/
def RealPolynomial := Polynomial ℝ

/-- Proposition that a polynomial is positive for all positive real numbers. -/
def IsPositiveForPositive (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, x > 0 → p.eval x > 0

/-- Proposition that a polynomial has nonnegative coefficients. -/
def HasNonnegativeCoeffs (p : RealPolynomial) : Prop :=
  ∀ n : ℕ, p.coeff n ≥ 0

/-- Main theorem: For any polynomial P that is positive for all positive real numbers,
    there exist polynomials Q and R with nonnegative coefficients such that
    P(x) = Q(x)/R(x) for all positive real numbers x. -/
theorem positive_polynomial_fraction_representation
  (P : RealPolynomial) (h : IsPositiveForPositive P) :
  ∃ (Q R : RealPolynomial), HasNonnegativeCoeffs Q ∧ HasNonnegativeCoeffs R ∧
    ∀ x : ℝ, x > 0 → P.eval x = (Q.eval x) / (R.eval x) := by
  sorry

end NUMINAMATH_CALUDE_positive_polynomial_fraction_representation_l3151_315130


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3151_315105

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + p*x^2 + q*x + r
  where
  p := -(a + b + c)
  q := a*b + b*c + c*a
  r := -a*b*c

theorem cubic_roots_sum (a b c : ℝ) :
  (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < 1) →
  CubicPolynomial a b c 0 = -1/8 →
  (∃ r : ℝ, b = a*r ∧ c = a*r^2) →
  (∑' k, (a^k + b^k + c^k)) = 9/2 →
  a + b + c = 19/12 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3151_315105


namespace NUMINAMATH_CALUDE_wednesday_to_tuesday_ratio_l3151_315122

/-- The amount of money Max's mom gave him on Tuesday, Wednesday, and Thursday --/
structure MoneyGiven where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the problem --/
def ProblemConditions (m : MoneyGiven) : Prop :=
  m.tuesday = 8 ∧
  ∃ k : ℕ, m.wednesday = k * m.tuesday ∧
  m.thursday = m.wednesday + 9 ∧
  m.thursday = m.tuesday + 41

/-- The theorem to be proved --/
theorem wednesday_to_tuesday_ratio
  (m : MoneyGiven)
  (h : ProblemConditions m) :
  m.wednesday / m.tuesday = 5 := by
sorry

end NUMINAMATH_CALUDE_wednesday_to_tuesday_ratio_l3151_315122


namespace NUMINAMATH_CALUDE_simplified_expression_l3151_315172

theorem simplified_expression (a : ℤ) (h : a = 2022) : 
  (a + 1 : ℚ) / a - 2 * (a : ℚ) / (a + 1) = (-a^2 + 2*a + 1 : ℚ) / (a * (a + 1)) ∧
  -a^2 + 2*a + 1 = -2022^2 + 4045 :=
sorry

end NUMINAMATH_CALUDE_simplified_expression_l3151_315172


namespace NUMINAMATH_CALUDE_equation_three_real_roots_l3151_315171

theorem equation_three_real_roots (k : ℂ) : 
  (∃! (r₁ r₂ r₃ : ℝ), ∀ (x : ℝ), 
    (x / (x + 3) + x / (x - 3) = k * x) ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ↔ 
  (k = Complex.I / 3 ∨ k = -Complex.I / 3) :=
sorry

end NUMINAMATH_CALUDE_equation_three_real_roots_l3151_315171


namespace NUMINAMATH_CALUDE_factor_implies_p_value_l3151_315197

theorem factor_implies_p_value (p : ℚ) : 
  (∀ x : ℚ, (3 * x + 4 = 0) → (4 * x^3 + p * x^2 + 17 * x + 24 = 0)) → 
  p = 13/4 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_p_value_l3151_315197


namespace NUMINAMATH_CALUDE_flagstaff_shadow_length_l3151_315107

/-- Given a flagstaff and a building casting shadows under similar conditions,
    this theorem proves the length of the flagstaff's shadow. -/
theorem flagstaff_shadow_length
  (h_flagstaff : ℝ)
  (h_building : ℝ)
  (s_building : ℝ)
  (h_flagstaff_pos : h_flagstaff > 0)
  (h_building_pos : h_building > 0)
  (s_building_pos : s_building > 0)
  (h_flagstaff_val : h_flagstaff = 17.5)
  (h_building_val : h_building = 12.5)
  (s_building_val : s_building = 28.75) :
  ∃ s_flagstaff : ℝ, s_flagstaff = 40.15 ∧ h_flagstaff / s_flagstaff = h_building / s_building :=
by
  sorry


end NUMINAMATH_CALUDE_flagstaff_shadow_length_l3151_315107
