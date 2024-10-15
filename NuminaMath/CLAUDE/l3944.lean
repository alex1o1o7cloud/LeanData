import Mathlib

namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3944_394485

theorem power_fraction_simplification : (25 ^ 40) / (125 ^ 20) = 5 ^ 20 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3944_394485


namespace NUMINAMATH_CALUDE_rectangular_frame_area_l3944_394447

theorem rectangular_frame_area : 
  let width : ℚ := 81 / 4
  let depth : ℚ := 148 / 9
  let area : ℚ := width * depth
  ⌊area⌋ = 333 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_frame_area_l3944_394447


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l3944_394489

theorem new_ratio_after_addition (a b : ℤ) : 
  (a : ℚ) / b = 1 / 4 →
  b = 72 →
  (a + 6 : ℚ) / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l3944_394489


namespace NUMINAMATH_CALUDE_geometric_sequence_10th_term_l3944_394470

theorem geometric_sequence_10th_term :
  let a₁ : ℚ := 5
  let r : ℚ := 5 / 3
  let n : ℕ := 10
  let aₙ := a₁ * r^(n - 1)
  aₙ = 9765625 / 19683 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_10th_term_l3944_394470


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l3944_394495

theorem smallest_factor_for_perfect_square : ∃ (y : ℕ), 
  (y > 0) ∧ 
  (∃ (n : ℕ), 76545 * y = n^2) ∧
  (y % 3 ≠ 0) ∧ 
  (y % 5 ≠ 0) ∧
  (∀ (z : ℕ), z > 0 ∧ z < y → ¬(∃ (m : ℕ), 76545 * z = m^2) ∨ (z % 3 = 0) ∨ (z % 5 = 0)) ∧
  y = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l3944_394495


namespace NUMINAMATH_CALUDE_alpha_plus_beta_is_75_degrees_l3944_394433

theorem alpha_plus_beta_is_75_degrees (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2)  -- α is acute
  (h2 : 0 < β ∧ β < π / 2)  -- β is acute
  (h3 : |Real.sin α - 1/2| + Real.sqrt (Real.tan β - 1) = 0) : 
  α + β = π / 2.4 := by  -- π/2.4 is equivalent to 75°
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_is_75_degrees_l3944_394433


namespace NUMINAMATH_CALUDE_soccer_team_wins_l3944_394492

/-- Given a soccer team that played 140 games and won 50 percent of them,
    prove that the number of games won is 70. -/
theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) :
  total_games = 140 →
  win_percentage = 1/2 →
  games_won = (total_games : ℚ) * win_percentage →
  games_won = 70 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_wins_l3944_394492


namespace NUMINAMATH_CALUDE_sector_angle_l3944_394425

theorem sector_angle (r : ℝ) (α : ℝ) (h1 : α * r = 5) (h2 : (1/2) * α * r^2 = 5) : α = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3944_394425


namespace NUMINAMATH_CALUDE_amy_photo_upload_l3944_394423

theorem amy_photo_upload (num_albums : ℕ) (photos_per_album : ℕ) 
  (h1 : num_albums = 9)
  (h2 : photos_per_album = 20) :
  num_albums * photos_per_album = 180 := by
sorry

end NUMINAMATH_CALUDE_amy_photo_upload_l3944_394423


namespace NUMINAMATH_CALUDE_triangular_front_view_solids_l3944_394471

/-- Enumeration of possible solids --/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- Definition of a solid with a triangular front view --/
def has_triangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

/-- Theorem stating that a solid with a triangular front view must be one of the specified solids --/
theorem triangular_front_view_solids (s : Solid) :
  has_triangular_front_view s →
  (s = Solid.TriangularPyramid ∨ s = Solid.SquarePyramid ∨ s = Solid.TriangularPrism ∨ s = Solid.Cone) :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_front_view_solids_l3944_394471


namespace NUMINAMATH_CALUDE_sequential_no_conditional_l3944_394448

-- Define the structures
inductive FlowchartStructure
  | Sequential
  | Loop
  | If
  | Until

-- Define a predicate for structures that generally contain a conditional judgment box
def hasConditionalJudgment : FlowchartStructure → Prop
  | FlowchartStructure.Sequential => False
  | FlowchartStructure.Loop => True
  | FlowchartStructure.If => True
  | FlowchartStructure.Until => True

theorem sequential_no_conditional : 
  ∀ (s : FlowchartStructure), ¬hasConditionalJudgment s ↔ s = FlowchartStructure.Sequential :=
by sorry

end NUMINAMATH_CALUDE_sequential_no_conditional_l3944_394448


namespace NUMINAMATH_CALUDE_extracurricular_activity_selection_l3944_394476

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem extracurricular_activity_selection 
  (total_people : ℕ) 
  (boys : ℕ) 
  (girls : ℕ) 
  (leaders : ℕ) 
  (to_select : ℕ) :
  total_people = 13 →
  boys = 8 →
  girls = 5 →
  leaders = 2 →
  to_select = 5 →
  (choose girls 1 * choose boys 4 = 350) ∧
  (choose (total_people - leaders) 3 = 165) ∧
  (choose total_people to_select - choose (total_people - leaders) to_select = 825) :=
by sorry

end NUMINAMATH_CALUDE_extracurricular_activity_selection_l3944_394476


namespace NUMINAMATH_CALUDE_qin_jiushao_area_formula_l3944_394455

theorem qin_jiushao_area_formula (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  let S := Real.sqrt ((c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2) / 4)
  a = 25 → b = 24 → c = 14 → S = (105 * Real.sqrt 39) / 4 :=
by sorry

end NUMINAMATH_CALUDE_qin_jiushao_area_formula_l3944_394455


namespace NUMINAMATH_CALUDE_certain_number_equation_l3944_394456

theorem certain_number_equation (x : ℝ) : 13 * x + 14 * x + 17 * x + 11 = 143 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3944_394456


namespace NUMINAMATH_CALUDE_rectangle_division_l3944_394486

/-- If a rectangle with an area of 59.6 square centimeters is divided into 4 equal parts, 
    then the area of one part is 14.9 square centimeters. -/
theorem rectangle_division (total_area : ℝ) (num_parts : ℕ) (area_of_part : ℝ) : 
  total_area = 59.6 → 
  num_parts = 4 → 
  area_of_part = total_area / num_parts → 
  area_of_part = 14.9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_division_l3944_394486


namespace NUMINAMATH_CALUDE_ronald_store_visits_l3944_394409

def store_visits (bananas_per_visit : ℕ) (total_bananas : ℕ) : ℕ :=
  total_bananas / bananas_per_visit

theorem ronald_store_visits :
  let bananas_per_visit := 10
  let total_bananas := 20
  store_visits bananas_per_visit total_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_ronald_store_visits_l3944_394409


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3944_394457

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 17600

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 1.76
    exponent := 4
    is_valid := by sorry }

/-- Theorem stating that the proposed notation correctly represents the original number -/
theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3944_394457


namespace NUMINAMATH_CALUDE_four_level_pyramid_books_l3944_394429

def pyramid_books (levels : ℕ) (ratio : ℝ) (top_level_books : ℕ) : ℝ :=
  let rec sum_levels (n : ℕ) : ℝ :=
    if n = 0 then 0
    else (top_level_books : ℝ) * (ratio ^ (n - 1)) + sum_levels (n - 1)
  sum_levels levels

theorem four_level_pyramid_books :
  pyramid_books 4 (1 / 0.8) 64 = 369 := by sorry

end NUMINAMATH_CALUDE_four_level_pyramid_books_l3944_394429


namespace NUMINAMATH_CALUDE_hotel_room_cost_l3944_394498

theorem hotel_room_cost (total_rooms : ℕ) (double_room_cost : ℕ) (total_revenue : ℕ) (single_rooms : ℕ) :
  total_rooms = 260 →
  double_room_cost = 60 →
  total_revenue = 14000 →
  single_rooms = 64 →
  ∃ (single_room_cost : ℕ),
    single_room_cost = 35 ∧
    single_room_cost * single_rooms + double_room_cost * (total_rooms - single_rooms) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_hotel_room_cost_l3944_394498


namespace NUMINAMATH_CALUDE_martha_jackets_bought_l3944_394451

theorem martha_jackets_bought (J : ℕ) : 
  (J + J / 2 : ℕ) + (9 + 9 / 3 : ℕ) = 18 → J = 4 :=
by sorry

end NUMINAMATH_CALUDE_martha_jackets_bought_l3944_394451


namespace NUMINAMATH_CALUDE_guessing_game_factor_l3944_394468

theorem guessing_game_factor (f : ℚ) : 33 * f = 2 * 51 - 3 → f = 3 := by
  sorry

end NUMINAMATH_CALUDE_guessing_game_factor_l3944_394468


namespace NUMINAMATH_CALUDE_camp_attendance_l3944_394490

theorem camp_attendance (stay_home : ℕ) (difference : ℕ) (camp : ℕ) : 
  stay_home = 777622 → difference = 574664 → camp + difference = stay_home → camp = 202958 := by
sorry

end NUMINAMATH_CALUDE_camp_attendance_l3944_394490


namespace NUMINAMATH_CALUDE_fir_trees_count_l3944_394408

theorem fir_trees_count :
  ∀ n : ℕ,
  n < 25 →
  n % 11 = 0 →
  n = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_fir_trees_count_l3944_394408


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3944_394453

theorem no_solution_for_equation : 
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 2 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3944_394453


namespace NUMINAMATH_CALUDE_path_length_for_73_segment_l3944_394413

/-- Represents a segment divided into smaller parts with squares constructed on each part --/
structure SegmentWithSquares where
  length : ℝ
  num_parts : ℕ

/-- Calculates the length of the path along the arrows for a given segment with squares --/
def path_length (s : SegmentWithSquares) : ℝ := 3 * s.length

theorem path_length_for_73_segment : 
  let s : SegmentWithSquares := { length := 73, num_parts := 2 }
  path_length s = 219 := by sorry

end NUMINAMATH_CALUDE_path_length_for_73_segment_l3944_394413


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l3944_394452

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-2, 0) and B(-1, 4) -/
theorem equidistant_point_y_coordinate :
  ∃ y : ℝ, ((-2 : ℝ) - 0)^2 + (0 - y)^2 = ((-1 : ℝ) - 0)^2 + (4 - y)^2 ∧ y = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l3944_394452


namespace NUMINAMATH_CALUDE_arun_weight_theorem_l3944_394472

def arun_weight_conditions (w : ℕ) : Prop :=
  64 < w ∧ w < 72 ∧ w % 3 = 0 ∧  -- Arun's condition
  60 < w ∧ w < 70 ∧ w % 2 = 0 ∧  -- Brother's condition
  w ≤ 67 ∧ Nat.Prime w ∧         -- Mother's condition
  63 ≤ w ∧ w ≤ 71 ∧ w % 5 = 0 ∧  -- Sister's condition
  62 < w ∧ w ≤ 73 ∧ w % 4 = 0    -- Father's condition

theorem arun_weight_theorem :
  ∃! w : ℕ, arun_weight_conditions w ∧ w = 66 := by
  sorry

end NUMINAMATH_CALUDE_arun_weight_theorem_l3944_394472


namespace NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3944_394440

theorem complex_equation_imaginary_part :
  ∀ z : ℂ, (4 + 3*I)*z = Complex.abs (3 - 4*I) → Complex.im z = -3/5 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3944_394440


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l3944_394441

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 5}

-- State the theorem
theorem intersection_equals_open_interval :
  M ∩ N = Set.Ioo 1 5 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l3944_394441


namespace NUMINAMATH_CALUDE_decimal_fraction_sum_equals_one_l3944_394415

theorem decimal_fraction_sum_equals_one : ∃ (a b c d e f g h : Nat),
  (a = 2 ∨ a = 3) ∧ (b = 2 ∨ b = 3) ∧
  (c = 2 ∨ c = 3) ∧ (d = 2 ∨ d = 3) ∧
  (e = 2 ∨ e = 3) ∧ (f = 2 ∨ f = 3) ∧
  (g = 2 ∨ g = 3) ∧ (h = 2 ∨ h = 3) ∧
  (a * 10 + b) / 100 + (c * 10 + d) / 100 + (e * 10 + f) / 100 + (g * 10 + h) / 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_decimal_fraction_sum_equals_one_l3944_394415


namespace NUMINAMATH_CALUDE_final_liquid_X_percentage_l3944_394481

/-- Composition of a solution -/
structure Solution :=
  (x : ℝ) -- Percentage of liquid X
  (water : ℝ) -- Percentage of water
  (z : ℝ) -- Percentage of liquid Z

/-- Given conditions -/
def solution_Y : Solution := ⟨20, 55, 25⟩
def initial_weight : ℝ := 12
def evaporated_water : ℝ := 4
def added_Y_weight : ℝ := 3
def solution_B : Solution := ⟨35, 15, 50⟩
def added_B_weight : ℝ := 2
def evaporation_factor : ℝ := 0.75
def solution_D : Solution := ⟨15, 60, 25⟩
def added_D_weight : ℝ := 6

/-- The theorem to prove -/
theorem final_liquid_X_percentage :
  let initial_X := solution_Y.x * initial_weight / 100
  let initial_Z := solution_Y.z * initial_weight / 100
  let remaining_water := solution_Y.water * initial_weight / 100 - evaporated_water
  let added_Y_X := solution_Y.x * added_Y_weight / 100
  let added_Y_water := solution_Y.water * added_Y_weight / 100
  let added_Y_Z := solution_Y.z * added_Y_weight / 100
  let added_B_X := solution_B.x * added_B_weight / 100
  let added_B_water := solution_B.water * added_B_weight / 100
  let added_B_Z := solution_B.z * added_B_weight / 100
  let total_before_evap := initial_X + initial_Z + remaining_water + added_Y_X + added_Y_water + added_Y_Z + added_B_X + added_B_water + added_B_Z
  let total_after_evap := total_before_evap * evaporation_factor
  let evaporated_water_2 := (1 - evaporation_factor) * (remaining_water + added_Y_water + added_B_water)
  let remaining_water_2 := remaining_water + added_Y_water + added_B_water - evaporated_water_2
  let added_D_X := solution_D.x * added_D_weight / 100
  let added_D_water := solution_D.water * added_D_weight / 100
  let added_D_Z := solution_D.z * added_D_weight / 100
  let final_X := initial_X + added_Y_X + added_B_X + added_D_X
  let final_water := remaining_water_2 + added_D_water
  let final_Z := initial_Z + added_Y_Z + added_B_Z + added_D_Z
  let final_total := final_X + final_water + final_Z
  final_X / final_total * 100 = 25.75 := by sorry

end NUMINAMATH_CALUDE_final_liquid_X_percentage_l3944_394481


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3944_394426

/-- An isosceles triangle with perimeter 18 and one side 4 has base length 7 -/
theorem isosceles_triangle_base_length : 
  ∀ (a b c : ℝ), 
    a + b + c = 18 →  -- perimeter is 18
    (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
    (a = 4 ∨ b = 4 ∨ c = 4) →  -- one side is 4
    (a + b > c ∧ b + c > a ∧ a + c > b) →  -- triangle inequality
    (if a = b then c else if b = c then a else b) = 7 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3944_394426


namespace NUMINAMATH_CALUDE_rem_negative_five_ninths_seven_thirds_l3944_394491

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_negative_five_ninths_seven_thirds :
  rem (-5/9 : ℚ) (7/3 : ℚ) = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_rem_negative_five_ninths_seven_thirds_l3944_394491


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3944_394405

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3944_394405


namespace NUMINAMATH_CALUDE_admission_fees_proof_l3944_394406

-- Define the given conditions
def child_fee : ℚ := 1.5
def adult_fee : ℚ := 4
def total_people : ℕ := 315
def num_children : ℕ := 180

-- Define the function to calculate total admission fees
def total_admission_fees : ℚ :=
  (child_fee * num_children) + (adult_fee * (total_people - num_children))

-- Theorem to prove
theorem admission_fees_proof : total_admission_fees = 810 := by
  sorry

end NUMINAMATH_CALUDE_admission_fees_proof_l3944_394406


namespace NUMINAMATH_CALUDE_f_properties_l3944_394479

open Real

noncomputable def f (x : ℝ) : ℝ := 2 / x + log x

theorem f_properties :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ = f x₂ → x₁ + x₂ > 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3944_394479


namespace NUMINAMATH_CALUDE_livestream_sales_scientific_notation_l3944_394419

/-- Proves that 1814 billion yuan is equal to 1.814 × 10^12 yuan -/
theorem livestream_sales_scientific_notation :
  (1814 : ℝ) * (10^9 : ℝ) = 1.814 * (10^12 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_livestream_sales_scientific_notation_l3944_394419


namespace NUMINAMATH_CALUDE_twentieth_base5_is_40_l3944_394487

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : ℕ :=
  if n < 5 then n
  else 10 * toBase5 (n / 5) + (n % 5)

/-- The 20th number in base 5 sequence -/
def twentieth_base5 : ℕ := toBase5 20

theorem twentieth_base5_is_40 : twentieth_base5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_base5_is_40_l3944_394487


namespace NUMINAMATH_CALUDE_binomial_representation_l3944_394497

theorem binomial_representation (n : ℕ) :
  ∃ x y z : ℕ, n = Nat.choose x 1 + Nat.choose y 2 + Nat.choose z 3 ∧
  ((0 ≤ x ∧ x < y ∧ y < z) ∨ (x = 0 ∧ y = 0 ∧ 0 < z)) :=
sorry

end NUMINAMATH_CALUDE_binomial_representation_l3944_394497


namespace NUMINAMATH_CALUDE_blood_donor_selection_l3944_394449

theorem blood_donor_selection (type_O : Nat) (type_A : Nat) (type_B : Nat) (type_AB : Nat)
  (h1 : type_O = 10)
  (h2 : type_A = 5)
  (h3 : type_B = 8)
  (h4 : type_AB = 3) :
  type_O * type_A * type_B * type_AB = 1200 := by
  sorry

end NUMINAMATH_CALUDE_blood_donor_selection_l3944_394449


namespace NUMINAMATH_CALUDE_cat_walking_time_l3944_394494

/-- Proves that the total time for Jenny's cat walking process is 28 minutes -/
theorem cat_walking_time (resisting_time : ℝ) (walking_distance : ℝ) (walking_rate : ℝ) : 
  resisting_time = 20 →
  walking_distance = 64 →
  walking_rate = 8 →
  resisting_time + walking_distance / walking_rate = 28 :=
by sorry

end NUMINAMATH_CALUDE_cat_walking_time_l3944_394494


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3944_394439

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3944_394439


namespace NUMINAMATH_CALUDE_histogram_group_width_l3944_394488

/-- Represents a group in a frequency histogram -/
structure HistogramGroup where
  a : ℝ
  b : ℝ
  m : ℝ  -- frequency
  h : ℝ  -- height
  h_pos : h > 0
  m_pos : m > 0
  a_lt_b : a < b

/-- 
The absolute value of the group width |a-b| in a frequency histogram 
is equal to the frequency m divided by the height h.
-/
theorem histogram_group_width (g : HistogramGroup) : 
  |g.b - g.a| = g.m / g.h := by
  sorry

end NUMINAMATH_CALUDE_histogram_group_width_l3944_394488


namespace NUMINAMATH_CALUDE_camera_tax_calculation_l3944_394442

/-- Calculate the tax amount given the base price and tax rate -/
def calculateTax (basePrice taxRate : ℝ) : ℝ :=
  basePrice * taxRate

/-- Prove that the tax amount for a $200 camera with 15% tax rate is $30 -/
theorem camera_tax_calculation :
  let basePrice : ℝ := 200
  let taxRate : ℝ := 0.15
  calculateTax basePrice taxRate = 30 := by
sorry

#eval calculateTax 200 0.15

end NUMINAMATH_CALUDE_camera_tax_calculation_l3944_394442


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_three_l3944_394410

theorem sqrt_equality_implies_one_three :
  ∀ a b : ℕ+,
  a < b →
  (Real.sqrt (1 + Real.sqrt (27 + 18 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 3) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_three_l3944_394410


namespace NUMINAMATH_CALUDE_eight_pow_plus_six_div_seven_l3944_394458

theorem eight_pow_plus_six_div_seven (n : ℕ) : 
  7 ∣ (8^n + 6) := by sorry

end NUMINAMATH_CALUDE_eight_pow_plus_six_div_seven_l3944_394458


namespace NUMINAMATH_CALUDE_regular_polygon_with_36_degree_exterior_angle_is_decagon_l3944_394462

/-- A regular polygon with exterior angles measuring 36° has 10 sides -/
theorem regular_polygon_with_36_degree_exterior_angle_is_decagon :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 36 →
    (n : ℝ) * exterior_angle = 360 →
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_36_degree_exterior_angle_is_decagon_l3944_394462


namespace NUMINAMATH_CALUDE_point_M_coordinates_l3944_394434

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem point_M_coordinates :
  ∀ x y : ℝ,
  y = curve x →
  curve_derivative x = -4 →
  (x = -1 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l3944_394434


namespace NUMINAMATH_CALUDE_equation_solution_l3944_394463

theorem equation_solution :
  let x : ℚ := -21/20
  (Real.sqrt (2 * x + 7) / Real.sqrt (4 * x + 7) = Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3944_394463


namespace NUMINAMATH_CALUDE_juanitas_dessert_cost_l3944_394421

/-- Represents the cost of a brownie dessert with various toppings -/
def brownieDessertCost (brownieCost iceCreamCost syrupCost nutsCost : ℚ)
  (iceCreamScoops syrupServings : ℕ) (includeNuts : Bool) : ℚ :=
  brownieCost +
  iceCreamCost * iceCreamScoops +
  syrupCost * syrupServings +
  (if includeNuts then nutsCost else 0)

/-- Proves that Juanita's dessert costs $7.00 given the prices and her order -/
theorem juanitas_dessert_cost :
  let brownieCost : ℚ := 5/2
  let iceCreamCost : ℚ := 1
  let syrupCost : ℚ := 1/2
  let nutsCost : ℚ := 3/2
  let iceCreamScoops : ℕ := 2
  let syrupServings : ℕ := 2
  let includeNuts : Bool := true
  brownieDessertCost brownieCost iceCreamCost syrupCost nutsCost
    iceCreamScoops syrupServings includeNuts = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_juanitas_dessert_cost_l3944_394421


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3944_394493

/-- The equation of an ellipse in its standard form -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 = 1

/-- The length of the major axis of an ellipse -/
def major_axis_length : ℝ := 8

/-- Theorem: The length of the major axis of the ellipse x^2/16 + y^2 = 1 is 8 -/
theorem ellipse_major_axis_length :
  ∀ x y : ℝ, is_ellipse x y → major_axis_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3944_394493


namespace NUMINAMATH_CALUDE_units_digit_52_cubed_plus_29_cubed_l3944_394431

theorem units_digit_52_cubed_plus_29_cubed : (52^3 + 29^3) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_52_cubed_plus_29_cubed_l3944_394431


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3944_394404

/-- Given three numbers A, B, and C, where A is a three-digit number and B and C are two-digit numbers,
    if the sum of numbers containing the digit seven is 208 and the sum of numbers containing the digit three is 76,
    then the sum of A, B, and C is 247. -/
theorem sum_of_numbers (A B C : ℕ) : 
  100 ≤ A ∧ A < 1000 ∧   -- A is a three-digit number
  10 ≤ B ∧ B < 100 ∧     -- B is a two-digit number
  10 ≤ C ∧ C < 100 ∧     -- C is a two-digit number
  ((A.repr.contains '7' ∨ B.repr.contains '7' ∨ C.repr.contains '7') → A + B + C = 208) ∧  -- Sum of numbers with 7
  (B.repr.contains '3' ∧ C.repr.contains '3' → B + C = 76)  -- Sum of numbers with 3
  → A + B + C = 247 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3944_394404


namespace NUMINAMATH_CALUDE_odd_periodic_function_difference_l3944_394430

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period 4 if f(x) = f(x + 4) for all x -/
def HasPeriod4 (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 4)

theorem odd_periodic_function_difference (f : ℝ → ℝ) 
  (h_odd : IsOdd f) 
  (h_period : HasPeriod4 f) 
  (h_def : ∀ x ∈ Set.Ioo (-2) 0, f x = 2^x) : 
  f 2016 - f 2015 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_difference_l3944_394430


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l3944_394445

/-- Calculates the total cost of John's purchase of soap and shampoo. -/
def total_cost (soap_bars : ℕ) (soap_weight : ℝ) (soap_price : ℝ)
                (shampoo_bottles : ℕ) (shampoo_weight : ℝ) (shampoo_price : ℝ) : ℝ :=
  (soap_bars : ℝ) * soap_weight * soap_price +
  (shampoo_bottles : ℝ) * shampoo_weight * shampoo_price

/-- Proves that John's total spending on soap and shampoo is $41.40. -/
theorem johns_purchase_cost : 
  total_cost 20 1.5 0.5 15 2.2 0.8 = 41.40 := by
  sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l3944_394445


namespace NUMINAMATH_CALUDE_power_multiplication_correct_equation_l3944_394466

theorem power_multiplication (a b : ℕ) : 2^a * 2^b = 2^(a + b) := by sorry

theorem correct_equation : 2^2 * 2^3 = 2^5 := by
  apply power_multiplication

end NUMINAMATH_CALUDE_power_multiplication_correct_equation_l3944_394466


namespace NUMINAMATH_CALUDE_seashells_sum_l3944_394436

/-- The number of seashells Joan found on the beach -/
def total_seashells : ℕ := 70

/-- The number of seashells Joan gave to Sam -/
def seashells_given : ℕ := 43

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 27

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem seashells_sum : total_seashells = seashells_given + seashells_left := by
  sorry

end NUMINAMATH_CALUDE_seashells_sum_l3944_394436


namespace NUMINAMATH_CALUDE_max_value_of_f_l3944_394400

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 12 * x - 5

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3944_394400


namespace NUMINAMATH_CALUDE_triangle_inequality_l3944_394422

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  |((a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (c + a))| < 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3944_394422


namespace NUMINAMATH_CALUDE_us_flag_stars_l3944_394414

theorem us_flag_stars (stripes : ℕ) (total_shapes : ℕ) : 
  stripes = 13 → 
  total_shapes = 54 → 
  ∃ (stars : ℕ), 
    (stars / 2 - 3 + 2 * stripes + 6 = total_shapes) ∧ 
    (stars = 50) := by
  sorry

end NUMINAMATH_CALUDE_us_flag_stars_l3944_394414


namespace NUMINAMATH_CALUDE_bobs_walking_rate_l3944_394465

/-- Proves that Bob's walking rate is 7 miles per hour given the conditions of the problem -/
theorem bobs_walking_rate 
  (total_distance : ℝ) 
  (yolanda_rate : ℝ) 
  (bob_distance : ℝ) 
  (head_start : ℝ) 
  (h1 : total_distance = 65) 
  (h2 : yolanda_rate = 5) 
  (h3 : bob_distance = 35) 
  (h4 : head_start = 1) : 
  (bob_distance / (total_distance - yolanda_rate * head_start - bob_distance) * yolanda_rate) = 7 :=
sorry

end NUMINAMATH_CALUDE_bobs_walking_rate_l3944_394465


namespace NUMINAMATH_CALUDE_inscribed_cone_volume_ratio_l3944_394477

/-- A right circular cone inscribed in a right prism -/
structure InscribedCone where
  /-- Radius of the cone's base -/
  r : ℝ
  /-- Height of both the cone and the prism -/
  h : ℝ
  /-- The radius and height are positive -/
  r_pos : r > 0
  h_pos : h > 0

/-- Theorem: The ratio of the volume of the inscribed cone to the volume of the prism is π/12 -/
theorem inscribed_cone_volume_ratio (c : InscribedCone) :
  (1 / 3 * π * c.r^2 * c.h) / (4 * c.r^2 * c.h) = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cone_volume_ratio_l3944_394477


namespace NUMINAMATH_CALUDE_base_10_to_base_8_l3944_394473

theorem base_10_to_base_8 : 
  (3 * 8^3 + 1 * 8^2 + 4 * 8^1 + 0 * 8^0 : ℕ) = 1632 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_l3944_394473


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l3944_394459

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l3944_394459


namespace NUMINAMATH_CALUDE_cylinder_and_cone_properties_l3944_394443

/-- Properties of a cylinder and a cone --/
theorem cylinder_and_cone_properties 
  (base_area : ℝ) 
  (height : ℝ) 
  (cylinder_volume : ℝ) 
  (cone_volume : ℝ) 
  (h1 : base_area = 72) 
  (h2 : height = 6) 
  (h3 : cylinder_volume = base_area * height) 
  (h4 : cone_volume = (1/3) * cylinder_volume) : 
  cylinder_volume = 432 ∧ cone_volume = 144 := by
  sorry

#check cylinder_and_cone_properties

end NUMINAMATH_CALUDE_cylinder_and_cone_properties_l3944_394443


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l3944_394418

theorem cubic_roots_sum_of_squares (a b c t : ℝ) : 
  (∀ x, x^3 - 12*x^2 + 20*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 24*t^2 - 16*t = -96 - 8*t := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l3944_394418


namespace NUMINAMATH_CALUDE_product_of_special_numbers_l3944_394411

theorem product_of_special_numbers (m n : ℕ) 
  (h1 : m + n = 20) 
  (h2 : (1 : ℚ) / m + (1 : ℚ) / n = 5 / 24) : 
  m * n = 96 := by
  sorry

end NUMINAMATH_CALUDE_product_of_special_numbers_l3944_394411


namespace NUMINAMATH_CALUDE_cubic_root_polynomial_l3944_394469

theorem cubic_root_polynomial (a b c : ℝ) (P : ℝ → ℝ) : 
  (a^3 + 4*a^2 + 7*a + 10 = 0) →
  (b^3 + 4*b^2 + 7*b + 10 = 0) →
  (c^3 + 4*c^2 + 7*c + 10 = 0) →
  (P a = b + c) →
  (P b = a + c) →
  (P c = a + b) →
  (P (a + b + c) = -22) →
  (∀ x, P x = 8/9*x^3 + 44/9*x^2 + 71/9*x + 2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_polynomial_l3944_394469


namespace NUMINAMATH_CALUDE_correct_robes_count_l3944_394435

/-- The number of robes a school already has for their choir. -/
def robes_already_have (total_singers : ℕ) (robe_cost : ℕ) (total_spend : ℕ) : ℕ :=
  total_singers - (total_spend / robe_cost)

/-- Theorem stating that the number of robes the school already has is correct. -/
theorem correct_robes_count :
  robes_already_have 30 2 36 = 12 := by sorry

end NUMINAMATH_CALUDE_correct_robes_count_l3944_394435


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l3944_394461

/-- The volume of a cube with space diagonal 5√3 is 125 -/
theorem cube_volume_from_diagonal : 
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 5 * Real.sqrt 3 → s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l3944_394461


namespace NUMINAMATH_CALUDE_valid_numbers_l3944_394432

def is_valid_number (n : ℕ) : Prop :=
  80 ≤ n ∧ n < 100 ∧ ∃ x : ℕ, n = 10 * x + (x - 1) ∧ 1 ≤ x ∧ x ≤ 9

theorem valid_numbers : 
  ∀ n : ℕ, is_valid_number n → n = 87 ∨ n = 98 :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l3944_394432


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3944_394446

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation 2x^2 - x - 3 = 0 -/
def f (x : ℝ) : ℝ := 2 * x^2 - x - 3

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l3944_394446


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l3944_394444

theorem painted_cube_theorem (n : ℕ) (h1 : n > 2) :
  (12 * (n - 2) : ℝ) = (n - 2)^3 → n = 2 * Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l3944_394444


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3944_394484

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3944_394484


namespace NUMINAMATH_CALUDE_unique_p_q_for_f_bounded_l3944_394467

def f (p q x : ℝ) := x^2 + p*x + q

theorem unique_p_q_for_f_bounded :
  ∃! p q : ℝ, ∀ x ∈ Set.Icc 1 5, |f p q x| ≤ 2 := by sorry

end NUMINAMATH_CALUDE_unique_p_q_for_f_bounded_l3944_394467


namespace NUMINAMATH_CALUDE_elroy_extra_miles_l3944_394483

/-- Proves that Elroy walks 5 more miles than last year's winner to collect the same amount -/
theorem elroy_extra_miles
  (last_year_rate : ℝ)
  (this_year_rate : ℝ)
  (last_year_amount : ℝ)
  (h1 : last_year_rate = 4)
  (h2 : this_year_rate = 2.75)
  (h3 : last_year_amount = 44) :
  (last_year_amount / this_year_rate) - (last_year_amount / last_year_rate) = 5 := by
sorry

end NUMINAMATH_CALUDE_elroy_extra_miles_l3944_394483


namespace NUMINAMATH_CALUDE_square_roots_sum_l3944_394437

theorem square_roots_sum (x y : ℝ) : 
  x^2 = 16 → y^2 = 9 → x^2 + y^2 + x + 2023 = 2052 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_l3944_394437


namespace NUMINAMATH_CALUDE_sphere_surface_inequality_l3944_394474

theorem sphere_surface_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  (x - y) * (y - z) * (x - z) ≤ 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_inequality_l3944_394474


namespace NUMINAMATH_CALUDE_calculate_expression_l3944_394401

theorem calculate_expression : (30 / (10 - 2 * 3))^2 = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3944_394401


namespace NUMINAMATH_CALUDE_more_numbers_with_one_l3944_394438

def range_upper_bound : ℕ := 10^10

def numbers_without_one (n : ℕ) : ℕ := 9^n - 1

theorem more_numbers_with_one :
  range_upper_bound - numbers_without_one 10 > numbers_without_one 10 := by
  sorry

end NUMINAMATH_CALUDE_more_numbers_with_one_l3944_394438


namespace NUMINAMATH_CALUDE_sqrt_6_over_3_properties_l3944_394403

theorem sqrt_6_over_3_properties : ∃ x : ℝ, x = (Real.sqrt 6) / 3 ∧ 0 < x ∧ x < 1 ∧ Irrational x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_over_3_properties_l3944_394403


namespace NUMINAMATH_CALUDE_exists_universal_transport_l3944_394402

/-- A graph where each pair of vertices is connected by exactly one edge of either type A or type B -/
structure TransportGraph (V : Type) :=
  (edges : V → V → Bool)
  (edge_type : V → V → Bool)
  (connect : ∀ (u v : V), u ≠ v → edges u v = true)
  (unique : ∀ (u v : V), edges u v = edges v u)

/-- A path in the graph with at most two intermediate vertices -/
def ShortPath {V : Type} (g : TransportGraph V) (t : Bool) (u v : V) : Prop :=
  ∃ (w x : V), (g.edges u w ∧ g.edge_type u w = t) ∧ 
               (g.edges w x ∧ g.edge_type w x = t) ∧ 
               (g.edges x v ∧ g.edge_type x v = t)

/-- Main theorem: There exists a transport type that allows short paths between all vertices -/
theorem exists_universal_transport {V : Type} (g : TransportGraph V) :
  ∃ (t : Bool), ∀ (u v : V), u ≠ v → ShortPath g t u v :=
sorry

end NUMINAMATH_CALUDE_exists_universal_transport_l3944_394402


namespace NUMINAMATH_CALUDE_chris_box_percentage_l3944_394407

theorem chris_box_percentage (k c : ℕ) (h : k = 2 * c / 3) : 
  (c : ℚ) / ((k : ℚ) + c) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_chris_box_percentage_l3944_394407


namespace NUMINAMATH_CALUDE_pencil_final_price_l3944_394454

/-- Given a pencil with an original cost and a discount, calculate the final price. -/
theorem pencil_final_price (original_cost discount : ℚ) 
  (h1 : original_cost = 4)
  (h2 : discount = 63 / 100) :
  original_cost - discount = 337 / 100 := by
  sorry

end NUMINAMATH_CALUDE_pencil_final_price_l3944_394454


namespace NUMINAMATH_CALUDE_total_earnings_l3944_394416

def weekly_earnings : ℕ := 16
def harvest_duration : ℕ := 76

theorem total_earnings : weekly_earnings * harvest_duration = 1216 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l3944_394416


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3944_394412

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3944_394412


namespace NUMINAMATH_CALUDE_point_on_line_with_vector_condition_l3944_394464

/-- Given two points in a 2D plane and a third point satisfying certain conditions,
    prove that the third point has specific coordinates. -/
theorem point_on_line_with_vector_condition (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (1, 3) →
  P₂ = (4, -6) →
  (∃ t : ℝ, P = (1 - t) • P₁ + t • P₂) →  -- P is on the line P₁P₂
  (P.1 - P₁.1, P.2 - P₁.2) = 2 • (P₂.1 - P.1, P₂.2 - P.2) →  -- Vector condition
  P = (3, -3) := by
sorry

end NUMINAMATH_CALUDE_point_on_line_with_vector_condition_l3944_394464


namespace NUMINAMATH_CALUDE_curve_fixed_point_l3944_394475

/-- The curve C passes through a fixed point for all k ≠ -1 -/
theorem curve_fixed_point (k : ℝ) (hk : k ≠ -1) :
  ∃ (x y : ℝ), ∀ (k : ℝ), k ≠ -1 →
    x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0 ∧ x = 1 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_curve_fixed_point_l3944_394475


namespace NUMINAMATH_CALUDE_land_area_scientific_notation_l3944_394417

theorem land_area_scientific_notation :
  let land_area : ℝ := 9600000
  9.6 * (10 ^ 6) = land_area := by
  sorry

end NUMINAMATH_CALUDE_land_area_scientific_notation_l3944_394417


namespace NUMINAMATH_CALUDE_exists_x_satisfying_conditions_l3944_394460

theorem exists_x_satisfying_conditions : ∃ x : ℝ,
  ({1, 3, x^2 - 2*x} : Set ℝ) = {1, 3, 0} ∧
  ({1, |2*x - 1|} : Set ℝ) = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_exists_x_satisfying_conditions_l3944_394460


namespace NUMINAMATH_CALUDE_stating_max_tulips_in_bouquet_l3944_394424

/-- Represents the cost of a yellow tulip in rubles -/
def yellow_cost : ℕ := 50

/-- Represents the cost of a red tulip in rubles -/
def red_cost : ℕ := 31

/-- Represents the maximum budget in rubles -/
def max_budget : ℕ := 600

/-- 
Theorem stating that the maximum number of tulips in a bouquet is 15,
given the specified conditions
-/
theorem max_tulips_in_bouquet :
  ∃ (y r : ℕ),
    -- The total number of tulips is odd
    Odd (y + r) ∧
    -- The difference between yellow and red tulips is exactly 1
    (y = r + 1 ∨ r = y + 1) ∧
    -- The total cost does not exceed the budget
    y * yellow_cost + r * red_cost ≤ max_budget ∧
    -- The total number of tulips is 15
    y + r = 15 ∧
    -- This is the maximum possible number of tulips
    ∀ (y' r' : ℕ),
      Odd (y' + r') →
      (y' = r' + 1 ∨ r' = y' + 1) →
      y' * yellow_cost + r' * red_cost ≤ max_budget →
      y' + r' ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_tulips_in_bouquet_l3944_394424


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l3944_394499

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-14/17, 96/17)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 4

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = -7 * x - 2

theorem intersection_point_is_unique :
  (∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l3944_394499


namespace NUMINAMATH_CALUDE_ab_four_necessary_not_sufficient_l3944_394428

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  b : ℝ

/-- The condition that the slopes are equal -/
def slopes_equal (l : TwoLines) : Prop :=
  l.a * l.b = 4

/-- The condition that the lines are parallel -/
def are_parallel (l : TwoLines) : Prop :=
  (2 * l.b = l.a * 2) ∧ ¬(2 * (l.b - 2) = l.a * (-1))

/-- The main theorem: ab = 4 is necessary but not sufficient for parallelism -/
theorem ab_four_necessary_not_sufficient :
  (∀ l : TwoLines, are_parallel l → slopes_equal l) ∧
  ¬(∀ l : TwoLines, slopes_equal l → are_parallel l) :=
sorry

end NUMINAMATH_CALUDE_ab_four_necessary_not_sufficient_l3944_394428


namespace NUMINAMATH_CALUDE_train_passing_bridge_l3944_394427

/-- Time for a train to pass a bridge -/
theorem train_passing_bridge 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 500)
  (h2 : train_speed_kmh = 72)
  (h3 : bridge_length = 200) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_bridge_l3944_394427


namespace NUMINAMATH_CALUDE_map_distance_calculation_l3944_394480

/-- Given a map with a scale of 1:1000000 and two points A and B that are 8 cm apart on the map,
    the actual distance between A and B is 80 km. -/
theorem map_distance_calculation (scale : ℚ) (map_distance : ℚ) (actual_distance : ℚ) :
  scale = 1 / 1000000 →
  map_distance = 8 →
  actual_distance = map_distance / scale →
  actual_distance = 80 * 100000 := by
  sorry


end NUMINAMATH_CALUDE_map_distance_calculation_l3944_394480


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3944_394482

theorem inequality_equivalence (x : ℝ) : (x - 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 5 ∪ {5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3944_394482


namespace NUMINAMATH_CALUDE_interval_preserving_linear_l3944_394450

-- Define the property that f maps intervals to intervals of the same length
def IntervalPreserving (f : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → ∃ c d, c < d ∧ f '' Set.Icc a b = Set.Icc c d ∧ d - c = b - a

-- State the theorem
theorem interval_preserving_linear (f : ℝ → ℝ) (h : IntervalPreserving f) :
  ∃ c : ℝ, (∀ x, f x = x + c) ∨ (∀ x, f x = -x + c) :=
sorry

end NUMINAMATH_CALUDE_interval_preserving_linear_l3944_394450


namespace NUMINAMATH_CALUDE_fruit_cost_price_l3944_394496

/-- Calculates the total cost price of fruits sold given their selling prices, loss ratios, and quantities. -/
def total_cost_price (apple_sp orange_sp banana_sp : ℚ) 
                     (apple_loss orange_loss banana_loss : ℚ) 
                     (apple_qty orange_qty banana_qty : ℕ) : ℚ :=
  let apple_cp := apple_sp / (1 - apple_loss)
  let orange_cp := orange_sp / (1 - orange_loss)
  let banana_cp := banana_sp / (1 - banana_loss)
  apple_cp * apple_qty + orange_cp * orange_qty + banana_cp * banana_qty

/-- The total cost price of fruits sold is 947.45 given the specified conditions. -/
theorem fruit_cost_price : 
  total_cost_price 18 24 12 (1/6) (1/8) (1/4) 10 15 20 = 947.45 := by
  sorry

#eval total_cost_price 18 24 12 (1/6) (1/8) (1/4) 10 15 20

end NUMINAMATH_CALUDE_fruit_cost_price_l3944_394496


namespace NUMINAMATH_CALUDE_problem_statement_l3944_394478

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem problem_statement (f : ℝ → ℝ) (m n : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f m - f n > f (-m) - f (-n)) :
  m - n < 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3944_394478


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3944_394420

theorem sum_of_numbers (a b c : ℝ) : 
  a = 0.8 → b = 1/2 → c = 0.5 → a < 2 ∧ b < 2 ∧ c < 2 → a + b + c = 1.8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3944_394420
