import Mathlib

namespace NUMINAMATH_GPT_complement_event_l1624_162435

def total_students : ℕ := 4
def males : ℕ := 2
def females : ℕ := 2
def choose2 (n : ℕ) := n * (n - 1) / 2

noncomputable def eventA : ℕ := males * females
noncomputable def eventB : ℕ := choose2 males
noncomputable def eventC : ℕ := choose2 females

theorem complement_event {total_students males females : ℕ}
  (h_total : total_students = 4)
  (h_males : males = 2)
  (h_females : females = 2) :
  (total_students.choose 2 - (eventB + eventC)) / total_students.choose 2 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_complement_event_l1624_162435


namespace NUMINAMATH_GPT_ratio_of_areas_l1624_162481

noncomputable def circumferences_equal_arcs (C1 C2 : ℝ) (k1 k2 : ℕ) : Prop :=
  (k1 : ℝ) / 360 * C1 = (k2 : ℝ) / 360 * C2

theorem ratio_of_areas (C1 C2 : ℝ) (h : circumferences_equal_arcs C1 C2 60 30) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1624_162481


namespace NUMINAMATH_GPT_probability_sum_less_than_product_l1624_162446

def set_of_even_integers : Set ℕ := {2, 4, 6, 8, 10}

def sum_less_than_product (a b : ℕ) : Prop :=
  a + b < a * b

theorem probability_sum_less_than_product :
  let total_combinations := 25
  let valid_combinations := 16
  (valid_combinations / total_combinations : ℚ) = 16 / 25 :=
by
  sorry

end NUMINAMATH_GPT_probability_sum_less_than_product_l1624_162446


namespace NUMINAMATH_GPT_cost_to_fill_pool_l1624_162428

-- Definitions based on the conditions
def cubic_foot_to_liters: ℕ := 25
def pool_depth: ℕ := 10
def pool_width: ℕ := 6
def pool_length: ℕ := 20
def cost_per_liter: ℕ := 3

-- Statement to be proved
theorem cost_to_fill_pool : 
  (pool_depth * pool_width * pool_length * cubic_foot_to_liters * cost_per_liter) = 90000 := 
by 
  sorry

end NUMINAMATH_GPT_cost_to_fill_pool_l1624_162428


namespace NUMINAMATH_GPT_max_height_of_table_l1624_162455

theorem max_height_of_table (BC CA AB : ℕ) (h : ℝ) :
  BC = 24 →
  CA = 28 →
  AB = 32 →
  h ≤ (49 * Real.sqrt 60) / 19 :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_height_of_table_l1624_162455


namespace NUMINAMATH_GPT_sum_of_solutions_l1624_162406

theorem sum_of_solutions (x : ℝ) (h1 : x^2 = 25) : ∃ S : ℝ, S = 0 ∧ (∀ x', x'^2 = 25 → x' = 5 ∨ x' = -5) := 
sorry

end NUMINAMATH_GPT_sum_of_solutions_l1624_162406


namespace NUMINAMATH_GPT_find_m_value_l1624_162492

-- Defining the hyperbola equation and the conditions
def hyperbola_eq (x y : ℝ) (m : ℝ) : Prop :=
  (x^2 / m) - (y^2 / 4) = 1

-- Definition of the focal distance
def focal_distance (c : ℝ) :=
  2 * c = 6

-- Definition of the relationship c^2 = a^2 + b^2 for hyperbolas
def hyperbola_focal_distance_eq (m : ℝ) (c b : ℝ) : Prop :=
  c^2 = m + b^2

-- Stating that the hyperbola has the given focal distance
def given_focal_distance : Prop :=
  focal_distance 3

-- Stating the given condition on b²
def given_b_squared : Prop :=
  4 = 4

-- The main theorem stating that m = 5 given the conditions.
theorem find_m_value (m : ℝ) : 
  (hyperbola_eq 1 1 m) → given_focal_distance → given_b_squared → m = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l1624_162492


namespace NUMINAMATH_GPT_Nick_raising_money_l1624_162488

theorem Nick_raising_money :
  let chocolate_oranges := 20
  let oranges_price := 10
  let candy_bars := 160
  let bars_price := 5
  let total_amount := chocolate_oranges * oranges_price + candy_bars * bars_price
  total_amount = 1000 := 
by
  sorry

end NUMINAMATH_GPT_Nick_raising_money_l1624_162488


namespace NUMINAMATH_GPT_radius_of_sphere_in_truncated_cone_l1624_162451

-- Definitions based on conditions
def radius_top_base := 5
def radius_bottom_base := 24

-- Theorem statement (without proof)
theorem radius_of_sphere_in_truncated_cone :
    (∃ (R_s : ℝ),
      (R_s = Real.sqrt 180.5) ∧
      ∀ (h : ℝ),
      (h^2 + (radius_bottom_base - radius_top_base)^2 = (h + R_s)^2 - R_s^2)) :=
sorry

end NUMINAMATH_GPT_radius_of_sphere_in_truncated_cone_l1624_162451


namespace NUMINAMATH_GPT_find_k_l1624_162482

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem find_k (k : ℝ) (h : deriv (f k) 0 = 27) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1624_162482


namespace NUMINAMATH_GPT_find_angle_A_find_area_l1624_162466

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def law_c1 (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + c * Real.cos A = -2 * b * Real.cos A

def law_c2 (a : ℝ) : Prop := a = 2 * Real.sqrt 3
def law_c3 (b c : ℝ) : Prop := b + c = 4

-- Questions
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c) : 
  A = 2 * Real.pi / 3 :=
sorry

theorem find_area (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c)
  (hA : A = 2 * Real.pi / 3) : 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_angle_A_find_area_l1624_162466


namespace NUMINAMATH_GPT_sqrt_fraction_evaluation_l1624_162440

theorem sqrt_fraction_evaluation :
  (Real.sqrt ((2 / 25) + (1 / 49) - (1 / 100)) = 3 / 10) :=
by sorry

end NUMINAMATH_GPT_sqrt_fraction_evaluation_l1624_162440


namespace NUMINAMATH_GPT_find_abcde_l1624_162456

noncomputable def find_five_digit_number (a b c d e : ℕ) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem find_abcde
  (a b c d e : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 0 ≤ e ∧ e ≤ 9)
  (h6 : a ≠ 0)
  (h7 : (10 * a + b + 10 * b + c) * (10 * b + c + 10 * c + d) * (10 * c + d + 10 * d + e) = 157605) :
  find_five_digit_number a b c d e = 12345 ∨ find_five_digit_number a b c d e = 21436 :=
sorry

end NUMINAMATH_GPT_find_abcde_l1624_162456


namespace NUMINAMATH_GPT_largest_square_l1624_162431

def sticks_side1 : List ℕ := [4, 4, 2, 3]
def sticks_side2 : List ℕ := [4, 4, 3, 1, 1]
def sticks_side3 : List ℕ := [4, 3, 3, 2, 1]
def sticks_side4 : List ℕ := [3, 3, 3, 2, 2]

def sum_of_sticks (sticks : List ℕ) : ℕ := sticks.foldl (· + ·) 0

theorem largest_square (h1 : sum_of_sticks sticks_side1 = 13)
                      (h2 : sum_of_sticks sticks_side2 = 13)
                      (h3 : sum_of_sticks sticks_side3 = 13)
                      (h4 : sum_of_sticks sticks_side4 = 13) :
  13 = 13 := by
  sorry

end NUMINAMATH_GPT_largest_square_l1624_162431


namespace NUMINAMATH_GPT_multiply_base5_234_75_l1624_162405

def to_base5 (n : ℕ) : ℕ := 
  let rec helper (n : ℕ) (acc : ℕ) (multiplier : ℕ) : ℕ := 
    if n = 0 then acc
    else
      let d := n % 5
      let q := n / 5
      helper q (acc + d * multiplier) (multiplier * 10)
  helper n 0 1

def base5_multiplication (a b : ℕ) : ℕ :=
  to_base5 ((a * b : ℕ))

theorem multiply_base5_234_75 : base5_multiplication 234 75 = 450620 := 
  sorry

end NUMINAMATH_GPT_multiply_base5_234_75_l1624_162405


namespace NUMINAMATH_GPT_distance_from_Martins_house_to_Lawrences_house_l1624_162439

def speed : ℝ := 2 -- Martin's speed is 2 miles per hour
def time : ℝ := 6  -- Time taken is 6 hours
def distance : ℝ := speed * time -- Distance formula

theorem distance_from_Martins_house_to_Lawrences_house : distance = 12 := by
  sorry

end NUMINAMATH_GPT_distance_from_Martins_house_to_Lawrences_house_l1624_162439


namespace NUMINAMATH_GPT_homothety_transformation_l1624_162424

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V]

/-- Definition of a homothety transformation -/
def homothety (S A A' : V) (k : ℝ) : Prop :=
  A' = k • A + (1 - k) • S

theorem homothety_transformation (S A A' : V) (k : ℝ) :
  homothety S A A' k ↔ A' = k • A + (1 - k) • S := 
by
  sorry

end NUMINAMATH_GPT_homothety_transformation_l1624_162424


namespace NUMINAMATH_GPT_friends_courses_l1624_162426

-- Define the notions of students and their properties
structure Student :=
  (first_name : String)
  (last_name : String)
  (year : ℕ)

-- Define the specific conditions from the problem
def students : List Student := [
  ⟨"Peter", "Krylov", 1⟩,
  ⟨"Nikolay", "Ivanov", 2⟩,
  ⟨"Boris", "Karpov", 3⟩,
  ⟨"Vasily", "Orlov", 4⟩
]

-- The main statement of the problem
theorem friends_courses :
  ∀ (s : Student), s ∈ students →
    (s.first_name = "Peter" → s.last_name = "Krylov" ∧ s.year = 1) ∧
    (s.first_name = "Nikolay" → s.last_name = "Ivanov" ∧ s.year = 2) ∧
    (s.first_name = "Boris" → s.last_name = "Karpov" ∧ s.year = 3) ∧
    (s.first_name = "Vasily" → s.last_name = "Orlov" ∧ s.year = 4) :=
by
  sorry

end NUMINAMATH_GPT_friends_courses_l1624_162426


namespace NUMINAMATH_GPT_ancient_chinese_wine_problem_l1624_162402

theorem ancient_chinese_wine_problem:
  ∃ x: ℝ, 10 * x + 3 * (5 - x) = 30 :=
by
  sorry

end NUMINAMATH_GPT_ancient_chinese_wine_problem_l1624_162402


namespace NUMINAMATH_GPT_maximize_volume_l1624_162453

-- Define the problem-specific constants
def bar_length : ℝ := 0.18
def length_to_width_ratio : ℝ := 2

-- Function to define volume of the rectangle frame
def volume (length width height : ℝ) : ℝ := length * width * height

theorem maximize_volume :
  ∃ (length width height : ℝ), 
  (length / width = length_to_width_ratio) ∧ 
  (2 * (length + width) = bar_length) ∧ 
  ((length = 2) ∧ (height = 1.5)) :=
sorry

end NUMINAMATH_GPT_maximize_volume_l1624_162453


namespace NUMINAMATH_GPT_check_line_properties_l1624_162489

-- Define the conditions
def line_equation (x y : ℝ) : Prop := y + 7 = -x - 3

-- Define the point and slope
def point_and_slope (x y : ℝ) (m : ℝ) : Prop := (x, y) = (-3, -7) ∧ m = -1

-- State the theorem to prove
theorem check_line_properties :
  ∃ x y m, line_equation x y ∧ point_and_slope x y m :=
sorry

end NUMINAMATH_GPT_check_line_properties_l1624_162489


namespace NUMINAMATH_GPT_evaluate_expression_l1624_162400

theorem evaluate_expression :
  (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 = 8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1624_162400


namespace NUMINAMATH_GPT_smores_cost_calculation_l1624_162434

variable (people : ℕ) (s'mores_per_person : ℕ) (s'mores_per_set : ℕ) (cost_per_set : ℕ)

theorem smores_cost_calculation
  (h1 : s'mores_per_person = 3)
  (h2 : people = 8)
  (h3 : s'mores_per_set = 4)
  (h4 : cost_per_set = 3):
  (people * s'mores_per_person / s'mores_per_set) * cost_per_set = 18 := 
by
  sorry

end NUMINAMATH_GPT_smores_cost_calculation_l1624_162434


namespace NUMINAMATH_GPT_quadratic_intersects_x_axis_l1624_162474

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_intersects_x_axis_l1624_162474


namespace NUMINAMATH_GPT_vehicle_worth_l1624_162423

-- Definitions from the conditions
def monthlyEarnings : ℕ := 4000
def savingFraction : ℝ := 0.5
def savingMonths : ℕ := 8

-- Theorem statement
theorem vehicle_worth : (monthlyEarnings * savingFraction * savingMonths : ℝ) = 16000 := 
by
  sorry

end NUMINAMATH_GPT_vehicle_worth_l1624_162423


namespace NUMINAMATH_GPT_baby_polar_bear_playing_hours_l1624_162476

-- Define the conditions
def total_hours_in_a_day : ℕ := 24
def total_central_angle : ℕ := 360
def angle_sleeping : ℕ := 130
def angle_eating : ℕ := 110

-- Main theorem statement
theorem baby_polar_bear_playing_hours :
  let angle_playing := total_central_angle - angle_sleeping - angle_eating
  let fraction_playing := angle_playing / total_central_angle
  let hours_playing := fraction_playing * total_hours_in_a_day
  hours_playing = 8 := by
  sorry

end NUMINAMATH_GPT_baby_polar_bear_playing_hours_l1624_162476


namespace NUMINAMATH_GPT_price_of_pants_l1624_162401

theorem price_of_pants
  (P S H : ℝ)
  (h1 : P + S + H = 340)
  (h2 : S = (3 / 4) * P)
  (h3 : H = P + 10) :
  P = 120 :=
by
  sorry

end NUMINAMATH_GPT_price_of_pants_l1624_162401


namespace NUMINAMATH_GPT_equation_transformation_l1624_162494

theorem equation_transformation (x y: ℝ) (h : 2 * x - 3 * y = 6) : 
  y = (2 * x - 6) / 3 := 
by
  sorry

end NUMINAMATH_GPT_equation_transformation_l1624_162494


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1624_162490

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ):
  (S 4 = S 8 - S 4) →
  (S 4 = S 12 - S 8) →
  (S 4 = S 16 - S 12) →
  S 16 / S 4 = 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1624_162490


namespace NUMINAMATH_GPT_bill_spots_39_l1624_162445

theorem bill_spots_39 (P : ℕ) (h1 : P + (2 * P - 1) = 59) : 2 * P - 1 = 39 :=
by sorry

end NUMINAMATH_GPT_bill_spots_39_l1624_162445


namespace NUMINAMATH_GPT_Clea_escalator_time_l1624_162470

variable {s : ℝ} -- speed of the escalator at its normal operating speed
variable {c : ℝ} -- speed of Clea walking down the escalator
variable {d : ℝ} -- distance of the escalator

theorem Clea_escalator_time :
  (30 * (c + s) = 72 * c) →
  (s = (7 * c) / 5) →
  (t = (72 * c) / ((3 / 2) * s)) →
  t = 240 / 7 :=
by
  sorry

end NUMINAMATH_GPT_Clea_escalator_time_l1624_162470


namespace NUMINAMATH_GPT_prove_ratio_l1624_162441

noncomputable def box_dimensions : ℝ × ℝ × ℝ := (2, 3, 5)
noncomputable def d := (2 * 3 * 5 : ℝ)
noncomputable def a := ((4 * Real.pi) / 3 : ℝ)
noncomputable def b := (10 * Real.pi : ℝ)
noncomputable def c := (62 : ℝ)

theorem prove_ratio :
  (b * c) / (a * d) = (15.5 : ℝ) :=
by
  unfold a b c d
  sorry

end NUMINAMATH_GPT_prove_ratio_l1624_162441


namespace NUMINAMATH_GPT_geometric_sequence_15th_term_l1624_162487

theorem geometric_sequence_15th_term :
  let a_1 := 27
  let r := (1 : ℚ) / 6
  let a_15 := a_1 * r ^ 14
  a_15 = 1 / 14155776 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_15th_term_l1624_162487


namespace NUMINAMATH_GPT_total_sandwiches_l1624_162421

theorem total_sandwiches :
  let billy := 49
  let katelyn := billy + 47
  let chloe := katelyn / 4
  billy + katelyn + chloe = 169 :=
by
  sorry

end NUMINAMATH_GPT_total_sandwiches_l1624_162421


namespace NUMINAMATH_GPT_tan_sum_l1624_162480

theorem tan_sum (A B : ℝ) (h₁ : A = 17) (h₂ : B = 28) :
  Real.tan (A) + Real.tan (B) + Real.tan (A) * Real.tan (B) = 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_sum_l1624_162480


namespace NUMINAMATH_GPT_p_at_5_l1624_162416

noncomputable def p (x : ℝ) : ℝ :=
  sorry

def p_cond (n : ℝ) : Prop :=
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → p n = 1 / n^3

theorem p_at_5 : (∀ n, p_cond n) → p 5 = -149 / 1500 :=
by
  intros
  sorry

end NUMINAMATH_GPT_p_at_5_l1624_162416


namespace NUMINAMATH_GPT_card_subsets_l1624_162442

theorem card_subsets (A : Finset ℕ) (hA_card : A.card = 3) : (A.powerset.card = 8) :=
sorry

end NUMINAMATH_GPT_card_subsets_l1624_162442


namespace NUMINAMATH_GPT_abs_sum_neq_zero_iff_or_neq_zero_l1624_162496

variable {x y : ℝ}

theorem abs_sum_neq_zero_iff_or_neq_zero (x y : ℝ) :
  (|x| + |y| ≠ 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_GPT_abs_sum_neq_zero_iff_or_neq_zero_l1624_162496


namespace NUMINAMATH_GPT_parallelepiped_volume_l1624_162475

open Real

noncomputable def volume_parallelepiped
  (a b : ℝ) (angle : ℝ) (S : ℝ) (sin_30 : angle = π / 6) : ℝ :=
  let h := S / (2 * (a + b))
  let base_area := (a * b * sin (π / 6)) / 2
  base_area * h

theorem parallelepiped_volume 
  (a b : ℝ) (S : ℝ) (h : S ≠ 0 ∧ a > 0 ∧ b > 0) :
  volume_parallelepiped a b (π / 6) S (rfl) = (a * b * S) / (4 * (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_parallelepiped_volume_l1624_162475


namespace NUMINAMATH_GPT_PeytonManning_total_distance_l1624_162404

noncomputable def PeytonManning_threw_distance : Prop :=
  let throw_distance_50 := 20
  let throw_times_sat := 20
  let throw_times_sun := 30
  let total_distance := 1600
  ∃ R : ℚ, 
    let throw_distance_80 := R * throw_distance_50
    let distance_sat := throw_distance_50 * throw_times_sat
    let distance_sun := throw_distance_80 * throw_times_sun
    distance_sat + distance_sun = total_distance

theorem PeytonManning_total_distance :
  PeytonManning_threw_distance := by
  sorry

end NUMINAMATH_GPT_PeytonManning_total_distance_l1624_162404


namespace NUMINAMATH_GPT_weight_of_lightest_dwarf_l1624_162469

noncomputable def weight_of_dwarf (n : ℕ) (x : ℝ) : ℝ := 5 - (n - 1) * x

theorem weight_of_lightest_dwarf :
  ∃ x : ℝ, 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 101 → weight_of_dwarf 1 x = 5) ∧
    (weight_of_dwarf 76 x + weight_of_dwarf 77 x + weight_of_dwarf 78 x + weight_of_dwarf 79 x + weight_of_dwarf 80 x =
     weight_of_dwarf 96 x + weight_of_dwarf 97 x + weight_of_dwarf 98 x + weight_of_dwarf 99 x + weight_of_dwarf 100 x + weight_of_dwarf 101 x) →
    weight_of_dwarf 101 x = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_lightest_dwarf_l1624_162469


namespace NUMINAMATH_GPT_divisible_bc_ad_l1624_162425

open Int

theorem divisible_bc_ad (a b c d m : ℤ) (hm : 0 < m)
  (h1 : m ∣ a * c)
  (h2 : m ∣ b * d)
  (h3 : m ∣ (b * c + a * d)) :
  m ∣ b * c ∧ m ∣ a * d :=
by
  sorry

end NUMINAMATH_GPT_divisible_bc_ad_l1624_162425


namespace NUMINAMATH_GPT_initial_black_beads_l1624_162432

theorem initial_black_beads (B : ℕ) : 
  let white_beads := 51
  let black_beads_removed := 1 / 6 * B
  let white_beads_removed := 1 / 3 * white_beads
  let total_beads_removed := 32
  white_beads_removed + black_beads_removed = total_beads_removed →
  B = 90 :=
by
  sorry

end NUMINAMATH_GPT_initial_black_beads_l1624_162432


namespace NUMINAMATH_GPT_rectangle_width_is_14_l1624_162433

noncomputable def rectangleWidth (areaOfCircle : ℝ) (length : ℝ) : ℝ :=
  let r := Real.sqrt (areaOfCircle / Real.pi)
  2 * r

theorem rectangle_width_is_14 :
  rectangleWidth 153.93804002589985 18 = 14 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_width_is_14_l1624_162433


namespace NUMINAMATH_GPT_Roe_saved_15_per_month_aug_nov_l1624_162430

-- Step 1: Define the given conditions
def savings_per_month_jan_jul : ℕ := 10
def months_jan_jul : ℕ := 7
def savings_dec : ℕ := 20
def total_savings_needed : ℕ := 150
def months_aug_nov : ℕ := 4

-- Step 2: Define the intermediary calculations based on the conditions
def total_saved_jan_jul := savings_per_month_jan_jul * months_jan_jul
def total_savings_aug_nov := total_savings_needed - total_saved_jan_jul - savings_dec

-- Step 3: Define what we need to prove
def savings_per_month_aug_nov : ℕ := total_savings_aug_nov / months_aug_nov

-- Step 4: State the proof goal
theorem Roe_saved_15_per_month_aug_nov :
  savings_per_month_aug_nov = 15 :=
by
  sorry

end NUMINAMATH_GPT_Roe_saved_15_per_month_aug_nov_l1624_162430


namespace NUMINAMATH_GPT_base_prime_representation_450_l1624_162477

-- Define prime factorization property for number 450
def prime_factorization_450 := (450 = 2^1 * 3^2 * 5^2)

-- Define base prime representation concept
def base_prime_representation (n : ℕ) : ℕ := 
  if n = 450 then 122 else 0

-- Prove that the base prime representation of 450 is 122
theorem base_prime_representation_450 : 
  prime_factorization_450 →
  base_prime_representation 450 = 122 :=
by
  intros
  sorry

end NUMINAMATH_GPT_base_prime_representation_450_l1624_162477


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l1624_162497

-- Define the quadratic function
def quadratic (x : ℝ) (k : ℝ) : ℝ :=
  -(x - 2) ^ 2 + k

-- Define the points A, B, and C
def A (y1 k : ℝ) := ∃ y1, quadratic (-1 / 2) k = y1
def B (y2 k : ℝ) := ∃ y2, quadratic (1) k = y2
def C (y3 k : ℝ) := ∃ y3, quadratic (4) k = y3

theorem relationship_y1_y2_y3 (y1 y2 y3 k: ℝ)
  (hA : A y1 k)
  (hB : B y2 k)
  (hC : C y3 k) :
  y1 < y3 ∧ y3 < y2 :=
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l1624_162497


namespace NUMINAMATH_GPT_largest_divisor_of_square_divisible_by_24_l1624_162444

theorem largest_divisor_of_square_divisible_by_24 (n : ℕ) (h₁ : n > 0) (h₂ : 24 ∣ n^2) (h₃ : ∀ k : ℕ, k ∣ n → k ≤ 8) : n = 24 := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_square_divisible_by_24_l1624_162444


namespace NUMINAMATH_GPT_FDI_in_rural_AndhraPradesh_l1624_162422

-- Definitions from conditions
def total_FDI : ℝ := 300 -- Total FDI calculated
def FDI_Gujarat : ℝ := 0.30 * total_FDI
def FDI_Gujarat_Urban : ℝ := 0.80 * FDI_Gujarat
def FDI_AndhraPradesh : ℝ := 0.20 * total_FDI
def FDI_AndhraPradesh_Rural : ℝ := 0.50 * FDI_AndhraPradesh 

-- Given the conditions, prove the size of FDI in rural Andhra Pradesh is 30 million
theorem FDI_in_rural_AndhraPradesh :
  FDI_Gujarat_Urban = 72 → FDI_AndhraPradesh_Rural = 30 :=
by
  sorry

end NUMINAMATH_GPT_FDI_in_rural_AndhraPradesh_l1624_162422


namespace NUMINAMATH_GPT_proof_of_problem_l1624_162462

theorem proof_of_problem (a b : ℝ) (h1 : a > b) (h2 : a * b = a / b) : b = 1 ∧ 0 < a :=
by
  sorry

end NUMINAMATH_GPT_proof_of_problem_l1624_162462


namespace NUMINAMATH_GPT_bottles_per_day_l1624_162493

theorem bottles_per_day (b d : ℕ) (h1 : b = 8066) (h2 : d = 74) : b / d = 109 :=
by {
  sorry
}

end NUMINAMATH_GPT_bottles_per_day_l1624_162493


namespace NUMINAMATH_GPT_base_of_hill_depth_l1624_162460

theorem base_of_hill_depth : 
  ∀ (H : ℕ), 
  (H = 900) → 
  (1 / 4 * H = 225) :=
by
  intros H h
  sorry

end NUMINAMATH_GPT_base_of_hill_depth_l1624_162460


namespace NUMINAMATH_GPT_sum_of_three_numbers_is_520_l1624_162478

noncomputable def sum_of_three_numbers (x y z : ℝ) : ℝ :=
  x + y + z

theorem sum_of_three_numbers_is_520 (x y z : ℝ) (h1 : z = (1848 / 1540) * x) (h2 : z = 0.4 * y) (h3 : x + y = 400) :
  sum_of_three_numbers x y z = 520 :=
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_is_520_l1624_162478


namespace NUMINAMATH_GPT_segment_length_l1624_162458

theorem segment_length (AB BC AC : ℝ) (hAB : AB = 4) (hBC : BC = 3) :
  AC = 7 ∨ AC = 1 :=
sorry

end NUMINAMATH_GPT_segment_length_l1624_162458


namespace NUMINAMATH_GPT_no_valid_middle_number_l1624_162499

theorem no_valid_middle_number
    (x : ℤ)
    (h1 : (x % 2 = 1))
    (h2 : 3 * x + 12 = x^2 + 20) :
    false :=
by
    sorry

end NUMINAMATH_GPT_no_valid_middle_number_l1624_162499


namespace NUMINAMATH_GPT_staff_member_pays_l1624_162465

noncomputable def calculate_final_price (d : ℝ) : ℝ :=
  let discounted_price := 0.55 * d
  let staff_discounted_price := 0.33 * d
  let final_price := staff_discounted_price + 0.08 * staff_discounted_price
  final_price

theorem staff_member_pays (d : ℝ) : calculate_final_price d = 0.3564 * d :=
by
  unfold calculate_final_price
  sorry

end NUMINAMATH_GPT_staff_member_pays_l1624_162465


namespace NUMINAMATH_GPT_cost_price_of_table_l1624_162420

theorem cost_price_of_table (CP SP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3000) : CP = 2500 := by
    sorry

end NUMINAMATH_GPT_cost_price_of_table_l1624_162420


namespace NUMINAMATH_GPT_tan_beta_is_six_over_seventeen_l1624_162413
-- Import the Mathlib library

-- Define the problem in Lean
theorem tan_beta_is_six_over_seventeen
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 4 / 5)
  (h2 : Real.tan (α - β) = 2 / 3) :
  Real.tan β = 6 / 17 := 
by
  sorry

end NUMINAMATH_GPT_tan_beta_is_six_over_seventeen_l1624_162413


namespace NUMINAMATH_GPT_mean_exterior_angles_l1624_162415

theorem mean_exterior_angles (a b c : ℝ) (ha : a = 45) (hb : b = 75) (hc : c = 60) :
  (180 - a + 180 - b + 180 - c) / 3 = 120 :=
by 
  sorry

end NUMINAMATH_GPT_mean_exterior_angles_l1624_162415


namespace NUMINAMATH_GPT_ratio_area_triangle_circle_l1624_162429

open Real

theorem ratio_area_triangle_circle
  (l r : ℝ)
  (h : ℝ := sqrt 2 * l)
  (h_eq_perimeter : 2 * l + h = 2 * π * r) :
  (1 / 2 * l^2) / (π * r^2) = (π * (3 - 2 * sqrt 2)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_area_triangle_circle_l1624_162429


namespace NUMINAMATH_GPT_income_on_fifth_day_l1624_162411

-- Define the incomes for the first four days
def income_day1 := 600
def income_day2 := 250
def income_day3 := 450
def income_day4 := 400

-- Define the average income
def average_income := 500

-- Define the length of days
def days := 5

-- Define the total income for the 5 days
def total_income : ℕ := days * average_income

-- Define the total income for the first 4 days
def total_income_first4 := income_day1 + income_day2 + income_day3 + income_day4

-- Define the income on the fifth day
def income_day5 := total_income - total_income_first4

-- The theorem to prove the income of the fifth day is $800
theorem income_on_fifth_day : income_day5 = 800 := by
  -- proof is not required, so we leave the proof section with sorry
  sorry

end NUMINAMATH_GPT_income_on_fifth_day_l1624_162411


namespace NUMINAMATH_GPT_sqrt_equality_l1624_162427

theorem sqrt_equality :
  Real.sqrt ((18: ℝ) * (17: ℝ) * (16: ℝ) * (15: ℝ) + 1) = 271 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_equality_l1624_162427


namespace NUMINAMATH_GPT_catriona_total_fish_eq_44_l1624_162409

-- Definitions based on conditions
def goldfish : ℕ := 8
def angelfish : ℕ := goldfish + 4
def guppies : ℕ := 2 * angelfish
def total_fish : ℕ := goldfish + angelfish + guppies

-- The theorem we need to prove
theorem catriona_total_fish_eq_44 : total_fish = 44 :=
by
  -- We are skipping the proof steps with 'sorry' for now
  sorry

end NUMINAMATH_GPT_catriona_total_fish_eq_44_l1624_162409


namespace NUMINAMATH_GPT_runs_twice_l1624_162412

-- Definitions of the conditions
def game_count : ℕ := 6
def runs_one : ℕ := 1
def runs_five : ℕ := 5
def average_runs : ℕ := 4

-- Assuming the number of runs scored twice is x
variable (x : ℕ)

-- Definition of total runs scored based on the conditions
def total_runs : ℕ := runs_one + 2 * x + 3 * runs_five

-- Statement to prove the number of runs scored twice
theorem runs_twice :
  (total_runs x) / game_count = average_runs → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_runs_twice_l1624_162412


namespace NUMINAMATH_GPT_sum_of_squares_l1624_162437

theorem sum_of_squares (n : Nat) (h : n = 2005^2) : 
  ∃ a1 b1 a2 b2 a3 b3 a4 b4 : Int, 
    (n = a1^2 + b1^2 ∧ a1 ≠ 0 ∧ b1 ≠ 0) ∧ 
    (n = a2^2 + b2^2 ∧ a2 ≠ 0 ∧ b2 ≠ 0) ∧ 
    (n = a3^2 + b3^2 ∧ a3 ≠ 0 ∧ b3 ≠ 0) ∧ 
    (n = a4^2 + b4^2 ∧ a4 ≠ 0 ∧ b4 ≠ 0) ∧ 
    (a1, b1) ≠ (a2, b2) ∧ 
    (a1, b1) ≠ (a3, b3) ∧ 
    (a1, b1) ≠ (a4, b4) ∧ 
    (a2, b2) ≠ (a3, b3) ∧ 
    (a2, b2) ≠ (a4, b4) ∧ 
    (a3, b3) ≠ (a4, b4) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1624_162437


namespace NUMINAMATH_GPT_multiply_divide_repeating_decimals_l1624_162414

theorem multiply_divide_repeating_decimals :
  (8 * (1 / 3) / 1) = 8 / 3 := by
  sorry

end NUMINAMATH_GPT_multiply_divide_repeating_decimals_l1624_162414


namespace NUMINAMATH_GPT_min_living_allowance_inequality_l1624_162463

variable (x : ℝ)

-- The regulation stipulates that the minimum living allowance should not be less than 300 yuan.
def min_living_allowance_regulation (x : ℝ) : Prop := x >= 300

theorem min_living_allowance_inequality (x : ℝ) :
  min_living_allowance_regulation x ↔ x ≥ 300 := by
  sorry

end NUMINAMATH_GPT_min_living_allowance_inequality_l1624_162463


namespace NUMINAMATH_GPT_range_of_solutions_l1624_162417

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end NUMINAMATH_GPT_range_of_solutions_l1624_162417


namespace NUMINAMATH_GPT_pipes_fill_time_l1624_162485

noncomputable def filling_time (P X Y Z : ℝ) : ℝ :=
  P / (X + Y + Z)

theorem pipes_fill_time (P : ℝ) (X Y Z : ℝ)
  (h1 : X + Y = P / 3) 
  (h2 : X + Z = P / 6) 
  (h3 : Y + Z = P / 4.5) :
  filling_time P X Y Z = 36 / 13 := by
  sorry

end NUMINAMATH_GPT_pipes_fill_time_l1624_162485


namespace NUMINAMATH_GPT_laptop_selection_l1624_162449

open Nat

theorem laptop_selection :
  ∃ (n : ℕ), n = (choose 4 2) * (choose 5 1) + (choose 4 1) * (choose 5 2) := 
sorry

end NUMINAMATH_GPT_laptop_selection_l1624_162449


namespace NUMINAMATH_GPT_chocolates_bought_l1624_162471

theorem chocolates_bought (C S : ℝ) (h1 : N * C = 45 * S) (h2 : 80 = ((S - C) / C) * 100) : 
  N = 81 :=
by
  sorry

end NUMINAMATH_GPT_chocolates_bought_l1624_162471


namespace NUMINAMATH_GPT_windows_per_floor_is_3_l1624_162467

-- Given conditions
variables (W : ℕ)
def windows_each_floor (W : ℕ) : Prop :=
  (3 * 2 * W) - 2 = 16

-- Correct answer
theorem windows_per_floor_is_3 : windows_each_floor 3 :=
by 
  sorry

end NUMINAMATH_GPT_windows_per_floor_is_3_l1624_162467


namespace NUMINAMATH_GPT_domain_real_iff_l1624_162484

noncomputable def is_domain_ℝ (m : ℝ) : Prop :=
  ∀ x : ℝ, (m * x^2 + 4 * m * x + 3 ≠ 0)

theorem domain_real_iff (m : ℝ) :
  is_domain_ℝ m ↔ (0 ≤ m ∧ m < 3 / 4) :=
sorry

end NUMINAMATH_GPT_domain_real_iff_l1624_162484


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l1624_162495

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l1624_162495


namespace NUMINAMATH_GPT_chosen_number_l1624_162473

theorem chosen_number (x : ℕ) (h : (x / 12) - 240 = 8) : x = 2976 :=
sorry

end NUMINAMATH_GPT_chosen_number_l1624_162473


namespace NUMINAMATH_GPT_team_a_wins_3_2_prob_l1624_162464

-- Definitions for the conditions in the problem
def prob_win_first_four : ℚ := 2 / 3
def prob_win_fifth : ℚ := 1 / 2

-- Definitions related to combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Main statement: Proving the probability of winning 3:2
theorem team_a_wins_3_2_prob :
  (C 4 2 * (prob_win_first_four ^ 2) * ((1 - prob_win_first_four) ^ 2) * prob_win_fifth) = 4 / 27 := 
sorry

end NUMINAMATH_GPT_team_a_wins_3_2_prob_l1624_162464


namespace NUMINAMATH_GPT_g_half_eq_neg_one_l1624_162436

noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem g_half_eq_neg_one : g (1/2) = -1 := by 
  sorry

end NUMINAMATH_GPT_g_half_eq_neg_one_l1624_162436


namespace NUMINAMATH_GPT_simplify_fraction_l1624_162486

theorem simplify_fraction (a b : ℕ) (h : a ≠ b) : (2 * a) / (2 * b) = a / b :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l1624_162486


namespace NUMINAMATH_GPT_range_of_a_l1624_162483

def is_in_third_quadrant (A : ℝ × ℝ) : Prop :=
  A.1 < 0 ∧ A.2 < 0

theorem range_of_a (a : ℝ) (h : is_in_third_quadrant (a, a - 1)) : a < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1624_162483


namespace NUMINAMATH_GPT_circle_line_no_intersection_l1624_162450

theorem circle_line_no_intersection (b : ℝ) :
  (∀ x y : ℝ, ¬ (x^2 + y^2 = 2 ∧ y = x + b)) ↔ (b > 2 ∨ b < -2) :=
by sorry

end NUMINAMATH_GPT_circle_line_no_intersection_l1624_162450


namespace NUMINAMATH_GPT_probability_all_same_color_l1624_162461

def total_marbles := 15
def red_marbles := 4
def white_marbles := 5
def blue_marbles := 6

def prob_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
def prob_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
def prob_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))

def prob_all_same_color := prob_all_red + prob_all_white + prob_all_blue

theorem probability_all_same_color :
  prob_all_same_color = (34/455) :=
by sorry

end NUMINAMATH_GPT_probability_all_same_color_l1624_162461


namespace NUMINAMATH_GPT_binom_identity_l1624_162454

-- Definition: Combinatorial coefficient (binomial coefficient)
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) (h : k ≤ n) :
  binom (n + 1) k = binom n k + binom n (k - 1) := by
  sorry

end NUMINAMATH_GPT_binom_identity_l1624_162454


namespace NUMINAMATH_GPT_negation_of_tan_one_l1624_162443

theorem negation_of_tan_one :
  (∃ x : ℝ, Real.tan x = 1) ↔ ¬ (∀ x : ℝ, Real.tan x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_tan_one_l1624_162443


namespace NUMINAMATH_GPT_boiling_point_of_water_l1624_162459

theorem boiling_point_of_water :
  (boiling_point_F : ℝ) = 212 →
  (boiling_point_C : ℝ) = (5 / 9) * (boiling_point_F - 32) →
  boiling_point_C = 100 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_boiling_point_of_water_l1624_162459


namespace NUMINAMATH_GPT_cost_of_socks_l1624_162438

theorem cost_of_socks (S : ℝ) (players : ℕ) (jersey : ℝ) (shorts : ℝ) 
                      (total_cost : ℝ) 
                      (h1 : players = 16) 
                      (h2 : jersey = 25) 
                      (h3 : shorts = 15.20) 
                      (h4 : total_cost = 752) 
                      (h5 : total_cost = players * (jersey + shorts + S)) 
                      : S = 6.80 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_socks_l1624_162438


namespace NUMINAMATH_GPT_teamA_worked_days_l1624_162479

def teamA_days_to_complete := 10
def teamB_days_to_complete := 15
def teamC_days_to_complete := 20
def total_days := 6
def teamA_halfway_withdrew := true

theorem teamA_worked_days : 
  ∀ (T_A T_B T_C total: ℕ) (halfway_withdrawal: Bool),
    T_A = teamA_days_to_complete ->
    T_B = teamB_days_to_complete ->
    T_C = teamC_days_to_complete ->
    total = total_days ->
    halfway_withdrawal = teamA_halfway_withdrew ->
    (total / 2) * (1 / T_A + 1 / T_B + 1 / T_C) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_teamA_worked_days_l1624_162479


namespace NUMINAMATH_GPT_scientific_notation_819000_l1624_162468

theorem scientific_notation_819000 :
  ∃ (a : ℝ) (n : ℤ), 819000 = a * 10 ^ n ∧ a = 8.19 ∧ n = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_scientific_notation_819000_l1624_162468


namespace NUMINAMATH_GPT_population_sampling_precision_l1624_162403

theorem population_sampling_precision (sample_size : ℕ → Prop) 
    (A : Prop) (B : Prop) (C : Prop) (D : Prop)
    (condition_A : A = (∀ n : ℕ, sample_size n → false))
    (condition_B : B = (∀ n : ℕ, sample_size n → n > 0 → true))
    (condition_C : C = (∀ n : ℕ, sample_size n → false))
    (condition_D : D = (∀ n : ℕ, sample_size n → false)) :
  B :=
by sorry

end NUMINAMATH_GPT_population_sampling_precision_l1624_162403


namespace NUMINAMATH_GPT_sum_of_a_c_l1624_162410

theorem sum_of_a_c (a b c d : ℝ) (h1 : -2 * abs (1 - a) + b = 7) (h2 : 2 * abs (1 - c) + d = 7)
    (h3 : -2 * abs (11 - a) + b = -1) (h4 : 2 * abs (11 - c) + d = -1) : a + c = 12 := by
  -- Definitions for conditions
  -- h1: intersection at (1, 7) for first graph
  -- h2: intersection at (1, 7) for second graph
  -- h3: intersection at (11, -1) for first graph
  -- h4: intersection at (11, -1) for second graph
  sorry

end NUMINAMATH_GPT_sum_of_a_c_l1624_162410


namespace NUMINAMATH_GPT_subtract_from_40_squared_l1624_162452

theorem subtract_from_40_squared : 39 * 39 = 40 * 40 - 79 := by
  sorry

end NUMINAMATH_GPT_subtract_from_40_squared_l1624_162452


namespace NUMINAMATH_GPT_polynomial_divisibility_by_6_l1624_162447

theorem polynomial_divisibility_by_6 (a b c : ℤ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_by_6_l1624_162447


namespace NUMINAMATH_GPT_systematic_sampling_interval_l1624_162472

-- Define the total number of students and sample size
def N : ℕ := 1200
def n : ℕ := 40

-- Define the interval calculation for systematic sampling
def k : ℕ := N / n

-- Prove that the interval k is 30
theorem systematic_sampling_interval : k = 30 := by
sorry

end NUMINAMATH_GPT_systematic_sampling_interval_l1624_162472


namespace NUMINAMATH_GPT_no_integer_solutions_l1624_162491

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1624_162491


namespace NUMINAMATH_GPT_walther_janous_inequality_equality_condition_l1624_162419

theorem walther_janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * y ∧ x / y = 2 ∧ y = z :=
sorry

end NUMINAMATH_GPT_walther_janous_inequality_equality_condition_l1624_162419


namespace NUMINAMATH_GPT_soybeans_in_jar_l1624_162407

theorem soybeans_in_jar
  (totalRedBeans : ℕ)
  (sampleSize : ℕ)
  (sampleRedBeans : ℕ)
  (totalBeans : ℕ)
  (proportion : sampleRedBeans / sampleSize = totalRedBeans / totalBeans)
  (h1 : totalRedBeans = 200)
  (h2 : sampleSize = 60)
  (h3 : sampleRedBeans = 5) :
  totalBeans = 2400 :=
by
  sorry

end NUMINAMATH_GPT_soybeans_in_jar_l1624_162407


namespace NUMINAMATH_GPT_parabola_y1_gt_y2_l1624_162498

variable {x1 x2 y1 y2 : ℝ}

theorem parabola_y1_gt_y2 
  (hx1 : -4 < x1 ∧ x1 < -2) 
  (hx2 : 0 < x2 ∧ x2 < 2) 
  (hy1 : y1 = x1^2) 
  (hy2 : y2 = x2^2) : 
  y1 > y2 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_y1_gt_y2_l1624_162498


namespace NUMINAMATH_GPT_amelia_jet_bars_l1624_162408

theorem amelia_jet_bars
    (required : ℕ) (sold_monday : ℕ) (sold_tuesday_less : ℕ) (total_sold : ℕ) (remaining : ℕ) :
    required = 90 →
    sold_monday = 45 →
    sold_tuesday_less = 16 →
    total_sold = sold_monday + (sold_monday - sold_tuesday_less) →
    remaining = required - total_sold →
    remaining = 16 :=
by
  intros
  sorry

end NUMINAMATH_GPT_amelia_jet_bars_l1624_162408


namespace NUMINAMATH_GPT_angle_between_diagonals_l1624_162448

variables (α β : ℝ) 

theorem angle_between_diagonals (α β : ℝ) :
  ∃ γ : ℝ, γ = Real.arccos (Real.sin α * Real.sin β) :=
by
  sorry

end NUMINAMATH_GPT_angle_between_diagonals_l1624_162448


namespace NUMINAMATH_GPT_power_sum_evaluation_l1624_162418

theorem power_sum_evaluation :
  (-1)^(4^3) + 2^(3^2) = 513 :=
by
  sorry

end NUMINAMATH_GPT_power_sum_evaluation_l1624_162418


namespace NUMINAMATH_GPT_relationship_between_A_B_C_l1624_162457

-- Definitions based on the problem conditions
def A : Set ℝ := {θ | ∃ k : ℤ, 2 * k * Real.pi < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2}
def B : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def C : Set ℝ := {θ | θ < Real.pi / 2}

-- Proof statement: Prove the specified relationship
theorem relationship_between_A_B_C : B ∪ C = C := by
  sorry

end NUMINAMATH_GPT_relationship_between_A_B_C_l1624_162457
