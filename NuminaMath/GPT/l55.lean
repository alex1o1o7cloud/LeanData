import Mathlib

namespace train_crossing_platform_time_l55_5520

theorem train_crossing_platform_time (train_length : ℝ) (platform_length : ℝ) (time_cross_post : ℝ) :
  train_length = 300 → platform_length = 350 → time_cross_post = 18 → 
  (train_length + platform_length) / (train_length / time_cross_post) = 39 :=
by
  intros
  sorry

end train_crossing_platform_time_l55_5520


namespace intersection_M_N_l55_5501

def M := { x : ℝ | |x| ≤ 1 }
def N := { x : ℝ | x^2 - x < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l55_5501


namespace anne_distance_l55_5504
  
theorem anne_distance (S T : ℕ) (H1 : S = 2) (H2 : T = 3) : S * T = 6 := by
  -- Given that speed S = 2 miles/hour and time T = 3 hours, we need to show the distance S * T = 6 miles.
  sorry

end anne_distance_l55_5504


namespace card_draw_probability_l55_5514

theorem card_draw_probability :
  (13 / 52) * (13 / 51) * (13 / 50) = 2197 / 132600 :=
by
  sorry

end card_draw_probability_l55_5514


namespace proportional_function_ratio_l55_5509

-- Let k be a constant, and y = kx be a proportional function.
-- We know that f(1) = 3 and f(a) = b where b ≠ 0.
-- We want to prove that a / b = 1 / 3.

theorem proportional_function_ratio (a b k : ℝ) :
  (∀ x, x = 1 → k * x = 3) →
  (∀ x, x = a → k * x = b) →
  b ≠ 0 →
  a / b = 1 / 3 :=
by
  intros h1 h2 h3
  -- the proof will follow but is not required here
  sorry

end proportional_function_ratio_l55_5509


namespace smallest_rational_number_l55_5580

theorem smallest_rational_number : ∀ (a b c d : ℚ), (a = -3) → (b = -1) → (c = 0) → (d = 1) → (a < b ∧ a < c ∧ a < d) :=
by
  intros a b c d h₁ h₂ h₃ h₄
  have h₅ : a = -3 := h₁
  have h₆ : b = -1 := h₂
  have h₇ : c = 0 := h₃
  have h₈ : d = 1 := h₄
  sorry

end smallest_rational_number_l55_5580


namespace jack_has_42_pounds_l55_5566

noncomputable def jack_pounds (P : ℕ) : Prop :=
  let euros := 11
  let yen := 3000
  let pounds_per_euro := 2
  let yen_per_pound := 100
  let total_yen := 9400
  let pounds_from_euros := euros * pounds_per_euro
  let pounds_from_yen := yen / yen_per_pound
  let total_pounds := P + pounds_from_euros + pounds_from_yen
  total_pounds * yen_per_pound = total_yen

theorem jack_has_42_pounds : jack_pounds 42 :=
  sorry

end jack_has_42_pounds_l55_5566


namespace smallest_n_for_terminating_decimal_l55_5572

theorem smallest_n_for_terminating_decimal : 
  ∃ n : ℕ, (0 < n) ∧ (∃ k m : ℕ, (n + 70 = 2 ^ k * 5 ^ m) ∧ k = 0 ∨ k = 1) ∧ n = 55 :=
by sorry

end smallest_n_for_terminating_decimal_l55_5572


namespace price_of_each_shirt_l55_5531

-- Defining the conditions
def total_pants_cost (pants_price : ℕ) (num_pants : ℕ) := num_pants * pants_price
def total_amount_spent (amount_given : ℕ) (change_received : ℕ) := amount_given - change_received
def total_shirts_cost (amount_spent : ℕ) (pants_cost : ℕ) := amount_spent - pants_cost
def price_per_shirt (shirts_total_cost : ℕ) (num_shirts : ℕ) := shirts_total_cost / num_shirts

-- The main statement
theorem price_of_each_shirt (pants_price num_pants amount_given change_received num_shirts : ℕ) :
  num_pants = 2 →
  pants_price = 54 →
  amount_given = 250 →
  change_received = 10 →
  num_shirts = 4 →
  price_per_shirt (total_shirts_cost (total_amount_spent amount_given change_received) 
                   (total_pants_cost pants_price num_pants)) num_shirts = 33
:= by
  sorry

end price_of_each_shirt_l55_5531


namespace find_z_l55_5565

-- Definitions from conditions
def x : ℕ := 22
def y : ℕ := 13
def total_boys_who_went_down_slide : ℕ := x + y
def ratio_slide_to_watch := 5 / 3

-- Statement we need to prove
theorem find_z : ∃ z : ℕ, (5 / 3 = total_boys_who_went_down_slide / z) ∧ z = 21 :=
by
  use 21
  sorry

end find_z_l55_5565


namespace xiaohong_height_l55_5507

theorem xiaohong_height 
  (father_height_cm : ℕ)
  (height_difference_dm : ℕ)
  (father_height : father_height_cm = 170)
  (height_difference : height_difference_dm = 4) :
  ∃ xiaohong_height_cm : ℕ, xiaohong_height_cm + height_difference_dm * 10 = father_height_cm :=
by
  use 130
  sorry

end xiaohong_height_l55_5507


namespace evaluate_expression_l55_5585

noncomputable def absoluteValue (x : ℝ) : ℝ := |x|

noncomputable def ceilingFunction (x : ℝ) : ℤ := ⌈x⌉

theorem evaluate_expression : ceilingFunction (absoluteValue (-52.7)) = 53 :=
by
  sorry

end evaluate_expression_l55_5585


namespace rectangle_and_square_problems_l55_5508

theorem rectangle_and_square_problems :
  ∃ (length width : ℝ), 
    (length / width = 2) ∧ 
    (length * width = 50) ∧ 
    (length = 10) ∧
    (width = 5) ∧
    ∃ (side_length : ℝ), 
      (side_length ^ 2 = 50) ∧ 
      (side_length - width = 5 * (Real.sqrt 2 - 1)) := 
by
  sorry

end rectangle_and_square_problems_l55_5508


namespace parabola_tangency_point_l55_5564

-- Definitions of the parabola equations
def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 20
def parabola2 (y : ℝ) : ℝ := y^2 + 36 * y + 380

-- The proof statement
theorem parabola_tangency_point : 
  ∃ (x y : ℝ), 
    parabola1 x = y ∧ parabola2 y = x ∧ x = -9 / 2 ∧ y = -35 / 2 :=
by
  sorry

end parabola_tangency_point_l55_5564


namespace two_colonies_reach_limit_l55_5597

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2^n

theorem two_colonies_reach_limit (days : ℕ) (h : bacteria_growth days = (2^20)) : 
  bacteria_growth days = bacteria_growth 20 := 
by sorry

end two_colonies_reach_limit_l55_5597


namespace double_luckiness_l55_5598

variable (oats marshmallows : ℕ)
variable (initial_luckiness doubled_luckiness : ℚ)

def luckiness (marshmallows total_pieces : ℕ) : ℚ :=
  marshmallows / total_pieces

theorem double_luckiness (h_oats : oats = 90) (h_marshmallows : marshmallows = 9)
  (h_initial : initial_luckiness = luckiness marshmallows (oats + marshmallows))
  (h_doubled : doubled_luckiness = 2 * initial_luckiness) :
  ∃ x : ℕ, doubled_luckiness = luckiness (marshmallows + x) (oats + marshmallows + x) :=
  sorry

#check double_luckiness

end double_luckiness_l55_5598


namespace no_point_in_punctured_disk_l55_5581

theorem no_point_in_punctured_disk (A B C D E F G : ℝ) (hB2_4AC : B^2 - 4 * A * C < 0) :
  ∃ δ > 0, ∀ x y : ℝ, 0 < x^2 + y^2 → x^2 + y^2 < δ^2 → 
    ¬(A * x^2 + B * x * y + C * y^2 + D * x^3 + E * x^2 * y + F * x * y^2 + G * y^3 = 0) :=
sorry

end no_point_in_punctured_disk_l55_5581


namespace possible_values_of_g_zero_l55_5546

variable {g : ℝ → ℝ}

theorem possible_values_of_g_zero (h : ∀ x : ℝ, g (2 * x) = g x ^ 2) : g 0 = 0 ∨ g 0 = 1 := 
sorry

end possible_values_of_g_zero_l55_5546


namespace correct_transformation_l55_5590

theorem correct_transformation (x y : ℤ) (h : x = y) : x - 2 = y - 2 :=
by
  sorry

end correct_transformation_l55_5590


namespace fraction_spent_on_candy_l55_5542

theorem fraction_spent_on_candy (initial_quarters : ℕ) (initial_cents remaining_cents cents_per_dollar : ℕ) (fraction_spent : ℝ) :
  initial_quarters = 14 ∧ remaining_cents = 300 ∧ initial_cents = initial_quarters * 25 ∧ cents_per_dollar = 100 →
  fraction_spent = (initial_cents - remaining_cents) / cents_per_dollar →
  fraction_spent = 1 / 2 :=
by
  intro h1 h2
  sorry

end fraction_spent_on_candy_l55_5542


namespace perfect_square_trinomial_m_eq_l55_5561

theorem perfect_square_trinomial_m_eq (
    m y : ℝ) (h : ∃ k : ℝ, 4*y^2 - m*y + 25 = (2*y - k)^2) :
  m = 20 ∨ m = -20 :=
by
  sorry

end perfect_square_trinomial_m_eq_l55_5561


namespace gcd_65_130_l55_5535

theorem gcd_65_130 : Int.gcd 65 130 = 65 := by
  sorry

end gcd_65_130_l55_5535


namespace no_partition_of_positive_integers_l55_5521

theorem no_partition_of_positive_integers :
  ∀ (A B C : Set ℕ), (∀ (x : ℕ), x ∈ A ∨ x ∈ B ∨ x ∈ C) →
  (∀ (x y : ℕ), x ∈ A ∧ y ∈ B → x^2 - x * y + y^2 ∈ C) →
  (∀ (x y : ℕ), x ∈ B ∧ y ∈ C → x^2 - x * y + y^2 ∈ A) →
  (∀ (x y : ℕ), x ∈ C ∧ y ∈ A → x^2 - x * y + y^2 ∈ B) →
  False := 
sorry

end no_partition_of_positive_integers_l55_5521


namespace beaver_hid_90_carrots_l55_5552

-- Defining the number of burrows and carrot condition homomorphic to the problem
def beaver_carrots (x : ℕ) := 5 * x
def rabbit_carrots (y : ℕ) := 7 * y

-- Stating the main theorem based on conditions derived from the problem
theorem beaver_hid_90_carrots (x y : ℕ) (h1 : beaver_carrots x = rabbit_carrots y) (h2 : y = x - 5) : 
  beaver_carrots x = 90 := 
by 
  sorry

end beaver_hid_90_carrots_l55_5552


namespace tan_sin_cos_proof_l55_5594

theorem tan_sin_cos_proof (h1 : Real.sin (Real.pi / 6) = 1 / 2)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2)
    (h3 : Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6)) :
    ((Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6))^2) / ((Real.tan (Real.pi / 6))^2 * (Real.cos (Real.pi / 6))^2) = 1 / 3 := by
  sorry

end tan_sin_cos_proof_l55_5594


namespace profit_equations_l55_5524

-- Define the conditions
def total_workers : ℕ := 150
def fabric_per_worker_per_day : ℕ := 30
def clothing_per_worker_per_day : ℕ := 4
def fabric_needed_per_clothing : ℝ := 1.5
def profit_per_meter : ℝ := 2
def profit_per_clothing : ℝ := 25

-- Define the profit functions
def profit_clothing (x : ℕ) : ℝ := profit_per_clothing * clothing_per_worker_per_day * x
def profit_fabric (x : ℕ) : ℝ := profit_per_meter * (fabric_per_worker_per_day * (total_workers - x) - fabric_needed_per_clothing * clothing_per_worker_per_day * x)

-- Define the total profit function
def total_profit (x : ℕ) : ℝ := profit_clothing x + profit_fabric x

-- Prove the given statements
theorem profit_equations (x : ℕ) :
  profit_clothing x = 100 * x ∧
  profit_fabric x = 9000 - 72 * x ∧
  total_profit 100 = 11800 :=
by
  -- Proof omitted
  sorry

end profit_equations_l55_5524


namespace volume_to_surface_area_ratio_l55_5526

-- Definitions based on the conditions
def unit_cube_volume : ℕ := 1
def num_unit_cubes : ℕ := 7
def unit_cube_total_volume : ℕ := num_unit_cubes * unit_cube_volume

def surface_area_of_central_cube : ℕ := 0
def exposed_faces_per_surrounding_cube : ℕ := 5
def num_surrounding_cubes : ℕ := 6
def total_surface_area : ℕ := num_surrounding_cubes * exposed_faces_per_surrounding_cube

-- Mathematical proof statement
theorem volume_to_surface_area_ratio : 
  (unit_cube_total_volume : ℚ) / (total_surface_area : ℚ) = 7 / 30 :=
by sorry

end volume_to_surface_area_ratio_l55_5526


namespace eval_expression_l55_5567

theorem eval_expression : 3^13 / 3^3 + 2^3 = 59057 := by
  sorry

end eval_expression_l55_5567


namespace midpoint_trajectory_l55_5548

theorem midpoint_trajectory (x y : ℝ) (h : ∃ (xₚ yₚ : ℝ), yₚ = 2 * xₚ^2 + 1 ∧ y = 4 * (xₚ / 2) ^ 2) : y = 4 * x ^ 2 :=
sorry

end midpoint_trajectory_l55_5548


namespace min_pie_pieces_l55_5562

theorem min_pie_pieces (p : ℕ) : 
  (∀ (k : ℕ), (k = 5 ∨ k = 7) → ∃ (m : ℕ), p = k * m ∨ p = m * k) → p = 11 := 
sorry

end min_pie_pieces_l55_5562


namespace sufficient_condition_for_A_l55_5591

variables {A B C : Prop}

theorem sufficient_condition_for_A (h1 : A ↔ B) (h2 : C → B) : C → A :=
sorry

end sufficient_condition_for_A_l55_5591


namespace best_sampling_method_l55_5570

/-- 
  Given a high school that wants to understand the psychological 
  pressure of students from three different grades, prove that 
  stratified sampling is the best method to use, assuming students
  from different grades may experience different levels of psychological
  pressure.
-/
theorem best_sampling_method
  (students_from_three_grades : Type)
  (survey_psychological_pressure : students_from_three_grades → ℝ)
  (potential_differences_by_grade : students_from_three_grades → ℝ → Prop):
  ∃ sampling_method, sampling_method = "stratified_sampling" :=
sorry

end best_sampling_method_l55_5570


namespace solidConstruction_l55_5568

-- Definitions
structure Solid where
  octagonal_faces : Nat
  hexagonal_faces : Nat
  square_faces : Nat

-- Conditions
def solidFromCube (S : Solid) : Prop :=
  S.octagonal_faces = 6 ∧ S.hexagonal_faces = 8 ∧ S.square_faces = 12

def circumscribedSphere (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

def solidFromOctahedron (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

-- Theorem statement
theorem solidConstruction {S : Solid} :
  solidFromCube S ∧ circumscribedSphere S → solidFromOctahedron S :=
by
  sorry

end solidConstruction_l55_5568


namespace triangle_area_is_six_l55_5596

-- Conditions
def line_equation (Q : ℝ) : Prop :=
  ∀ (x y : ℝ), 12 * x - 4 * y + (Q - 305) = 0

def area_of_triangle (Q R : ℝ) : Prop :=
  R = (305 - Q) ^ 2 / 96

-- Question: Given a line equation forming a specific triangle, prove the area R equals 6.
theorem triangle_area_is_six (Q : ℝ) (h1 : Q = 281 ∨ Q = 329) :
  ∃ R : ℝ, line_equation Q → area_of_triangle Q R → R = 6 :=
by {
  sorry -- Proof to be provided
}

end triangle_area_is_six_l55_5596


namespace minimum_value_2sqrt5_l55_5553

theorem minimum_value_2sqrt5 : ∀ x : ℝ, 
  ∃ m : ℝ, (∀ x : ℝ, m ≤ (x^2 + 10) / (Real.sqrt (x^2 + 5))) ∧ (m = 2 * Real.sqrt 5) := by
  sorry

end minimum_value_2sqrt5_l55_5553


namespace distance_between_first_and_last_stop_in_km_l55_5544

-- Define the total number of stops
def num_stops := 12

-- Define the distance between the third and sixth stops in meters
def dist_3_to_6 := 3300

-- The distance between consecutive stops is the same
def distance_between_first_and_last_stop : ℕ := (num_stops - 1) * (dist_3_to_6 / 3)

-- The distance in kilometers (1 kilometer = 1000 meters)
noncomputable def distance_km : ℝ := distance_between_first_and_last_stop / 1000

-- Statement to prove
theorem distance_between_first_and_last_stop_in_km : distance_km = 12.1 :=
by
  -- Theorem proof should go here
  sorry

end distance_between_first_and_last_stop_in_km_l55_5544


namespace cardinality_union_l55_5599

open Finset

theorem cardinality_union (A B : Finset ℕ) (h : 2 ^ A.card + 2 ^ B.card - 2 ^ (A ∩ B).card = 144) : (A ∪ B).card = 8 := 
by 
  sorry

end cardinality_union_l55_5599


namespace length_AD_l55_5511

theorem length_AD (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 13) (h3 : x * (13 - x) = 36) : x = 4 ∨ x = 9 :=
by sorry

end length_AD_l55_5511


namespace ground_beef_total_cost_l55_5505

-- Define the conditions
def price_per_kg : ℝ := 5.00
def quantity_in_kg : ℝ := 12

-- The total cost calculation
def total_cost (price_per_kg quantity_in_kg : ℝ) : ℝ := price_per_kg * quantity_in_kg

-- Theorem statement
theorem ground_beef_total_cost :
  total_cost price_per_kg quantity_in_kg = 60.00 :=
sorry

end ground_beef_total_cost_l55_5505


namespace arithmetic_seq_sum_l55_5579

theorem arithmetic_seq_sum (a_n : ℕ → ℝ) (h_arith_seq : ∃ d, ∀ n, a_n (n + 1) = a_n n + d)
    (h_sum : a_n 5 + a_n 8 = 24) : a_n 6 + a_n 7 = 24 := by
  sorry

end arithmetic_seq_sum_l55_5579


namespace right_side_longer_l55_5549

/-- The sum of the three sides of a triangle is 50. 
    The right side of the triangle is a certain length longer than the left side, which has a value of 12 cm. 
    The triangle base has a value of 24 cm. 
    Prove that the right side is 2 cm longer than the left side. -/
theorem right_side_longer (L R B : ℝ) (hL : L = 12) (hB : B = 24) (hSum : L + B + R = 50) : R = L + 2 :=
by
  sorry

end right_side_longer_l55_5549


namespace tables_in_conference_hall_l55_5593

theorem tables_in_conference_hall (c t : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : 4 * c + 4 * t = 648) : 
  t = 18 :=
by sorry

end tables_in_conference_hall_l55_5593


namespace eval_g_at_2_l55_5574

def g (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem eval_g_at_2 : g 2 = 3 :=
by {
  -- This is the place for proof steps, currently it is filled with sorry.
  sorry
}

end eval_g_at_2_l55_5574


namespace enclosed_area_of_curve_l55_5522

theorem enclosed_area_of_curve :
  let side_length := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let arc_length := Real.pi
  let arc_angle := Real.pi / 2
  let arc_radius := arc_length / arc_angle
  let sector_area := (arc_angle / (2 * Real.pi)) * Real.pi * arc_radius^2
  let total_sector_area := 12 * sector_area
  let enclosed_area := octagon_area + total_sector_area + 3 * Real.pi
  enclosed_area = 54 + 38.4 * Real.sqrt 2 + 3 * Real.pi :=
by
  -- We will use sorry to indicate the proof is omitted.
  sorry

end enclosed_area_of_curve_l55_5522


namespace complement_intersection_l55_5551

-- Definitions
def A : Set ℝ := { x | x^2 + x - 6 < 0 }
def B : Set ℝ := { x | x > 1 }

-- Stating the problem
theorem complement_intersection (x : ℝ) : x ∈ (Aᶜ ∩ B) ↔ x ∈ Set.Ici 2 :=
by sorry

end complement_intersection_l55_5551


namespace rebus_solution_l55_5529

theorem rebus_solution (A B C D : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : D ≠ 0) 
  (h5 : A ≠ B) (h6 : A ≠ C) (h7 : A ≠ D) (h8 : B ≠ C) (h9 : B ≠ D) (h10 : C ≠ D) :
  1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D) → 
  A = 2 ∧ B = 9 ∧ C = 1 ∧ D = 6 :=
by
  intros h
  sorry

end rebus_solution_l55_5529


namespace number_of_distinct_possible_values_for_c_l55_5556

variables {a b r s t : ℂ}
variables (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
variables (h_transform : ∀ z, (a * z + b - r) * (a * z + b - s) * (a * z + b - t) = (z - c * r) * (z - c * s) * (z - c * t))

theorem number_of_distinct_possible_values_for_c (h_nonzero : a ≠ 0) : 
  ∃ (n : ℕ), n = 4 := sorry

end number_of_distinct_possible_values_for_c_l55_5556


namespace trajectory_of_M_l55_5569

open Real

-- Define the endpoints A and B
variable {A B M : Real × Real}

-- Given conditions
def segment_length (A B : Real × Real) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25

def on_axes (A B : Real × Real) : Prop :=
  A.2 = 0 ∧ B.1 = 0

def point_m_relationship (A B M : Real × Real) : Prop :=
  let AM := (M.1 - A.1, M.2 - A.2)
  let MB := (M.1 - B.1, M.2 - B.2)
  AM.1 = (2 / 3) * MB.1 ∧ AM.2 = (2 / 3) * MB.2 ∧
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4

theorem trajectory_of_M (A B M : Real × Real)
  (h1 : segment_length A B)
  (h2 : on_axes A B)
  (h3 : point_m_relationship A B M) :
  (M.1^2 / 9) + (M.2^2 / 4) = 1 :=
sorry

end trajectory_of_M_l55_5569


namespace zongzi_packing_l55_5586

theorem zongzi_packing (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (8 * x + 10 * y = 200) ↔ (x, y) = (5, 16) ∨ (x, y) = (10, 12) ∨ (x, y) = (15, 8) ∨ (x, y) = (20, 4) := 
sorry

end zongzi_packing_l55_5586


namespace vector_addition_proof_l55_5563

def u : ℝ × ℝ × ℝ := (-3, 2, 5)
def v : ℝ × ℝ × ℝ := (4, -7, 1)
def result : ℝ × ℝ × ℝ := (-2, -3, 11)

theorem vector_addition_proof : (2 • u + v) = result := by
  sorry

end vector_addition_proof_l55_5563


namespace ceil_floor_eq_zero_implies_sum_l55_5516

theorem ceil_floor_eq_zero_implies_sum (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ + ⌊x⌋ = 2 * x :=
by
  sorry

end ceil_floor_eq_zero_implies_sum_l55_5516


namespace no_integer_solutions_l55_5519

theorem no_integer_solutions (x y : ℤ) : x^3 + 3 ≠ 4 * y * (y + 1) :=
sorry

end no_integer_solutions_l55_5519


namespace math_problem_l55_5500

theorem math_problem
  (x y z : ℤ)
  (hz : z ≠ 0)
  (eq1 : 2 * x - 3 * y - z = 0)
  (eq2 : x + 3 * y - 14 * z = 0) :
  (x^2 - x * y) / (y^2 + 2 * z^2) = 10 / 11 := 
by 
  sorry

end math_problem_l55_5500


namespace solve_inequality_l55_5573

open Real

theorem solve_inequality (x : ℝ) : (x ≠ 3) ∧ (x * (x + 1) / (x - 3) ^ 2 ≥ 9) ↔ (2.13696 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4.73804) :=
by
  sorry

end solve_inequality_l55_5573


namespace neg_of_all_men_are_honest_l55_5530

variable {α : Type} (man honest : α → Prop)

theorem neg_of_all_men_are_honest :
  ¬ (∀ x, man x → honest x) ↔ ∃ x, man x ∧ ¬ honest x :=
by
  sorry

end neg_of_all_men_are_honest_l55_5530


namespace intersection_lg_1_x_squared_zero_t_le_one_l55_5540

theorem intersection_lg_1_x_squared_zero_t_le_one  :
  let M := {x | 0 ≤ x ∧ x ≤ 2}
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_lg_1_x_squared_zero_t_le_one_l55_5540


namespace number_of_teachers_in_school_l55_5543

-- Definitions based on provided conditions
def number_of_girls : ℕ := 315
def number_of_boys : ℕ := 309
def total_number_of_people : ℕ := 1396

-- Proof goal: Number of teachers in the school
theorem number_of_teachers_in_school : 
  total_number_of_people - (number_of_girls + number_of_boys) = 772 :=
by
  sorry

end number_of_teachers_in_school_l55_5543


namespace power_of_prime_implies_n_prime_l55_5559

theorem power_of_prime_implies_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Nat.Prime p) :
  3^n - 2^n = p^k → Nat.Prime n :=
by
  sorry

end power_of_prime_implies_n_prime_l55_5559


namespace ineq_a3b3c3_l55_5595

theorem ineq_a3b3c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ a^2 * b + b^2 * c + c^2 * a ∧ (a^3 + b^3 + c^3 = a^2 * b + b^2 * c + c^2 * a ↔ a = b ∧ b = c) :=
by
  sorry

end ineq_a3b3c3_l55_5595


namespace problem1_solution_problem2_solution_l55_5525

theorem problem1_solution (x : ℝ): 2 * x^2 + x - 3 = 0 → (x = 1 ∨ x = -3 / 2) :=
by
  intro h
  -- Proof skipped
  sorry

theorem problem2_solution (x : ℝ): (x - 3)^2 = 2 * x * (3 - x) → (x = 3 ∨ x = 1) :=
by
  intro h
  -- Proof skipped
  sorry

end problem1_solution_problem2_solution_l55_5525


namespace sequence_b_n_l55_5506

theorem sequence_b_n (b : ℕ → ℝ) (h₁ : b 1 = 2) (h₂ : ∀ n, (b (n + 1))^3 = 64 * (b n)^3) : 
    b 50 = 2 * 4^49 :=
sorry

end sequence_b_n_l55_5506


namespace cosine_values_count_l55_5537

theorem cosine_values_count (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 360) (h3 : Real.cos x = -0.65) : 
  ∃ (n : ℕ), n = 2 := by
  sorry

end cosine_values_count_l55_5537


namespace difference_between_sums_l55_5545

-- Define the arithmetic sequence sums
def sum_seq (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Define sets A and B
def sumA : ℕ := sum_seq 10 75
def sumB : ℕ := sum_seq 76 125

-- State the problem
theorem difference_between_sums : sumB - sumA = 2220 :=
by
  -- The proof is omitted
  sorry

end difference_between_sums_l55_5545


namespace expand_expression_l55_5576

theorem expand_expression (x y : ℝ) : 
  5 * (4 * x^2 + 3 * x * y - 4) = 20 * x^2 + 15 * x * y - 20 := 
by 
  sorry

end expand_expression_l55_5576


namespace cookie_sheet_perimeter_l55_5502

def width : ℕ := 10
def length : ℕ := 2

def perimeter (w l : ℕ) : ℕ := 2 * w + 2 * l

theorem cookie_sheet_perimeter : 
  perimeter width length = 24 := by
  sorry

end cookie_sheet_perimeter_l55_5502


namespace find_a_l55_5517

theorem find_a (a : ℝ) : 
  (a + 3)^2 = (a + 1)^2 + (a + 2)^2 → a = 2 := 
by
  intro h
  -- Proof should go here
  sorry

end find_a_l55_5517


namespace days_at_sister_l55_5528

def total_days_vacation : ℕ := 21
def days_plane : ℕ := 2
def days_grandparents : ℕ := 5
def days_train : ℕ := 1
def days_brother : ℕ := 5
def days_car_to_sister : ℕ := 1
def days_bus_to_sister : ℕ := 1
def extra_days_due_to_time_zones : ℕ := 1
def days_bus_back : ℕ := 1
def days_car_back : ℕ := 1

theorem days_at_sister : 
  total_days_vacation - (days_plane + days_grandparents + days_train + days_brother + days_car_to_sister + days_bus_to_sister + extra_days_due_to_time_zones + days_bus_back + days_car_back) = 3 :=
by
  sorry

end days_at_sister_l55_5528


namespace time_to_cross_bridge_l55_5541

theorem time_to_cross_bridge 
  (speed_kmhr : ℕ) 
  (bridge_length_m : ℕ) 
  (h1 : speed_kmhr = 10)
  (h2 : bridge_length_m = 2500) :
  (bridge_length_m / (speed_kmhr * 1000 / 60) = 15) :=
by
  sorry

end time_to_cross_bridge_l55_5541


namespace proof_aim_l55_5584

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + (2 - a) = 0

theorem proof_aim (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
sorry

end proof_aim_l55_5584


namespace sum_of_numbers_l55_5527

variable (x y : ℝ)

def condition1 := 0.45 * x = 2700
def condition2 := y = 2 * x

theorem sum_of_numbers (h1 : condition1 x) (h2 : condition2 x y) : x + y = 18000 :=
by {
  sorry
}

end sum_of_numbers_l55_5527


namespace remainder_mod_68_l55_5589

theorem remainder_mod_68 (n : ℕ) (h : 67^67 + 67 ≡ 66 [MOD n]) : n = 68 := 
by 
  sorry

end remainder_mod_68_l55_5589


namespace Alfred_spent_on_repairs_l55_5571

noncomputable def AlfredRepairCost (purchase_price selling_price gain_percent : ℚ) : ℚ :=
  let R := (selling_price - purchase_price * (1 + gain_percent)) / (1 + gain_percent)
  R

theorem Alfred_spent_on_repairs :
  AlfredRepairCost 4700 5800 0.017543859649122806 = 1000 := by
  sorry

end Alfred_spent_on_repairs_l55_5571


namespace tan_theta_eq_neg_4_over_3_expression_eval_l55_5578

theorem tan_theta_eq_neg_4_over_3 (θ : ℝ) (h₁ : Real.sin θ = 4 / 5) (h₂ : Real.pi / 2 < θ ∧ θ < Real.pi) :
  Real.tan θ = -4 / 3 :=
sorry

theorem expression_eval (θ : ℝ) (h₁ : Real.sin θ = 4 / 5) (h₂ : Real.pi / 2 < θ ∧ θ < Real.pi) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (3 * Real.sin θ ^ 2 + Real.cos θ ^ 2) = 8 / 25 :=
sorry

end tan_theta_eq_neg_4_over_3_expression_eval_l55_5578


namespace megans_candy_l55_5558

variable (M : ℕ)

theorem megans_candy (h1 : M * 3 + 10 = 25) : M = 5 :=
by sorry

end megans_candy_l55_5558


namespace budget_for_equipment_l55_5577

theorem budget_for_equipment 
    (transportation_p : ℝ := 20)
    (r_d_p : ℝ := 9)
    (utilities_p : ℝ := 5)
    (supplies_p : ℝ := 2)
    (salaries_degrees : ℝ := 216)
    (total_degrees : ℝ := 360)
    (total_budget : ℝ := 100)
    :
    (total_budget - (transportation_p + r_d_p + utilities_p + supplies_p +
    (salaries_degrees / total_degrees * total_budget))) = 4 := 
sorry

end budget_for_equipment_l55_5577


namespace distance_to_x_axis_l55_5539

theorem distance_to_x_axis (x y : ℝ) (h : (x, y) = (3, -4)) : abs y = 4 := sorry

end distance_to_x_axis_l55_5539


namespace driver_speed_l55_5575

theorem driver_speed (v t : ℝ) (h1 : t > 0) (h2 : v > 0) (h3 : v * t = (v + 37.5) * (3 / 8) * t) : v = 22.5 :=
by
  sorry

end driver_speed_l55_5575


namespace probability_at_least_one_correct_l55_5532

-- Define the probability of missing a single question
def prob_miss_one : ℚ := 3 / 4

-- Define the probability of missing all six questions
def prob_miss_six : ℚ := prob_miss_one ^ 6

-- Define the probability of getting at least one correct answer
def prob_at_least_one : ℚ := 1 - prob_miss_six

-- The problem statement
theorem probability_at_least_one_correct :
  prob_at_least_one = 3367 / 4096 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_one_correct_l55_5532


namespace nuts_to_raisins_ratio_l55_5534

/-- 
Given that Chris mixed 3 pounds of raisins with 4 pounds of nuts 
and the total cost of the raisins was 0.15789473684210525 of the total cost of the mixture, 
prove that the ratio of the cost of a pound of nuts to the cost of a pound of raisins is 4:1. 
-/
theorem nuts_to_raisins_ratio (R N : ℝ)
    (h1 : 3 * R = 0.15789473684210525 * (3 * R + 4 * N)) :
    N / R = 4 :=
sorry  -- proof skipped

end nuts_to_raisins_ratio_l55_5534


namespace complex_fraction_evaluation_l55_5513

open Complex

theorem complex_fraction_evaluation (c d : ℂ) (hz : c ≠ 0) (hz' : d ≠ 0) (h : c^2 + c * d + d^2 = 0) :
  (c^12 + d^12) / (c^3 + d^3)^4 = 1 / 8 := 
by sorry

end complex_fraction_evaluation_l55_5513


namespace parametric_plane_equation_l55_5512

-- Definitions to translate conditions
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ := (2 + 2 * s - t, 4 - 2 * s, 6 + s - 3 * t)

-- Theorem to prove the equivalence to plane equation
theorem parametric_plane_equation : 
  ∃ A B C D, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧ 
  (∀ s t x y z, parametric_plane s t = (x, y, z) → 6 * x - 5 * y - 2 * z + 20 = 0) := by
  sorry

end parametric_plane_equation_l55_5512


namespace not_possible_arrangement_l55_5518

theorem not_possible_arrangement : 
  ¬ ∃ (f : Fin 4026 → Fin 2014), 
    (∀ k : Fin 2014, ∃ i j : Fin 4026, i < j ∧ f i = k ∧ f j = k ∧ (j.val - i.val - 1) = k.val) :=
sorry

end not_possible_arrangement_l55_5518


namespace max_children_tickets_l55_5503

theorem max_children_tickets 
  (total_budget : ℕ) (adult_ticket_cost : ℕ) 
  (child_ticket_cost_individual : ℕ) (child_ticket_cost_group : ℕ) (min_group_tickets : ℕ) 
  (remaining_budget : ℕ) :
  total_budget = 75 →
  adult_ticket_cost = 12 →
  child_ticket_cost_individual = 6 →
  child_ticket_cost_group = 4 →
  min_group_tickets = 5 →
  (remaining_budget = total_budget - adult_ticket_cost) →
  ∃ (n : ℕ), n = 15 ∧ n * child_ticket_cost_group ≤ remaining_budget :=
by
  intros h_total_budget h_adult_ticket_cost h_child_ticket_cost_individual h_child_ticket_cost_group h_min_group_tickets h_remaining_budget
  sorry

end max_children_tickets_l55_5503


namespace number_multiplies_xz_l55_5592

theorem number_multiplies_xz (x y z w A B : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  A * B = 4 :=
sorry

end number_multiplies_xz_l55_5592


namespace option_D_correct_l55_5557

theorem option_D_correct (x : ℝ) : 2 * x^2 * (3 * x)^2 = 18 * x^4 :=
by sorry

end option_D_correct_l55_5557


namespace cos_seven_pi_over_six_l55_5538

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l55_5538


namespace ten_pow_n_plus_one_divisible_by_eleven_l55_5523

theorem ten_pow_n_plus_one_divisible_by_eleven (n : ℕ) (h : n % 2 = 1) : 11 ∣ (10 ^ n + 1) :=
sorry

end ten_pow_n_plus_one_divisible_by_eleven_l55_5523


namespace find_k_l55_5560

-- Assume three lines in the form of equations
def line1 (x y k : ℝ) := x + k * y = 0
def line2 (x y : ℝ) := 2 * x + 3 * y + 8 = 0
def line3 (x y : ℝ) := x - y - 1 = 0

-- Assume the intersection point exists
def intersection_point (x y : ℝ) := 
  line2 x y ∧ line3 x y

-- The main theorem statement
theorem find_k (k : ℝ) (x y : ℝ) (h : intersection_point x y) : 
  line1 x y k ↔ k = -1/2 := 
sorry

end find_k_l55_5560


namespace lowest_score_jack_l55_5533

noncomputable def lowest_possible_score (mean : ℝ) (std_dev : ℝ) := 
  max ((1.28 * std_dev) + mean) (mean + 2 * std_dev)

theorem lowest_score_jack (mean : ℝ := 60) (std_dev : ℝ := 10) :
  lowest_possible_score mean std_dev = 73 := 
by
  -- We need to show that the minimum score Jack could get is 73 based on problem conditions
  sorry

end lowest_score_jack_l55_5533


namespace no_even_and_increasing_function_l55_5536

-- Definition of a function being even
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Definition of a function being increasing
def is_increasing_function (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≤ f y

-- Theorem stating the non-existence of a function that is both even and increasing
theorem no_even_and_increasing_function : ¬ ∃ f : ℝ → ℝ, is_even_function f ∧ is_increasing_function f :=
by
  sorry

end no_even_and_increasing_function_l55_5536


namespace cost_of_two_sandwiches_l55_5515

theorem cost_of_two_sandwiches (J S : ℝ) 
  (h1 : 5 * J = 10) 
  (h2 : S + J = 5) :
  2 * S = 6 := 
sorry

end cost_of_two_sandwiches_l55_5515


namespace eggs_left_after_capital_recovered_l55_5582

-- Conditions as definitions
def eggs_in_crate := 30
def crate_cost_dollars := 5
def price_per_egg_cents := 20

-- The amount of cents in a dollar
def cents_per_dollar := 100

-- Total cost in cents
def crate_cost_cents := crate_cost_dollars * cents_per_dollar

-- The number of eggs needed to recover the capital
def eggs_to_recover_capital := crate_cost_cents / price_per_egg_cents

-- The number of eggs left
def eggs_left := eggs_in_crate - eggs_to_recover_capital

-- The theorem stating the problem
theorem eggs_left_after_capital_recovered : eggs_left = 5 :=
by
  sorry

end eggs_left_after_capital_recovered_l55_5582


namespace problem_III_l55_5583

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / x

theorem problem_III
  (a x1 x2 : ℝ)
  (h_a : 0 < a ∧ a < 1)
  (h_roots : f x1 = a ∧ f x2 = a)
  (h_order : x1 < x2)
  (h_bounds : Real.exp (-1) < x1 ∧ x1 < 1 ∧ 1 < x2) :
  x2 - x1 > 1 / a - 1 :=
sorry

end problem_III_l55_5583


namespace equilateral_triangle_percentage_l55_5554

theorem equilateral_triangle_percentage (s : Real) :
  let area_square := s^2
  let area_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area := area_square + area_triangle
  area_triangle / total_area * 100 = (4 * Real.sqrt 3 - 3) / 13 * 100 := by
  sorry

end equilateral_triangle_percentage_l55_5554


namespace miles_to_drive_l55_5555

def total_miles : ℕ := 1200
def miles_driven : ℕ := 768
def miles_remaining : ℕ := total_miles - miles_driven

theorem miles_to_drive : miles_remaining = 432 := by
  -- Proof goes here, omitted as per instructions
  sorry

end miles_to_drive_l55_5555


namespace center_of_circle_l55_5550

theorem center_of_circle (x y : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 - 6 * y = 12) → ((x + 2)^2 + (y - 3)^2 = 25) :=
by
  sorry

end center_of_circle_l55_5550


namespace star_value_example_l55_5510

def star (a b c : ℤ) : ℤ := (a + b + c) ^ 2

theorem star_value_example : star 3 (-5) 2 = 0 :=
by
  sorry

end star_value_example_l55_5510


namespace range_of_a_l55_5587

namespace ProofProblem

theorem range_of_a (a : ℝ) (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → ∃ y : ℝ, y = a * x + 2 * a + 1 ∧ y > 0 ∧ y < 0) : 
  -1 < a ∧ a < -1/3 := 
sorry

end ProofProblem

end range_of_a_l55_5587


namespace theta_plus_2phi_l55_5547

theorem theta_plus_2phi (θ φ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hφ : 0 < φ ∧ φ < π / 2)
  (h_tan_θ : Real.tan θ = 1 / 7) (h_sin_φ : Real.sin φ = 1 / Real.sqrt 10) :
  θ + 2 * φ = π / 4 := 
sorry

end theta_plus_2phi_l55_5547


namespace base_9_units_digit_of_sum_l55_5588

def base_n_units_digit (n : ℕ) (a : ℕ) : ℕ :=
a % n

theorem base_9_units_digit_of_sum : base_n_units_digit 9 (45 + 76) = 2 :=
by
  sorry

end base_9_units_digit_of_sum_l55_5588
