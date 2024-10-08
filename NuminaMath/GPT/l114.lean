import Mathlib

namespace no_triangle_100_sticks_yes_triangle_99_sticks_l114_114619

-- Definitions for the sums of lengths of sticks
def sum_lengths (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Conditions and questions for the problem
def is_divisible_by_3 (x : ℕ) : Prop := x % 3 = 0

-- Proof problem for n = 100
theorem no_triangle_100_sticks : ¬ (is_divisible_by_3 (sum_lengths 100)) := by
  sorry

-- Proof problem for n = 99
theorem yes_triangle_99_sticks : is_divisible_by_3 (sum_lengths 99) := by
  sorry

end no_triangle_100_sticks_yes_triangle_99_sticks_l114_114619


namespace smallest_multiple_1_10_is_2520_l114_114620

noncomputable def smallest_multiple_1_10 : ℕ :=
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))

theorem smallest_multiple_1_10_is_2520 : smallest_multiple_1_10 = 2520 :=
  sorry

end smallest_multiple_1_10_is_2520_l114_114620


namespace number_of_ways_to_represent_1500_l114_114595

theorem number_of_ways_to_represent_1500 :
  ∃ (count : ℕ), count = 30 ∧ ∀ (a b c : ℕ), a * b * c = 1500 :=
sorry

end number_of_ways_to_represent_1500_l114_114595


namespace integer_roots_iff_floor_square_l114_114694

variable (α β : ℝ)
variable (m n : ℕ)
variable (real_roots : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0)

noncomputable def are_integers (α β : ℝ) : Prop := (∃ (a b : ℤ), α = a ∧ β = b)

theorem integer_roots_iff_floor_square (m n : ℕ) (α β : ℝ)
  (hmn : 0 ≤ m ∧ 0 ≤ n)
  (roots_real : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0) :
  (are_integers α β) ↔ (∃ k : ℤ, (⌊m * α⌋ + ⌊m * β⌋) = k^2) :=
sorry

end integer_roots_iff_floor_square_l114_114694


namespace family_member_bites_count_l114_114563

-- Definitions based on the given conditions
def cyrus_bites_arms_and_legs : Nat := 14
def cyrus_bites_body : Nat := 10
def family_size : Nat := 6
def total_bites_cyrus : Nat := cyrus_bites_arms_and_legs + cyrus_bites_body
def total_bites_family : Nat := total_bites_cyrus / 2

-- Translation of the question to a theorem statement
theorem family_member_bites_count : (total_bites_family / family_size) = 2 := by
  -- use sorry to indicate the proof is skipped
  sorry

end family_member_bites_count_l114_114563


namespace tan_product_identity_l114_114257

theorem tan_product_identity (A B : ℝ) (hA : A = 20) (hB : B = 25) (hSum : A + B = 45) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 := 
  by
  sorry

end tan_product_identity_l114_114257


namespace candies_eaten_l114_114526

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l114_114526


namespace distinct_points_4_l114_114227

theorem distinct_points_4 (x y : ℝ) :
  (x + y = 7 ∨ 3 * x - 2 * y = -6) ∧ (x - y = -2 ∨ 4 * x + y = 10) →
  (x, y) =
    (5 / 2, 9 / 2) ∨ 
    (x, y) = (1, 6) ∨
    (x, y) = (-2, 0) ∨ 
    (x, y) = (14 / 11, 74 / 11) :=
sorry

end distinct_points_4_l114_114227


namespace greatest_integer_is_8_l114_114782

theorem greatest_integer_is_8 {a b : ℤ} (h_sum : a + b + 8 = 21) : max a (max b 8) = 8 :=
by
  sorry

end greatest_integer_is_8_l114_114782


namespace average_speed_l114_114847

theorem average_speed :
  ∀ (initial_odometer final_odometer total_time : ℕ), 
    initial_odometer = 2332 →
    final_odometer = 2772 →
    total_time = 8 →
    (final_odometer - initial_odometer) / total_time = 55 :=
by
  intros initial_odometer final_odometer total_time h_initial h_final h_time
  sorry

end average_speed_l114_114847


namespace total_groups_correct_l114_114771

-- Definitions from conditions
def eggs := 57
def egg_group_size := 7

def bananas := 120
def banana_group_size := 10

def marbles := 248
def marble_group_size := 8

-- Calculate the number of groups for each type of object
def egg_groups := eggs / egg_group_size
def banana_groups := bananas / banana_group_size
def marble_groups := marbles / marble_group_size

-- Total number of groups
def total_groups := egg_groups + banana_groups + marble_groups

-- Proof statement
theorem total_groups_correct : total_groups = 51 := by
  sorry

end total_groups_correct_l114_114771


namespace necessary_but_not_sufficient_condition_l114_114035

-- Define the sets M and P
def M (x : ℝ) : Prop := x > 2
def P (x : ℝ) : Prop := x < 3

-- Statement of the problem
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (M x ∨ P x) → (x ∈ { y : ℝ | 2 < y ∧ y < 3 }) :=
sorry

end necessary_but_not_sufficient_condition_l114_114035


namespace range_of_a_l114_114474

theorem range_of_a (a : ℝ) (h₁ : 1/2 ≤ 1) (h₂ : a ≤ a + 1)
    (h_condition : ∀ x:ℝ, (1/2 ≤ x ∧ x ≤ 1) → (a ≤ x ∧ x ≤ a + 1)) :
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l114_114474


namespace exists_positive_integer_m_l114_114265

theorem exists_positive_integer_m (m : ℕ) (h_positive : m > 0) : 
  ∃ (m : ℕ), m > 0 ∧ ∃ k : ℕ, 8 * m = k^2 := 
sorry

end exists_positive_integer_m_l114_114265


namespace complement_A_in_U_l114_114625

noncomputable def U : Set ℝ := {x | x > -Real.sqrt 3}
noncomputable def A : Set ℝ := {x | 1 < 4 - x^2 ∧ 4 - x^2 ≤ 2}

theorem complement_A_in_U :
  (U \ A) = {x | -Real.sqrt 3 < x ∧ x ≤ -Real.sqrt 2} ∪ {x | Real.sqrt 2 ≤ x ∧ x < (Real.sqrt 3) ∨ Real.sqrt 3 ≤ x} :=
by
  sorry

end complement_A_in_U_l114_114625


namespace sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l114_114660

theorem sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : 100^2 + 1^2 = p * q ∧ 65^2 + 76^2 = p * q) : p + q = 210 := 
sorry

end sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l114_114660


namespace cylinder_volume_increase_l114_114569

theorem cylinder_volume_increase {R H : ℕ} (x : ℚ) (C : ℝ) (π : ℝ) 
  (hR : R = 8) (hH : H = 3) (hπ : π = Real.pi)
  (hV : ∃ C > 0, π * (R + x)^2 * (H + x) = π * R^2 * H + C) :
  x = 16 / 3 :=
by
  sorry

end cylinder_volume_increase_l114_114569


namespace range_of_a_l114_114338

-- Defining the problem conditions
def f (x : ℝ) : ℝ := sorry -- The function f : ℝ → ℝ is defined elsewhere such that its range is [0, 4]
def g (a x : ℝ) : ℝ := a * x - 1

-- Theorem to prove the range of 'a'
theorem range_of_a (a : ℝ) : (a ≥ 1/2) ∨ (a ≤ -1/2) :=
sorry

end range_of_a_l114_114338


namespace greatest_drop_in_price_is_May_l114_114339

def priceChangeJan := -1.25
def priceChangeFeb := 2.75
def priceChangeMar := -0.75
def priceChangeApr := 1.50
def priceChangeMay := -3.00
def priceChangeJun := -1.00

theorem greatest_drop_in_price_is_May :
  priceChangeMay < priceChangeJan ∧
  priceChangeMay < priceChangeMar ∧
  priceChangeMay < priceChangeApr ∧
  priceChangeMay < priceChangeJun ∧
  priceChangeMay < priceChangeFeb :=
by sorry

end greatest_drop_in_price_is_May_l114_114339


namespace students_at_start_of_year_l114_114295

variable (S : ℕ)

def initial_students := S
def students_left := 6
def students_new := 42
def end_year_students := 47

theorem students_at_start_of_year :
  initial_students + (students_new - students_left) = end_year_students → initial_students = 11 :=
by
  sorry

end students_at_start_of_year_l114_114295


namespace combined_alloy_tin_amount_l114_114245

theorem combined_alloy_tin_amount
  (weight_A weight_B weight_C : ℝ)
  (ratio_lead_tin_A : ℝ)
  (ratio_tin_copper_B : ℝ)
  (ratio_copper_tin_C : ℝ)
  (amount_tin : ℝ) :
  weight_A = 150 → weight_B = 200 → weight_C = 250 →
  ratio_lead_tin_A = 5/3 → ratio_tin_copper_B = 2/3 → ratio_copper_tin_C = 4 →
  amount_tin = ((3/8) * weight_A) + ((2/5) * weight_B) + ((1/5) * weight_C) →
  amount_tin = 186.25 :=
by sorry

end combined_alloy_tin_amount_l114_114245


namespace parabola_focus_coordinates_parabola_distance_to_directrix_l114_114489

-- Define constants and variables
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

noncomputable def focus_coordinates : ℝ × ℝ := (1, 0)

noncomputable def point : ℝ × ℝ := (4, 4)

noncomputable def directrix : ℝ := -1

noncomputable def distance_to_directrix : ℝ := 5

-- Proof statements
theorem parabola_focus_coordinates (x y : ℝ) (h : parabola_equation x y) : 
  focus_coordinates = (1, 0) :=
sorry

theorem parabola_distance_to_directrix (p : ℝ × ℝ) (d : ℝ) (h : p = point) (h_line : d = directrix) : 
  distance_to_directrix = 5 :=
  by
    -- Define and use the distance between point and vertical line formula
    sorry

end parabola_focus_coordinates_parabola_distance_to_directrix_l114_114489


namespace shaded_area_l114_114803

/-- Prove that the shaded area of a shape formed by removing four right triangles of legs 2 from each corner of a 6 × 6 square is equal to 28 square units -/
theorem shaded_area (a b c d : ℕ) (square_side_length : ℕ) (triangle_leg_length : ℕ)
  (h1 : square_side_length = 6)
  (h2 : triangle_leg_length = 2)
  (h3 : a = 1)
  (h4 : b = 2)
  (h5 : c = b)
  (h6 : d = 4*a) : 
  a * square_side_length * square_side_length - d * (b * b / 2) = 28 := 
sorry

end shaded_area_l114_114803


namespace alice_forest_walks_l114_114729

theorem alice_forest_walks
  (morning_distance : ℕ)
  (total_distance : ℕ)
  (days_per_week : ℕ)
  (forest_distance : ℕ) :
  morning_distance = 10 →
  total_distance = 110 →
  days_per_week = 5 →
  (total_distance - morning_distance * days_per_week) / days_per_week = forest_distance →
  forest_distance = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_forest_walks_l114_114729


namespace minimize_expression_l114_114556

theorem minimize_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 30) :
  (a, b) = (15 / 4, 15) ↔ (∀ x y : ℝ, 0 < x → 0 < y → (4 * x + y = 30) → (1 / x + 4 / y) ≥ (1 / (15 / 4) + 4 / 15)) := by
sorry

end minimize_expression_l114_114556


namespace teresa_total_marks_l114_114876

theorem teresa_total_marks :
  let science_marks := 70
  let music_marks := 80
  let social_studies_marks := 85
  let physics_marks := 1 / 2 * music_marks
  science_marks + music_marks + social_studies_marks + physics_marks = 275 :=
by
  sorry

end teresa_total_marks_l114_114876


namespace Cary_height_is_72_l114_114974

variable (Cary_height Bill_height Jan_height : ℕ)

-- Conditions
axiom Bill_height_is_half_Cary_height : Bill_height = Cary_height / 2
axiom Jan_height_is_6_inches_taller_than_Bill : Jan_height = Bill_height + 6
axiom Jan_height_is_42 : Jan_height = 42

-- Theorem statement
theorem Cary_height_is_72 : Cary_height = 72 := 
by
  sorry

end Cary_height_is_72_l114_114974


namespace cos_2alpha_plus_5pi_by_12_l114_114097

open Real

noncomputable def alpha : ℝ := sorry

axiom alpha_obtuse : π / 2 < alpha ∧ alpha < π

axiom sin_alpha_plus_pi_by_3 : sin (alpha + π / 3) = -4 / 5

theorem cos_2alpha_plus_5pi_by_12 : 
  cos (2 * alpha + 5 * π / 12) = 17 * sqrt 2 / 50 :=
by sorry

end cos_2alpha_plus_5pi_by_12_l114_114097


namespace intersection_A_B_l114_114105

noncomputable def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by 
  sorry

end intersection_A_B_l114_114105


namespace metal_detector_time_on_less_crowded_days_l114_114642

variable (find_parking_time walk_time crowded_metal_detector_time total_time_per_week : ℕ)
variable (week_days crowded_days less_crowded_days : ℕ)

theorem metal_detector_time_on_less_crowded_days
  (h1 : find_parking_time = 5)
  (h2 : walk_time = 3)
  (h3 : crowded_metal_detector_time = 30)
  (h4 : total_time_per_week = 130)
  (h5 : week_days = 5)
  (h6 : crowded_days = 2)
  (h7 : less_crowded_days = 3) :
  (total_time_per_week = (find_parking_time * week_days) + (walk_time * week_days) + (crowded_metal_detector_time * crowded_days) + (10 * less_crowded_days)) :=
sorry

end metal_detector_time_on_less_crowded_days_l114_114642


namespace remaining_perimeter_of_square_with_cutouts_l114_114559

theorem remaining_perimeter_of_square_with_cutouts 
  (square_side : ℝ) (green_square_side : ℝ) (init_perimeter : ℝ) 
  (green_square_perimeter_increase : ℝ) (final_perimeter : ℝ) :
  square_side = 10 → green_square_side = 2 →
  init_perimeter = 4 * square_side → green_square_perimeter_increase = 4 * green_square_side →
  final_perimeter = init_perimeter + green_square_perimeter_increase →
  final_perimeter = 44 :=
by
  intros hsquare_side hgreen_square_side hinit_perimeter hgreen_incr hfinal_perimeter
  -- Proof steps can be added here
  sorry

end remaining_perimeter_of_square_with_cutouts_l114_114559


namespace area_of_isosceles_trapezoid_l114_114200

variable (a b c d : ℝ) -- Variables for sides and bases of the trapezoid

-- Define isosceles trapezoid with given sides and bases
def is_isosceles_trapezoid (a b c d : ℝ) (h : ℝ) :=
  a = b ∧ c = 10 ∧ d = 16 ∧ (∃ (h : ℝ), a^2 = h^2 + ((d - c) / 2)^2 ∧ a = 5)

-- Lean theorem for the area of the isosceles trapezoid
theorem area_of_isosceles_trapezoid :
  ∀ (a b c d : ℝ) (h : ℝ), is_isosceles_trapezoid a b c d h
  → (1 / 2) * (c + d) * h = 52 :=
by
  sorry

end area_of_isosceles_trapezoid_l114_114200


namespace determine_d_l114_114148

variables (a b c d : ℝ)

-- Conditions given in the problem
def condition1 (a b d : ℝ) : Prop := d / a = (d - 25) / b
def condition2 (b c d : ℝ) : Prop := d / b = (d - 15) / c
def condition3 (a c d : ℝ) : Prop := d / a = (d - 35) / c

-- Final statement to prove
theorem determine_d (a b c : ℝ) (d : ℝ) :
    condition1 a b d ∧ condition2 b c d ∧ condition3 a c d → d = 75 :=
by sorry

end determine_d_l114_114148


namespace flour_needed_for_bread_l114_114607

-- Definitions based on conditions
def flour_per_loaf : ℝ := 2.5
def number_of_loaves : ℕ := 2

-- Theorem statement
theorem flour_needed_for_bread : flour_per_loaf * number_of_loaves = 5 :=
by sorry

end flour_needed_for_bread_l114_114607


namespace evaluate_expression_l114_114849

theorem evaluate_expression : 3002^3 - 3001 * 3002^2 - 3001^2 * 3002 + 3001^3 + 1 = 6004 :=
by
  sorry

end evaluate_expression_l114_114849


namespace convex_polygon_sides_eq_49_l114_114006

theorem convex_polygon_sides_eq_49 
  (n : ℕ)
  (hn : n > 0) 
  (h : (n * (n - 3)) / 2 = 23 * n) : n = 49 :=
sorry

end convex_polygon_sides_eq_49_l114_114006


namespace frog_return_prob_A_after_2022_l114_114308

def initial_prob_A : ℚ := 1
def transition_prob_A_to_adj : ℚ := 1/3
def transition_prob_adj_to_A : ℚ := 1/3
def transition_prob_adj_to_adj : ℚ := 2/3

noncomputable def prob_A_return (n : ℕ) : ℚ :=
if (n % 2 = 0) then
  (2/9) * (1/2^(n/2)) + (1/9)
else
  0

theorem frog_return_prob_A_after_2022 : prob_A_return 2022 = (2/9) * (1/2^1010) + (1/9) :=
by
  sorry

end frog_return_prob_A_after_2022_l114_114308


namespace positive_value_of_m_l114_114392

theorem positive_value_of_m (m : ℝ) (h : (64 * m^2 - 60 * m) = 0) : m = 15 / 16 :=
sorry

end positive_value_of_m_l114_114392


namespace depth_of_lost_ship_l114_114490

theorem depth_of_lost_ship (rate_of_descent : ℕ) (time_taken : ℕ) (h1 : rate_of_descent = 60) (h2 : time_taken = 60) :
  rate_of_descent * time_taken = 3600 :=
by {
  /-
  Proof steps would go here.
  -/
  sorry
}

end depth_of_lost_ship_l114_114490


namespace deceased_member_income_l114_114649

theorem deceased_member_income
  (initial_income_4_members : ℕ)
  (initial_members : ℕ := 4)
  (initial_average_income : ℕ := 840)
  (final_income_3_members : ℕ)
  (remaining_members : ℕ := 3)
  (final_average_income : ℕ := 650)
  (total_income_initial : initial_income_4_members = initial_average_income * initial_members)
  (total_income_final : final_income_3_members = final_average_income * remaining_members)
  (income_deceased : ℕ) :
  income_deceased = initial_income_4_members - final_income_3_members :=
by
  -- sorry indicates this part of the proof is left as an exercise
  sorry

end deceased_member_income_l114_114649


namespace greatest_integer_less_than_M_over_100_l114_114418

theorem greatest_integer_less_than_M_over_100 :
  (1 / (Nat.factorial 3 * Nat.factorial 16) +
   1 / (Nat.factorial 4 * Nat.factorial 15) +
   1 / (Nat.factorial 5 * Nat.factorial 14) +
   1 / (Nat.factorial 6 * Nat.factorial 13) +
   1 / (Nat.factorial 7 * Nat.factorial 12) +
   1 / (Nat.factorial 8 * Nat.factorial 11) +
   1 / (Nat.factorial 9 * Nat.factorial 10) = M / (Nat.factorial 2 * Nat.factorial 17)) →
  (⌊(M : ℚ) / 100⌋ = 27) := 
sorry

end greatest_integer_less_than_M_over_100_l114_114418


namespace minimum_value_of_f_ge_7_l114_114305

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem minimum_value_of_f_ge_7 {x : ℝ} (hx : x > 0) : f x ≥ 7 := 
by
  sorry

end minimum_value_of_f_ge_7_l114_114305


namespace max_gcd_2015xy_l114_114909

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end max_gcd_2015xy_l114_114909


namespace train_speed_is_60_kmph_l114_114252

noncomputable def train_length : ℝ := 110
noncomputable def time_to_pass_man : ℝ := 5.999520038396929
noncomputable def man_speed_kmph : ℝ := 6

theorem train_speed_is_60_kmph :
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / time_to_pass_man
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph = 60 :=
by
  sorry

end train_speed_is_60_kmph_l114_114252


namespace tomatoes_first_shipment_l114_114404

theorem tomatoes_first_shipment :
  ∃ X : ℕ, 
    (∀Y : ℕ, 
      (Y = 300) → -- Saturday sale
      (X - Y = X - 300) ∧
      (∀Z : ℕ, 
        (Z = 200) → -- Sunday rotting
        (X - 300 - Z = X - 500) ∧
        (∀W : ℕ, 
          (W = 2 * X) → -- Monday new shipment
          (X - 500 + W = 2500) →
          (X = 1000)
        )
      )
    ) :=
by
  sorry

end tomatoes_first_shipment_l114_114404


namespace prove_trig_values_l114_114347

/-- Given angles A and B, where both are acute angles,
  and their sine values are known,
  we aim to prove the cosine of (A + B) and the measure
  of angle C in triangle ABC. -/
theorem prove_trig_values (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2)
  (sin_A_eq : Real.sin A = (Real.sqrt 5) / 5)
  (sin_B_eq : Real.sin B = (Real.sqrt 10) / 10) :
  Real.cos (A + B) = (Real.sqrt 2) / 2 ∧ (π - (A + B)) = 3 * π / 4 := by
sorry

end prove_trig_values_l114_114347


namespace find_date_behind_l114_114372

variables (x y : ℕ)
-- Conditions
def date_behind_C := x
def date_behind_A := x + 1
def date_behind_B := x + 13
def date_behind_P := x + 14

-- Statement to prove
theorem find_date_behind : (x + y = (x + 1) + (x + 13)) → (y = date_behind_P) :=
by
  sorry

end find_date_behind_l114_114372


namespace cells_at_end_of_12th_day_l114_114011

def initial_organisms : ℕ := 8
def initial_cells_per_organism : ℕ := 4
def total_initial_cells : ℕ := initial_organisms * initial_cells_per_organism
def division_period_days : ℕ := 3
def total_duration_days : ℕ := 12
def complete_periods : ℕ := total_duration_days / division_period_days
def common_ratio : ℕ := 3

theorem cells_at_end_of_12th_day :
  total_initial_cells * common_ratio^(complete_periods - 1) = 864 := by
  sorry

end cells_at_end_of_12th_day_l114_114011


namespace symmetric_line_eq_l114_114912

-- Defining a structure for a line using its standard equation form "ax + by + c = 0"
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Definition: A line is symmetric with respect to y-axis if it can be obtained
-- by replacing x with -x in its equation form.

def isSymmetricToYAxis (l₁ l₂ : Line) : Prop :=
  l₂.a = -l₁.a ∧ l₂.b = l₁.b ∧ l₂.c = l₁.c

-- The given condition: line1 is 4x - 3y + 5 = 0
def line1 : Line := { a := 4, b := -3, c := 5 }

-- The expected line l symmetric to y-axis should satisfy our properties
def expected_line_l : Line := { a := 4, b := 3, c := -5 }

-- The theorem we need to prove
theorem symmetric_line_eq : ∃ l : Line,
  isSymmetricToYAxis line1 l ∧ l = { a := 4, b := 3, c := -5 } :=
by
  sorry

end symmetric_line_eq_l114_114912


namespace all_of_the_above_were_used_as_money_l114_114774

-- Defining the conditions that each type was used as money
def gold_used_as_money : Prop := True
def stones_used_as_money : Prop := True
def horses_used_as_money : Prop := True
def dried_fish_used_as_money : Prop := True
def mollusk_shells_used_as_money : Prop := True

-- The statement to prove
theorem all_of_the_above_were_used_as_money :
  gold_used_as_money ∧
  stones_used_as_money ∧
  horses_used_as_money ∧
  dried_fish_used_as_money ∧
  mollusk_shells_used_as_money ↔
  (∀ x, (x = "gold" ∨ x = "stones" ∨ x = "horses" ∨ x = "dried fish" ∨ x = "mollusk shells") → 
  (x = "gold" ∧ gold_used_as_money) ∨ 
  (x = "stones" ∧ stones_used_as_money) ∨ 
  (x = "horses" ∧ horses_used_as_money) ∨ 
  (x = "dried fish" ∧ dried_fish_used_as_money) ∨ 
  (x = "mollusk shells" ∧ mollusk_shells_used_as_money)) :=
by
  sorry

end all_of_the_above_were_used_as_money_l114_114774


namespace number_of_new_galleries_l114_114923

-- Definitions based on conditions
def number_of_pictures_first_gallery := 9
def number_of_pictures_per_new_gallery := 2
def pencils_per_picture := 4
def pencils_per_exhibition_signature := 2
def total_pencils_used := 88

-- Theorem statement according to the correct answer
theorem number_of_new_galleries 
  (number_of_pictures_first_gallery : ℕ)
  (number_of_pictures_per_new_gallery : ℕ)
  (pencils_per_picture : ℕ)
  (pencils_per_exhibition_signature : ℕ)
  (total_pencils_used : ℕ)
  (drawing_pencils_first_gallery := number_of_pictures_first_gallery * pencils_per_picture)
  (signing_pencils_first_gallery := pencils_per_exhibition_signature)
  (total_pencils_first_gallery := drawing_pencils_first_gallery + signing_pencils_first_gallery)
  (pencils_for_new_galleries := total_pencils_used - total_pencils_first_gallery)
  (pencils_per_new_gallery := (number_of_pictures_per_new_gallery * pencils_per_picture) + pencils_per_exhibition_signature) :
  pencils_per_new_gallery > 0 → pencils_for_new_galleries / pencils_per_new_gallery = 5 :=
sorry

end number_of_new_galleries_l114_114923


namespace quadratic_roots_square_diff_l114_114437

theorem quadratic_roots_square_diff (α β : ℝ) (h : α ≠ β)
    (hα : α^2 - 3 * α + 2 = 0) (hβ : β^2 - 3 * β + 2 = 0) :
    (α - β)^2 = 1 :=
sorry

end quadratic_roots_square_diff_l114_114437


namespace exists_n_divisible_l114_114487

theorem exists_n_divisible (k : ℕ) (m : ℤ) (hk : k > 0) (hm : m % 2 = 1) : 
  ∃ n : ℕ, n > 0 ∧ 2^k ∣ (n^n - m) :=
by
  sorry

end exists_n_divisible_l114_114487


namespace ravioli_to_tortellini_ratio_l114_114867

-- Definitions from conditions
def total_students : ℕ := 800
def ravioli_students : ℕ := 300
def tortellini_students : ℕ := 150

-- Ratio calculation as a theorem
theorem ravioli_to_tortellini_ratio : 2 = ravioli_students / Nat.gcd ravioli_students tortellini_students :=
by
  -- Given the defined values
  have gcd_val : Nat.gcd ravioli_students tortellini_students = 150 := by
    sorry
  have ratio_simp : ravioli_students / 150 = 2 := by
    sorry
  exact ratio_simp

end ravioli_to_tortellini_ratio_l114_114867


namespace range_of_a_for_negative_root_l114_114506

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 4^x - 2^(x-1) + a = 0) →
  - (1/2 : ℝ) < a ∧ a ≤ (1/16 : ℝ) :=
by
  sorry

end range_of_a_for_negative_root_l114_114506


namespace convert_and_compute_l114_114807

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  if n = 231 then 2 * 4^2 + 3 * 4^1 + 1 * 4^0
  else if n = 21 then 2 * 4^1 + 1 * 4^0
  else if n = 3 then 3
  else 0

noncomputable def base10_to_base4 (n : ℕ) : ℕ :=
  if n = 135 then 2 * 4^2 + 1 * 4^1 + 3 * 4^0
  else 0

theorem convert_and_compute :
  base10_to_base4 ((base4_to_base10 231 / base4_to_base10 3) * base4_to_base10 21) = 213 :=
by {
  sorry
}

end convert_and_compute_l114_114807


namespace gillian_spent_multiple_of_sandi_l114_114251

theorem gillian_spent_multiple_of_sandi
  (sandi_had : ℕ := 600)
  (gillian_spent : ℕ := 1050)
  (sandi_spent : ℕ := sandi_had / 2)
  (diff : ℕ := gillian_spent - sandi_spent)
  (extra : ℕ := 150)
  (multiple_of_sandi : ℕ := (diff - extra) / sandi_spent) : 
  multiple_of_sandi = 1 := 
  by sorry

end gillian_spent_multiple_of_sandi_l114_114251


namespace remainder_division_l114_114442

theorem remainder_division (x : ℂ) (β : ℂ) (hβ : β^7 = 1) :
  (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 0 ->
  (x^63 + x^49 + x^35 + x^14 + 1) % (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 5 :=
by
  intro h
  sorry

end remainder_division_l114_114442


namespace profit_percent_is_correct_l114_114587

noncomputable def profit_percent : ℝ := 
  let marked_price_per_pen := 1 
  let pens_bought := 56 
  let effective_payment := 46 
  let discount := 0.01
  let cost_price_per_pen := effective_payment / pens_bought
  let selling_price_per_pen := marked_price_per_pen * (1 - discount)
  let total_selling_price := pens_bought * selling_price_per_pen
  let profit := total_selling_price - effective_payment
  (profit / effective_payment) * 100

theorem profit_percent_is_correct : abs (profit_percent - 20.52) < 0.01 :=
by
  sorry

end profit_percent_is_correct_l114_114587


namespace remainder_when_add_13_l114_114865

theorem remainder_when_add_13 (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 :=
sorry

end remainder_when_add_13_l114_114865


namespace construction_rates_construction_cost_l114_114510

-- Defining the conditions as Lean hypotheses

def length := 1650
def diff_rate := 30
def time_ratio := 3/2

-- Daily construction rates (questions answered as hypotheses as well)
def daily_rate_A := 60
def daily_rate_B := 90

-- Additional conditions for cost calculations
def cost_A_per_day := 90000
def cost_B_per_day := 120000
def total_days := 14
def alone_days_A := 5

-- Problem stated as proofs to be completed
theorem construction_rates :
  (∀ (x : ℕ), x = daily_rate_A ∧ (x + diff_rate) = daily_rate_B ∧ 
  (1650 / (x + diff_rate)) * (3/2) = (1650 / x) → 
  60 = daily_rate_A ∧ (60 + 30) = daily_rate_B ) :=
by sorry

theorem construction_cost :
  (∀ (m : ℕ), m = alone_days_A ∧ 
  (cost_A_per_day * total_days + cost_B_per_day * (total_days - alone_days_A)) / 1000 = 2340) :=
by sorry

end construction_rates_construction_cost_l114_114510


namespace tangent_line_eq_max_f_val_in_interval_a_le_2_l114_114518

-- Definitions based on given conditions
def f (x : ℝ) (a : ℝ) : ℝ := x ^ 3 - a * x ^ 2

def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x ^ 2 - 2 * a * x

-- (I) (i) Proof that the tangent line equation is y = 3x - 2 at (1, f(1))
theorem tangent_line_eq (a : ℝ) (h : f_prime 1 a = 3) : y = 3 * x - 2 :=
by sorry

-- (I) (ii) Proof that the max value of f(x) in [0,2] is 8
theorem max_f_val_in_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x 0 ≤ f 2 0 :=
by sorry

-- (II) Proof that a ≤ 2 if f(x) + x ≥ 0 for all x ∈ [0,2]
theorem a_le_2 (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x a + x ≥ 0) : a ≤ 2 :=
by sorry

end tangent_line_eq_max_f_val_in_interval_a_le_2_l114_114518


namespace quadratic_has_real_roots_iff_l114_114367

theorem quadratic_has_real_roots_iff (m : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + m + 5 = 0) ↔ m ≤ -1 :=
by
  -- Proof omitted
  sorry

end quadratic_has_real_roots_iff_l114_114367


namespace largest_n_for_inequality_l114_114515

theorem largest_n_for_inequality :
  ∃ n : ℕ, 3 * n^2007 < 3^4015 ∧ ∀ m : ℕ, 3 * m^2007 < 3^4015 → m ≤ 8 ∧ n = 8 :=
by
  sorry

end largest_n_for_inequality_l114_114515


namespace molecular_weight_compound_l114_114359

theorem molecular_weight_compound :
  let weight_H := 1.008
  let weight_Cr := 51.996
  let weight_O := 15.999
  let n_H := 2
  let n_Cr := 1
  let n_O := 4
  (n_H * weight_H) + (n_Cr * weight_Cr) + (n_O * weight_O) = 118.008 :=
by
  sorry

end molecular_weight_compound_l114_114359


namespace alpha_value_l114_114866

theorem alpha_value (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.logb 3 (x + 1)) (h2 : f α = 1) : α = 2 := by
  sorry

end alpha_value_l114_114866


namespace age_ratio_l114_114878

theorem age_ratio (V A : ℕ) (h1 : V - 5 = 16) (h2 : V * 2 = 7 * A) :
  (V + 4) * 2 = (A + 4) * 5 := 
sorry

end age_ratio_l114_114878


namespace train_crossing_time_l114_114784

open Real

noncomputable def time_to_cross_bridge 
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000/3600)
  total_distance / speed_train_ms

theorem train_crossing_time
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ)
  (h_length_train : length_train = 160)
  (h_speed_train_kmh : speed_train_kmh = 45)
  (h_length_bridge : length_bridge = 215) :
  time_to_cross_bridge length_train speed_train_kmh length_bridge = 30 :=
sorry

end train_crossing_time_l114_114784


namespace race_total_distance_l114_114228

theorem race_total_distance (D : ℝ) 
  (A_time : D / 20 = D / 25 + 1) 
  (beat_distance : D / 20 * 25 = D + 20) : 
  D = 80 :=
sorry

end race_total_distance_l114_114228


namespace find_third_side_length_l114_114718

noncomputable def triangle_third_side_length (a b c : ℝ) (B C : ℝ) 
  (h1 : B = 3 * C) (h2 : b = 12) (h3 : c = 20) : Prop :=
a = 16

theorem find_third_side_length (a b c : ℝ) (B C : ℝ)
  (h1 : B = 3 * C) (h2 : b = 12) (h3 : c = 20) :
  triangle_third_side_length a b c B C h1 h2 h3 :=
sorry

end find_third_side_length_l114_114718


namespace total_packs_l114_114643

theorem total_packs (cards_per_person cards_per_pack : ℕ) (num_people : ℕ) 
  (h1 : cards_per_person = 540) 
  (h2 : cards_per_pack = 20) 
  (h3 : num_people = 4) : 
  (cards_per_person / cards_per_pack) * num_people = 108 := 
by
  sorry

end total_packs_l114_114643


namespace anna_age_l114_114516

-- Define the conditions as given in the problem
variable (x : ℕ)
variable (m n : ℕ)

-- Translate the problem statement into Lean
axiom perfect_square_condition : x - 4 = m^2
axiom perfect_cube_condition : x + 3 = n^3

-- The proof problem statement in Lean 4
theorem anna_age : x = 5 :=
by
  sorry

end anna_age_l114_114516


namespace cross_country_meet_winning_scores_l114_114657

theorem cross_country_meet_winning_scores :
  ∃ (scores : Finset ℕ), scores.card = 13 ∧
    ∀ s ∈ scores, s ≥ 15 ∧ s ≤ 27 :=
by
  sorry

end cross_country_meet_winning_scores_l114_114657


namespace total_cost_of_shirt_and_sweater_l114_114665

-- Define the given conditions
def price_of_shirt := 36.46
def diff_price_shirt_sweater := 7.43
def price_of_sweater := price_of_shirt + diff_price_shirt_sweater

-- Statement to prove
theorem total_cost_of_shirt_and_sweater :
  price_of_shirt + price_of_sweater = 80.35 :=
by
  -- Proof goes here
  sorry

end total_cost_of_shirt_and_sweater_l114_114665


namespace minimum_milk_candies_l114_114114

/-- A supermarket needs to purchase candies with the following conditions:
 1. The number of watermelon candies is at most 3 times the number of chocolate candies.
 2. The number of milk candies is at least 4 times the number of chocolate candies.
 3. The sum of chocolate candies and watermelon candies is at least 2020.

 Prove that the minimum number of milk candies that need to be purchased is 2020. -/
theorem minimum_milk_candies (x y z : ℕ)
  (h1 : y ≤ 3 * x)
  (h2 : z ≥ 4 * x)
  (h3 : x + y ≥ 2020) :
  z ≥ 2020 :=
sorry

end minimum_milk_candies_l114_114114


namespace non_consecutive_heads_probability_l114_114349

-- Define the total number of basic events (n).
def total_events : ℕ := 2^4

-- Define the number of events where heads do not appear consecutively (m).
def non_consecutive_heads_events : ℕ := 1 + (Nat.choose 4 1) + (Nat.choose 3 2)

-- Define the probability of heads not appearing consecutively.
def probability_non_consecutive_heads : ℚ := non_consecutive_heads_events / total_events

-- The theorem we seek to prove
theorem non_consecutive_heads_probability :
  probability_non_consecutive_heads = 1 / 2 :=
by
  sorry

end non_consecutive_heads_probability_l114_114349


namespace john_trip_l114_114578

theorem john_trip (t : ℝ) (h : t ≥ 0) : 
  ∀ t : ℝ, 60 * t + 90 * ((7 / 2) - t) = 300 :=
by sorry

end john_trip_l114_114578


namespace sin_half_angle_product_lt_quarter_l114_114125

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end sin_half_angle_product_lt_quarter_l114_114125


namespace percentage_increase_l114_114012

theorem percentage_increase (W E : ℝ) (P : ℝ) :
  W = 200 →
  E = 204 →
  (∃ P, E = W * (1 + P / 100) * 0.85) →
  P = 20 :=
by
  intros hW hE hP
  -- Proof could be added here.
  sorry

end percentage_increase_l114_114012


namespace kara_forgot_medication_times_l114_114467

theorem kara_forgot_medication_times :
  let ounces_per_medication := 4
  let medication_times_per_day := 3
  let days_per_week := 7
  let total_weeks := 2
  let total_water_intaken := 160
  let expected_total_water := (ounces_per_medication * medication_times_per_day * days_per_week * total_weeks)
  let water_difference := expected_total_water - total_water_intaken
  let forget_times := water_difference / ounces_per_medication
  forget_times = 2 := by sorry

end kara_forgot_medication_times_l114_114467


namespace arlene_hike_distance_l114_114770

-- Define the conditions: Arlene's pace and the time she spent hiking
def arlene_pace : ℝ := 4 -- miles per hour
def arlene_time_hiking : ℝ := 6 -- hours

-- Define the problem statement and provide the mathematical proof
theorem arlene_hike_distance :
  arlene_pace * arlene_time_hiking = 24 :=
by
  -- This is where the proof would go
  sorry

end arlene_hike_distance_l114_114770


namespace hillary_descending_rate_is_1000_l114_114928

-- Definitions from the conditions
def base_to_summit_distance : ℕ := 5000
def hillary_departure_time : ℕ := 6
def hillary_climbing_rate : ℕ := 800
def eddy_climbing_rate : ℕ := 500
def hillary_stop_distance_from_summit : ℕ := 1000
def hillary_and_eddy_pass_time : ℕ := 12

-- Derived definitions
def hillary_climbing_time : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) / hillary_climbing_rate
def hillary_stop_time : ℕ := hillary_departure_time + hillary_climbing_time
def eddy_climbing_time_at_pass : ℕ := hillary_and_eddy_pass_time - hillary_departure_time
def eddy_climbed_distance : ℕ := eddy_climbing_rate * eddy_climbing_time_at_pass
def hillary_distance_descended_at_pass : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) - eddy_climbed_distance
def hillary_descending_time : ℕ := hillary_and_eddy_pass_time - hillary_stop_time 

def hillary_descending_rate : ℕ := hillary_distance_descended_at_pass / hillary_descending_time

-- Statement to prove
theorem hillary_descending_rate_is_1000 : hillary_descending_rate = 1000 := 
by
  sorry

end hillary_descending_rate_is_1000_l114_114928


namespace probability_beautiful_equation_l114_114781

def tetrahedron_faces : Set ℕ := {1, 2, 3, 4}

def is_beautiful_equation (a b : ℕ) : Prop :=
    ∃ m ∈ tetrahedron_faces, a = m + 1 ∨ a = m + 2 ∨ a = m + 3 ∨ a = m + 4 ∧ b = m * (a - m)

theorem probability_beautiful_equation : 
  (∃ a b1 b2, is_beautiful_equation a b1 ∧ is_beautiful_equation a b2) ∧
  (∃ a b1 b2, tetrahedron_faces ⊆ {a} ∧ tetrahedron_faces ⊆ {b1} ∧ tetrahedron_faces ⊆ {b2}) :=
  sorry

end probability_beautiful_equation_l114_114781


namespace problem_statement_l114_114913

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for f

-- Theorem stating the axis of symmetry and increasing interval for the transformed function
theorem problem_statement (hf_even : ∀ x, f x = f (-x))
  (hf_increasing : ∀ x₁ x₂, 3 < x₁ → x₁ < x₂ → x₂ < 5 → f x₁ < f x₂) :
  -- For y = f(x - 1), the following holds:
  (∀ x, (f (x - 1)) = f (-(x - 1))) ∧
  (∀ x₁ x₂, 4 < x₁ → x₁ < x₂ → x₂ < 6 → f (x₁ - 1) < f (x₂ - 1)) :=
sorry

end problem_statement_l114_114913


namespace find_semi_perimeter_l114_114092

noncomputable def semi_perimeter_of_rectangle (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : ℝ :=
  (a + b) / 2

theorem find_semi_perimeter (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : semi_perimeter_of_rectangle a b h₁ h₂ = (3 / 2) * Real.sqrt 2012 :=
  sorry

end find_semi_perimeter_l114_114092


namespace find_remainder_l114_114561

theorem find_remainder (S : Finset ℕ) (h : ∀ n ∈ S, ∃ m, n^2 + 10 * n - 2010 = m^2) :
  (S.sum id) % 1000 = 304 := by
  sorry

end find_remainder_l114_114561


namespace erica_has_correct_amount_l114_114884

-- Definitions for conditions
def total_money : ℕ := 91
def sam_money : ℕ := 38

-- Definition for the question regarding Erica's money
def erica_money := total_money - sam_money

-- The theorem stating the proof problem
theorem erica_has_correct_amount : erica_money = 53 := sorry

end erica_has_correct_amount_l114_114884


namespace circle_symmetry_line_l114_114009

theorem circle_symmetry_line :
  ∃ l: ℝ → ℝ → Prop, 
    (∀ x y, l x y → x - y + 2 = 0) ∧ 
    (∀ x y, l x y ↔ (x + 2)^2 + (y - 2)^2 = 4) :=
sorry

end circle_symmetry_line_l114_114009


namespace play_role_assignments_l114_114555

def specific_role_assignments (men women remaining either_gender_roles : ℕ) : ℕ :=
  men * women * Nat.choose remaining either_gender_roles

theorem play_role_assignments :
  specific_role_assignments 6 7 11 4 = 13860 := by
  -- The given problem statement implies evaluating the specific role assignments
  sorry

end play_role_assignments_l114_114555


namespace remainder_of_2_pow_2005_mod_7_l114_114410

theorem remainder_of_2_pow_2005_mod_7 :
  2 ^ 2005 % 7 = 2 :=
sorry

end remainder_of_2_pow_2005_mod_7_l114_114410


namespace total_earnings_l114_114221

theorem total_earnings (x y : ℝ) (h1 : 20 * x * y - 18 * x * y = 120) : 
  18 * x * y + 20 * x * y + 20 * x * y = 3480 := 
by
  sorry

end total_earnings_l114_114221


namespace proof_S_squared_l114_114817

variables {a b c p S r r_a r_b r_c : ℝ}

-- Conditions
axiom cond1 : r * p = r_a * (p - a)
axiom cond2 : r * r_a = (p - b) * (p - c)
axiom cond3 : r_b * r_c = p * (p - a)
axiom heron : S^2 = p * (p - a) * (p - b) * (p - c)

-- Proof statement
theorem proof_S_squared : S^2 = r * r_a * r_b * r_c :=
by sorry

end proof_S_squared_l114_114817


namespace pizza_sales_calculation_l114_114611

def pizzas_sold_in_spring (total_sales : ℝ) (summer_sales : ℝ) (fall_percentage : ℝ) (winter_percentage : ℝ) : ℝ :=
  total_sales - summer_sales - (fall_percentage * total_sales) - (winter_percentage * total_sales)

theorem pizza_sales_calculation :
  let summer_sales := 5;
  let fall_percentage := 0.1;
  let winter_percentage := 0.2;
  ∃ (total_sales : ℝ), 0.4 * total_sales = summer_sales ∧
    pizzas_sold_in_spring total_sales summer_sales fall_percentage winter_percentage = 3.75 :=
by
  sorry

end pizza_sales_calculation_l114_114611


namespace find_four_numbers_proportion_l114_114870

theorem find_four_numbers_proportion :
  ∃ (a b c d : ℝ), 
  a + d = 14 ∧
  b + c = 11 ∧
  a^2 + b^2 + c^2 + d^2 = 221 ∧
  a * d = b * c ∧
  a = 12 ∧
  b = 8 ∧
  c = 3 ∧
  d = 2 :=
by
  sorry

end find_four_numbers_proportion_l114_114870


namespace students_per_table_correct_l114_114276

-- Define the number of tables and students
def num_tables := 34
def num_students := 204

-- Define x as the number of students per table
def students_per_table := 6

-- State the theorem
theorem students_per_table_correct : num_students / num_tables = students_per_table :=
by
  sorry

end students_per_table_correct_l114_114276


namespace speed_against_current_l114_114965

theorem speed_against_current (V_m V_c : ℝ) (h1 : V_m + V_c = 20) (h2 : V_c = 1) :
  V_m - V_c = 18 :=
by
  sorry

end speed_against_current_l114_114965


namespace range_of_m_n_l114_114222

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  m * Real.exp x + x^2 + n * x

theorem range_of_m_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0) ∧ (∀ x : ℝ, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
sorry

end range_of_m_n_l114_114222


namespace find_c_eq_neg_9_over_4_l114_114997

theorem find_c_eq_neg_9_over_4 (c x : ℚ) (h₁ : 3 * x + 5 = 1) (h₂ : c * x - 8 = -5) :
  c = -9 / 4 :=
sorry

end find_c_eq_neg_9_over_4_l114_114997


namespace intersection_point_l114_114017

-- Mathematical problem translated to Lean 4 statement

theorem intersection_point : 
  ∃ x y : ℝ, y = -3 * x + 1 ∧ y + 1 = 15 * x ∧ x = 1 / 9 ∧ y = 2 / 3 := 
by
  sorry

end intersection_point_l114_114017


namespace probability_selecting_both_types_X_distribution_correct_E_X_correct_l114_114459

section DragonBoatFestival

/-- The total number of zongzi on the plate -/
def total_zongzi : ℕ := 10

/-- The total number of red bean zongzi -/
def red_bean_zongzi : ℕ := 2

/-- The total number of plain zongzi -/
def plain_zongzi : ℕ := 8

/-- The number of zongzi to select -/
def zongzi_to_select : ℕ := 3

/-- Probability of selecting at least one red bean zongzi and at least one plain zongzi -/
def probability_selecting_both : ℚ := 8 / 15

/-- Distribution of the number of red bean zongzi selected (X) -/
def X_distribution : ℕ → ℚ
| 0 => 7 / 15
| 1 => 7 / 15
| 2 => 1 / 15
| _ => 0

/-- Mathematical expectation of the number of red bean zongzi selected (E(X)) -/
def E_X : ℚ := 3 / 5

/-- Theorem stating the probability of selecting both types of zongzi -/
theorem probability_selecting_both_types :
  let p := probability_selecting_both
  p = 8 / 15 :=
by
  let p := probability_selecting_both
  sorry

/-- Theorem stating the probability distribution of the number of red bean zongzi selected -/
theorem X_distribution_correct :
  (X_distribution 0 = 7 / 15) ∧
  (X_distribution 1 = 7 / 15) ∧
  (X_distribution 2 = 1 / 15) :=
by
  sorry

/-- Theorem stating the mathematical expectation of the number of red bean zongzi selected -/
theorem E_X_correct :
  let E := E_X
  E = 3 / 5 :=
by
  let E := E_X
  sorry

end DragonBoatFestival

end probability_selecting_both_types_X_distribution_correct_E_X_correct_l114_114459


namespace gcd_lcm_sum_l114_114610

variable (a b : ℕ)

-- Definition for gcd
def gcdOf (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition for lcm
def lcmOf (a b : ℕ) : ℕ := Nat.lcm a b

-- Statement of the problem
theorem gcd_lcm_sum (h1 : a = 8) (h2 : b = 12) : gcdOf a b + lcmOf a b = 28 := by
  sorry

end gcd_lcm_sum_l114_114610


namespace call_center_agents_ratio_l114_114966

theorem call_center_agents_ratio
  (a b : ℕ) -- Number of agents in teams A and B
  (x : ℝ) -- Calls each member of team B processes
  (h1 : (a : ℝ) / (b : ℝ) = 5 / 8)
  (h2 : b * x * 4 / 7 + a * 6 / 5 * x * 3 / 7 = b * x + a * 6 / 5 * x) :
  (a : ℝ) / (b : ℝ) = 5 / 8 :=
by
  sorry

end call_center_agents_ratio_l114_114966


namespace fox_initial_coins_l114_114290

theorem fox_initial_coins :
  ∃ (x : ℕ), ∀ (c1 c2 c3 : ℕ),
    c1 = 3 * x - 50 ∧
    c2 = 3 * c1 - 50 ∧
    c3 = 3 * c2 - 50 ∧
    3 * c3 - 50 = 20 →
    x = 25 :=
by
  sorry

end fox_initial_coins_l114_114290


namespace solve_for_k_l114_114199

-- Define the hypotheses as Lean statements
theorem solve_for_k (x k : ℝ) (h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
by {
  sorry
}

end solve_for_k_l114_114199


namespace new_batting_average_l114_114492

def initial_runs (A : ℕ) := 16 * A
def additional_runs := 85
def increased_average := 3
def runs_in_5_innings := 100 + 120 + 45 + 75 + 65
def total_runs_17_innings (A : ℕ) := 17 * (A + increased_average)
def A : ℕ := 34
def total_runs_22_innings := total_runs_17_innings A + runs_in_5_innings
def number_of_innings := 22
def new_average := total_runs_22_innings / number_of_innings

theorem new_batting_average : new_average = 47 :=
by sorry

end new_batting_average_l114_114492


namespace arrange_numbers_l114_114008

variable {a : ℝ}

theorem arrange_numbers (h1 : -1 < a) (h2 : a < 0) : (1 / a < a) ∧ (a < a ^ 2) ∧ (a ^ 2 < |a|) :=
by 
  sorry

end arrange_numbers_l114_114008


namespace ratio_of_sopranos_to_altos_l114_114684

theorem ratio_of_sopranos_to_altos (S A : ℕ) :
  (10 = 5 * S) ∧ (15 = 5 * A) → (S : ℚ) / (A : ℚ) = 2 / 3 :=
by sorry

end ratio_of_sopranos_to_altos_l114_114684


namespace savings_percentage_correct_l114_114638

def coat_price : ℝ := 120
def hat_price : ℝ := 30
def gloves_price : ℝ := 50

def coat_discount : ℝ := 0.20
def hat_discount : ℝ := 0.40
def gloves_discount : ℝ := 0.30

def original_total : ℝ := coat_price + hat_price + gloves_price
def coat_savings : ℝ := coat_price * coat_discount
def hat_savings : ℝ := hat_price * hat_discount
def gloves_savings : ℝ := gloves_price * gloves_discount
def total_savings : ℝ := coat_savings + hat_savings + gloves_savings

theorem savings_percentage_correct :
  (total_savings / original_total) * 100 = 25.5 := by
  sorry

end savings_percentage_correct_l114_114638


namespace two_people_lying_l114_114190

def is_lying (A B C D : Prop) : Prop :=
  (A ↔ ¬B) ∧ (B ↔ ¬C) ∧ (C ↔ ¬B) ∧ (D ↔ ¬A)

theorem two_people_lying (A B C D : Prop) (LA LB LC LD : Prop) :
  is_lying A B C D → (LA → ¬A) → (LB → ¬B) → (LC → ¬C) → (LD → ¬D) → (LA ∧ LC ∧ ¬LB ∧ ¬LD) :=
by
  sorry

end two_people_lying_l114_114190


namespace functional_equation_solution_l114_114172

noncomputable def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * x + f x * f y) = x * f (x + y)

theorem functional_equation_solution (f : ℝ → ℝ) :
  func_equation f →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end functional_equation_solution_l114_114172


namespace quadratic_roots_distinct_real_l114_114528

theorem quadratic_roots_distinct_real (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = 0)
    (Δ : ℝ := b^2 - 4 * a * c) (hΔ : Δ > 0) :
    (∀ r1 r2 : ℝ, r1 ≠ r2) :=
by
  sorry

end quadratic_roots_distinct_real_l114_114528


namespace rectangles_greater_than_one_area_l114_114065

theorem rectangles_greater_than_one_area (n : ℕ) (H : n = 5) : ∃ r, r = 84 :=
by
  sorry

end rectangles_greater_than_one_area_l114_114065


namespace painters_work_l114_114933

theorem painters_work (w1 w2 : ℕ) (d1 d2 : ℚ) (C : ℚ) (h1 : w1 * d1 = C) (h2 : w2 * d2 = C) (p : w1 = 5) (t : d1 = 1.6) (a : w2 = 4) : d2 = 2 := 
by
  sorry

end painters_work_l114_114933


namespace spoon_less_than_fork_l114_114960

-- Define the initial price of spoon and fork in kopecks
def initial_price (x : ℕ) : Prop :=
  x > 100 -- ensuring the spoon's sale price remains positive

-- Define the sale price of the spoon
def spoon_sale_price (x : ℕ) : ℕ :=
  x - 100

-- Define the sale price of the fork
def fork_sale_price (x : ℕ) : ℕ :=
  x / 10

-- Prove that the spoon's sale price can be less than the fork's sale price
theorem spoon_less_than_fork (x : ℕ) (h : initial_price x) : 
  spoon_sale_price x < fork_sale_price x :=
by
  sorry

end spoon_less_than_fork_l114_114960


namespace arcsin_one_half_eq_pi_six_l114_114341

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l114_114341


namespace range_of_f_l114_114460

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - 2 * x)

theorem range_of_f : ∀ y, (∃ x, x ≤ (1 / 2) ∧ f x = y) ↔ y ∈ Set.Iic 1 := by
  sorry

end range_of_f_l114_114460


namespace tetrahedron_volume_l114_114322

noncomputable def volume_of_tetrahedron (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABC_ABD : ℝ) : ℝ :=
  (1/3) * area_ABC * area_ABD * (Real.sin angle_ABC_ABD) * (AB / (Real.sqrt 2))

theorem tetrahedron_volume :
  let AB := 5 -- edge AB length in cm
  let area_ABC := 18 -- area of face ABC in cm^2
  let area_ABD := 24 -- area of face ABD in cm^2
  let angle_ABC_ABD := Real.pi / 4 -- 45 degrees in radians
  volume_of_tetrahedron AB area_ABC area_ABD angle_ABC_ABD = 43.2 :=
by
  sorry

end tetrahedron_volume_l114_114322


namespace y_minus_x_eq_seven_point_five_l114_114602

theorem y_minus_x_eq_seven_point_five (x y : ℚ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) :
  y - x = 7.5 :=
by sorry

end y_minus_x_eq_seven_point_five_l114_114602


namespace math_problem_l114_114985

theorem math_problem (n d : ℕ) (h1 : 0 < n) (h2 : d < 10)
  (h3 : 3 * n^2 + 2 * n + d = 263)
  (h4 : 3 * n^2 + 2 * n + 4 = 396 + 7 * d) :
  n + d = 11 :=
by {
  sorry
}

end math_problem_l114_114985


namespace average_children_in_families_with_children_l114_114761

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l114_114761


namespace system_solution_l114_114202

theorem system_solution (x y z : ℝ) :
    x + y + z = 2 ∧ 
    x^2 + y^2 + z^2 = 26 ∧
    x^3 + y^3 + z^3 = 38 →
    (x = 1 ∧ y = 4 ∧ z = -3) ∨
    (x = 1 ∧ y = -3 ∧ z = 4) ∨
    (x = 4 ∧ y = 1 ∧ z = -3) ∨
    (x = 4 ∧ y = -3 ∧ z = 1) ∨
    (x = -3 ∧ y = 1 ∧ z = 4) ∨
    (x = -3 ∧ y = 4 ∧ z = 1) := by
  sorry

end system_solution_l114_114202


namespace trigonometric_expression_in_third_quadrant_l114_114229

theorem trigonometric_expression_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α < 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.tan α > 0) : 
  ¬ (Real.tan α - Real.sin α < 0) :=
sorry

end trigonometric_expression_in_third_quadrant_l114_114229


namespace union_M_N_l114_114964

-- Definitions based on conditions
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 2 * a}

-- The theorem to be proven
theorem union_M_N : M ∪ N = {0, 1, 2, 4} := by
  sorry

end union_M_N_l114_114964


namespace value_of_expression_l114_114379

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by
sorry

end value_of_expression_l114_114379


namespace exam_total_questions_l114_114680

/-- 
In an examination, a student scores 4 marks for every correct answer 
and loses 1 mark for every wrong answer. The student secures 140 marks 
in total. Given that the student got 40 questions correct, 
prove that the student attempted a total of 60 questions. 
-/
theorem exam_total_questions (C W T : ℕ) 
  (score_correct : C = 40)
  (total_score : 4 * C - W = 140)
  (total_questions : T = C + W) : 
  T = 60 := 
by 
  -- Proof omitted
  sorry

end exam_total_questions_l114_114680


namespace find_cost_price_l114_114656

theorem find_cost_price (C : ℝ) (h1 : C * 1.05 = C + 0.05 * C)
  (h2 : 0.95 * C = C - 0.05 * C)
  (h3 : 1.05 * C - 4 = 1.045 * C) :
  C = 800 := sorry

end find_cost_price_l114_114656


namespace find_n_l114_114752

variable {a : ℕ → ℝ} (h1 : a 4 = 7) (h2 : a 3 + a 6 = 16)

theorem find_n (n : ℕ) (h3 : a n = 31) : n = 16 := by
  sorry

end find_n_l114_114752


namespace remaining_fish_l114_114439

theorem remaining_fish (initial_fish : ℝ) (moved_fish : ℝ) (remaining_fish : ℝ) : initial_fish = 212.0 → moved_fish = 68.0 → remaining_fish = 144.0 → initial_fish - moved_fish = remaining_fish := by sorry

end remaining_fish_l114_114439


namespace max_area_rectangle_l114_114678

theorem max_area_rectangle (perimeter : ℕ) (a b : ℕ) (h1 : perimeter = 30) 
  (h2 : b = a + 3) : a * b = 54 :=
by
  sorry

end max_area_rectangle_l114_114678


namespace ratio_of_60_to_12_l114_114911

theorem ratio_of_60_to_12 : 60 / 12 = 5 := 
by 
  sorry

end ratio_of_60_to_12_l114_114911


namespace calc_hash_2_5_3_l114_114779

def operation_hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem calc_hash_2_5_3 : operation_hash 2 5 3 = 1 := by
  sorry

end calc_hash_2_5_3_l114_114779


namespace find_number_l114_114961

theorem find_number (y : ℝ) (h : 0.25 * 820 = 0.15 * y - 20) : y = 1500 :=
by
  sorry

end find_number_l114_114961


namespace solve_equations_l114_114107

-- Prove that the solutions to the given equations are correct.
theorem solve_equations :
  (∀ x : ℝ, (x * (x - 4) = 2 * x - 8) ↔ (x = 4 ∨ x = 2)) ∧
  (∀ x : ℝ, ((2 * x) / (2 * x - 3) - (4 / (2 * x + 3)) = 1) ↔ (x = 10.5)) :=
by
  sorry

end solve_equations_l114_114107


namespace min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l114_114054

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  sorry

theorem min_value_x_plus_reciprocal_equality_at_one : (1 : ℝ) + 1 / 1 = 2 :=
by
  norm_num

end min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l114_114054


namespace shared_vertex_angle_of_triangle_and_square_l114_114197

theorem shared_vertex_angle_of_triangle_and_square (α β γ δ ε ζ η θ : ℝ) :
  (α = 60 ∧ β = 60 ∧ γ = 60 ∧ δ = 90 ∧ ε = 90 ∧ ζ = 90 ∧ η = 90 ∧ θ = 90) →
  θ = 90 :=
by
  sorry

end shared_vertex_angle_of_triangle_and_square_l114_114197


namespace not_q_is_false_l114_114075

variable (n : ℤ)

-- Definition of the propositions
def p (n : ℤ) : Prop := 2 * n - 1 % 2 = 1 -- 2n - 1 is odd
def q (n : ℤ) : Prop := (2 * n + 1) % 2 = 0 -- 2n + 1 is even

-- Proof statement: Not q is false, meaning q is false
theorem not_q_is_false (n : ℤ) : ¬ q n = False := sorry

end not_q_is_false_l114_114075


namespace non_working_games_count_l114_114600

def total_games : ℕ := 16
def price_each : ℕ := 7
def total_earnings : ℕ := 56

def working_games : ℕ := total_earnings / price_each
def non_working_games : ℕ := total_games - working_games

theorem non_working_games_count : non_working_games = 8 := by
  sorry

end non_working_games_count_l114_114600


namespace weights_problem_l114_114851

theorem weights_problem
  (a b c d : ℕ)
  (h1 : a + b = 280)
  (h2 : b + c = 255)
  (h3 : c + d = 290) 
  : a + d = 315 := 
  sorry

end weights_problem_l114_114851


namespace gcd_7920_14553_l114_114355

theorem gcd_7920_14553 : Int.gcd 7920 14553 = 11 := by
  sorry

end gcd_7920_14553_l114_114355


namespace bakery_combinations_l114_114553

theorem bakery_combinations 
  (total_breads : ℕ) (bread_types : Finset ℕ) (purchases : Finset ℕ)
  (h_total : total_breads = 8)
  (h_bread_types : bread_types.card = 5)
  (h_purchases : purchases.card = 2) : 
  ∃ (combinations : ℕ), combinations = 70 := 
sorry

end bakery_combinations_l114_114553


namespace percentage_apples_sold_l114_114819

theorem percentage_apples_sold (A P : ℕ) (h1 : A = 600) (h2 : A * (100 - P) / 100 = 420) : P = 30 := 
by {
  sorry
}

end percentage_apples_sold_l114_114819


namespace ellipse_foci_distance_l114_114945

noncomputable def distance_between_foci : ℝ :=
  let a := 20
  let b := 10
  2 * Real.sqrt (a ^ 2 - b ^ 2)

theorem ellipse_foci_distance : distance_between_foci = 20 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l114_114945


namespace total_wheels_of_four_wheelers_l114_114132

-- Define the number of four-wheelers and wheels per four-wheeler
def number_of_four_wheelers : ℕ := 13
def wheels_per_four_wheeler : ℕ := 4

-- Prove the total number of wheels for the 13 four-wheelers
theorem total_wheels_of_four_wheelers : (number_of_four_wheelers * wheels_per_four_wheeler) = 52 :=
by sorry

end total_wheels_of_four_wheelers_l114_114132


namespace initial_marbles_l114_114688

variable (C_initial : ℕ)
variable (marbles_given : ℕ := 42)
variable (marbles_left : ℕ := 5)

theorem initial_marbles :
  C_initial = marbles_given + marbles_left :=
sorry

end initial_marbles_l114_114688


namespace bobby_position_after_100_turns_l114_114057

def movement_pattern (start_pos : ℤ × ℤ) (n : ℕ) : (ℤ × ℤ) :=
  let x := start_pos.1 - ((2 * (n / 4 + 1) + 3 * (n / 4)) * ((n + 1) / 4))
  let y := start_pos.2 + ((2 * (n / 4 + 1) + 2 * (n / 4)) * ((n + 1) / 4))
  if n % 4 == 0 then (x, y)
  else if n % 4 == 1 then (x, y + 2 * ((n + 3) / 4) + 1)
  else if n % 4 == 2 then (x - 3 * ((n + 5) / 4), y + 2 * ((n + 3) / 4) + 1)
  else (x - 3 * ((n + 5) / 4) + 3, y + 2 * ((n + 3) / 4) - 2)

theorem bobby_position_after_100_turns :
  movement_pattern (10, -10) 100 = (-667, 640) :=
sorry

end bobby_position_after_100_turns_l114_114057


namespace find_x_l114_114895

theorem find_x (x : ℤ) (h : 3 * x = (26 - x) + 10) : x = 9 :=
by
  -- proof steps would be provided here
  sorry

end find_x_l114_114895


namespace intersection_of_M_and_complement_N_l114_114682

def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | 2 * x < 2 }
def complement_N : Set ℝ := { x | x ≥ 1 }

theorem intersection_of_M_and_complement_N : M ∩ complement_N = { x | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_M_and_complement_N_l114_114682


namespace time_to_hit_ground_l114_114362

theorem time_to_hit_ground : ∃ t : ℝ, 
  (y = -4.9 * t^2 + 7.2 * t + 8) → (y - (-0.6 * t) * t = 0) → t = 223/110 :=
by
  sorry

end time_to_hit_ground_l114_114362


namespace perpendicular_tangent_l114_114756

noncomputable def f (x a : ℝ) := (x + a) * Real.exp x -- Defines the function

theorem perpendicular_tangent (a : ℝ) : 
  ∀ (tangent_slope perpendicular_slope : ℝ), 
  (tangent_slope = 1) → 
  (perpendicular_slope = -1) →
  tangent_slope = Real.exp 0 * (a + 1) →
  tangent_slope + perpendicular_slope = 0 → 
  a = 0 := by 
  intros tangent_slope perpendicular_slope htangent hperpendicular hderiv hperpendicular_slope
  sorry

end perpendicular_tangent_l114_114756


namespace area_correct_l114_114126

open BigOperators

def Rectangle (PQ RS : ℕ) := PQ * RS

def PointOnSegment (a b : ℕ) (ratio : ℚ) : ℚ :=
ratio * (b - a)

def area_of_PTUS : ℚ :=
Rectangle 10 6 - (0.5 * 6 * (10 / 3) + 0.5 * 10 * 6)

theorem area_correct :
  area_of_PTUS = 20 := by
  sorry

end area_correct_l114_114126


namespace triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l114_114428

noncomputable def sum_sine_3A_3B_3C (A B C : ℝ) : ℝ :=
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C)

theorem triangle_inequality_sine_three_times {A B C : ℝ} (h : A + B + C = Real.pi) (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) : 
  (-2 : ℝ) ≤ sum_sine_3A_3B_3C A B C ∧ sum_sine_3A_3B_3C A B C ≤ (3 * Real.sqrt 3 / 2) :=
by
  sorry

theorem equality_sine_three_times_lower_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = 0) (h2: B = Real.pi / 2) (h3: C = Real.pi / 2) :
  sum_sine_3A_3B_3C A B C = -2 :=
by
  sorry

theorem equality_sine_three_times_upper_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = Real.pi / 3) (h2: B = Real.pi / 3) (h3: C = Real.pi / 3) :
  sum_sine_3A_3B_3C A B C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l114_114428


namespace max_leap_years_in_200_years_l114_114457

-- Definitions based on conditions
def leap_year_occurrence (years : ℕ) : ℕ :=
  years / 4

-- Define the problem statement based on the given conditions and required proof
theorem max_leap_years_in_200_years : leap_year_occurrence 200 = 50 := 
by
  sorry

end max_leap_years_in_200_years_l114_114457


namespace fraction_addition_l114_114156

/--
The value of 2/5 + 1/3 is 11/15.
-/
theorem fraction_addition :
  (2 / 5 : ℚ) + (1 / 3) = 11 / 15 := 
sorry

end fraction_addition_l114_114156


namespace potion_kit_cost_is_18_l114_114370

def price_spellbook : ℕ := 5
def count_spellbooks : ℕ := 5
def price_owl : ℕ := 28
def count_potion_kits : ℕ := 3
def payment_total_silver : ℕ := 537
def silver_per_gold : ℕ := 9

def cost_each_potion_kit_in_silver (payment_total_silver : ℕ)
                                   (price_spellbook : ℕ)
                                   (count_spellbooks : ℕ)
                                   (price_owl : ℕ)
                                   (count_potion_kits : ℕ)
                                   (silver_per_gold : ℕ) : ℕ :=
  let total_gold := payment_total_silver / silver_per_gold
  let cost_spellbooks := count_spellbooks * price_spellbook
  let cost_remaining_gold := total_gold - cost_spellbooks - price_owl
  let cost_each_potion_kit_gold := cost_remaining_gold / count_potion_kits
  cost_each_potion_kit_gold * silver_per_gold

theorem potion_kit_cost_is_18 :
  cost_each_potion_kit_in_silver payment_total_silver
                                 price_spellbook
                                 count_spellbooks
                                 price_owl
                                 count_potion_kits
                                 silver_per_gold = 18 :=
by sorry

end potion_kit_cost_is_18_l114_114370


namespace average_eq_16_l114_114892

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 12

theorem average_eq_16 (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : (x + y) / 2 = 16 := by
  sorry

end average_eq_16_l114_114892


namespace inclination_angle_l114_114546

theorem inclination_angle (θ : ℝ) (h : 0 ≤ θ ∧ θ < 180) :
  (∀ x y : ℝ, x - y + 3 = 0 → θ = 45) :=
sorry

end inclination_angle_l114_114546


namespace find_power_of_4_l114_114838

theorem find_power_of_4 (x : Nat) : 
  (2 * x + 5 + 2 = 29) -> 
  (x = 11) :=
by
  sorry

end find_power_of_4_l114_114838


namespace theme_park_ratio_l114_114599

theorem theme_park_ratio (a c : ℕ) (h_cost_adult : 20 * a + 15 * c = 1600) (h_eq_ratio : a * 28 = c * 59) :
  a / c = 59 / 28 :=
by
  /-
  Proof steps would go here.
  -/
  sorry

end theme_park_ratio_l114_114599


namespace range_of_a_l114_114195

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x ≤ f a (x + ε))) → a ≤ -1/4 :=
sorry

end range_of_a_l114_114195


namespace loss_percentage_remaining_stock_l114_114769

noncomputable def total_worth : ℝ := 9999.999999999998
def overall_loss : ℝ := 200
def profit_percentage_20 : ℝ := 0.1
def sold_20_percentage : ℝ := 0.2
def remaining_percentage : ℝ := 0.8

theorem loss_percentage_remaining_stock :
  ∃ L : ℝ, 0.8 * total_worth * (L / 100) - 0.02 * total_worth = overall_loss ∧ L = 5 :=
by sorry

end loss_percentage_remaining_stock_l114_114769


namespace problem_statement_l114_114424

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ ⦃x y⦄, x > 4 → y > x → f y < f x)
                          (h2 : ∀ x, f (4 + x) = f (4 - x)) : f 3 > f 6 :=
by 
  sorry

end problem_statement_l114_114424


namespace Gary_final_amount_l114_114465

theorem Gary_final_amount
(initial_amount dollars_snake dollars_hamster dollars_supplies : ℝ)
(h1 : initial_amount = 73.25)
(h2 : dollars_snake = 55.50)
(h3 : dollars_hamster = 25.75)
(h4 : dollars_supplies = 12.40) :
  initial_amount + dollars_snake - dollars_hamster - dollars_supplies = 90.60 :=
by
  sorry

end Gary_final_amount_l114_114465


namespace parabola_equations_l114_114042

theorem parabola_equations (x y : ℝ) (h₁ : (0, 0) = (0, 0)) (h₂ : (-2, 3) = (-2, 3)) :
  (x^2 = 4 / 3 * y) ∨ (y^2 = - 9 / 2 * x) :=
sorry

end parabola_equations_l114_114042


namespace triangle_area_l114_114543

theorem triangle_area (a b c : ℝ) (K : ℝ) (m n p : ℕ) (h1 : a = 10) (h2 : b = 12) (h3 : c = 15)
  (h4 : K = 240 * Real.sqrt 7 / 7)
  (h5 : Int.gcd m p = 1) -- m and p are relatively prime
  (h6 : n ≠ 1 ∧ ¬ (∃ x, x^2 ∣ n ∧ x > 1)) -- n is not divisible by the square of any prime
  : m + n + p = 254 := sorry

end triangle_area_l114_114543


namespace no_real_roots_of_quadratic_l114_114700

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem no_real_roots_of_quadratic :
  let a := 2
  let b := -5
  let c := 6
  discriminant a b c < 0 → ¬∃ x : ℝ, 2 * x ^ 2 - 5 * x + 6 = 0 :=
by {
  -- Proof skipped
  sorry
}

end no_real_roots_of_quadratic_l114_114700


namespace transform_point_c_l114_114184

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_diag (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem transform_point_c :
  let C := (3, 2)
  let C' := reflect_x C
  let C'' := reflect_y C'
  let C''' := reflect_diag C''
  C''' = (-2, -3) :=
by
  sorry

end transform_point_c_l114_114184


namespace range_of_f_l114_114134

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f : Set.Icc (-(3 / 2)) 3 = Set.image f (Set.Icc 0 (Real.pi / 2)) :=
  sorry

end range_of_f_l114_114134


namespace distinct_left_views_l114_114335

/-- Consider 10 small cubes each having dimension 1 cm × 1 cm × 1 cm.
    Each pair of adjacent cubes shares at least one edge (1 cm) or one face (1 cm × 1 cm).
    The cubes must not be suspended in the air and each cube's edges should be either
    perpendicular or parallel to the horizontal lines. Prove that the number of distinct
    left views of any arrangement of these 10 cubes is 16. -/
theorem distinct_left_views (cube_count : ℕ) (dimensions : ℝ) 
  (shared_edge : (ℝ × ℝ) → Prop) (no_suspension : Prop) (alignment : Prop) :
  cube_count = 10 →
  dimensions = 1 →
  (∀ x y, shared_edge (x, y) ↔ x = y ∨ x - y = 1) →
  no_suspension →
  alignment →
  distinct_left_views_count = 16 :=
by
  sorry

end distinct_left_views_l114_114335


namespace largest_possible_a_l114_114883

theorem largest_possible_a (a b c e : ℕ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 5 * e) (h4 : e < 100) : a ≤ 2961 :=
by
  sorry

end largest_possible_a_l114_114883


namespace equation_of_line_l114_114306

-- Define the points P and Q
def P : (ℝ × ℝ) := (3, 2)
def Q : (ℝ × ℝ) := (4, 7)

-- Prove that the equation of the line passing through points P and Q is 5x - y - 13 = 0
theorem equation_of_line : ∃ (A B C : ℝ), A = 5 ∧ B = -1 ∧ C = -13 ∧
  ∀ x y : ℝ, (y - 2) / (7 - 2) = (x - 3) / (4 - 3) → 5 * x - y - 13 = 0 :=
by
  sorry

end equation_of_line_l114_114306


namespace fixed_fee_1430_l114_114541

def fixed_monthly_fee (f p : ℝ) : Prop :=
  f + p = 20.60 ∧ f + 3 * p = 33.20

theorem fixed_fee_1430 (f p: ℝ) (h : fixed_monthly_fee f p) : 
  f = 14.30 :=
by
  sorry

end fixed_fee_1430_l114_114541


namespace equal_area_condition_l114_114801

variable {θ : ℝ} (h1 : 0 < θ) (h2 : θ < π / 2)

theorem equal_area_condition : 2 * θ = (Real.tan θ) * (Real.tan (2 * θ)) :=
by {
  sorry
}

end equal_area_condition_l114_114801


namespace customer_payment_eq_3000_l114_114277

theorem customer_payment_eq_3000 (cost_price : ℕ) (markup_percentage : ℕ) (payment : ℕ)
  (h1 : cost_price = 2500)
  (h2 : markup_percentage = 20)
  (h3 : payment = cost_price + (markup_percentage * cost_price / 100)) :
  payment = 3000 :=
by
  sorry

end customer_payment_eq_3000_l114_114277


namespace total_walnut_trees_in_park_l114_114723

-- Define initial number of walnut trees in the park
def initial_walnut_trees : ℕ := 22

-- Define number of walnut trees planted by workers
def planted_walnut_trees : ℕ := 33

-- Prove the total number of walnut trees in the park
theorem total_walnut_trees_in_park : initial_walnut_trees + planted_walnut_trees = 55 := by
  sorry

end total_walnut_trees_in_park_l114_114723


namespace scientific_notation_example_l114_114495

theorem scientific_notation_example : (8485000 : ℝ) = 8.485 * 10 ^ 6 := 
by 
  sorry

end scientific_notation_example_l114_114495


namespace system_solution_l114_114020

theorem system_solution (m n : ℝ) (h1 : -2 * m * 5 + 5 * 2 = 15) (h2 : 5 + 7 * n * 2 = 14) :
  ∃ (a b : ℝ), (-2 * m * (a + b) + 5 * (a - 2 * b) = 15) ∧ ((a + b) + 7 * n * (a - 2 * b) = 14) ∧ (a = 4) ∧ (b = 1) :=
by
  -- The proof is intentionally omitted
  sorry

end system_solution_l114_114020


namespace middle_of_7_consecutive_nat_sum_63_l114_114823

theorem middle_of_7_consecutive_nat_sum_63 (x : ℕ) (h : 7 * x = 63) : x = 9 :=
by
  sorry

end middle_of_7_consecutive_nat_sum_63_l114_114823


namespace sum_incorrect_correct_l114_114470

theorem sum_incorrect_correct (x : ℕ) (h : x + 9 = 39) :
  ((x - 5 + 14) + (x * 5 + 14)) = 203 :=
sorry

end sum_incorrect_correct_l114_114470


namespace total_cookies_l114_114038

theorem total_cookies
  (num_bags : ℕ)
  (cookies_per_bag : ℕ)
  (h_num_bags : num_bags = 286)
  (h_cookies_per_bag : cookies_per_bag = 452) :
  num_bags * cookies_per_bag = 129272 :=
by
  sorry

end total_cookies_l114_114038


namespace Tony_temp_above_fever_threshold_l114_114687

def normal_temp : ℕ := 95
def illness_A : ℕ := 10
def illness_B : ℕ := 4
def illness_C : Int := -2
def fever_threshold : ℕ := 100

theorem Tony_temp_above_fever_threshold :
  let T := normal_temp + illness_A + illness_B + illness_C
  T = 107 ∧ (T - fever_threshold) = 7 := by
  -- conditions
  let t_0 := normal_temp
  let T_A := illness_A
  let T_B := illness_B
  let T_C := illness_C
  let F := fever_threshold
  -- calculations
  let T := t_0 + T_A + T_B + T_C
  show T = 107 ∧ (T - F) = 7
  sorry

end Tony_temp_above_fever_threshold_l114_114687


namespace quadratic_inequality_solution_set_l114_114941

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2 * x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end quadratic_inequality_solution_set_l114_114941


namespace distance_from_mo_l114_114275

-- Definitions based on conditions
-- 1. Grid squares have side length 1 cm.
-- 2. Shape shaded gray on the grid.
-- 3. The total shaded area needs to be divided into two equal parts.
-- 4. The line to be drawn is parallel to line MO.

noncomputable def grid_side_length : ℝ := 1.0
noncomputable def shaded_area : ℝ := 10.0
noncomputable def line_mo_distance (d : ℝ) : Prop := 
  ∃ parallel_line_distance, parallel_line_distance = d ∧ 
    ∃ equal_area, 2 * equal_area = shaded_area ∧ equal_area = 5.0

-- Theorem: The parallel line should be drawn at 2.6 cm 
theorem distance_from_mo (d : ℝ) : 
  d = 2.6 ↔ line_mo_distance d := 
by
  sorry

end distance_from_mo_l114_114275


namespace seymour_flats_of_roses_l114_114235

-- Definitions used in conditions
def flats_of_petunias := 4
def petunias_per_flat := 8
def venus_flytraps := 2
def fertilizer_per_petunia := 8
def fertilizer_per_rose := 3
def fertilizer_per_venus_flytrap := 2
def total_fertilizer := 314

-- Compute the total fertilizer for petunias and Venus flytraps
def total_fertilizer_petunias := flats_of_petunias * petunias_per_flat * fertilizer_per_petunia
def total_fertilizer_venus_flytraps := venus_flytraps * fertilizer_per_venus_flytrap

-- Remaining fertilizer for roses
def remaining_fertilizer_for_roses := total_fertilizer - total_fertilizer_petunias - total_fertilizer_venus_flytraps

-- Define roses per flat and the fertilizer used per flat of roses
def roses_per_flat := 6
def fertilizer_per_flat_of_roses := roses_per_flat * fertilizer_per_rose

-- The number of flats of roses
def flats_of_roses := remaining_fertilizer_for_roses / fertilizer_per_flat_of_roses

-- The proof problem statement
theorem seymour_flats_of_roses : flats_of_roses = 3 := by
  sorry

end seymour_flats_of_roses_l114_114235


namespace part_a_part_b_part_c_l114_114683

-- Define the conditions
inductive Color
| blue
| red
| green
| yellow

-- Each square can be painted in one of the colors: blue, red, or green.
def square_colors : List Color := [Color.blue, Color.red, Color.green]

-- Each triangle can be painted in one of the colors: blue, red, or yellow.
def triangle_colors : List Color := [Color.blue, Color.red, Color.yellow]

-- Condition that polygons with a common side cannot share the same color
def different_color (c1 c2 : Color) : Prop := c1 ≠ c2

-- Part (a)
theorem part_a : ∃ n : Nat, n = 7 := sorry

-- Part (b)
theorem part_b : ∃ n : Nat, n = 43 := sorry

-- Part (c)
theorem part_c : ∃ n : Nat, n = 667 := sorry

end part_a_part_b_part_c_l114_114683


namespace shifted_graph_sum_l114_114303

theorem shifted_graph_sum :
  let f (x : ℝ) := 3*x^2 - 2*x + 8
  let g (x : ℝ) := f (x - 6)
  let a := 3
  let b := -38
  let c := 128
  a + b + c = 93 :=
by
  sorry

end shifted_graph_sum_l114_114303


namespace expression_evaluates_to_4_l114_114250

theorem expression_evaluates_to_4 :
  2 * Real.cos (Real.pi / 6) + (- 1 / 2 : ℝ)⁻¹ + |Real.sqrt 3 - 2| + (2 * Real.sqrt (9 / 4))^0 + Real.sqrt 9 = 4 := 
by
  sorry

end expression_evaluates_to_4_l114_114250


namespace mild_numbers_with_mild_squares_count_l114_114071

def is_mild (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 3, d = 0 ∨ d = 1

theorem mild_numbers_with_mild_squares_count :
  ∃ count : ℕ, count = 7 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → is_mild n → is_mild (n * n)) → count = 7 := by
  sorry

end mild_numbers_with_mild_squares_count_l114_114071


namespace SmartMart_science_kits_l114_114498

theorem SmartMart_science_kits (sc pz : ℕ) (h1 : pz = sc - 9) (h2 : pz = 36) : sc = 45 := by
  sorry

end SmartMart_science_kits_l114_114498


namespace log_graph_passes_fixed_point_l114_114239

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem log_graph_passes_fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  log_a a (-1 + 2) = 0 :=
by
  sorry

end log_graph_passes_fixed_point_l114_114239


namespace fergus_entry_exit_l114_114397

theorem fergus_entry_exit (n : ℕ) (hn : n = 8) : 
  n * (n - 1) = 56 := 
by
  sorry

end fergus_entry_exit_l114_114397


namespace probability_60_or_more_points_l114_114905

theorem probability_60_or_more_points :
  let five_choose k := Nat.choose 5 k
  let prob_correct (k : Nat) := (five_choose k) * (1 / 2)^5
  let prob_at_least_3_correct := prob_correct 3 + prob_correct 4 + prob_correct 5
  prob_at_least_3_correct = 1 / 2 := 
sorry

end probability_60_or_more_points_l114_114905


namespace fibonacci_coprime_l114_114720

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_coprime (n : ℕ) (hn : n ≥ 1) :
  Nat.gcd (fibonacci n) (fibonacci (n - 1)) = 1 := by
  sorry

end fibonacci_coprime_l114_114720


namespace solution_l114_114321

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ),
    x - y = 1 ∧
    x^3 - y^3 = 2 ∧
    x^4 + y^4 = 23 / 9 ∧
    x^5 - y^5 = 29 / 9

theorem solution : problem_statement := sorry

end solution_l114_114321


namespace molecular_weight_correct_l114_114747

noncomputable def molecular_weight : ℝ := 
  let N_count := 2
  let H_count := 6
  let Br_count := 1
  let O_count := 1
  let C_count := 3
  let N_weight := 14.01
  let H_weight := 1.01
  let Br_weight := 79.90
  let O_weight := 16.00
  let C_weight := 12.01
  N_count * N_weight + 
  H_count * H_weight + 
  Br_count * Br_weight + 
  O_count * O_weight +
  C_count * C_weight

theorem molecular_weight_correct :
  molecular_weight = 166.01 := 
by
  sorry

end molecular_weight_correct_l114_114747


namespace binomial_square_b_value_l114_114421

theorem binomial_square_b_value (b : ℝ) (h : ∃ c : ℝ, (9 * x^2 + 24 * x + b) = (3 * x + c) ^ 2) : b = 16 :=
sorry

end binomial_square_b_value_l114_114421


namespace boards_nailing_l114_114284

variables {x y a b : ℕ} 

theorem boards_nailing :
  (2 * x + 3 * y = 87) ∧
  (3 * a + 5 * b = 94) →
  (x + y = 30) ∧ (a + b = 30) :=
by
  sorry

end boards_nailing_l114_114284


namespace evaluate_nested_fraction_l114_114053

-- We start by defining the complex nested fraction
def nested_fraction : Rat :=
  1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))

-- We assert that the value of the nested fraction is 8/21 
theorem evaluate_nested_fraction : nested_fraction = 8 / 21 := by
  sorry

end evaluate_nested_fraction_l114_114053


namespace max_value_expression_l114_114934

theorem max_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
by sorry

end max_value_expression_l114_114934


namespace max_subset_no_ap_l114_114189

theorem max_subset_no_ap (n : ℕ) (H : n ≥ 4) :
  ∃ (s : Finset ℝ), (s.card ≥ ⌊Real.sqrt (2 * n / 3)⌋₊ + 1) ∧
  ∀ (a b c : ℝ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → a ≠ c → b ≠ c → (a, b, c) ≠ (a + b - c, b, c) :=
sorry

end max_subset_no_ap_l114_114189


namespace car_a_has_higher_avg_speed_l114_114983

-- Definitions of the conditions for Car A
def distance_car_a : ℕ := 120
def speed_segment_1_car_a : ℕ := 60
def distance_segment_1_car_a : ℕ := 40
def speed_segment_2_car_a : ℕ := 40
def distance_segment_2_car_a : ℕ := 40
def speed_segment_3_car_a : ℕ := 80
def distance_segment_3_car_a : ℕ := distance_car_a - distance_segment_1_car_a - distance_segment_2_car_a

-- Definitions of the conditions for Car B
def distance_car_b : ℕ := 120
def time_segment_1_car_b : ℕ := 1
def speed_segment_1_car_b : ℕ := 60
def time_segment_2_car_b : ℕ := 1
def speed_segment_2_car_b : ℕ := 40
def total_time_car_b : ℕ := 3
def distance_segment_1_car_b := speed_segment_1_car_b * time_segment_1_car_b
def distance_segment_2_car_b := speed_segment_2_car_b * time_segment_2_car_b
def time_segment_3_car_b := total_time_car_b - time_segment_1_car_b - time_segment_2_car_b
def distance_segment_3_car_b := distance_car_b - distance_segment_1_car_b - distance_segment_2_car_b
def speed_segment_3_car_b := distance_segment_3_car_b / time_segment_3_car_b

-- Total Time for Car A
def time_car_a := distance_segment_1_car_a / speed_segment_1_car_a
                + distance_segment_2_car_a / speed_segment_2_car_a
                + distance_segment_3_car_a / speed_segment_3_car_a

-- Average Speed for Car A
def avg_speed_car_a := distance_car_a / time_car_a

-- Total Time for Car B
def time_car_b := total_time_car_b

-- Average Speed for Car B
def avg_speed_car_b := distance_car_b / time_car_b

-- Proof that Car A has a higher average speed than Car B
theorem car_a_has_higher_avg_speed : avg_speed_car_a > avg_speed_car_b := by sorry

end car_a_has_higher_avg_speed_l114_114983


namespace tangent_parabola_line_l114_114390

theorem tangent_parabola_line (a x₀ y₀ : ℝ) 
  (h_line : x₀ - y₀ - 1 = 0)
  (h_parabola : y₀ = a * x₀^2)
  (h_tangent_slope : 2 * a * x₀ = 1) : 
  a = 1 / 4 :=
sorry

end tangent_parabola_line_l114_114390


namespace union_A_B_l114_114856

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | x^2 - 1 < 0}
def A_union_B := {x : ℝ | (Real.log x ≤ 0) ∨ (x^2 - 1 < 0)}

theorem union_A_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- proof to be added
  sorry

end union_A_B_l114_114856


namespace lines_perpendicular_l114_114888

-- Define the conditions: lines not parallel to the coordinate planes 
-- (which translates to k_1 and k_2 not being infinite, but we can code it directly as a statement on the product being -1)
variable {k1 k2 l1 l2 : ℝ} 

-- Define the theorem statement 
theorem lines_perpendicular (hk : k1 * k2 = -1) : 
  ∀ (x : ℝ), (k1 ≠ 0) ∧ (k2 ≠ 0) → 
  (∀ (y1 y2 : ℝ), y1 = k1 * x + l1 → y2 = k2 * x + l2 → 
  (k1 * k2 = -1)) :=
sorry

end lines_perpendicular_l114_114888


namespace prime_count_60_to_70_l114_114951

theorem prime_count_60_to_70 : ∃ primes : Finset ℕ, primes.card = 2 ∧ ∀ p ∈ primes, 60 < p ∧ p < 70 ∧ Nat.Prime p :=
by
  sorry

end prime_count_60_to_70_l114_114951


namespace speed_difference_is_zero_l114_114732

theorem speed_difference_is_zero :
  let distance_bike := 72
  let time_bike := 9
  let distance_truck := 72
  let time_truck := 9
  let speed_bike := distance_bike / time_bike
  let speed_truck := distance_truck / time_truck
  (speed_truck - speed_bike) = 0 := by
  sorry

end speed_difference_is_zero_l114_114732


namespace composite_proposition_l114_114538

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 5 ≤ 4

noncomputable def q : Prop := ∀ x : ℝ, 0 < x ∧ x < Real.pi / 2 → ¬ (∀ v : ℝ, v = (Real.sin x + 4 / Real.sin x) → v = 4)

theorem composite_proposition : p ∧ ¬q := 
by 
  sorry

end composite_proposition_l114_114538


namespace determine_a_l114_114312

open Complex

noncomputable def complex_eq_real_im_part (a : ℝ) : Prop :=
  let z := (a - I) * (1 + I) / I
  (z.re, z.im) = ((a - 1 : ℝ), -(a + 1 : ℝ))

theorem determine_a (a : ℝ) (h : complex_eq_real_im_part a) : a = -1 :=
sorry

end determine_a_l114_114312


namespace larger_page_sum_137_l114_114438

theorem larger_page_sum_137 (x y : ℕ) (h1 : x + y = 137) (h2 : y = x + 1) : y = 69 :=
sorry

end larger_page_sum_137_l114_114438


namespace largest_n_for_negative_sum_l114_114217

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ} -- common difference of the arithmetic sequence

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

theorem largest_n_for_negative_sum
  (h_arith_seq : is_arithmetic_sequence a d)
  (h_first_term : a 0 < 0)
  (h_sum_2015_2016 : a 2014 + a 2015 > 0)
  (h_product_2015_2016 : a 2014 * a 2015 < 0) :
  (∀ n, sum_of_first_n_terms a n < 0 → n ≤ 4029) ∧ (sum_of_first_n_terms a 4029 < 0) :=
sorry

end largest_n_for_negative_sum_l114_114217


namespace sum_first_11_terms_of_arithmetic_sequence_l114_114889

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 an : ℤ) : ℤ :=
  n * (a1 + an) / 2

theorem sum_first_11_terms_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : S n = sum_arithmetic_sequence n (a 1) (a n))
  (h2 : a 3 + a 6 + a 9 = 60) : S 11 = 220 :=
sorry

end sum_first_11_terms_of_arithmetic_sequence_l114_114889


namespace quadratic_equation_with_given_root_l114_114337

theorem quadratic_equation_with_given_root : 
  ∃ p q : ℤ, (∀ x : ℝ, x^2 + (p : ℝ) * x + (q : ℝ) = 0 ↔ x = 2 - Real.sqrt 7 ∨ x = 2 + Real.sqrt 7) 
  ∧ (p = -4) ∧ (q = -3) :=
by
  sorry

end quadratic_equation_with_given_root_l114_114337


namespace necessary_but_not_sufficient_l114_114286

theorem necessary_but_not_sufficient :
    (∀ (x y : ℝ), x > 2 ∧ y > 3 → x + y > 5 ∧ x * y > 6) ∧ 
    ¬(∀ (x y : ℝ), x + y > 5 ∧ x * y > 6 → x > 2 ∧ y > 3) := by
  sorry

end necessary_but_not_sufficient_l114_114286


namespace find_k_l114_114479

theorem find_k (x y k : ℤ) (h₁ : x = -1) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) :
  k = 4 :=
by
  sorry

end find_k_l114_114479


namespace audrey_sleep_time_l114_114690

theorem audrey_sleep_time (T : ℝ) (h1 : (3 / 5) * T = 6) : T = 10 :=
by
  sorry

end audrey_sleep_time_l114_114690


namespace bridge_length_l114_114345

theorem bridge_length (length_train : ℝ) (speed_train : ℝ) (time : ℝ) (h1 : length_train = 15) (h2 : speed_train = 275) (h3 : time = 48) : 
    (speed_train / 100) * time - length_train = 117 := 
by
    -- these are the provided conditions, enabling us to skip actual proof steps with 'sorry'
    sorry

end bridge_length_l114_114345


namespace black_equals_sum_of_white_l114_114972

theorem black_equals_sum_of_white :
  ∃ (a b c d : ℤ) (a_neq_zero : a ≠ 0) (b_neq_zero : b ≠ 0) (c_neq_zero : c ≠ 0) (d_neq_zero : d ≠ 0),
    (c + d * Real.sqrt 7 = (Real.sqrt (a + b * Real.sqrt 2) + Real.sqrt (a - b * Real.sqrt 2))^2) :=
by
  sorry

end black_equals_sum_of_white_l114_114972


namespace distance_PQ_is_12_miles_l114_114137

-- Define the conditions
def average_speed_PQ := 40 -- mph
def average_speed_QP := 45 -- mph
def time_difference := 2 -- minutes

-- Main proof statement to show that the distance is 12 miles
theorem distance_PQ_is_12_miles 
    (x : ℝ) 
    (h1 : average_speed_PQ > 0) 
    (h2 : average_speed_QP > 0) 
    (h3 : abs ((x / average_speed_PQ * 60) - (x / average_speed_QP * 60)) = time_difference) 
    : x = 12 := 
by
  sorry

end distance_PQ_is_12_miles_l114_114137


namespace sum_of_m_and_n_l114_114514

noncomputable section

variable {a b m n : ℕ}

theorem sum_of_m_and_n 
  (h1 : a = n * b)
  (h2 : (a + b) = m * (a - b)) :
  m + n = 5 :=
sorry

end sum_of_m_and_n_l114_114514


namespace perpendicular_slope_of_line_l114_114536

theorem perpendicular_slope_of_line (x y : ℤ) : 
    (5 * x - 4 * y = 20) → 
    ∃ m : ℚ, m = -4 / 5 := 
by 
    sorry

end perpendicular_slope_of_line_l114_114536


namespace equal_five_digit_number_sets_l114_114887

def five_digit_numbers_not_div_5 : ℕ :=
  9 * 10^3 * 8

def five_digit_numbers_first_two_not_5 : ℕ :=
  8 * 9 * 10^3

theorem equal_five_digit_number_sets :
  five_digit_numbers_not_div_5 = five_digit_numbers_first_two_not_5 :=
by
  repeat { sorry }

end equal_five_digit_number_sets_l114_114887


namespace inequality_solutions_l114_114242

theorem inequality_solutions (p p' q q' : ℕ) (hp : p ≠ p') (hq : q ≠ q') (hp_pos : 0 < p) (hp'_pos : 0 < p') (hq_pos : 0 < q) (hq'_pos : 0 < q') :
  (-(q : ℚ) / p > -(q' : ℚ) / p') ↔ (q * p' < p * q') :=
by
  sorry

end inequality_solutions_l114_114242


namespace power_increase_fourfold_l114_114044

theorem power_increase_fourfold 
    (F v : ℝ)
    (k : ℝ)
    (R : ℝ := k * v)
    (P_initial : ℝ := F * v)
    (v' : ℝ := 2 * v)
    (F' : ℝ := 2 * F)
    (R' : ℝ := k * v')
    (P_final : ℝ := F' * v') :
    P_final = 4 * P_initial := 
by
  sorry

end power_increase_fourfold_l114_114044


namespace speed_of_second_train_equivalent_l114_114230

noncomputable def relative_speed_in_m_per_s (time_seconds : ℝ) (total_distance_m : ℝ) : ℝ :=
total_distance_m / time_seconds

noncomputable def relative_speed_in_km_per_h (relative_speed_m_per_s : ℝ) : ℝ :=
relative_speed_m_per_s * 3.6

noncomputable def speed_of_second_train (relative_speed_km_per_h : ℝ) (speed_of_first_train_km_per_h : ℝ) : ℝ :=
relative_speed_km_per_h - speed_of_first_train_km_per_h

theorem speed_of_second_train_equivalent
  (length_of_first_train length_of_second_train : ℝ)
  (speed_of_first_train_km_per_h : ℝ)
  (time_of_crossing_seconds : ℝ) :
  speed_of_second_train
    (relative_speed_in_km_per_h (relative_speed_in_m_per_s time_of_crossing_seconds (length_of_first_train + length_of_second_train)))
    speed_of_first_train_km_per_h = 36 := by
  sorry

end speed_of_second_train_equivalent_l114_114230


namespace count_ways_to_exhaust_black_matches_l114_114093

theorem count_ways_to_exhaust_black_matches 
  (n r g : ℕ) 
  (h_r_le_n : r ≤ n) 
  (h_g_le_n : g ≤ n) 
  (h_r_ge_0 : 0 ≤ r) 
  (h_g_ge_0 : 0 ≤ g) 
  (h_n_ge_0 : 0 < n) :
  ∃ ways : ℕ, ways = (Nat.factorial (3 * n - r - g - 1)) / (Nat.factorial (n - 1) * Nat.factorial (n - r) * Nat.factorial (n - g)) :=
by
  sorry

end count_ways_to_exhaust_black_matches_l114_114093


namespace freezer_temp_calculation_l114_114297

def refrigerator_temp : ℝ := 4
def freezer_temp (rt : ℝ) (d : ℝ) : ℝ := rt - d

theorem freezer_temp_calculation :
  (freezer_temp refrigerator_temp 22) = -18 :=
by
  sorry

end freezer_temp_calculation_l114_114297


namespace solve_quadratic_l114_114055

theorem solve_quadratic {x : ℚ} (h1 : x > 0) (h2 : 3 * x ^ 2 + 11 * x - 20 = 0) : x = 4 / 3 :=
sorry

end solve_quadratic_l114_114055


namespace fernandez_family_children_l114_114106

-- Conditions definition
variables (m : ℕ) -- age of the mother
variables (x : ℕ) -- number of children
variables (y : ℕ) -- average age of the children

-- Given conditions
def average_age_family (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + 50 + 70 + x * y) / (3 + x) = 25

def average_age_mother_children (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + x * y) / (1 + x) = 18

-- Goal statement
theorem fernandez_family_children
  (m : ℕ) (x : ℕ) (y : ℕ)
  (h1 : average_age_family m x y)
  (h2 : average_age_mother_children m x y) :
  x = 9 :=
sorry

end fernandez_family_children_l114_114106


namespace circle_touching_y_axis_radius_5_k_value_l114_114790

theorem circle_touching_y_axis_radius_5_k_value :
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) →
    (∃ r : ℝ, r = 5 ∧ (∀ c : ℝ × ℝ, (c.1 + 4)^2 + (c.2 + 2)^2 = r^2) ∧
      (∃ x : ℝ, x + 4 = 0)) :=
by
  sorry

end circle_touching_y_axis_radius_5_k_value_l114_114790


namespace limit_proof_l114_114272

open Real

-- Define the conditions
axiom sin_6x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |sin (6 * x) / (6 * x) - 1| < ε
axiom arctg_2x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |arctan (2 * x) / (2 * x) - 1| < ε

-- State the limit proof problem
theorem limit_proof :
  (∃ ε > 0, ∀ x : ℝ, |x| < ε → x ≠ 0 →
  |(x * sin (6 * x)) / (arctan (2 * x)) ^ 2 - (3 / 2)| < ε) :=
sorry

end limit_proof_l114_114272


namespace probability_green_ball_l114_114139

/-- 
Given three containers with specific numbers of red and green balls, 
and the probability of selecting each container being equal, 
the probability of picking a green ball when choosing a container randomly is 7/12.
-/
theorem probability_green_ball :
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  (green_I + green_II + green_III) = 7 / 12 :=
by 
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  have : (green_I + green_II + green_III) = (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) := by rfl
  have : (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) = (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) := by rfl
  have : (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) = (1 / 9 + 2 / 9 + 1 / 4) := by rfl
  have : (1 / 9 + 2 / 9 + 1 / 4) = (4 / 36 + 8 / 36 + 9 / 36) := by rfl
  have : (4 / 36 + 8 / 36 + 9 / 36) = 21 / 36 := by rfl
  have : 21 / 36 = 7 / 12 := by rfl
  rfl

end probability_green_ball_l114_114139


namespace original_price_of_cycle_l114_114944

theorem original_price_of_cycle (P : ℝ) (h1 : 1440 = P + 0.6 * P) : P = 900 :=
by
  sorry

end original_price_of_cycle_l114_114944


namespace pentagon_area_l114_114864

variable (a b c d e : ℕ)
variable (r s : ℕ)

-- Given conditions
axiom H₁: a = 14
axiom H₂: b = 35
axiom H₃: c = 42
axiom H₄: d = 14
axiom H₅: e = 35
axiom H₆: r = 21
axiom H₇: s = 28
axiom H₈: r^2 + s^2 = e^2

-- Question: Prove that the area of the pentagon is 1176
theorem pentagon_area : b * c - (1 / 2) * r * s = 1176 := 
by 
  sorry

end pentagon_area_l114_114864


namespace fourth_term_expansion_l114_114154

def binomial_term (n r : ℕ) (a b : ℚ) : ℚ :=
  (Nat.descFactorial n r) / (Nat.factorial r) * a^(n - r) * b^r

theorem fourth_term_expansion (x : ℚ) (hx : x ≠ 0) : 
  binomial_term 6 3 2 (-(1 / (x^(1/3)))) = (-160 / x) :=
by
  sorry

end fourth_term_expansion_l114_114154


namespace sqrt_of_square_neg_three_l114_114223

theorem sqrt_of_square_neg_three : Real.sqrt ((-3 : ℝ)^2) = 3 := by
  sorry

end sqrt_of_square_neg_three_l114_114223


namespace find_point_C_l114_114971

def point := ℝ × ℝ
def is_midpoint (M A B : point) : Prop := (2 * M.1 = A.1 + B.1) ∧ (2 * M.2 = A.2 + B.2)

-- Variables for known points
def A : point := (2, 8)
def M : point := (4, 11)
def L : point := (6, 6)

-- The proof problem: Prove the coordinates of point C
theorem find_point_C (C : point) (B : point) :
  is_midpoint M A B →
  -- (additional conditions related to the angle bisector can be added if specified)
  C = (14, 2) :=
sorry

end find_point_C_l114_114971


namespace cos_alpha_plus_pi_div_4_value_l114_114455

noncomputable def cos_alpha_plus_pi_div_4 (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) : Real :=
  Real.cos (α + π / 4)

theorem cos_alpha_plus_pi_div_4_value (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) :
  cos_alpha_plus_pi_div_4 α h1 h2 = -4 / 5 :=
sorry

end cos_alpha_plus_pi_div_4_value_l114_114455


namespace maximum_n_l114_114193

/-- Definition of condition (a): For any three people, there exist at least two who know each other. -/
def condition_a (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 3 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), G.Adj a b

/-- Definition of condition (b): For any four people, there exist at least two who do not know each other. -/
def condition_b (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 4 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), ¬ G.Adj a b

theorem maximum_n (G : SimpleGraph V) [Fintype V] (h1 : condition_a G) (h2 : condition_b G) : 
  Fintype.card V ≤ 8 :=
by
  sorry

end maximum_n_l114_114193


namespace parabola_vertex_point_l114_114869

theorem parabola_vertex_point (a b c : ℝ) 
    (h_vertex : ∃ a b c : ℝ, ∀ x : ℝ, y = a * x^2 + b * x + c) 
    (h_vertex_coord : ∃ (h k : ℝ), h = 3 ∧ k = -5) 
    (h_pass : ∃ (x y : ℝ), x = 0 ∧ y = -2) :
    c = -2 := by
  sorry

end parabola_vertex_point_l114_114869


namespace find_n_modulo_l114_114641

theorem find_n_modulo :
  ∀ n : ℤ, (0 ≤ n ∧ n < 25 ∧ -175 % 25 = n % 25) → n = 0 :=
by
  intros n h
  sorry

end find_n_modulo_l114_114641


namespace multiplication_sequence_result_l114_114274

theorem multiplication_sequence_result : (1 * 3 * 5 * 7 * 9 * 11 = 10395) :=
by
  sorry

end multiplication_sequence_result_l114_114274


namespace find_R_position_l114_114860

theorem find_R_position :
  ∀ (P Q R : ℤ), P = -6 → Q = -1 → Q = (P + R) / 2 → R = 4 :=
by
  intros P Q R hP hQ hQ_halfway
  sorry

end find_R_position_l114_114860


namespace find_a_l114_114564

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_eq (x y a : ℝ) : Prop := x + a * y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y → line_eq x y a → (x - 1)^2 + (y - 2)^2 = 4) →
  ∃ a, (a = -1) :=
sorry

end find_a_l114_114564


namespace muffins_divide_equally_l114_114931

theorem muffins_divide_equally (friends : ℕ) (total_muffins : ℕ) (Jessie_and_friends : ℕ) (muffins_per_person : ℕ) :
  friends = 6 →
  total_muffins = 35 →
  Jessie_and_friends = friends + 1 →
  muffins_per_person = total_muffins / Jessie_and_friends →
  muffins_per_person = 5 :=
by
  intros h_friends h_muffins h_people h_division
  sorry

end muffins_divide_equally_l114_114931


namespace range_of_a_l114_114043

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) :
  (a > -2 ∧ a < -1) ∨ (a ≥ 1) :=
by
  sorry

end range_of_a_l114_114043


namespace prove_range_of_xyz_l114_114256

variable (x y z a : ℝ)

theorem prove_range_of_xyz 
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2 / 2)
  (ha : 0 < a) :
  (0 ≤ x ∧ x ≤ 2 * a / 3) ∧ (0 ≤ y ∧ y ≤ 2 * a / 3) ∧ (0 ≤ z ∧ z ≤ 2 * a / 3) :=
sorry

end prove_range_of_xyz_l114_114256


namespace distance_between_chords_l114_114499

-- Definitions based on the conditions
structure CircleGeometry where
  radius: ℝ
  d1: ℝ -- distance from the center to the closest chord (34 units)
  d2: ℝ -- distance from the center to the second chord (38 units)
  d3: ℝ -- distance from the center to the outermost chord (38 units)

-- The problem itself
theorem distance_between_chords (circle: CircleGeometry) (h1: circle.d2 = 3) (h2: circle.d1 = 3 * circle.d2) (h3: circle.d3 = circle.d2) :
  2 * circle.d2 = 6 :=
by
  sorry

end distance_between_chords_l114_114499


namespace students_in_only_one_subject_l114_114662

variables (A B C : ℕ) 
variables (A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ)

def students_in_one_subject (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ) : ℕ :=
  A + B + C - A_inter_B - A_inter_C - B_inter_C + A_inter_B_inter_C - 2 * A_inter_B_inter_C

theorem students_in_only_one_subject :
  ∀ (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ),
    A = 29 →
    B = 28 →
    C = 27 →
    A_inter_B = 13 →
    A_inter_C = 12 →
    B_inter_C = 11 →
    A_inter_B_inter_C = 5 →
    students_in_one_subject A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C = 27 :=
by
  intros A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C hA hB hC hAB hAC hBC hABC
  unfold students_in_one_subject
  rw [hA, hB, hC, hAB, hAC, hBC, hABC]
  norm_num
  sorry

end students_in_only_one_subject_l114_114662


namespace evaluate_expression_l114_114897

theorem evaluate_expression (x : ℤ) (h : x = 2) : 20 - 2 * (3 * x^2 - 4 * x + 8) = -4 :=
by
  rw [h]
  sorry

end evaluate_expression_l114_114897


namespace angle_E_measure_l114_114712

theorem angle_E_measure {D E F : Type} (angle_D angle_E angle_F : ℝ) 
  (h1 : angle_E = angle_F)
  (h2 : angle_F = 3 * angle_D)
  (h3 : angle_D = (1/2) * angle_E) 
  (h_sum : angle_D + angle_E + angle_F = 180) :
  angle_E = 540 / 7 := 
by
  sorry

end angle_E_measure_l114_114712


namespace determine_x_l114_114920

theorem determine_x (x : ℝ) (hx : 0 < x) (h : (⌊x⌋ : ℝ) * x = 120) : x = 120 / 11 := 
sorry

end determine_x_l114_114920


namespace problem1_problem2_l114_114879

-- Problem 1 equivalent proof problem
theorem problem1 : 
  (Real.sqrt 3 * Real.sqrt 6 - (Real.sqrt (1 / 2) - Real.sqrt 8)) = (9 * Real.sqrt 2 / 2) :=
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2 (x : Real) (hx : x = Real.sqrt 5) : 
  ((1 + 1 / x) / ((x^2 + x) / x)) = (Real.sqrt 5 / 5) :=
by
  sorry

end problem1_problem2_l114_114879


namespace equilateral_triangle_of_ap_angles_gp_sides_l114_114719

theorem equilateral_triangle_of_ap_angles_gp_sides
  (A B C : ℝ)
  (α β γ : ℝ)
  (hαβγ_sum : α + β + γ = 180)
  (h_ap_angles : 2 * β = α + γ)
  (a b c : ℝ)
  (h_gp_sides : b^2 = a * c) :
  α = β ∧ β = γ ∧ a = b ∧ b = c :=
sorry

end equilateral_triangle_of_ap_angles_gp_sides_l114_114719


namespace packs_needed_l114_114534

-- Define the problem conditions
def bulbs_bedroom : ℕ := 2
def bulbs_bathroom : ℕ := 1
def bulbs_kitchen : ℕ := 1
def bulbs_basement : ℕ := 4
def bulbs_pack : ℕ := 2

def total_bulbs_main_areas : ℕ := bulbs_bedroom + bulbs_bathroom + bulbs_kitchen + bulbs_basement
def bulbs_garage : ℕ := total_bulbs_main_areas / 2

def total_bulbs : ℕ := total_bulbs_main_areas + bulbs_garage

def total_packs : ℕ := total_bulbs / bulbs_pack

-- The proof statement
theorem packs_needed : total_packs = 6 :=
by
  sorry

end packs_needed_l114_114534


namespace oranges_thrown_away_l114_114939

theorem oranges_thrown_away (original_oranges: ℕ) (new_oranges: ℕ) (total_oranges: ℕ) (x: ℕ)
  (h1: original_oranges = 5) (h2: new_oranges = 28) (h3: total_oranges = 31) :
  original_oranges - x + new_oranges = total_oranges → x = 2 :=
by
  intros h_eq
  -- Proof omitted
  sorry

end oranges_thrown_away_l114_114939


namespace manager_salary_l114_114146

def avg_salary_employees := 1500
def num_employees := 20
def avg_salary_increase := 600
def num_total_people := num_employees + 1

def total_salary_employees := num_employees * avg_salary_employees
def new_avg_salary := avg_salary_employees + avg_salary_increase
def total_salary_with_manager := num_total_people * new_avg_salary

theorem manager_salary : total_salary_with_manager - total_salary_employees = 14100 :=
by
  sorry

end manager_salary_l114_114146


namespace shelby_rain_drive_time_eq_3_l114_114212

-- Definitions as per the conditions
def distance (v : ℝ) (t : ℝ) : ℝ := v * t
def total_distance := 24 -- in miles
def total_time := 50 / 60 -- in hours (converted to minutes)
def non_rainy_speed := 30 / 60 -- in miles per minute
def rainy_speed := 20 / 60 -- in miles per minute

-- Lean statement of the proof problem
theorem shelby_rain_drive_time_eq_3 :
  ∃ x : ℝ,
  (distance non_rainy_speed (total_time - x / 60) + distance rainy_speed (x / 60) = total_distance)
  ∧ (0 ≤ x) ∧ (x ≤ total_time * 60) →
  x = 3 := 
sorry

end shelby_rain_drive_time_eq_3_l114_114212


namespace conference_center_people_count_l114_114672

-- Definition of the conditions
def rooms : ℕ := 6
def capacity_per_room : ℕ := 80
def fraction_full : ℚ := 2/3

-- Total capacity of the conference center
def total_capacity := rooms * capacity_per_room

-- Number of people in the conference center when 2/3 full
def num_people := fraction_full * total_capacity

-- The theorem stating the problem
theorem conference_center_people_count :
  num_people = 320 := 
by
  -- This is a placeholder for the proof
  sorry

end conference_center_people_count_l114_114672


namespace squares_form_acute_triangle_l114_114937

theorem squares_form_acute_triangle (a b c x y z d : ℝ)
    (h_triangle : ∀ x y z : ℝ, (x > 0 ∧ y > 0 ∧ z > 0) → (x + y > z) ∧ (x + z > y) ∧ (y + z > x))
    (h_acute : ∀ x y z : ℝ, (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2))
    (h_inscribed_squares : x = a ^ 2 * b * c / (d * a + b * c) ∧
                           y = b ^ 2 * a * c / (d * b + a * c) ∧
                           z = c ^ 2 * a * b / (d * c + a * b)) :
    (x + y > z) ∧ (x + z > y) ∧ (y + z > x) ∧
    (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2) :=
sorry

end squares_form_acute_triangle_l114_114937


namespace cone_cylinder_volume_ratio_l114_114777

theorem cone_cylinder_volume_ratio :
  let π := Real.pi
  let Vcylinder := π * (3:ℝ)^2 * (15:ℝ)
  let Vcone := (1/3:ℝ) * π * (2:ℝ)^2 * (5:ℝ)
  (Vcone / Vcylinder) = (4 / 81) :=
by
  let π := Real.pi
  let r_cylinder := (3:ℝ)
  let h_cylinder := (15:ℝ)
  let r_cone := (2:ℝ)
  let h_cone := (5:ℝ)
  let Vcylinder := π * r_cylinder^2 * h_cylinder
  let Vcone := (1/3:ℝ) * π * r_cone^2 * h_cone
  have h1 : Vcylinder = 135 * π := by sorry
  have h2 : Vcone = (20 / 3) * π := by sorry
  have h3 : (Vcone / Vcylinder) = (4 / 81) := by sorry
  exact h3

end cone_cylinder_volume_ratio_l114_114777


namespace price_of_shoes_on_Monday_l114_114389

noncomputable def priceOnThursday : ℝ := 50

noncomputable def increasedPriceOnFriday : ℝ := priceOnThursday * 1.2

noncomputable def discountedPriceOnMonday : ℝ := increasedPriceOnFriday * 0.85

noncomputable def finalPriceOnMonday : ℝ := discountedPriceOnMonday * 1.05

theorem price_of_shoes_on_Monday :
  finalPriceOnMonday = 53.55 :=
by
  sorry

end price_of_shoes_on_Monday_l114_114389


namespace system1_solution_system2_solution_l114_114280

theorem system1_solution (p q : ℝ) 
  (h1 : p + q = 4)
  (h2 : 2 * p - q = 5) : 
  p = 3 ∧ q = 1 := 
sorry

theorem system2_solution (v t : ℝ)
  (h3 : 2 * v + t = 3)
  (h4 : 3 * v - 2 * t = 3) :
  v = 9 / 7 ∧ t = 3 / 7 :=
sorry

end system1_solution_system2_solution_l114_114280


namespace packs_of_green_bouncy_balls_l114_114149

/-- Maggie bought 10 bouncy balls in each pack of red, yellow, and green bouncy balls.
    She bought 4 packs of red bouncy balls, 8 packs of yellow bouncy balls, and some 
    packs of green bouncy balls. In total, she bought 160 bouncy balls. This theorem 
    aims to prove how many packs of green bouncy balls Maggie bought. 
 -/
theorem packs_of_green_bouncy_balls (red_packs : ℕ) (yellow_packs : ℕ) (total_balls : ℕ) (balls_per_pack : ℕ) 
(pack : ℕ) :
  red_packs = 4 →
  yellow_packs = 8 →
  balls_per_pack = 10 →
  total_balls = 160 →
  red_packs * balls_per_pack + yellow_packs * balls_per_pack + pack * balls_per_pack = total_balls →
  pack = 4 :=
by
  intros h_red h_yellow h_balls_per_pack h_total_balls h_eq
  sorry

end packs_of_green_bouncy_balls_l114_114149


namespace inverse_proportion_incorrect_D_l114_114259

theorem inverse_proportion_incorrect_D :
  ∀ (x y x1 y1 x2 y2 : ℝ), (y = -3 / x) ∧ (y1 = -3 / x1) ∧ (y2 = -3 / x2) ∧ (x1 < x2) → ¬(y1 < y2) :=
by
  sorry

end inverse_proportion_incorrect_D_l114_114259


namespace probability_not_in_square_b_l114_114583

theorem probability_not_in_square_b (area_A : ℝ) (perimeter_B : ℝ) 
  (area_A_eq : area_A = 30) (perimeter_B_eq : perimeter_B = 16) : 
  (14 / 30 : ℝ) = (7 / 15 : ℝ) :=
by
  sorry

end probability_not_in_square_b_l114_114583


namespace total_items_washed_l114_114443

def towels := 15
def shirts := 10
def loads := 20

def items_per_load : Nat := towels + shirts
def total_items : Nat := items_per_load * loads

theorem total_items_washed : total_items = 500 :=
by
  rw [total_items, items_per_load]
  -- step expansion:
  -- unfold items_per_load
  -- calc 
  -- 15 + 10 = 25  -- from definition
  -- 25 * 20 = 500  -- from multiplication
  sorry

end total_items_washed_l114_114443


namespace g_neg_9_equiv_78_l114_114651

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (y : ℝ) : ℝ := 3 * (y / 2 - 3 / 2)^2 + 4 * (y / 2 - 3 / 2) - 6

theorem g_neg_9_equiv_78 : g (-9) = 78 := by
  sorry

end g_neg_9_equiv_78_l114_114651


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l114_114296

-- Definitions based on given conditions
variables (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions
axiom a_4 : a 4 = 6
axiom a_6 : a 6 = 10
axiom all_positive_b : ∀ n, 0 < b n
axiom b_3 : b 3 = a 3
axiom T_2 : T 2 = 3

-- Required to prove
theorem arithmetic_sequence_general_formula : ∀ n, a n = 2 * n - 2 :=
sorry

theorem geometric_sequence_sum_formula : ∀ n, T n = 2^n - 1 :=
sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l114_114296


namespace compare_series_l114_114004

theorem compare_series (x y : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) : 
  (1 / (1 - x^2) + 1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by
  sorry

end compare_series_l114_114004


namespace range_of_a_for_propositions_p_and_q_l114_114539

theorem range_of_a_for_propositions_p_and_q :
  {a : ℝ | ∃ x, (x^2 + 2 * a * x + 4 = 0) ∧ (3 - 2 * a > 1)} = {a | a ≤ -2} := sorry

end range_of_a_for_propositions_p_and_q_l114_114539


namespace ellipse_condition_l114_114841

theorem ellipse_condition (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m) = 1) →
  (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end ellipse_condition_l114_114841


namespace tangent_line_slope_l114_114101

theorem tangent_line_slope (h : ℝ → ℝ) (a : ℝ) (P : ℝ × ℝ) 
  (tangent_eq : ∀ x y, 2 * x + y + 1 = 0 ↔ (x, y) = (a, h a)) : 
  deriv h a < 0 :=
sorry

end tangent_line_slope_l114_114101


namespace income_recording_l114_114523

theorem income_recording (exp_200 : Int := -200) (income_60 : Int := 60) : exp_200 = -200 → income_60 = 60 →
  (income_60 > 0) :=
by
  intro h_exp h_income
  sorry

end income_recording_l114_114523


namespace roots_conditions_l114_114973

theorem roots_conditions (α β m n : ℝ) (h_pos : β > 0)
  (h1 : α + 2 * β = -m)
  (h2 : 2 * α * β + β^2 = -3)
  (h3 : α * β^2 = -n)
  (h4 : α^2 + 2 * β^2 = 6) : 
  m = 0 ∧ n = 2 := by
  sorry

end roots_conditions_l114_114973


namespace hawks_points_l114_114612

theorem hawks_points (E H : ℕ) (h₁ : E + H = 82) (h₂ : E = H + 18) (h₃ : H ≥ 9) : H = 32 :=
sorry

end hawks_points_l114_114612


namespace cannot_use_square_difference_formula_l114_114103

theorem cannot_use_square_difference_formula (x y : ℝ) :
  ¬ ∃ a b : ℝ, (2 * x + 3 * y) * (-3 * y - 2 * x) = (a + b) * (a - b) :=
sorry

end cannot_use_square_difference_formula_l114_114103


namespace triangle_area_is_96_l114_114309

-- Definitions of radii and sides being congruent
def tangent_circles (radius1 radius2 : ℝ) : Prop :=
  ∃ (O O' : ℝ × ℝ), dist O O' = radius1 + radius2

-- Given conditions
def radius_small : ℝ := 2
def radius_large : ℝ := 4
def sides_congruent (AB AC : ℝ) : Prop :=
  AB = AC

-- Theorem stating the goal
theorem triangle_area_is_96 
  (O O' : ℝ × ℝ)
  (AB AC : ℝ)
  (circ_tangent : tangent_circles radius_small radius_large)
  (sides_tangent : sides_congruent AB AC) :
  ∃ (BC : ℝ), ∃ (AF : ℝ), (1/2) * BC * AF = 96 := 
by
  sorry

end triangle_area_is_96_l114_114309


namespace total_test_points_l114_114152

theorem total_test_points (total_questions two_point_questions four_point_questions points_per_two_question points_per_four_question : ℕ) 
  (h1 : total_questions = 40)
  (h2 : four_point_questions = 10)
  (h3 : points_per_two_question = 2)
  (h4 : points_per_four_question = 4)
  (h5 : two_point_questions = total_questions - four_point_questions)
  : (two_point_questions * points_per_two_question) + (four_point_questions * points_per_four_question) = 100 :=
by
  sorry

end total_test_points_l114_114152


namespace tenth_term_arithmetic_sequence_l114_114207

theorem tenth_term_arithmetic_sequence (a d : ℕ) 
  (h1 : a + 2 * d = 10) 
  (h2 : a + 5 * d = 16) : 
  a + 9 * d = 24 := 
by 
  sorry

end tenth_term_arithmetic_sequence_l114_114207


namespace inequality_transformation_range_of_a_l114_114360

-- Define the given function f(x) = |x + 2|
def f (x : ℝ) : ℝ := abs (x + 2)

-- State the inequality transformation problem
theorem inequality_transformation (x : ℝ) :  (2 * abs (x + 2) < 4 - abs (x - 1)) ↔ (-7 / 3 < x ∧ x < -1) :=
by sorry

-- State the implication problem involving m, n, and a
theorem range_of_a (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (hmn : m + n = 1) (a : ℝ) :
  (∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) → (-6 ≤ a ∧ a ≤ 2) :=
by sorry

end inequality_transformation_range_of_a_l114_114360


namespace expand_expression_l114_114218

theorem expand_expression : ∀ (x : ℝ), 2 * (x + 3) * (x^2 - 2*x + 7) = 2*x^3 + 2*x^2 + 2*x + 42 := 
by
  intro x
  sorry

end expand_expression_l114_114218


namespace find_number_l114_114472

theorem find_number (n x : ℤ) (h1 : n * x + 3 = 10 * x - 17) (h2 : x = 4) : n = 5 :=
by
  sorry

end find_number_l114_114472


namespace find_monotonic_bijections_l114_114425

variable {f : ℝ → ℝ}

-- Define the properties of the function f
def bijective (f : ℝ → ℝ) : Prop :=
  Function.Bijective f

def condition (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f t + f (f t) = 2 * t

theorem find_monotonic_bijections (f : ℝ → ℝ) (hf_bij : bijective f) (hf_cond : condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end find_monotonic_bijections_l114_114425


namespace multiple_7_proposition_l114_114917

theorem multiple_7_proposition : (47 % 7 ≠ 0 ∨ 49 % 7 = 0) → True :=
by
  intros h
  sorry

end multiple_7_proposition_l114_114917


namespace mean_temperature_l114_114726

def temperatures : List ℚ := [80, 79, 81, 85, 87, 89, 87, 90, 89, 88]

theorem mean_temperature :
  let n := temperatures.length
  let sum := List.sum temperatures
  (sum / n : ℚ) = 85.5 :=
by
  sorry

end mean_temperature_l114_114726


namespace factor_expression_l114_114540

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) :=
by
  sorry

end factor_expression_l114_114540


namespace factorize_3x2_minus_3y2_l114_114852

theorem factorize_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factorize_3x2_minus_3y2_l114_114852


namespace problem1_problem2_l114_114249

theorem problem1 : 101 * 99 = 9999 := 
by sorry

theorem problem2 : 32 * 2^2 + 14 * 2^3 + 10 * 2^4 = 400 := 
by sorry

end problem1_problem2_l114_114249


namespace josh_initial_money_l114_114348

/--
Josh spent $1.75 on a drink, and then spent another $1.25, and has $6.00 left. 
Prove that initially Josh had $9.00.
-/
theorem josh_initial_money : 
  ∃ (initial : ℝ), (initial - 1.75 - 1.25 = 6) ∧ initial = 9 := 
sorry

end josh_initial_money_l114_114348


namespace change_positions_of_three_out_of_eight_l114_114773

theorem change_positions_of_three_out_of_eight :
  (Nat.choose 8 3) * (Nat.factorial 3) = (Nat.choose 8 3) * 6 :=
by
  sorry

end change_positions_of_three_out_of_eight_l114_114773


namespace max_unsuccessful_attempts_l114_114557

theorem max_unsuccessful_attempts (n_rings letters_per_ring : ℕ) (h_rings : n_rings = 3) (h_letters : letters_per_ring = 6) : 
  (letters_per_ring ^ n_rings) - 1 = 215 := 
by 
  -- conditions
  rw [h_rings, h_letters]
  -- necessary imports and proof generation
  sorry

end max_unsuccessful_attempts_l114_114557


namespace negation_of_statement_equivalence_l114_114086

-- Definitions of the math club and enjoyment of puzzles
def member_of_math_club (x : Type) : Prop := sorry
def enjoys_puzzles (x : Type) : Prop := sorry

-- Original statement: All members of the math club enjoy puzzles
def original_statement : Prop :=
∀ x, member_of_math_club x → enjoys_puzzles x

-- Negation of the original statement
def negated_statement : Prop :=
∃ x, member_of_math_club x ∧ ¬ enjoys_puzzles x

-- Proof problem statement
theorem negation_of_statement_equivalence :
  ¬ original_statement ↔ negated_statement :=
sorry

end negation_of_statement_equivalence_l114_114086


namespace mary_income_more_than_tim_income_l114_114449

variables (J T M : ℝ)
variables (h1 : T = 0.60 * J) (h2 : M = 0.8999999999999999 * J)

theorem mary_income_more_than_tim_income : (M - T) / T * 100 = 50 :=
by
  sorry

end mary_income_more_than_tim_income_l114_114449


namespace vending_machine_problem_l114_114417

variable (x n : ℕ)

theorem vending_machine_problem (h : 25 * x + 10 * 15 + 5 * 30 = 25 * 25 + 10 * 5 + 5 * n) (hx : x = 25) :
  n = 50 := by
sorry

end vending_machine_problem_l114_114417


namespace tree_leaves_not_shed_l114_114150

-- Definitions of conditions based on the problem.
variable (initial_leaves : ℕ) (shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves : ℕ)

-- Setting the conditions
def conditions :=
  initial_leaves = 5000 ∧
  shed_week1 = initial_leaves / 5 ∧
  shed_week2 = 30 * (initial_leaves - shed_week1) / 100 ∧
  shed_week3 = 60 * shed_week2 / 100 ∧
  shed_week4 = 50 * (initial_leaves - shed_week1 - shed_week2 - shed_week3) / 100 ∧
  shed_week5 = 2 * shed_week3 / 3 ∧
  remaining_leaves = initial_leaves - shed_week1 - shed_week2 - shed_week3 - shed_week4 - shed_week5

-- The proof statement
theorem tree_leaves_not_shed (h : conditions initial_leaves shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves) :
  remaining_leaves = 560 :=
sorry

end tree_leaves_not_shed_l114_114150


namespace sum_fractions_l114_114124

theorem sum_fractions : 
  (1/2 + 1/6 + 1/12 + 1/20 + 1/30 + 1/42 = 6/7) :=
by
  sorry

end sum_fractions_l114_114124


namespace fencing_required_for_field_l114_114069

noncomputable def fence_length (L W : ℕ) : ℕ := 2 * W + L

theorem fencing_required_for_field :
  ∀ (L W : ℕ), (L = 20) → (440 = L * W) → fence_length L W = 64 :=
by
  intros L W hL hA
  sorry

end fencing_required_for_field_l114_114069


namespace minimum_balls_to_draw_l114_114356

-- Defining the sizes for the different colors of balls
def red_balls : Nat := 40
def green_balls : Nat := 25
def yellow_balls : Nat := 20
def blue_balls : Nat := 15
def purple_balls : Nat := 10
def orange_balls : Nat := 5

-- Given conditions
def max_red_balls_before_18 : Nat := 17
def max_green_balls_before_18 : Nat := 17
def max_yellow_balls_before_18 : Nat := 17
def max_blue_balls_before_18 : Nat := 15
def max_purple_balls_before_18 : Nat := 10
def max_orange_balls_before_18 : Nat := 5

-- Sum of maximum balls of each color that can be drawn without ensuring 18 of any color
def max_balls_without_18 : Nat := 
  max_red_balls_before_18 + 
  max_green_balls_before_18 + 
  max_yellow_balls_before_18 + 
  max_blue_balls_before_18 + 
  max_purple_balls_before_18 + 
  max_orange_balls_before_18

theorem minimum_balls_to_draw {n : Nat} (h : n = max_balls_without_18 + 1) :
  n = 82 := by
  sorry

end minimum_balls_to_draw_l114_114356


namespace find_f1_verify_function_l114_114327

theorem find_f1 (f : ℝ → ℝ) (h_mono : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2)
    (h1_pos : ∀ x : ℝ, 0 < x → f x > 1 / x^2)
    (h_eq : ∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) :
    f 1 = 2 := sorry

theorem verify_function (f : ℝ → ℝ) (h_def : ∀ x : ℝ, 0 < x → f x = 2 / x^2) :
    (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2) ∧ (∀ x : ℝ, 0 < x → f x > 1 / x^2) ∧
    (∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) := sorry

end find_f1_verify_function_l114_114327


namespace find_coals_per_bag_l114_114135

open Nat

variable (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ)

def coal_per_bag (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ) : ℕ :=
  (totalTime / timePerSet) * burnRate / totalBags

theorem find_coals_per_bag :
  coal_per_bag 15 20 240 3 = 60 :=
by
  sorry

end find_coals_per_bag_l114_114135


namespace Yvonne_laps_l114_114840

-- Definitions of the given conditions
def laps_swim_by_Yvonne (l_y : ℕ) : Prop := 
  ∃ l_s l_j, 
  l_s = l_y / 2 ∧ 
  l_j = 3 * l_s ∧ 
  l_j = 15

-- Theorem statement
theorem Yvonne_laps (l_y : ℕ) (h : laps_swim_by_Yvonne l_y) : l_y = 10 :=
sorry

end Yvonne_laps_l114_114840


namespace inequality_solution_inequality_proof_l114_114776

def f (x: ℝ) := |x - 5|

theorem inequality_solution : {x : ℝ | f x + f (x + 2) ≤ 3} = {x | 5 / 2 ≤ x ∧ x ≤ 11 / 2} :=
sorry

theorem inequality_proof (a x : ℝ) (h : a < 0) : f (a * x) - f (5 * a) ≥ a * f x :=
sorry

end inequality_solution_inequality_proof_l114_114776


namespace composite_quotient_l114_114304

def first_eight_composites := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) := l.foldl (· * ·) 1

theorem composite_quotient :
  let numerator := product first_eight_composites
  let denominator := product next_eight_composites
  numerator / denominator = (1 : ℚ)/(1430 : ℚ) :=
by
  sorry

end composite_quotient_l114_114304


namespace focus_of_parabola_l114_114129

theorem focus_of_parabola (a k : ℝ) (h_eq : ∀ x : ℝ, k = 6 ∧ a = 9) :
  (0, (1 / (4 * a)) + k) = (0, 217 / 36) := sorry

end focus_of_parabola_l114_114129


namespace minimum_m_plus_n_l114_114940

theorem minimum_m_plus_n
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_ellipse : 1 / m + 4 / n = 1) :
  m + n = 9 :=
sorry

end minimum_m_plus_n_l114_114940


namespace range_of_a_l114_114141

noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x < 1 then a^x else (a - 3) * x + 4 * a

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 0 < a ∧ a ≤ 3 / 4 :=
sorry

end range_of_a_l114_114141


namespace boat_license_combinations_l114_114307

theorem boat_license_combinations :
  let letters := ['A', 'M', 'S']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let any_digit := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  3 * 9 * 10^4 = 270000 := 
by 
  sorry

end boat_license_combinations_l114_114307


namespace total_candies_is_90_l114_114763

-- Defining the conditions
def boxes_chocolate := 6
def boxes_caramel := 4
def pieces_per_box := 9

-- Defining the total number of boxes
def total_boxes := boxes_chocolate + boxes_caramel

-- Defining the total number of candies
def total_candies := total_boxes * pieces_per_box

-- Theorem stating the proof problem
theorem total_candies_is_90 : total_candies = 90 := by
  -- Provide a placeholder for the proof
  sorry

end total_candies_is_90_l114_114763


namespace product_mod5_is_zero_l114_114609

theorem product_mod5_is_zero :
  (2023 * 2024 * 2025 * 2026) % 5 = 0 :=
by
  sorry

end product_mod5_is_zero_l114_114609


namespace least_value_of_q_minus_p_l114_114996

variables (y p q : ℝ)

/-- Triangle side lengths -/
def BC := y + 7
def AC := y + 3
def AB := 2 * y + 1

/-- Given conditions for triangle inequalities and angle B being the largest -/
def triangle_inequality_conditions :=
  (y + 7 + (y + 3) > 2 * y + 1) ∧
  (y + 7 + (2 * y + 1) > y + 3) ∧
  ((y + 3) + (2 * y + 1) > y + 7)

def angle_largest_conditions :=
  (2 * y + 1 > y + 3) ∧
  (2 * y + 1 > y + 7)

/-- Prove the least possible value of q - p given the conditions -/
theorem least_value_of_q_minus_p
  (h1 : triangle_inequality_conditions y)
  (h2 : angle_largest_conditions y)
  (h3 : 6 < y)
  (h4 : y < 8) :
  q - p = 2 := sorry

end least_value_of_q_minus_p_l114_114996


namespace fraction_expression_l114_114434

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 + 1 / 2) = 1 / 10 :=
by
  sorry

end fraction_expression_l114_114434


namespace consumer_installment_credit_value_l114_114063

variable (consumer_installment_credit : ℝ) 

noncomputable def automobile_installment_credit := 0.36 * consumer_installment_credit

noncomputable def finance_company_credit := 35

theorem consumer_installment_credit_value :
  (∃ C : ℝ, automobile_installment_credit C = 0.36 * C ∧ finance_company_credit = (1 / 3) * automobile_installment_credit C) →
  consumer_installment_credit = 291.67 :=
by
  sorry

end consumer_installment_credit_value_l114_114063


namespace total_students_l114_114433

theorem total_students (girls boys : ℕ) (h1 : girls = 300) (h2 : boys = 8 * (girls / 5)) : girls + boys = 780 := by
  sorry

end total_students_l114_114433


namespace find_other_number_l114_114269

theorem find_other_number (HCF LCM one_number other_number : ℤ)
  (hHCF : HCF = 12)
  (hLCM : LCM = 396)
  (hone_number : one_number = 48)
  (hrelation : HCF * LCM = one_number * other_number) :
  other_number = 99 :=
by
  sorry

end find_other_number_l114_114269


namespace a_values_l114_114033

def A (a : ℤ) : Set ℤ := {2, a^2 - a + 2, 1 - a}

theorem a_values (a : ℤ) (h : 4 ∈ A a) : a = 2 ∨ a = -3 :=
sorry

end a_values_l114_114033


namespace value_of_x2_y2_z2_l114_114702

variable (x y z : ℝ)

theorem value_of_x2_y2_z2 (h1 : x^2 + 3 * y = 4) 
                          (h2 : y^2 - 5 * z = 5) 
                          (h3 : z^2 - 7 * x = -8) : 
                          x^2 + y^2 + z^2 = 20.75 := 
by
  sorry

end value_of_x2_y2_z2_l114_114702


namespace necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l114_114634

-- Problem 1
theorem necessary_condition_for_q_implies_m_bounds (m : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 20 ≤ 0 → 1 - m^2 ≤ x ∧ x ≤ 1 + m^2) → (- Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
sorry

-- Problem 2
theorem necessary_but_not_sufficient_condition_for_not_q (m : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - 8 * x - 20 ≤ 0) → ¬ (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) → (m ≥ 3 ∨ m ≤ -3) :=
sorry

end necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l114_114634


namespace sequence_formula_l114_114992

theorem sequence_formula (a : ℕ → ℕ) (n : ℕ) (h : ∀ n ≥ 1, a n = a (n - 1) + n^3) : 
  a n = (n * (n + 1) / 2) ^ 2 := sorry

end sequence_formula_l114_114992


namespace expression_divisible_by_41_l114_114520

theorem expression_divisible_by_41 (n : ℕ) : 41 ∣ (5 * 7^(2*(n+1)) + 2^(3*n)) :=
  sorry

end expression_divisible_by_41_l114_114520


namespace sum_arithmetic_series_remainder_l114_114037

theorem sum_arithmetic_series_remainder :
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S % 9 = 5 :=
by
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  show S % 9 = 5
  sorry

end sum_arithmetic_series_remainder_l114_114037


namespace decimal_to_base8_conversion_l114_114930

-- Define the base and the number in decimal.
def base : ℕ := 8
def decimal_number : ℕ := 127

-- Define the expected representation in base 8.
def expected_base8_representation : ℕ := 177

-- Theorem stating that conversion of 127 in base 10 to base 8 yields 177
theorem decimal_to_base8_conversion : Nat.ofDigits base (Nat.digits base decimal_number) = expected_base8_representation := 
by
  sorry

end decimal_to_base8_conversion_l114_114930


namespace odd_nat_existence_l114_114740

theorem odd_nat_existence (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (n : ℕ) :
  ∃ m : ℕ, (a^m * b^2 - 1) % 2^n = 0 ∨ (b^m * a^2 - 1) % 2^n = 0 := 
by
  sorry

end odd_nat_existence_l114_114740


namespace calculate_angle_l114_114007

def degrees_to_seconds (d m s : ℕ) : ℕ :=
  d * 3600 + m * 60 + s

def seconds_to_degrees (s : ℕ) : (ℕ × ℕ × ℕ) :=
  (s / 3600, (s % 3600) / 60, s % 60)

theorem calculate_angle : 
  (let d1 := 50
   let m1 := 24
   let angle1_sec := degrees_to_seconds d1 m1 0
   let angle1_sec_tripled := 3 * angle1_sec
   let (d1', m1', s1') := seconds_to_degrees angle1_sec_tripled

   let d2 := 98
   let m2 := 12
   let s2 := 25
   let angle2_sec := degrees_to_seconds d2 m2 s2
   let angle2_sec_divided := angle2_sec / 5
   let (d2', m2', s2') := seconds_to_degrees angle2_sec_divided

   let total_sec := degrees_to_seconds d1' m1' s1' + degrees_to_seconds d2' m2' s2'
   let (final_d, final_m, final_s) := seconds_to_degrees total_sec
   (final_d, final_m, final_s)) = (170, 50, 29) := by sorry

end calculate_angle_l114_114007


namespace seating_arrangements_l114_114406

theorem seating_arrangements :
  let total_arrangements := Nat.factorial 8
  let jwp_together := (Nat.factorial 6) * (Nat.factorial 3)
  total_arrangements - jwp_together = 36000 := by
  sorry

end seating_arrangements_l114_114406


namespace pond_length_l114_114045

theorem pond_length (L W S : ℝ) (h1 : L = 2 * W) (h2 : L = 80) (h3 : S^2 = (1/50) * (L * W)) : S = 8 := 
by 
  -- Insert proof here 
  sorry

end pond_length_l114_114045


namespace triangle_side_s_l114_114062

/-- The sides of a triangle have lengths 8, 13, and s where s is a whole number.
    What is the smallest possible value of s?
    We need to show that the minimum possible value of s such that 8 + s > 13,
    s < 21, and 13 + s > 8 is s = 6. -/
theorem triangle_side_s (s : ℕ) : 
  (8 + s > 13) ∧ (8 + 13 > s) ∧ (13 + s > 8) → s = 6 :=
by
  sorry

end triangle_side_s_l114_114062


namespace find_intersection_l114_114361

variable (A : Set ℝ)
variable (B : Set ℝ := {1, 2})
variable (f : ℝ → ℝ := λ x => x^2)

theorem find_intersection (h : ∀ x, x ∈ A → f x ∈ B) : A ∩ B = ∅ ∨ A ∩ B = {1} :=
by
  sorry

end find_intersection_l114_114361


namespace arthur_money_left_l114_114904

theorem arthur_money_left {initial_amount spent_fraction : ℝ} (h_initial : initial_amount = 200) (h_fraction : spent_fraction = 4 / 5) : 
  (initial_amount - spent_fraction * initial_amount = 40) :=
by
  sorry

end arthur_money_left_l114_114904


namespace pen_and_notebook_cost_l114_114949

theorem pen_and_notebook_cost (pen_cost : ℝ) (notebook_cost : ℝ) 
  (h1 : pen_cost = 4.5) 
  (h2 : pen_cost = notebook_cost + 1.8) : 
  pen_cost + notebook_cost = 7.2 := 
  by
    sorry

end pen_and_notebook_cost_l114_114949


namespace find_first_m_gt_1959_l114_114244

theorem find_first_m_gt_1959 :
  ∃ m n : ℕ, 8 * m - 7 = n^2 ∧ m > 1959 ∧ m = 2017 :=
by
  sorry

end find_first_m_gt_1959_l114_114244


namespace solve_fractional_equation_l114_114948

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x / (x - 1) = 3) ↔ x = 3 := 
by
  sorry

end solve_fractional_equation_l114_114948


namespace y_greater_than_one_l114_114021

variable (x y : ℝ)

theorem y_greater_than_one (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
sorry

end y_greater_than_one_l114_114021


namespace solve_system_of_equations_l114_114873

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x * y * (x + y) = 30 ∧ x^3 + y^3 = 35 ∧ ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
sorry

end solve_system_of_equations_l114_114873


namespace range_of_a_for_critical_points_l114_114450

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + a * x + 3

theorem range_of_a_for_critical_points : 
  ∀ a : ℝ, (∃ x : ℝ, deriv (f a) x = 0) ↔ (a < 0 ∨ a > 3) :=
by
  sorry

end range_of_a_for_critical_points_l114_114450


namespace find_missing_digit_l114_114378

theorem find_missing_digit 
  (x : Nat) 
  (h : 16 + x ≡ 0 [MOD 9]) : 
  x = 2 :=
sorry

end find_missing_digit_l114_114378


namespace fraction_of_original_price_l114_114266

theorem fraction_of_original_price
  (CP SP : ℝ)
  (h1 : SP = 1.275 * CP)
  (f: ℝ)
  (h2 : f * SP = 0.85 * CP)
  : f = 17 / 25 :=
by
  sorry

end fraction_of_original_price_l114_114266


namespace minimum_value_l114_114837

theorem minimum_value : 
  ∀ a b : ℝ, 0 < a → 0 < b → a + 2 * b = 3 → (1 / a + 1 / b) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
by
  sorry

end minimum_value_l114_114837


namespace man_has_2_nickels_l114_114026

theorem man_has_2_nickels
  (d n : ℕ)
  (h1 : 10 * d + 5 * n = 70)
  (h2 : d + n = 8) :
  n = 2 := 
by
  -- omit the proof
  sorry

end man_has_2_nickels_l114_114026


namespace number_of_extremum_points_of_f_l114_114505

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (x + 1)^3 * Real.exp (x + 1) else (-(x + 1))^3 * Real.exp (-(x + 1))

theorem number_of_extremum_points_of_f :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    ((f (x1 - epsilon) < f x1 ∧ f x1 > f (x1 + epsilon)) ∨ (f (x1 - epsilon) > f x1 ∧ f x1 < f (x1 + epsilon))) ∧
    ((f (x2 - epsilon) < f x2 ∧ f x2 > f (x2 + epsilon)) ∨ (f (x2 - epsilon) > f x2 ∧ f x2 < f (x2 + epsilon))) ∧
    ((f (x3 - epsilon) < f x3 ∧ f x3 > f (x3 + epsilon)) ∨ (f (x3 - epsilon) > f x3 ∧ f x3 < f (x3 + epsilon)))) :=
sorry

end number_of_extremum_points_of_f_l114_114505


namespace circleAtBottomAfterRotation_l114_114099

noncomputable def calculateFinalCirclePosition (initialPosition : String) (sides : ℕ) : String :=
  if (sides = 8) then (if initialPosition = "bottom" then "bottom" else "unknown") else "unknown"

theorem circleAtBottomAfterRotation :
  calculateFinalCirclePosition "bottom" 8 = "bottom" :=
by
  sorry

end circleAtBottomAfterRotation_l114_114099


namespace sum_of_integers_is_18_l114_114027

theorem sum_of_integers_is_18 (a b c d : ℕ) 
  (h1 : a * b + c * d = 38)
  (h2 : a * c + b * d = 34)
  (h3 : a * d + b * c = 43) : 
  a + b + c + d = 18 := 
  sorry

end sum_of_integers_is_18_l114_114027


namespace number_of_purple_balls_l114_114346

theorem number_of_purple_balls (k : ℕ) (h : k > 0) (E : (24 - k) / (8 + k) = 1) : k = 8 :=
by {
  sorry
}

end number_of_purple_balls_l114_114346


namespace arrange_6_books_l114_114469

theorem arrange_6_books :
  Nat.factorial 6 = 720 :=
by
  sorry

end arrange_6_books_l114_114469


namespace roger_forgot_lawns_l114_114993

theorem roger_forgot_lawns
  (dollars_per_lawn : ℕ)
  (total_lawns : ℕ)
  (total_earned : ℕ)
  (actual_mowed_lawns : ℕ)
  (forgotten_lawns : ℕ)
  (h1 : dollars_per_lawn = 9)
  (h2 : total_lawns = 14)
  (h3 : total_earned = 54)
  (h4 : actual_mowed_lawns = total_earned / dollars_per_lawn) :
  forgotten_lawns = total_lawns - actual_mowed_lawns :=
  sorry

end roger_forgot_lawns_l114_114993


namespace sufficiency_of_p_for_q_not_necessity_of_p_for_q_l114_114419

noncomputable def p (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m
noncomputable def q (m : ℝ) := ∀ x : ℝ, (- (5 - 2 * m)) ^ x < 0

theorem sufficiency_of_p_for_q : ∀ m : ℝ, (m < 1 → m < 2) :=
by sorry

theorem not_necessity_of_p_for_q : ∀ m : ℝ, ¬ (m < 2 → m < 1) :=
by sorry

end sufficiency_of_p_for_q_not_necessity_of_p_for_q_l114_114419


namespace average_age_inhabitants_Campo_Verde_l114_114331

theorem average_age_inhabitants_Campo_Verde
  (H M : ℕ)
  (ratio_h_m : H / M = 2 / 3)
  (avg_age_men : ℕ := 37)
  (avg_age_women : ℕ := 42) :
  ((37 * H + 42 * M) / (H + M) : ℕ) = 40 := 
sorry

end average_age_inhabitants_Campo_Verde_l114_114331


namespace weekend_price_is_correct_l114_114998

-- Define the original price of the jacket
def original_price : ℝ := 250

-- Define the first discount rate (40%)
def first_discount_rate : ℝ := 0.40

-- Define the additional weekend discount rate (10%)
def additional_discount_rate : ℝ := 0.10

-- Define a function to apply the first discount
def apply_first_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Define a function to apply the additional discount
def apply_additional_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Using both discounts, calculate the final weekend price
def weekend_price : ℝ :=
  apply_additional_discount (apply_first_discount original_price first_discount_rate) additional_discount_rate

-- The final theorem stating the expected weekend price is $135
theorem weekend_price_is_correct : weekend_price = 135 := by
  sorry

end weekend_price_is_correct_l114_114998


namespace fruit_order_count_l114_114954

-- Define the initial conditions
def apples := 3
def oranges := 2
def bananas := 2
def totalFruits := apples + oranges + bananas -- which is 7

-- Calculate the factorial of a number
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Noncomputable definition to skip proof
noncomputable def distinctOrders : ℕ :=
  fact totalFruits / (fact apples * fact oranges * fact bananas)

-- Lean statement expressing that the number of distinct orders is 210
theorem fruit_order_count : distinctOrders = 210 :=
by
  sorry

end fruit_order_count_l114_114954


namespace not_equal_77_l114_114357

theorem not_equal_77 (x y : ℤ) : x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end not_equal_77_l114_114357


namespace triangle_sum_is_16_l114_114633

-- Definition of the triangle operation
def triangle (a b c : ℕ) : ℕ := a * b - c

-- Lean theorem statement
theorem triangle_sum_is_16 : 
  triangle 2 4 3 + triangle 3 6 7 = 16 := 
by 
  sorry

end triangle_sum_is_16_l114_114633


namespace minimum_guests_l114_114654

-- Define the conditions as variables
def total_food : ℕ := 4875
def max_food_per_guest : ℕ := 3

-- Define the theorem we need to prove
theorem minimum_guests : ∃ g : ℕ, g * max_food_per_guest = total_food ∧ g >= 1625 := by
  sorry

end minimum_guests_l114_114654


namespace evaluate_expression_l114_114822

theorem evaluate_expression (m n : ℝ) (h : 4 * m - 4 + n = 2) : 
  (m * (-2)^2 - 2 * (-2) + n = 10) :=
by
  sorry

end evaluate_expression_l114_114822


namespace quadratic_roots_product_l114_114059

theorem quadratic_roots_product :
  ∀ (x1 x2: ℝ), (x1^2 - 4 * x1 - 2 = 0 ∧ x2^2 - 4 * x2 - 2 = 0) → (x1 * x2 = -2) :=
by
  -- Assume x1 and x2 are roots of the quadratic equation
  intros x1 x2 h
  sorry

end quadratic_roots_product_l114_114059


namespace max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l114_114288

-- Problem (1)
theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x*y ≤ 4 :=
sorry

-- Additional statement to show when the maximum is achieved
theorem max_xy_is_4 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x = 4 ∧ y = 1 ↔ x*y = 4 :=
sorry

-- Problem (2)
theorem min_x_plus_y (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x + y ≥ 9 :=
sorry

-- Additional statement to show when the minimum is achieved
theorem min_x_plus_y_is_9 (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x = 6 ∧ y = 3 ↔ x + y = 9 :=
sorry

end max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l114_114288


namespace inequality_proof_l114_114711

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem inequality_proof :
  (a / (a + b)) * ((a + 2 * b) / (a + 3 * b)) < Real.sqrt (a / (a + 4 * b)) :=
sorry

end inequality_proof_l114_114711


namespace length_of_shop_proof_l114_114380

-- Given conditions
def monthly_rent : ℝ := 1440
def width : ℝ := 20
def annual_rent_per_sqft : ℝ := 48

-- Correct answer to be proved
def length_of_shop : ℝ := 18

-- The following statement is the proof problem in Lean 4
theorem length_of_shop_proof (h1 : monthly_rent = 1440) 
                            (h2 : width = 20) 
                            (h3 : annual_rent_per_sqft = 48) : 
  length_of_shop = 18 := 
  sorry

end length_of_shop_proof_l114_114380


namespace sum_of_ages_is_220_l114_114686

-- Definitions based on the conditions
def father_age (S : ℕ) := (7 * S) / 4
def sum_ages (F S : ℕ) := F + S

-- The proof statement
theorem sum_of_ages_is_220 (F S : ℕ) (h1 : 4 * F = 7 * S)
  (h2 : 3 * (F + 10) = 5 * (S + 10)) : sum_ages F S = 220 :=
by
  sorry

end sum_of_ages_is_220_l114_114686


namespace inverse_linear_intersection_l114_114646

theorem inverse_linear_intersection (m n : ℝ) 
  (h1 : n = 2 / m) 
  (h2 : n = m + 3) 
  : (1 / m) - (1 / n) = 3 / 2 := 
by sorry

end inverse_linear_intersection_l114_114646


namespace factories_checked_by_second_group_l114_114808

theorem factories_checked_by_second_group 
(T : ℕ) (G1 : ℕ) (R : ℕ) 
(hT : T = 169) 
(hG1 : G1 = 69) 
(hR : R = 48) : 
T - (G1 + R) = 52 :=
by {
  sorry
}

end factories_checked_by_second_group_l114_114808


namespace sum_of_decimals_l114_114529

theorem sum_of_decimals :
  5.467 + 2.349 + 3.785 = 11.751 :=
sorry

end sum_of_decimals_l114_114529


namespace range_of_a_l114_114921

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l114_114921


namespace find_a_l114_114420

theorem find_a (a r : ℝ) (h1 : a * r = 24) (h2 : a * r^4 = 3) : a = 48 :=
sorry

end find_a_l114_114420


namespace problem1_problem2_l114_114936

-- First proof problem
theorem problem1 (a b : ℝ) : a^4 + 6 * a^2 * b^2 + b^4 ≥ 4 * a * b * (a^2 + b^2) :=
by sorry

-- Second proof problem
theorem problem2 (a b : ℝ) : ∃ (x : ℝ), 
  (∀ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| ≥ 1) ∧
  ∃ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| = 1 :=
by sorry

end problem1_problem2_l114_114936


namespace sum_of_four_digit_numbers_l114_114087

open Nat

theorem sum_of_four_digit_numbers (s : Finset ℤ) :
  (∀ x, x ∈ s → (∃ k, x = 30 * k + 2) ∧ 1000 ≤ x ∧ x ≤ 9999) →
  s.sum id = 1652100 := by
  sorry

end sum_of_four_digit_numbers_l114_114087


namespace probability_of_Z_l114_114441

namespace ProbabilityProof

def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_X_or_Y_or_Z : ℚ := 0.4583333333333333

theorem probability_of_Z :
  ∃ P_Z : ℚ, P_Z = 0.0833333333333333 ∧ 
  P_X_or_Y_or_Z = P_X + P_Y + P_Z :=
by
  sorry

end ProbabilityProof

end probability_of_Z_l114_114441


namespace tom_age_ratio_l114_114504

-- Define the constants T and N with the given conditions
variables (T N : ℕ)
-- Tom's age T years, sum of three children's ages is also T
-- N years ago, Tom's age was three times the sum of children's ages then

-- We need to prove that T / N = 4 under these conditions
theorem tom_age_ratio (h1 : T = 3 * T - 8 * N) : T / N = 4 :=
sorry

end tom_age_ratio_l114_114504


namespace initial_amount_is_53_l114_114881

variable (X : ℕ) -- Initial amount of money Olivia had
variable (ATM_collect : ℕ := 91) -- Money collected from ATM
variable (supermarket_spent_diff : ℕ := 39) -- Spent 39 dollars more at the supermarket
variable (money_left : ℕ := 14) -- Money left after supermarket

-- Define the final amount Olivia had
def final_amount (X ATM_collect supermarket_spent_diff : ℕ) : ℕ :=
  X + ATM_collect - (ATM_collect + supermarket_spent_diff)

-- Theorem stating that the initial amount X was 53 dollars
theorem initial_amount_is_53 : final_amount X ATM_collect supermarket_spent_diff = money_left → X = 53 :=
by
  intros h
  sorry

end initial_amount_is_53_l114_114881


namespace find_b_and_sinA_find_sin_2A_plus_pi_over_4_l114_114340

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinB : ℝ)

-- Conditions
def triangle_conditions :=
  (a > b) ∧
  (a = 5) ∧
  (c = 6) ∧
  (sinB = 3 / 5)

-- Question 1: Prove b = sqrt 13 and sin A = (3 * sqrt 13) / 13
theorem find_b_and_sinA (h : triangle_conditions a b c sinB) :
  b = Real.sqrt 13 ∧
  ∃ sinA : ℝ, sinA = (3 * Real.sqrt 13) / 13 :=
  sorry

-- Question 2: Prove sin (2A + π/4) = 7 * sqrt 2 / 26
theorem find_sin_2A_plus_pi_over_4 (h : triangle_conditions a b c sinB)
  (hb : b = Real.sqrt 13)
  (sinA : ℝ)
  (h_sinA : sinA = (3 * Real.sqrt 13) / 13) :
  ∃ sin2Aπ4 : ℝ, sin2Aπ4 = (7 * Real.sqrt 2) / 26 :=
  sorry

end find_b_and_sinA_find_sin_2A_plus_pi_over_4_l114_114340


namespace factor_polynomial_l114_114293

theorem factor_polynomial (x y z : ℤ) :
  x * (y - z) ^ 3 + y * (z - x) ^ 3 + z * (x - y) ^ 3 = (x - y) * (y - z) * (z - x) * (x + y + z) := 
by
  sorry

end factor_polynomial_l114_114293


namespace triangle_side_inequality_l114_114791

theorem triangle_side_inequality (y : ℕ) (h : 3 < y^2 ∧ y^2 < 19) : 
  y = 2 ∨ y = 3 ∨ y = 4 :=
sorry

end triangle_side_inequality_l114_114791


namespace train_average_speed_with_stoppages_l114_114733

theorem train_average_speed_with_stoppages :
  (∀ d t_without_stops t_with_stops : ℝ, t_without_stops = d / 400 → 
  t_with_stops = d / (t_without_stops * (10/9)) → 
  t_with_stops = d / 360) :=
sorry

end train_average_speed_with_stoppages_l114_114733


namespace ounces_per_cup_l114_114412

theorem ounces_per_cup (total_ounces : ℕ) (total_cups : ℕ) 
  (h : total_ounces = 264 ∧ total_cups = 33) : total_ounces / total_cups = 8 :=
by
  sorry

end ounces_per_cup_l114_114412


namespace hyperbola_asymptotes_l114_114830

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (y^2 / 9 - x^2 / 4 = 1 →
  (y = (3 / 2) * x ∨ y = - (3 / 2) * x)) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l114_114830


namespace determine_lunch_break_duration_lunch_break_duration_in_minutes_l114_114946

noncomputable def painter_lunch_break_duration (j h L : ℝ) : Prop :=
  (10 - L) * (j + h) = 0.6 ∧
  (8 - L) * h = 0.3 ∧
  (5 - L) * j = 0.1

theorem determine_lunch_break_duration (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L = 0.8 :=
by sorry

theorem lunch_break_duration_in_minutes (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L * 60 = 48 :=
by sorry

end determine_lunch_break_duration_lunch_break_duration_in_minutes_l114_114946


namespace expr_undefined_iff_l114_114486

theorem expr_undefined_iff (x : ℝ) : (x^2 - 9 = 0) ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end expr_undefined_iff_l114_114486


namespace part1_part2_l114_114533

def f (x a : ℝ) : ℝ := |x - a| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x 2 > 5 ↔ x < - 1 / 3 ∨ x > 3 :=
by sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x a ≤ |a - 2|) → a ≤ 3 / 2 :=
by sorry

end part1_part2_l114_114533


namespace compute_a2004_l114_114247

def recurrence_sequence (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 0
  else sorry -- We'll define recurrence operations in the proofs

theorem compute_a2004 : recurrence_sequence 2004 = -2^1002 := 
sorry -- Proof omitted

end compute_a2004_l114_114247


namespace solve_x_value_l114_114962
-- Import the necessary libraries

-- Define the problem and the main theorem
theorem solve_x_value (x : ℝ) (h : 3 / x^2 = x / 27) : x = 3 * Real.sqrt 3 :=
by
  sorry

end solve_x_value_l114_114962


namespace find_larger_number_l114_114753

theorem find_larger_number :
  ∃ (x y : ℝ), (y = x + 10) ∧ (x = y / 2) ∧ (x + y = 34) → y = 20 :=
by
  sorry

end find_larger_number_l114_114753


namespace bus_stop_time_l114_114714

theorem bus_stop_time (v_exclude_stop v_include_stop : ℕ) (h1 : v_exclude_stop = 54) (h2 : v_include_stop = 36) : 
  ∃ t: ℕ, t = 20 :=
by
  sorry

end bus_stop_time_l114_114714


namespace parallel_lines_iff_a_eq_neg3_l114_114833

theorem parallel_lines_iff_a_eq_neg3 (a : ℝ) :
  (∀ x y : ℝ, a * x + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 ≠ 0) ↔ a = -3 :=
sorry

end parallel_lines_iff_a_eq_neg3_l114_114833


namespace parents_years_in_america_before_aziz_birth_l114_114509

noncomputable def aziz_birth_year (current_year : ℕ) (aziz_age : ℕ) : ℕ :=
  current_year - aziz_age

noncomputable def years_parents_in_america_before_aziz_birth (arrival_year : ℕ) (aziz_birth_year : ℕ) : ℕ :=
  aziz_birth_year - arrival_year

theorem parents_years_in_america_before_aziz_birth 
  (current_year : ℕ := 2021) 
  (aziz_age : ℕ := 36) 
  (arrival_year : ℕ := 1982) 
  (expected_years : ℕ := 3) :
  years_parents_in_america_before_aziz_birth arrival_year (aziz_birth_year current_year aziz_age) = expected_years :=
by 
  sorry

end parents_years_in_america_before_aziz_birth_l114_114509


namespace tomatoes_for_5_liters_l114_114291

theorem tomatoes_for_5_liters (kg_per_3_liters : ℝ) (liters_needed : ℝ) :
  (kg_per_3_liters = 69 / 3) → (liters_needed = 5) → (kg_per_3_liters * liters_needed = 115) := 
by
  intros h1 h2
  sorry

end tomatoes_for_5_liters_l114_114291


namespace tournament_games_count_l114_114786

-- We define the conditions
def number_of_players : ℕ := 6

-- Function to calculate the number of games played in a tournament where each player plays twice with each opponent
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Now we state the theorem
theorem tournament_games_count : total_games number_of_players = 60 := by
  -- Proof goes here
  sorry

end tournament_games_count_l114_114786


namespace sufficient_and_necessary_condition_l114_114147

variable (a_n : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
variable (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
variable (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d))

theorem sufficient_and_necessary_condition (d : ℚ) (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d)) :
  (d > 0) ↔ (S 4 + S 6 > 2 * S 5) := by
  sorry

end sufficient_and_necessary_condition_l114_114147


namespace y_minus_x_value_l114_114285

theorem y_minus_x_value (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 55.56 :=
sorry

end y_minus_x_value_l114_114285


namespace greatest_value_a_plus_b_l114_114929

theorem greatest_value_a_plus_b (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) : a + b = 2 * Real.sqrt 55 :=
by
  sorry

end greatest_value_a_plus_b_l114_114929


namespace find_N_l114_114821

theorem find_N :
  ∃ N : ℕ,
  (5 + 6 + 7 + 8 + 9) / 5 = (2005 + 2006 + 2007 + 2008 + 2009) / (N : ℝ) ∧ N = 1433 :=
sorry

end find_N_l114_114821


namespace find_num_20_paise_coins_l114_114427

def num_20_paise_coins (x y : ℕ) : Prop :=
  x + y = 334 ∧ 20 * x + 25 * y = 7100

theorem find_num_20_paise_coins (x y : ℕ) (h : num_20_paise_coins x y) : x = 250 :=
by
  sorry

end find_num_20_paise_coins_l114_114427


namespace geometric_sequence_angle_count_l114_114979

theorem geometric_sequence_angle_count :
  (∃ θs : Finset ℝ, (∀ θ ∈ θs, 0 < θ ∧ θ < 2 * π ∧ ¬ ∃ k : ℕ, θ = k * (π / 2)) 
                    ∧ θs.card = 4
                    ∧ ∀ θ ∈ θs, ∃ a b c : ℝ, (a, b, c) = (Real.sin θ, Real.cos θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.sin θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.cos θ, Real.tan θ)
                                             ∧ b = a * c) :=
sorry

end geometric_sequence_angle_count_l114_114979


namespace sum_of_marked_angles_l114_114209

theorem sum_of_marked_angles (sum_of_angles_around_vertex : ℕ := 360) 
    (vertices : ℕ := 7) (triangles : ℕ := 3) 
    (sum_of_interior_angles_triangle : ℕ := 180) :
    (vertices * sum_of_angles_around_vertex - triangles * sum_of_interior_angles_triangle) = 1980 :=
by
  sorry

end sum_of_marked_angles_l114_114209


namespace inequality_may_not_hold_l114_114029

theorem inequality_may_not_hold (m n : ℝ) (h : m > n) : ¬ (m^2 > n^2) :=
by
  -- Leaving the proof out according to the instructions.
  sorry

end inequality_may_not_hold_l114_114029


namespace proof_part_a_l114_114814

variable {α : Type} [LinearOrder α]

structure ConvexQuadrilateral (α : Type) :=
(a b c d : α)
(a'b'c'd' : α)
(ab_eq_a'b' : α)
(bc_eq_b'c' : α)
(cd_eq_c'd' : α)
(da_eq_d'a' : α)
(angle_A_gt_angle_A' : Prop)
(angle_B_lt_angle_B' : Prop)
(angle_C_gt_angle_C' : Prop)
(angle_D_lt_angle_D' : Prop)

theorem proof_part_a (Quad : ConvexQuadrilateral ℝ) : 
  Quad.angle_A_gt_angle_A' → 
  Quad.angle_B_lt_angle_B' ∧ Quad.angle_C_gt_angle_C' ∧ Quad.angle_D_lt_angle_D' := sorry

end proof_part_a_l114_114814


namespace labor_hired_l114_114074

noncomputable def Q_d (P : ℝ) : ℝ := 60 - 14 * P
noncomputable def Q_s (P : ℝ) : ℝ := 20 + 6 * P
noncomputable def MPL (L : ℝ) : ℝ := 160 / (L^2)
def wage : ℝ := 5

theorem labor_hired (L P : ℝ) (h_eq_price: 60 - 14 * P = 20 + 6 * P) (h_eq_wage: 160 / (L^2) * 2 = wage) :
  L = 8 :=
by
  have h1 : 60 - 14 * P = 20 + 6 * P := h_eq_price
  have h2 : 160 / (L^2) * 2 = wage := h_eq_wage
  sorry

end labor_hired_l114_114074


namespace income_second_day_l114_114028

theorem income_second_day (x : ℕ) 
  (h_condition : (200 + x + 750 + 400 + 500) / 5 = 400) : x = 150 :=
by 
  -- Proof omitted.
  sorry

end income_second_day_l114_114028


namespace y_n_sq_eq_3_x_n_sq_add_1_l114_114023

def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 1) => 4 * x n - x (n - 1)

def y : ℕ → ℤ
| 0       => 1
| 1       => 2
| (n + 1) => 4 * y n - y (n - 1)

theorem y_n_sq_eq_3_x_n_sq_add_1 (n : ℕ) : y n ^ 2 = 3 * (x n) ^ 2 + 1 :=
sorry

end y_n_sq_eq_3_x_n_sq_add_1_l114_114023


namespace downstream_speed_l114_114358

variable (Vu Vs Vd Vc : ℝ)

theorem downstream_speed
  (h1 : Vu = 25)
  (h2 : Vs = 32)
  (h3 : Vu = Vs - Vc)
  (h4 : Vd = Vs + Vc) :
  Vd = 39 := by
  sorry

end downstream_speed_l114_114358


namespace roots_of_polynomial_l114_114088

noncomputable def polynomial : Polynomial ℤ := Polynomial.X^3 - 4 * Polynomial.X^2 - Polynomial.X + 4

theorem roots_of_polynomial :
  (Polynomial.X - 1) * (Polynomial.X + 1) * (Polynomial.X - 4) = polynomial :=
by
  sorry

end roots_of_polynomial_l114_114088


namespace arithmetic_seq_max_sum_l114_114572

noncomputable def max_arith_seq_sum_lemma (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_seq_max_sum :
  ∀ (a1 d : ℤ),
    (3 * a1 + 6 * d = 9) →
    (a1 + 5 * d = -9) →
    max_arith_seq_sum_lemma a1 d 3 = 21 :=
by
  sorry

end arithmetic_seq_max_sum_l114_114572


namespace total_charging_time_l114_114727

def charge_smartphone_full : ℕ := 26
def charge_tablet_full : ℕ := 53
def charge_phone_half : ℕ := charge_smartphone_full / 2
def charge_tablet : ℕ := charge_tablet_full

theorem total_charging_time : 
  charge_phone_half + charge_tablet = 66 := by
  sorry

end total_charging_time_l114_114727


namespace right_triangle_hypotenuse_l114_114039

theorem right_triangle_hypotenuse (a b c : ℕ) (h1 : a^2 + b^2 = c^2) 
  (h2 : b = c - 1575) (h3 : b < 1991) : c = 1800 :=
sorry

end right_triangle_hypotenuse_l114_114039


namespace relationship_among_abc_l114_114153

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1/2)

theorem relationship_among_abc : a > b ∧ b > c :=
by {
  sorry
}

end relationship_among_abc_l114_114153


namespace cylinder_height_same_volume_as_cone_l114_114551

theorem cylinder_height_same_volume_as_cone
    (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V : ℝ)
    (h_volume_cone_eq : V = (1 / 3) * Real.pi * r_cone ^ 2 * h_cone)
    (r_cone_val : r_cone = 2)
    (h_cone_val : h_cone = 6)
    (r_cylinder_val : r_cylinder = 1) :
    ∃ h_cylinder : ℝ, (V = Real.pi * r_cylinder ^ 2 * h_cylinder) ∧ h_cylinder = 8 :=
by
  -- Here you would provide the proof for the theorem.
  sorry

end cylinder_height_same_volume_as_cone_l114_114551


namespace ashley_champagne_bottles_l114_114423

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end ashley_champagne_bottles_l114_114423


namespace probability_A_l114_114981

variable (A B : Prop)
variable (P : Prop → ℝ)

axiom prob_B : P B = 0.4
axiom prob_A_and_B : P (A ∧ B) = 0.15
axiom prob_notA_and_notB : P (¬ A ∧ ¬ B) = 0.5499999999999999

theorem probability_A : P A = 0.20 :=
by sorry

end probability_A_l114_114981


namespace students_not_enrolled_in_either_l114_114644

variable (total_students french_students german_students both_students : ℕ)

theorem students_not_enrolled_in_either (h1 : total_students = 60)
                                        (h2 : french_students = 41)
                                        (h3 : german_students = 22)
                                        (h4 : both_students = 9) :
    total_students - (french_students + german_students - both_students) = 6 := by
  sorry

end students_not_enrolled_in_either_l114_114644


namespace proof_fraction_l114_114016

def find_fraction (x : ℝ) : Prop :=
  (2 / 9) * x = 10 → (2 / 5) * x = 18

-- Optional, you can define x based on the condition:
noncomputable def certain_number : ℝ := 10 * (9 / 2)

theorem proof_fraction :
  find_fraction certain_number :=
by
  intro h
  sorry

end proof_fraction_l114_114016


namespace find_d_share_l114_114219

def money_distribution (a b c d : ℕ) (x : ℕ) := 
  a = 5 * x ∧ 
  b = 2 * x ∧ 
  c = 4 * x ∧ 
  d = 3 * x ∧ 
  (c = d + 500)

theorem find_d_share (a b c d x : ℕ) (h : money_distribution a b c d x) : d = 1500 :=
by
  --proof would go here
  sorry

end find_d_share_l114_114219


namespace monotone_decreasing_f_find_a_value_l114_114896

-- Condition declarations
variables (a b : ℝ) (h_a_pos : a > 0) (max_val min_val : ℝ)
noncomputable def f (x : ℝ) := x + (a / x) + b

-- Problem 1: Prove that f is monotonically decreasing in (0, sqrt(a)]
theorem monotone_decreasing_f : 
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a → f a b x1 > f a b x2) :=
sorry

-- Conditions for Problem 2
variable (hf_inc : ∀ x1 x2 : ℝ, Real.sqrt a ≤ x1 ∧ x1 < x2 → f a b x1 < f a b x2)
variable (h_max : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≤ 5)
variable (h_min : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≥ 3)

-- Problem 2: Find the value of a
theorem find_a_value : a = 6 :=
sorry

end monotone_decreasing_f_find_a_value_l114_114896


namespace faye_total_crayons_l114_114329

-- Define the number of rows and the number of crayons per row as given conditions.
def num_rows : ℕ := 7
def crayons_per_row : ℕ := 30

-- State the theorem we need to prove.
theorem faye_total_crayons : (num_rows * crayons_per_row) = 210 :=
by
  sorry

end faye_total_crayons_l114_114329


namespace intersection_M_N_l114_114639

def M : Set ℝ := { x | Real.exp (x - 1) > 1 }
def N : Set ℝ := { x | x^2 - 2*x - 3 < 0 }

theorem intersection_M_N :
  (M ∩ N : Set ℝ) = { x | 1 < x ∧ x < 3 } := 
by
  sorry

end intersection_M_N_l114_114639


namespace monthly_income_of_B_l114_114110

variable (x y : ℝ)

-- Monthly incomes in the ratio 5:6
axiom income_ratio (A_income B_income : ℝ) : A_income = 5 * x ∧ B_income = 6 * x

-- Monthly expenditures in the ratio 3:4
axiom expenditure_ratio (A_expenditure B_expenditure : ℝ) : A_expenditure = 3 * y ∧ B_expenditure = 4 * y

-- Savings of A and B
axiom savings_A (A_income A_expenditure : ℝ) : 1800 = A_income - A_expenditure
axiom savings_B (B_income B_expenditure : ℝ) : 1600 = B_income - B_expenditure

-- The theorem to prove
theorem monthly_income_of_B (B_income : ℝ) (x y : ℝ) 
  (h1 : A_income = 5 * x)
  (h2 : B_income = 6 * x)
  (h3: A_expenditure = 3 * y)
  (h4: B_expenditure = 4 * y)
  (h5 : 1800 = 5 * x - 3 * y)
  (h6 : 1600 = 6 * x - 4 * y)
  : B_income = 7200 := by
  sorry

end monthly_income_of_B_l114_114110


namespace cylinder_ellipse_major_axis_l114_114664

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ), r = 2 →
  ∀ (minor_axis : ℝ), minor_axis = 2 * r →
  ∀ (major_axis : ℝ), major_axis = 1.4 * minor_axis →
  major_axis = 5.6 :=
by
  intros r hr minor_axis hminor major_axis hmajor
  sorry

end cylinder_ellipse_major_axis_l114_114664


namespace find_m_l114_114743

theorem find_m (m : ℝ) :
  (∃ x : ℝ, x^2 - m * x + m^2 - 19 = 0 ∧ (x = 2 ∨ x = 3))
  ∧ (∀ x : ℝ, x^2 - m * x + m^2 - 19 = 0 → x ≠ 2 ∧ x ≠ -4) 
  → m = -2 :=
by
  sorry

end find_m_l114_114743


namespace tetrahedron_edge_length_l114_114289

-- Definitions corresponding to the conditions of the problem.
def radius : ℝ := 2

def diameter : ℝ := 2 * radius

/-- Centers of four mutually tangent balls -/
def center_distance : ℝ := diameter

/-- The side length of the square formed by the centers of four balls on the floor. -/
def side_length_of_square : ℝ := center_distance

/-- The edge length of the tetrahedron circumscribed around the four balls. -/
def edge_length_tetrahedron : ℝ := side_length_of_square

-- The statement to be proved.
theorem tetrahedron_edge_length :
  edge_length_tetrahedron = 4 :=
by
  sorry  -- Proof to be constructed

end tetrahedron_edge_length_l114_114289


namespace find_sum_a100_b100_l114_114709

-- Definitions of arithmetic sequences and their properties
structure arithmetic_sequence (an : ℕ → ℝ) :=
  (a1 : ℝ)
  (d : ℝ)
  (def_seq : ∀ n, an n = a1 + (n - 1) * d)

-- Given conditions
variables (a_n b_n : ℕ → ℝ)
variables (ha : arithmetic_sequence a_n)
variables (hb : arithmetic_sequence b_n)

-- Specified conditions
axiom cond1 : a_n 5 + b_n 5 = 3
axiom cond2 : a_n 9 + b_n 9 = 19

-- The goal to be proved
theorem find_sum_a100_b100 : a_n 100 + b_n 100 = 383 :=
sorry

end find_sum_a100_b100_l114_114709


namespace largest_four_digit_number_prop_l114_114699

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end largest_four_digit_number_prop_l114_114699


namespace max_k_value_l114_114225

noncomputable def circle_equation (x y : ℝ) : Prop :=
x^2 + y^2 - 8 * x + 15 = 0

noncomputable def point_on_line (k x y : ℝ) : Prop :=
y = k * x - 2

theorem max_k_value (k : ℝ) :
  (∃ x y, circle_equation x y ∧ point_on_line k x y ∧ (x - 4)^2 + y^2 = 1) →
  k ≤ 4 / 3 :=
by
  sorry

end max_k_value_l114_114225


namespace aarti_three_times_work_l114_114679

theorem aarti_three_times_work (d : ℕ) (h : d = 5) : 3 * d = 15 :=
by
  sorry

end aarti_three_times_work_l114_114679


namespace adjacent_product_negative_l114_114177

noncomputable def a_seq : ℕ → ℚ
| 0 => 15
| (n+1) => (a_seq n) - (2 / 3)

theorem adjacent_product_negative :
  ∃ n : ℕ, a_seq 22 * a_seq 23 < 0 :=
by
  -- From the conditions, it is known that a_seq satisfies the recursive definition
  --
  -- We seek to prove that a_seq 22 * a_seq 23 < 0
  sorry

end adjacent_product_negative_l114_114177


namespace value_of_x_plus_inv_x_l114_114798

theorem value_of_x_plus_inv_x (x : ℝ) (hx : x ≠ 0) (t : ℝ) (ht : t = x^2 + (1 / x)^2) : x + (1 / x) = 5 :=
by
  have ht_val : t = 23 := by
    rw [ht] -- assuming t = 23 by condition
    sorry -- proof continuation placeholder

  -- introduce y and relate it to t
  let y := x + (1 / x)

  -- express t in terms of y and handle the algebra:
  have t_expr : t = y^2 - 2 := by
    sorry -- proof continuation placeholder

  -- show that y^2 = 25 and therefore y = 5 as the only valid solution:
  have y_val : y = 5 := by
    sorry -- proof continuation placeholder

  -- hence, the required value is found:
  exact y_val

end value_of_x_plus_inv_x_l114_114798


namespace max_hours_is_70_l114_114174

-- Define the conditions
def regular_hourly_rate : ℕ := 8
def first_20_hours : ℕ := 20
def max_weekly_earnings : ℕ := 660
def overtime_rate_multiplier : ℕ := 25

-- Define the overtime hourly rate
def overtime_hourly_rate : ℕ := regular_hourly_rate + (regular_hourly_rate * overtime_rate_multiplier / 100)

-- Define the earnings for the first 20 hours
def earnings_first_20_hours : ℕ := regular_hourly_rate * first_20_hours

-- Define the maximum overtime earnings
def max_overtime_earnings : ℕ := max_weekly_earnings - earnings_first_20_hours

-- Define the maximum overtime hours
def max_overtime_hours : ℕ := max_overtime_earnings / overtime_hourly_rate

-- Define the maximum total hours
def max_total_hours : ℕ := first_20_hours + max_overtime_hours

-- Theorem to prove that the maximum number of hours is 70
theorem max_hours_is_70 : max_total_hours = 70 :=
by
  sorry

end max_hours_is_70_l114_114174


namespace solve_problems_l114_114663

theorem solve_problems (x y : ℕ) (hx : x + y = 14) (hy : 7 * x - 12 * y = 60) : x = 12 :=
sorry

end solve_problems_l114_114663


namespace circle_parabola_intersect_l114_114730

theorem circle_parabola_intersect (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = 1 ∧ y = a * x^2 → (x ≠ 0 ∨ y ≠ 0)) ↔ a > 1 / 2 :=
by
  sorry

end circle_parabola_intersect_l114_114730


namespace avg_annual_growth_rate_optimal_selling_price_l114_114916

-- Define the conditions and question for the first problem: average annual growth rate.
theorem avg_annual_growth_rate (initial final : ℝ) (years : ℕ) (growth_rate : ℝ) :
  initial = 200 ∧ final = 288 ∧ years = 2 ∧ (final = initial * (1 + growth_rate)^years) →
  growth_rate = 0.2 :=
by
  -- Proof will come here
  sorry

-- Define the conditions and question for the second problem: setting the selling price.
theorem optimal_selling_price (cost initial_volume : ℕ) (initial_price : ℝ) 
(additional_sales_per_dollar : ℕ) (desired_profit : ℝ) (optimal_price : ℝ) :
  cost = 50 ∧ initial_volume = 50 ∧ initial_price = 100 ∧ additional_sales_per_dollar = 5 ∧
  desired_profit = 4000 ∧ 
  (∃ p : ℝ, (p - cost) * (initial_volume + additional_sales_per_dollar * (initial_price - p)) = desired_profit ∧ p = optimal_price) →
  optimal_price = 70 :=
by
  -- Proof will come here
  sorry

end avg_annual_growth_rate_optimal_selling_price_l114_114916


namespace ava_legs_count_l114_114999

-- Conditions:
-- There are a total of 9 animals in the farm.
-- There are only chickens and buffalos in the farm.
-- There are 5 chickens in the farm.

def total_animals : Nat := 9
def num_chickens : Nat := 5
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

-- Proof statement: Ava counted 26 legs.
theorem ava_legs_count (num_buffalos : Nat) 
  (H1 : total_animals = num_chickens + num_buffalos) : 
  num_chickens * legs_per_chicken + num_buffalos * legs_per_buffalo = 26 :=
by 
  have H2 : num_buffalos = total_animals - num_chickens := by sorry
  sorry

end ava_legs_count_l114_114999


namespace profit_percent_is_20_l114_114283

variable (C S : ℝ)

-- Definition from condition: The cost price of 60 articles is equal to the selling price of 50 articles
def condition : Prop := 60 * C = 50 * S

-- Definition of profit percent to be proven as 20%
def profit_percent_correct : Prop := ((S - C) / C) * 100 = 20

theorem profit_percent_is_20 (h : condition C S) : profit_percent_correct C S :=
sorry

end profit_percent_is_20_l114_114283


namespace fuel_tank_capacity_l114_114794

theorem fuel_tank_capacity (x : ℝ) 
  (h1 : (5 / 6) * x - (2 / 3) * x = 15) : x = 90 :=
sorry

end fuel_tank_capacity_l114_114794


namespace parabola_coefficients_l114_114788

theorem parabola_coefficients (a b c : ℝ) 
  (h_vertex : ∀ x, a * (x - 4) * (x - 4) + 3 = a * x * x + b * x + c) 
  (h_pass_point : 1 = a * (2 - 4) * (2 - 4) + 3) :
  (a = -1/2) ∧ (b = 4) ∧ (c = -5) :=
by
  sorry

end parabola_coefficients_l114_114788


namespace total_invested_expression_l114_114166

variables (x y T : ℝ)

axiom annual_income_exceed_65 : 0.10 * x - 0.08 * y = 65
axiom total_invested_is_T : x + y = T

theorem total_invested_expression :
  T = 1.8 * y + 650 :=
sorry

end total_invested_expression_l114_114166


namespace m_range_iff_four_distinct_real_roots_l114_114530

noncomputable def four_distinct_real_roots (m : ℝ) : Prop :=
∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
(x1^2 - 4 * |x1| + 5 = m) ∧
(x2^2 - 4 * |x2| + 5 = m) ∧
(x3^2 - 4 * |x3| + 5 = m) ∧
(x4^2 - 4 * |x4| + 5 = m)

theorem m_range_iff_four_distinct_real_roots (m : ℝ) :
  four_distinct_real_roots m ↔ 1 < m ∧ m < 5 :=
sorry

end m_range_iff_four_distinct_real_roots_l114_114530


namespace geom_seq_common_ratio_l114_114383

theorem geom_seq_common_ratio (a₁ a₂ a₃ a₄ q : ℝ) 
  (h1 : a₁ + a₄ = 18)
  (h2 : a₂ * a₃ = 32)
  (h3 : a₂ = a₁ * q)
  (h4 : a₃ = a₁ * q^2)
  (h5 : a₄ = a₁ * q^3) : 
  q = 2 ∨ q = (1 / 2) :=
by {
  sorry
}

end geom_seq_common_ratio_l114_114383


namespace largest_common_term_in_range_1_to_200_l114_114617

theorem largest_common_term_in_range_1_to_200 :
  ∃ (a : ℕ), a < 200 ∧ (∃ (n₁ n₂ : ℕ), a = 3 + 8 * n₁ ∧ a = 5 + 9 * n₂) ∧ a = 179 :=
by
  sorry

end largest_common_term_in_range_1_to_200_l114_114617


namespace math_problem_l114_114326

noncomputable def problem_statement : Prop := (7^2 - 5^2)^4 = 331776

theorem math_problem : problem_statement := by
  sorry

end math_problem_l114_114326


namespace inequality_am_gm_l114_114882

theorem inequality_am_gm (a b : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : 0 < b) (h₃ : b < 1) :
  1 + a + b > 3 * Real.sqrt (a * b) :=
by
  sorry

end inequality_am_gm_l114_114882


namespace unique_2_digit_cyclic_permutation_divisible_l114_114880

def is_cyclic_permutation (n : ℕ) (M : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → M i = M j

def M (a : Fin 2 → ℕ) : ℕ := a 0 * 10 + a 1

theorem unique_2_digit_cyclic_permutation_divisible (a : Fin 2 → ℕ) (h0 : ∀ i, a i ≠ 0) :
  (M a) % (a 1 * 10 + a 0) = 0 → 
  (M a = 11) :=
by
  sorry

end unique_2_digit_cyclic_permutation_divisible_l114_114880


namespace percentage_equivalence_l114_114813

theorem percentage_equivalence (x : ℝ) : 0.3 * 0.6 * 0.7 * x = 0.126 * x :=
by
  sorry

end percentage_equivalence_l114_114813


namespace nature_of_roots_Q_l114_114703

noncomputable def Q (x : ℝ) : ℝ := x^6 - 4 * x^5 + 3 * x^4 - 7 * x^3 - x^2 + x + 10

theorem nature_of_roots_Q : 
  ∃ (negative_roots positive_roots : Finset ℝ),
    (∀ r ∈ negative_roots, r < 0) ∧
    (∀ r ∈ positive_roots, r > 0) ∧
    negative_roots.card = 1 ∧
    positive_roots.card > 1 ∧
    ∀ r, r ∈ negative_roots ∨ r ∈ positive_roots → Q r = 0 :=
sorry

end nature_of_roots_Q_l114_114703


namespace wheel_rpm_l114_114175

noncomputable def radius : ℝ := 175
noncomputable def speed_kmh : ℝ := 66
noncomputable def speed_cmm := speed_kmh * 100000 / 60 -- convert from km/h to cm/min
noncomputable def circumference := 2 * Real.pi * radius -- circumference of the wheel
noncomputable def rpm := speed_cmm / circumference -- revolutions per minute

theorem wheel_rpm : rpm = 1000 := by
  sorry

end wheel_rpm_l114_114175


namespace quadratic_solution_l114_114260

theorem quadratic_solution (x : ℝ) (h_eq : x^2 - 3 * x - 6 = 0) (h_neq : x ≠ 0) :
    x = (3 + Real.sqrt 33) / 2 ∨ x = (3 - Real.sqrt 33) / 2 :=
by
  sorry

end quadratic_solution_l114_114260


namespace find_interval_l114_114138

theorem find_interval (n : ℕ) 
  (h1 : n < 500) 
  (h2 : n ∣ 9999) 
  (h3 : n + 4 ∣ 99) : (1 ≤ n) ∧ (n ≤ 125) := 
sorry

end find_interval_l114_114138


namespace find_n_if_pow_eqn_l114_114464

theorem find_n_if_pow_eqn (n : ℕ) :
  6 ^ 3 = 9 ^ n → n = 3 :=
by 
  sorry

end find_n_if_pow_eqn_l114_114464


namespace creative_sum_l114_114061

def letterValue (ch : Char) : Int :=
  let n := (ch.toNat - 'a'.toNat + 1) % 12
  if n = 0 then 2
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 2
  else if n = 5 then 1
  else if n = 6 then 0
  else if n = 7 then -1
  else if n = 8 then -2
  else if n = 9 then -3
  else if n = 10 then -2
  else if n = 11 then -1
  else 0 -- this should never happen

def wordValue (word : String) : Int :=
  word.foldl (λ acc ch => acc + letterValue ch) 0

theorem creative_sum : wordValue "creative" = -2 :=
  by
    sorry

end creative_sum_l114_114061


namespace fraction_of_crop_brought_to_BC_l114_114832

/-- Consider a kite-shaped field with sides AB = 120 m, BC = CD = 80 m, DA = 120 m.
    The angle between sides AB and BC is 120°, and between sides CD and DA is also 120°.
    Prove that the fraction of the crop brought to the longest side BC is 1/2. -/
theorem fraction_of_crop_brought_to_BC :
  ∀ (AB BC CD DA : ℝ) (α β : ℝ),
  AB = 120 ∧ BC = 80 ∧ CD = 80 ∧ DA = 120 ∧ α = 120 ∧ β = 120 →
  ∃ (frac : ℝ), frac = 1 / 2 :=
by
  intros AB BC CD DA α β h
  sorry

end fraction_of_crop_brought_to_BC_l114_114832


namespace sector_area_l114_114140

theorem sector_area (r : ℝ) (α : ℝ) (h1 : 2 * r + α * r = 16) (h2 : α = 2) :
  1 / 2 * α * r^2 = 16 :=
by
  sorry

end sector_area_l114_114140


namespace total_of_three_new_observations_l114_114091

theorem total_of_three_new_observations (avg9 : ℕ) (num9 : ℕ) 
(new_obs : ℕ) (new_avg_diff : ℕ) (new_num : ℕ) 
(total9 : ℕ) (new_avg : ℕ) (total12 : ℕ) : 
avg9 = 15 ∧ num9 = 9 ∧ new_obs = 3 ∧ new_avg_diff = 2 ∧
new_num = num9 + new_obs ∧ new_avg = avg9 - new_avg_diff ∧
total9 = num9 * avg9 ∧ total9 + 3 * (new_avg) = total12 → 
total12 - total9 = 21 := by sorry

end total_of_three_new_observations_l114_114091


namespace farm_needs_12880_ounces_of_horse_food_per_day_l114_114211

-- Define the given conditions
def ratio_sheep_to_horses : ℕ × ℕ := (1, 7)
def food_per_horse_per_day : ℕ := 230
def number_of_sheep : ℕ := 8

-- Define the proof goal
theorem farm_needs_12880_ounces_of_horse_food_per_day :
  let number_of_horses := number_of_sheep * ratio_sheep_to_horses.2
  number_of_horses * food_per_horse_per_day = 12880 :=
by
  sorry

end farm_needs_12880_ounces_of_horse_food_per_day_l114_114211


namespace Zain_coins_total_l114_114261

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end Zain_coins_total_l114_114261


namespace orange_jellybeans_count_l114_114724

theorem orange_jellybeans_count (total blue purple red : Nat)
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_red : red = 120) :
  ∃ orange : Nat, orange = total - (blue + purple + red) ∧ orange = 40 :=
by
  sorry

end orange_jellybeans_count_l114_114724


namespace modulo_11_residue_l114_114588

theorem modulo_11_residue : 
  (341 + 6 * 50 + 4 * 156 + 3 * 12^2) % 11 = 4 := 
by
  sorry

end modulo_11_residue_l114_114588


namespace total_carrots_l114_114532

def sally_carrots : ℕ := 6
def fred_carrots : ℕ := 4
def mary_carrots : ℕ := 10

theorem total_carrots : sally_carrots + fred_carrots + mary_carrots = 20 := by
  sorry

end total_carrots_l114_114532


namespace chocolate_ice_cream_ordered_l114_114648

theorem chocolate_ice_cream_ordered (V C : ℕ) (total_ice_cream : ℕ) (percentage_vanilla : ℚ) 
  (h_total : total_ice_cream = 220) 
  (h_percentage : percentage_vanilla = 0.20) 
  (h_vanilla_total : V = percentage_vanilla * total_ice_cream) 
  (h_vanilla_chocolate : V = 2 * C) 
  : C = 22 := 
by 
  sorry

end chocolate_ice_cream_ordered_l114_114648


namespace melanie_attended_games_l114_114178

/-- Melanie attended 5 football games if there were 12 total games and she missed 7. -/
theorem melanie_attended_games (totalGames : ℕ) (missedGames : ℕ) (h₁ : totalGames = 12) (h₂ : missedGames = 7) :
  totalGames - missedGames = 5 := 
sorry

end melanie_attended_games_l114_114178


namespace jerry_bought_one_pound_of_pasta_sauce_l114_114938

-- Definitions of the given conditions
def cost_mustard_oil_per_liter : ℕ := 13
def liters_mustard_oil : ℕ := 2
def cost_pasta_per_pound : ℕ := 4
def pounds_pasta : ℕ := 3
def cost_pasta_sauce_per_pound : ℕ := 5
def leftover_amount : ℕ := 7
def initial_amount : ℕ := 50

-- The goal to prove
theorem jerry_bought_one_pound_of_pasta_sauce :
  (initial_amount - leftover_amount - liters_mustard_oil * cost_mustard_oil_per_liter 
  - pounds_pasta * cost_pasta_per_pound) / cost_pasta_sauce_per_pound = 1 :=
by
  sorry

end jerry_bought_one_pound_of_pasta_sauce_l114_114938


namespace find_f_six_l114_114243

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the function definition

axiom f_property : ∀ x y : ℝ, f (x - y) = f x * f y
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0
axiom f_two : f 2 = 5

theorem find_f_six : f 6 = 1 / 5 :=
sorry

end find_f_six_l114_114243


namespace short_sleeve_shirts_l114_114000

theorem short_sleeve_shirts (total_shirts long_sleeve_shirts short_sleeve_shirts : ℕ) 
  (h1 : total_shirts = 9) 
  (h2 : long_sleeve_shirts = 5)
  (h3 : short_sleeve_shirts = total_shirts - long_sleeve_shirts) : 
  short_sleeve_shirts = 4 :=
by 
  sorry

end short_sleeve_shirts_l114_114000


namespace ant_trip_ratio_l114_114024

theorem ant_trip_ratio (A B : ℕ) (x c : ℕ) (h1 : A * x = c) (h2 : B * (3 / 2 * x) = 3 * c) :
  B = 2 * A :=
by
  sorry

end ant_trip_ratio_l114_114024


namespace remainder_of_multiple_of_n_mod_7_l114_114484

theorem remainder_of_multiple_of_n_mod_7
  (n m : ℤ)
  (h1 : n % 7 = 1)
  (h2 : m % 7 = 3) :
  (m * n) % 7 = 3 :=
by
  sorry

end remainder_of_multiple_of_n_mod_7_l114_114484


namespace evaluate_expression_l114_114224

variable (x y : ℝ)
variable (h₀ : x ≠ 0)
variable (h₁ : y ≠ 0)
variable (h₂ : 5 * x ≠ 3 * y)

theorem evaluate_expression : 
  (5 * x - 3 * y)⁻¹ * ((5 * x)⁻¹ - (3 * y)⁻¹) = -1 / (15 * x * y) :=
sorry

end evaluate_expression_l114_114224


namespace yolanda_walking_rate_correct_l114_114161

-- Definitions and conditions
def distance_XY : ℕ := 65
def bobs_walking_rate : ℕ := 7
def bobs_distance_when_met : ℕ := 35
def yolanda_start_time (t: ℕ) : ℕ := t + 1 -- Yolanda starts walking 1 hour earlier

-- Yolanda's walking rate calculation
def yolandas_walking_rate : ℕ := 5

theorem yolanda_walking_rate_correct { time_bob_walked : ℕ } 
  (h1 : distance_XY = 65)
  (h2 : bobs_walking_rate = 7)
  (h3 : bobs_distance_when_met = 35) 
  (h4 : time_bob_walked = bobs_distance_when_met / bobs_walking_rate)
  (h5 : yolanda_start_time time_bob_walked = 6) -- since bob walked 5 hours, yolanda walked 6 hours
  (h6 : distance_XY - bobs_distance_when_met = 30) :
  yolandas_walking_rate = ((distance_XY - bobs_distance_when_met) / yolanda_start_time time_bob_walked) := 
sorry

end yolanda_walking_rate_correct_l114_114161


namespace find_third_number_l114_114816

noncomputable def averageFirstSet (x : ℝ) : ℝ := (20 + 40 + x) / 3
noncomputable def averageSecondSet : ℝ := (10 + 70 + 16) / 3

theorem find_third_number (x : ℝ) (h : averageFirstSet x = averageSecondSet + 8) : x = 60 :=
by
  sorry

end find_third_number_l114_114816


namespace transformation_identity_l114_114580

theorem transformation_identity (a b : ℝ) 
    (h1 : ∃ a b : ℝ, ∀ x y : ℝ, (y, -x) = (-7, 3) → (x, y) = (3, 7))
    (h2 : ∃ a b : ℝ, ∀ c d : ℝ, (d, c) = (3, -7) → (c, d) = (-7, 3)) :
    b - a = 4 :=
by
    sorry

end transformation_identity_l114_114580


namespace parallelogram_area_ratio_l114_114191

theorem parallelogram_area_ratio (
  AB CD BC AD AP CQ BP DQ: ℝ)
  (h1 : AB = 13)
  (h2 : CD = 13)
  (h3 : BC = 15)
  (h4 : AD = 15)
  (h5 : AP = 10 / 3)
  (h6 : CQ = 10 / 3)
  (h7 : BP = 29 / 3)
  (h8 : DQ = 29 / 3)
  : ((area_APDQ / area_BPCQ) = 19) :=
sorry

end parallelogram_area_ratio_l114_114191


namespace find_AC_l114_114614

theorem find_AC (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (max_val : A - C = 3) (min_val : -A - C = -1) : 
  A = 2 ∧ C = 1 :=
by
  sorry

end find_AC_l114_114614


namespace women_count_l114_114745

def total_passengers : Nat := 54
def men : Nat := 18
def children : Nat := 10
def women : Nat := total_passengers - men - children

theorem women_count : women = 26 :=
sorry

end women_count_l114_114745


namespace solve_fractional_equation_l114_114238

theorem solve_fractional_equation (x : ℝ) (h : (x + 1) / (4 * x^2 - 1) = (3 / (2 * x + 1)) - (4 / (4 * x - 2))) : x = 6 := 
by
  sorry

end solve_fractional_equation_l114_114238


namespace sum_of_three_digit_numbers_l114_114677

theorem sum_of_three_digit_numbers :
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  let Sum := n / 2 * (first_term + last_term)
  Sum = 494550 :=
by {
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  have n_def : n = 900 := by norm_num [n]
  let Sum := n / 2 * (first_term + last_term)
  have sum_def : Sum = 450 * (100 + 999) := by norm_num [Sum, first_term, last_term, n_def]
  have final_sum : Sum = 494550 := by norm_num [sum_def]
  exact final_sum
}

end sum_of_three_digit_numbers_l114_114677


namespace find_range_of_a_l114_114952

noncomputable def range_of_a : Set ℝ :=
  {a | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)}

theorem find_range_of_a :
  {a : ℝ | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)} = 
  {a | (-2 < a ∧ a < -1) ∨ (1 ≤ a)} :=
by
  sorry

end find_range_of_a_l114_114952


namespace largest_not_sum_of_two_composites_l114_114693

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end largest_not_sum_of_two_composites_l114_114693


namespace tan_expression_value_l114_114827

noncomputable def sequence_properties (a b : ℕ → ℝ) :=
  (a 0 * a 5 * a 10 = -3 * Real.sqrt 3) ∧
  (b 0 + b 5 + b 10 = 7 * Real.pi) ∧
  (∀ n, a (n + 1) = a n * a 1) ∧
  (∀ n, b (n + 1) = b n + (b 1 - b 0))

theorem tan_expression_value (a b : ℕ → ℝ) (h : sequence_properties a b) :
  Real.tan (b 2 + b 8) / (1 - a 3 * a 7) = -Real.sqrt 3 :=
sorry

end tan_expression_value_l114_114827


namespace oscar_leap_vs_elmer_stride_l114_114082

theorem oscar_leap_vs_elmer_stride :
  ∀ (num_poles : ℕ) (distance : ℝ) (elmer_strides_per_gap : ℕ) (oscar_leaps_per_gap : ℕ)
    (elmer_stride_time_mult : ℕ) (total_distance_poles : ℕ)
    (elmer_total_strides : ℕ) (oscar_total_leaps : ℕ) (elmer_stride_length : ℝ)
    (oscar_leap_length : ℝ) (expected_diff : ℝ),
    num_poles = 81 →
    distance = 10560 →
    elmer_strides_per_gap = 60 →
    oscar_leaps_per_gap = 15 →
    elmer_stride_time_mult = 2 →
    total_distance_poles = 2 →
    elmer_total_strides = elmer_strides_per_gap * (num_poles - 1) →
    oscar_total_leaps = oscar_leaps_per_gap * (num_poles - 1) →
    elmer_stride_length = distance / elmer_total_strides →
    oscar_leap_length = distance / oscar_total_leaps →
    expected_diff = oscar_leap_length - elmer_stride_length →
    expected_diff = 6.6
:= sorry

end oscar_leap_vs_elmer_stride_l114_114082


namespace bobby_shoes_multiple_l114_114872

theorem bobby_shoes_multiple (B M : ℕ) (hBonny : 13 = 2 * B - 5) (hBobby : 27 = M * B) : 
  M = 3 :=
by 
  sorry

end bobby_shoes_multiple_l114_114872


namespace number_of_teams_l114_114795

-- Define the problem context
variables (n : ℕ)

-- Define the conditions
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The theorem we want to prove
theorem number_of_teams (h : total_games n = 55) : n = 11 :=
sorry

end number_of_teams_l114_114795


namespace composite_shape_sum_l114_114627

def triangular_prism_faces := 5
def triangular_prism_edges := 9
def triangular_prism_vertices := 6

def pentagonal_prism_additional_faces := 7
def pentagonal_prism_additional_edges := 10
def pentagonal_prism_additional_vertices := 5

def pyramid_additional_faces := 5
def pyramid_additional_edges := 5
def pyramid_additional_vertices := 1

def resulting_shape_faces := triangular_prism_faces - 1 + pentagonal_prism_additional_faces + pyramid_additional_faces
def resulting_shape_edges := triangular_prism_edges + pentagonal_prism_additional_edges + pyramid_additional_edges
def resulting_shape_vertices := triangular_prism_vertices + pentagonal_prism_additional_vertices + pyramid_additional_vertices

def sum_faces_edges_vertices := resulting_shape_faces + resulting_shape_edges + resulting_shape_vertices

theorem composite_shape_sum : sum_faces_edges_vertices = 51 :=
by
  unfold sum_faces_edges_vertices resulting_shape_faces resulting_shape_edges resulting_shape_vertices
  unfold triangular_prism_faces triangular_prism_edges triangular_prism_vertices
  unfold pentagonal_prism_additional_faces pentagonal_prism_additional_edges pentagonal_prism_additional_vertices
  unfold pyramid_additional_faces pyramid_additional_edges pyramid_additional_vertices
  simp
  sorry

end composite_shape_sum_l114_114627


namespace gerald_bars_l114_114246

theorem gerald_bars (G : ℕ) 
  (H1 : ∀ G, ∀ teacher_bars : ℕ, teacher_bars = 2 * G → total_bars = G + teacher_bars) 
  (H2 : ∀ total_bars : ℕ, total_squares = total_bars * 8 → total_squares_needed = 24 * 7) 
  (H3 : ∀ total_squares : ℕ, total_squares_needed = 24 * 7) 
  : G = 7 :=
by
  sorry

end gerald_bars_l114_114246


namespace derivative_at_two_l114_114220

theorem derivative_at_two {f : ℝ → ℝ} (f_deriv : ∀x, deriv f x = 2 * x - 4) : deriv f 2 = 0 := 
by sorry

end derivative_at_two_l114_114220


namespace problem1_problem2_problem3_l114_114744

-- Definition of given quantities and conditions
variables (a b x : ℝ) (α β : ℝ)

-- Given Conditions
@[simp] def cond1 := true
@[simp] def cond2 := true
@[simp] def cond3 := true
@[simp] def cond4 := true

-- First Question
theorem problem1 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    a * Real.sin α = b * Real.sin β := sorry

-- Second Question
theorem problem2 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    Real.sin β ≤ a / b := sorry

-- Third Question
theorem problem3 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    x = a * (1 - Real.cos α) + b * (1 - Real.cos β) := sorry

end problem1_problem2_problem3_l114_114744


namespace thirty_ml_of_one_liter_is_decimal_fraction_l114_114850

-- We define the known conversion rule between liters and milliliters.
def liter_to_ml := 1000

-- We define the volume in milliliters that we are considering.
def volume_ml := 30

-- We state the main theorem which asserts that 30 ml of a liter is equal to the decimal fraction 0.03.
theorem thirty_ml_of_one_liter_is_decimal_fraction : (volume_ml / (liter_to_ml : ℝ)) = 0.03 := by
  -- insert proof here
  sorry

end thirty_ml_of_one_liter_is_decimal_fraction_l114_114850


namespace hexagon_arithmetic_sum_l114_114320

theorem hexagon_arithmetic_sum (a n : ℝ) (h : 6 * a + 15 * n = 720) : 2 * a + 5 * n = 240 :=
by
  sorry

end hexagon_arithmetic_sum_l114_114320


namespace range_of_x_l114_114735

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = (2 / (Real.sqrt (x - 1)))) → (x > 1) :=
by
  sorry

end range_of_x_l114_114735


namespace ants_meeting_points_l114_114031

/-- Definition for the problem setup: two ants running at constant speeds around a circle. -/
structure AntsRunningCircle where
  laps_ant1 : ℕ
  laps_ant2 : ℕ

/-- Theorem stating that given the laps completed by two ants in opposite directions on a circle, 
    they will meet at a specific number of distinct points. -/
theorem ants_meeting_points 
  (ants : AntsRunningCircle)
  (h1 : ants.laps_ant1 = 9)
  (h2 : ants.laps_ant2 = 6) : 
    ∃ n : ℕ, n = 5 := 
by
  -- Proof goes here
  sorry

end ants_meeting_points_l114_114031


namespace readers_both_l114_114180

-- Definitions of the number of readers
def total_readers : ℕ := 150
def readers_science_fiction : ℕ := 120
def readers_literary_works : ℕ := 90

-- Statement of the proof problem
theorem readers_both :
  (readers_science_fiction + readers_literary_works - total_readers) = 60 :=
by
  -- Proof omitted
  sorry

end readers_both_l114_114180


namespace peter_completes_remaining_work_in_14_days_l114_114603

-- Define the conditions and the theorem
variable (W : ℕ) (work_done : ℕ) (remaining_work : ℕ)

theorem peter_completes_remaining_work_in_14_days
  (h1 : Matt_and_Peter_rate = (W/20))
  (h2 : Peter_rate = (W/35))
  (h3 : Work_done_in_12_days = (12 * (W/20)))
  (h4 : Remaining_work = (W - (12 * (W/20))))
  : (remaining_work / Peter_rate)  = 14 := sorry

end peter_completes_remaining_work_in_14_days_l114_114603


namespace problem_l114_114669

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

axiom universal_set : U = {1, 2, 3, 4, 5, 6, 7}
axiom set_M : M = {3, 4, 5}
axiom set_N : N = {1, 3, 6}

def complement (U M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

theorem problem :
  {1, 6} = (complement U M) ∩ N :=
by
  sorry

end problem_l114_114669


namespace area_of_region_enclosed_by_parabolas_l114_114594

-- Define the given parabolas
def parabola1 (y : ℝ) : ℝ := -3 * y^2
def parabola2 (y : ℝ) : ℝ := 1 - 4 * y^2

-- Define the integral representing the area between the parabolas
noncomputable def areaBetweenParabolas : ℝ :=
  2 * (∫ y in (0 : ℝ)..1, (parabola2 y - parabola1 y))

-- The statement to be proved
theorem area_of_region_enclosed_by_parabolas :
  areaBetweenParabolas = 4 / 3 := 
sorry

end area_of_region_enclosed_by_parabolas_l114_114594


namespace contradiction_method_l114_114471

theorem contradiction_method (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) → a < 1 :=
sorry

end contradiction_method_l114_114471


namespace int_fraction_not_integer_l114_114058

theorem int_fraction_not_integer (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ (k : ℤ), a^2 + b^2 = k * (a^2 - b^2) := 
sorry

end int_fraction_not_integer_l114_114058


namespace pool_depth_l114_114590

theorem pool_depth 
  (length width : ℝ) 
  (chlorine_per_120_cubic_feet chlorine_cost : ℝ) 
  (total_spent volume_per_quart_of_chlorine : ℝ) 
  (H_length : length = 10) 
  (H_width : width = 8)
  (H_chlorine_per_120_cubic_feet : chlorine_per_120_cubic_feet = 1 / 120)
  (H_chlorine_cost : chlorine_cost = 3)
  (H_total_spent : total_spent = 12)
  (H_volume_per_quart_of_chlorine : volume_per_quart_of_chlorine = 120) :
  ∃ depth : ℝ, total_spent / chlorine_cost * volume_per_quart_of_chlorine = length * width * depth ∧ depth = 6 :=
by 
  sorry

end pool_depth_l114_114590


namespace count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l114_114698

noncomputable def count_equally_spaced_integers : ℕ := 
  sorry

theorem count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle:
  count_equally_spaced_integers = 4 :=
sorry

end count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l114_114698


namespace arithmetic_square_root_of_3_neg_2_l114_114118

theorem arithmetic_square_root_of_3_neg_2 : Real.sqrt (3 ^ (-2: Int)) = 1 / 3 := 
by 
  sorry

end arithmetic_square_root_of_3_neg_2_l114_114118


namespace arithmetic_sequences_count_l114_114503

noncomputable def countArithmeticSequences (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4

theorem arithmetic_sequences_count :
  ∀ n : ℕ, countArithmeticSequences n = if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4 :=
by sorry

end arithmetic_sequences_count_l114_114503


namespace min_value_of_c_l114_114170

theorem min_value_of_c (a b c : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_ineq1 : a < b) 
  (h_ineq2 : b < 2 * b) 
  (h_ineq3 : 2 * b < c)
  (h_unique_sol : ∃ x : ℝ, 3 * x + (|x - a| + |x - b| + |x - (2 * b)| + |x - c|) = 3000) :
  c = 502 := sorry

end min_value_of_c_l114_114170


namespace no_such_function_l114_114451

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, (∀ y x : ℝ, 0 < x → x < y → f y > (y - x) * (f x)^2) :=
by
  sorry

end no_such_function_l114_114451


namespace probability_number_greater_than_3_from_0_5_l114_114676

noncomputable def probability_number_greater_than_3_in_0_5 : ℝ :=
  let total_interval_length := 5 - 0
  let event_interval_length := 5 - 3
  event_interval_length / total_interval_length

theorem probability_number_greater_than_3_from_0_5 :
  probability_number_greater_than_3_in_0_5 = 2 / 5 :=
by
  sorry

end probability_number_greater_than_3_from_0_5_l114_114676


namespace min_shirts_to_save_money_l114_114279

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 75 + 8 * x < 12 * x ∧ x = 19 :=
sorry

end min_shirts_to_save_money_l114_114279


namespace find_i_value_for_S_i_l114_114955

theorem find_i_value_for_S_i :
  ∃ (i : ℕ), (3 * 6 - 2 ≤ i ∧ i < 3 * 6 + 1) ∧ (1000 ≤ 31 * 2^6) ∧ (31 * 2^6 ≤ 3000) ∧ i = 2 :=
by sorry

end find_i_value_for_S_i_l114_114955


namespace friends_team_division_l114_114089

theorem friends_team_division :
  let num_friends : ℕ := 8
  let num_teams : ℕ := 4
  let ways_to_divide := num_teams ^ num_friends
  ways_to_divide = 65536 :=
by
  sorry

end friends_team_division_l114_114089


namespace quotient_larger_than_dividend_l114_114375

-- Define the problem conditions
variables {a b : ℝ}

-- State the theorem corresponding to the problem
theorem quotient_larger_than_dividend (h : b ≠ 0) : ¬ (∀ a : ℝ, ∀ b : ℝ, (a / b > a) ) :=
by
  sorry

end quotient_larger_than_dividend_l114_114375


namespace length_of_segment_correct_l114_114759

noncomputable def length_of_segment (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem length_of_segment_correct :
  length_of_segment 5 (-1) 13 11 = 4 * Real.sqrt 13 := by
  sorry

end length_of_segment_correct_l114_114759


namespace contrapositive_necessary_condition_l114_114395

theorem contrapositive_necessary_condition {p q : Prop} (h : p → q) : ¬p → ¬q :=
  by sorry

end contrapositive_necessary_condition_l114_114395


namespace most_irregular_acute_triangle_l114_114187

theorem most_irregular_acute_triangle :
  ∃ (α β γ : ℝ), α ≤ β ∧ β ≤ γ ∧ γ ≤ (90:ℝ) ∧ 
  ((β - α ≤ 15) ∧ (γ - β ≤ 15) ∧ (90 - γ ≤ 15)) ∧
  (α + β + γ = 180) ∧ 
  (α = 45 ∧ β = 60 ∧ γ = 75) := sorry

end most_irregular_acute_triangle_l114_114187


namespace solve_equation_l114_114524

theorem solve_equation (x : ℝ) (h : 16 * x^2 = 81) : x = 9 / 4 ∨ x = - (9 / 4) :=
by
  sorry

end solve_equation_l114_114524


namespace smallest_value_of_Q_l114_114863

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - 4*x^2 + 2*x - 3

theorem smallest_value_of_Q :
  min (-10) (min 3 (-2)) = -10 :=
by
  -- Skip the proof
  sorry

end smallest_value_of_Q_l114_114863


namespace solve_system_l114_114608

/-- Given the system of equations:
    3 * (x + y) - 4 * (x - y) = 5
    (x + y) / 2 + (x - y) / 6 = 0
  Prove that the solution is x = -1/3 and y = 2/3 
-/
theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x + y) - 4 * (x - y) = 5)
  (h2 : (x + y) / 2 + (x - y) / 6 = 0) : 
  x = -1 / 3 ∧ y = 2 / 3 := 
sorry

end solve_system_l114_114608


namespace division_proof_l114_114637

-- Define the given condition
def given_condition : Prop :=
  2084.576 / 135.248 = 15.41

-- Define the problem statement we want to prove
def problem_statement : Prop :=
  23.8472 / 13.5786 = 1.756

-- Main theorem stating that under the given condition, the problem statement holds
theorem division_proof (h : given_condition) : problem_statement :=
by sorry

end division_proof_l114_114637


namespace no_equal_prob_for_same_color_socks_l114_114640

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l114_114640


namespace drop_volume_l114_114316

theorem drop_volume :
  let leak_rate := 3 -- drops per minute
  let pot_volume := 3 * 1000 -- volume in milliliters
  let time := 50 -- minutes
  let total_drops := leak_rate * time -- total number of drops
  (pot_volume / total_drops) = 20 := 
by
  let leak_rate : ℕ := 3
  let pot_volume : ℕ := 3 * 1000
  let time : ℕ := 50
  let total_drops := leak_rate * time
  have h : (pot_volume / total_drops) = 20 := by sorry
  exact h

end drop_volume_l114_114316


namespace find_arithmetic_mean_l114_114749

theorem find_arithmetic_mean (σ μ : ℝ) (hσ : σ = 1.5) (h : 11 = μ - 2 * σ) : μ = 14 :=
by
  sorry

end find_arithmetic_mean_l114_114749


namespace total_number_of_outfits_l114_114019

noncomputable def number_of_outfits (shirts pants ties jackets : ℕ) :=
  shirts * pants * ties * jackets

theorem total_number_of_outfits :
  number_of_outfits 8 5 5 3 = 600 :=
by
  sorry

end total_number_of_outfits_l114_114019


namespace homework_duration_equation_l114_114369

-- Define the initial and final durations and the rate of decrease
def initial_duration : ℝ := 100
def final_duration : ℝ := 70
def rate_of_decrease (x : ℝ) : ℝ := x

-- Statement of the proof problem
theorem homework_duration_equation (x : ℝ) :
  initial_duration * (1 - rate_of_decrease x) ^ 2 = final_duration :=
sorry

end homework_duration_equation_l114_114369


namespace value_of_x_m_minus_n_l114_114513

variables {x : ℝ} {m n : ℝ}

theorem value_of_x_m_minus_n (hx_m : x^m = 6) (hx_n : x^n = 3) : x^(m - n) = 2 := 
by 
  sorry

end value_of_x_m_minus_n_l114_114513


namespace arithmetic_sequence_x_y_sum_l114_114604

theorem arithmetic_sequence_x_y_sum :
  ∀ (a d x y: ℕ), 
  a = 3 → d = 6 → 
  (∀ (n: ℕ), n ≥ 1 → a + (n-1) * d = 3 + (n-1) * 6) →
  (a + 5 * d = x) → (a + 6 * d = y) → 
  (y = 45 - d) → x + y = 72 :=
by
  intros a d x y h_a h_d h_seq h_x h_y h_y_equals
  sorry

end arithmetic_sequence_x_y_sum_l114_114604


namespace garden_strawberry_yield_l114_114562

-- Definitions from the conditions
def garden_length : ℝ := 10
def garden_width : ℝ := 15
def plants_per_sq_ft : ℝ := 5
def strawberries_per_plant : ℝ := 12

-- Expected total number of strawberries
def expected_strawberries : ℝ := 9000

-- Proof statement
theorem garden_strawberry_yield : 
  (garden_length * garden_width * plants_per_sq_ft * strawberries_per_plant) = expected_strawberries :=
by sorry

end garden_strawberry_yield_l114_114562


namespace one_fourth_more_than_x_equals_twenty_percent_less_than_80_l114_114715

theorem one_fourth_more_than_x_equals_twenty_percent_less_than_80 :
  ∃ n : ℝ, (80 - 0.30 * 80 = 56) ∧ (5 / 4 * n = 56) ∧ (n = 45) :=
by
  sorry

end one_fourth_more_than_x_equals_twenty_percent_less_than_80_l114_114715


namespace sales_tax_difference_l114_114885

theorem sales_tax_difference (price : ℝ) (tax_rate1 tax_rate2 : ℝ) :
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.065 →
  (price * tax_rate1 - price * tax_rate2 = 0.5) :=
by
  intros
  sorry

end sales_tax_difference_l114_114885


namespace toothpicks_grid_total_l114_114323

theorem toothpicks_grid_total (L W : ℕ) (hL : L = 60) (hW : W = 32) : 
  (L + 1) * W + (W + 1) * L = 3932 := 
by 
  sorry

end toothpicks_grid_total_l114_114323


namespace min_total_fund_Required_l114_114127

noncomputable def sell_price_A (x : ℕ) : ℕ := x + 10
noncomputable def cost_A (x : ℕ) : ℕ := 600
noncomputable def cost_B (x : ℕ) : ℕ := 400

def num_barrels_A_B_purchased (x : ℕ) := cost_A x / (sell_price_A x) = cost_B x / x

noncomputable def total_cost (m : ℕ) : ℕ := 10 * m + 10000

theorem min_total_fund_Required (price_A price_B m total : ℕ) :
  price_B = 20 →
  price_A = 30 →
  price_A = price_B + 10 →
  (num_barrels_A_B_purchased price_B) →
  total = total_cost m →
  m = 250 →
  total = 12500 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end min_total_fund_Required_l114_114127


namespace molecular_weight_compound_l114_114635

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def molecular_weight (n_H n_Br n_O : ℕ) : ℝ :=
  n_H * atomic_weight_H + n_Br * atomic_weight_Br + n_O * atomic_weight_O

theorem molecular_weight_compound : 
  molecular_weight 1 1 3 = 128.91 :=
by
  -- This is where the proof would go
  sorry

end molecular_weight_compound_l114_114635


namespace consecutive_integers_sqrt19_sum_l114_114386

theorem consecutive_integers_sqrt19_sum :
  ∃ a b : ℤ, (a < ⌊Real.sqrt 19⌋ ∧ ⌊Real.sqrt 19⌋ < b ∧ a + 1 = b) ∧ a + b = 9 := 
by
  sorry

end consecutive_integers_sqrt19_sum_l114_114386


namespace regular_polygon_sides_and_exterior_angle_l114_114364

theorem regular_polygon_sides_and_exterior_angle (n : ℕ) (exterior_sum : ℝ) :
  (180 * (n - 2) = 360 + exterior_sum) → (exterior_sum = 360) → n = 6 ∧ (360 / n = 60) :=
by
  intro h1 h2
  sorry

end regular_polygon_sides_and_exterior_angle_l114_114364


namespace siamese_cats_initially_l114_114350

theorem siamese_cats_initially (house_cats: ℕ) (cats_sold: ℕ) (cats_left: ℕ) (initial_siamese: ℕ) :
  house_cats = 5 → 
  cats_sold = 10 → 
  cats_left = 8 → 
  (initial_siamese + house_cats - cats_sold = cats_left) → 
  initial_siamese = 13 :=
by
  intros h1 h2 h3 h4
  sorry

end siamese_cats_initially_l114_114350


namespace certain_number_is_17_l114_114233

theorem certain_number_is_17 (x : ℤ) (h : 2994 / x = 177) : x = 17 :=
by
  sorry

end certain_number_is_17_l114_114233


namespace tangent_line_at_one_l114_114382

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_one : ∀ x y, (x = 1 ∧ y = 0) → (x - y - 1 = 0) :=
by 
  intro x y h
  sorry

end tangent_line_at_one_l114_114382


namespace find_z_plus_one_over_y_l114_114377

variable {x y z : ℝ}

theorem find_z_plus_one_over_y (h1 : x * y * z = 1) 
                                (h2 : x + 1 / z = 7) 
                                (h3 : y + 1 / x = 31) 
                                (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : 
                              z + 1 / y = 5 / 27 := 
by
  sorry

end find_z_plus_one_over_y_l114_114377


namespace min_value_lemma_min_value_achieved_l114_114133

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2)

theorem min_value_lemma : ∀ (x : ℝ), f x ≥ Real.sqrt 5 := 
by
  intro x
  sorry

theorem min_value_achieved : ∃ (x : ℝ), f x = Real.sqrt 5 :=
by
  use 1 / 3
  sorry

end min_value_lemma_min_value_achieved_l114_114133


namespace bob_investment_correct_l114_114366

noncomputable def initial_investment_fundA : ℝ := 2000
noncomputable def interest_rate_fundA : ℝ := 0.12
noncomputable def initial_investment_fundB : ℝ := 1000
noncomputable def interest_rate_fundB : ℝ := 0.30
noncomputable def fundA_after_two_years := initial_investment_fundA * (1 + interest_rate_fundA)
noncomputable def fundB_after_two_years (B : ℝ) := B * (1 + interest_rate_fundB)^2
noncomputable def extra_value : ℝ := 549.9999999999998

theorem bob_investment_correct :
  fundA_after_two_years = fundB_after_two_years initial_investment_fundB + extra_value :=
by
  sorry

end bob_investment_correct_l114_114366


namespace part1_part2_l114_114778

-- Define the first part of the problem
theorem part1 (a b : ℝ) :
  (∀ x : ℝ, |x^2 + a * x + b| ≤ 2 * |x - 4| * |x + 2|) → (a = -2 ∧ b = -8) :=
sorry

-- Define the second part of the problem
theorem part2 (a b m : ℝ) :
  (∀ x : ℝ, x > 1 → x^2 + a * x + b ≥ (m + 2) * x - m - 15) → m ≤ 2 :=
sorry

end part1_part2_l114_114778


namespace exists_positive_integer_divisible_by_15_and_sqrt_in_range_l114_114691

theorem exists_positive_integer_divisible_by_15_and_sqrt_in_range :
  ∃ (n : ℕ), (n % 15 = 0) ∧ (28 < Real.sqrt n) ∧ (Real.sqrt n < 28.5) ∧ (n = 795) :=
by
  sorry

end exists_positive_integer_divisible_by_15_and_sqrt_in_range_l114_114691


namespace perpendicular_vectors_l114_114843

def vec := ℝ × ℝ

def dot_product (a b : vec) : ℝ :=
  a.1 * b.1 + a.2 * b.2

variables (m : ℝ)
def a : vec := (1, 2)
def b : vec := (m, 1)

theorem perpendicular_vectors (h : dot_product a (b m) = 0) : m = -2 :=
sorry

end perpendicular_vectors_l114_114843


namespace tangent_line_equation_at_1_2_l114_114415

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem tangent_line_equation_at_1_2 :
  let x₀ := 1
  let y₀ := 2
  let slope := -2
  ∀ (x y : ℝ),
    y - y₀ = slope * (x - x₀) →
    2 * x + y - 4 = 0 :=
by
  sorry

end tangent_line_equation_at_1_2_l114_114415


namespace alice_marble_groups_l114_114003

-- Define the number of each colored marble Alice has
def pink_marble := 1
def blue_marble := 1
def white_marble := 1
def black_marbles := 4

-- The function to count the number of different groups of two marbles Alice can choose
noncomputable def count_groups : Nat :=
  let total_colors := 4  -- Pink, Blue, White, and one representative black
  1 + (total_colors.choose 2)

-- The theorem statement 
theorem alice_marble_groups : count_groups = 7 := by 
  sorry

end alice_marble_groups_l114_114003


namespace crab_ratio_l114_114666

theorem crab_ratio 
  (oysters_day1 : ℕ) 
  (crabs_day1 : ℕ) 
  (total_days : ℕ) 
  (oysters_ratio : ℕ) 
  (oysters_day2 : ℕ) 
  (total_oysters_crabs : ℕ) 
  (crabs_day2 : ℕ) 
  (ratio : ℚ) :
  oysters_day1 = 50 →
  crabs_day1 = 72 →
  oysters_ratio = 2 →
  oysters_day2 = oysters_day1 / oysters_ratio →
  total_oysters_crabs = 195 →
  total_oysters_crabs = oysters_day1 + crabs_day1 + oysters_day2 + crabs_day2 →
  crabs_day2 = total_oysters_crabs - (oysters_day1 + crabs_day1 + oysters_day2) →
  ratio = (crabs_day2 : ℚ) / crabs_day1 →
  ratio = 2 / 3 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end crab_ratio_l114_114666


namespace total_trip_time_l114_114453

-- Definitions: conditions from the problem
def time_in_first_country : Nat := 2
def time_in_second_country := 2 * time_in_first_country
def time_in_third_country := 2 * time_in_first_country

-- Statement: prove that the total time spent is 10 weeks
theorem total_trip_time : time_in_first_country + time_in_second_country + time_in_third_country = 10 := by
  sorry

end total_trip_time_l114_114453


namespace arithmetic_seq_a5_l114_114820

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem arithmetic_seq_a5 (h1 : is_arithmetic_sequence a) (h2 : a 2 + a 8 = 12) :
  a 5 = 6 :=
by
  sorry

end arithmetic_seq_a5_l114_114820


namespace ball_min_bounces_reach_target_height_l114_114041

noncomputable def minimum_bounces (initial_height : ℝ) (ratio : ℝ) (target_height : ℝ) : ℕ :=
  Nat.ceil (Real.log (target_height / initial_height) / Real.log ratio)

theorem ball_min_bounces_reach_target_height :
  minimum_bounces 20 (2 / 3) 2 = 6 :=
by
  -- This is where the proof would go, but we use sorry to skip it
  sorry

end ball_min_bounces_reach_target_height_l114_114041


namespace problem_p_3_l114_114398

theorem problem_p_3 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) (hn : n = (2^(2*p) - 1) / 3) : n ∣ 2^n - 2 := by
  sorry

end problem_p_3_l114_114398


namespace net_income_calculation_l114_114547

-- Definitions based on conditions
def rent_per_hour := 20
def monday_hours := 8
def wednesday_hours := 8
def friday_hours := 6
def sunday_hours := 5
def maintenance_cost := 35
def insurance_fee := 15
def rental_days := 4

-- Derived values based on conditions
def total_income_per_week :=
  (monday_hours + wednesday_hours) * rent_per_hour * 2 + 
  friday_hours * rent_per_hour + 
  sunday_hours * rent_per_hour

def total_expenses_per_week :=
  maintenance_cost + 
  insurance_fee * rental_days

def net_income_per_week := 
  total_income_per_week - total_expenses_per_week

-- The final proof statement
theorem net_income_calculation : net_income_per_week = 445 := by
  sorry

end net_income_calculation_l114_114547


namespace necessary_but_not_sufficient_for_odd_function_l114_114713

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f (x)

theorem necessary_but_not_sufficient_for_odd_function (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f :=
sorry

end necessary_but_not_sufficient_for_odd_function_l114_114713


namespace circle_center_and_radius_l114_114585

theorem circle_center_and_radius (x y : ℝ) (h : x^2 + y^2 - 6*x = 0) :
  (∃ c : ℝ × ℝ, c = (3, 0)) ∧ (∃ r : ℝ, r = 3) := 
sorry

end circle_center_and_radius_l114_114585


namespace number_of_multiples_of_3003_l114_114775

theorem number_of_multiples_of_3003 (i j : ℕ) (h : 0 ≤ i ∧ i < j ∧ j ≤ 199): 
  (∃ n : ℕ, n = 3003 * k ∧ n = 10^j - 10^i) → 
  (number_of_solutions = 1568) :=
sorry

end number_of_multiples_of_3003_l114_114775


namespace absolute_value_condition_l114_114894

theorem absolute_value_condition (a : ℝ) (h : |a| = -a) : a = 0 ∨ a < 0 :=
by
  sorry

end absolute_value_condition_l114_114894


namespace solve_for_y_l114_114034

theorem solve_for_y (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := 
by 
  sorry

end solve_for_y_l114_114034


namespace shaded_square_cover_columns_l114_114989

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2

theorem shaded_square_cover_columns :
  ∃ n : Nat, 
    triangular_number n = 136 ∧ 
    ∀ i : Fin 10, ∃ k ≤ n, (triangular_number k) % 10 = i.val :=
sorry

end shaded_square_cover_columns_l114_114989


namespace smaller_number_is_72_l114_114811

theorem smaller_number_is_72
  (x : ℝ)
  (h1 : (3 * x - 24) / (8 * x - 24) = 4 / 9)
  : 3 * x = 72 :=
sorry

end smaller_number_is_72_l114_114811


namespace distinct_arrangements_l114_114158

-- Defining the conditions as constants
def num_women : ℕ := 9
def num_men : ℕ := 3
def total_slots : ℕ := num_women + num_men

-- Using the combination formula directly as part of the statement
theorem distinct_arrangements : Nat.choose total_slots num_men = 220 := by
  sorry

end distinct_arrangements_l114_114158


namespace midpoint_sum_of_coordinates_l114_114183

theorem midpoint_sum_of_coordinates
  (M : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hmx : (C.1 + D.1) / 2 = M.1)
  (hmy : (C.2 + D.2) / 2 = M.2)
  (hM : M = (3, 5))
  (hC : C = (5, 3)) :
  D.1 + D.2 = 8 :=
by
  sorry

end midpoint_sum_of_coordinates_l114_114183


namespace smallest_b_for_N_fourth_power_l114_114157

theorem smallest_b_for_N_fourth_power : 
  ∃ (b : ℤ), (∀ n : ℤ, 7 * b^2 + 7 * b + 7 = n^4) ∧ b = 18 :=
by
  sorry

end smallest_b_for_N_fourth_power_l114_114157


namespace trains_cross_time_l114_114704

def length_train1 := 140 -- in meters
def length_train2 := 160 -- in meters

def speed_train1_kmph := 60 -- in km/h
def speed_train2_kmph := 48 -- in km/h

def kmph_to_mps (speed : ℕ) := speed * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

def relative_speed_mps := speed_train1_mps + speed_train2_mps

def total_length := length_train1 + length_train2

def time_to_cross := total_length / relative_speed_mps

theorem trains_cross_time : time_to_cross = 10 :=
  by sorry

end trains_cross_time_l114_114704


namespace find_f_100_l114_114622

-- Define the function f such that it satisfies the condition f(10^x) = x
noncomputable def f : ℝ → ℝ := sorry

-- Define the main theorem to prove f(100) = 2 given the condition f(10^x) = x
theorem find_f_100 (h : ∀ x : ℝ, f (10^x) = x) : f 100 = 2 :=
by {
  sorry
}

end find_f_100_l114_114622


namespace original_polygon_sides_l114_114095

theorem original_polygon_sides {n : ℕ} 
    (hn : (n - 2) * 180 = 1080) : n = 7 ∨ n = 8 ∨ n = 9 :=
sorry

end original_polygon_sides_l114_114095


namespace eq_circle_value_of_k_l114_114131

noncomputable def circle_center : Prod ℝ ℝ := (2, 3)
noncomputable def circle_radius := 2
noncomputable def line_equation (k : ℝ) : ℝ → ℝ := fun x => k * x - 1
noncomputable def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

theorem eq_circle (x y : ℝ) : 
  circle_equation x y ↔ (x - 2)^2 + (y - 3)^2 = 4 := 
by sorry

theorem value_of_k (k : ℝ) : 
  (∀ M N : Prod ℝ ℝ, 
  circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 ∧ 
  line_equation k M.1 = M.2 ∧ line_equation k N.1 = N.2 ∧ 
  M ≠ N ∧ 
  (circle_center.1 - M.1) * (circle_center.1 - N.1) + 
  (circle_center.2 - M.2) * (circle_center.2 - N.2) = 0) → 
  (k = 1 ∨ k = 7) := 
by sorry

end eq_circle_value_of_k_l114_114131


namespace shaggy_seeds_l114_114237

theorem shaggy_seeds {N : ℕ} (h1 : 50 < N) (h2 : N < 65) (h3 : N = 60) : 
  ∃ L : ℕ, L = 54 := by
  let L := 54
  sorry

end shaggy_seeds_l114_114237


namespace part1_part2_part3_l114_114854

open Real

-- Definitions of points
structure Point :=
(x : ℝ)
(y : ℝ)

def M (m : ℝ) : Point := ⟨m - 2, 2 * m - 7⟩
def N (n : ℝ) : Point := ⟨n, 3⟩

-- Part 1
theorem part1 : 
  (M (7 / 2)).y = 0 ∧ (M (7 / 2)).x = 3 / 2 :=
by
  sorry

-- Part 2
theorem part2 (m : ℝ) : abs (m - 2) = abs (2 * m - 7) → (m = 5 ∨ m = 3) :=
by
  sorry

-- Part 3
theorem part3 (m n : ℝ) : abs ((M m).y - 3) = 2 ∧ (M m).x = n - 2 → (n = 4 ∨ n = 2) :=
by
  sorry

end part1_part2_part3_l114_114854


namespace minimum_value_l114_114787

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (∃ (m : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → m ≤ (y / x + 1 / y)) ∧
   m = 3 ∧ (∀ (x : ℝ), 0 < x → 0 < (1 - x) → (1 - x) + x = 1 → (y / x + 1 / y = m) ↔ x = 1 / 2)) :=
by
  sorry

end minimum_value_l114_114787


namespace single_intersection_point_l114_114839

theorem single_intersection_point (k : ℝ) :
  (∃! x : ℝ, x^2 - 2 * x - k = 0) ↔ k = 0 :=
by
  sorry

end single_intersection_point_l114_114839


namespace sum_of_ages_is_correct_l114_114010

-- Define the present ages of A, B, and C
def present_age_A : ℕ := 11

-- Define the ratio conditions from 3 years ago
def three_years_ago_ratio (A B C : ℕ) : Prop :=
  B - 3 = 2 * (A - 3) ∧ C - 3 = 3 * (A - 3)

-- The statement we want to prove
theorem sum_of_ages_is_correct {A B C : ℕ} (hA : A = 11)
  (h_ratio : three_years_ago_ratio A B C) :
  A + B + C = 57 :=
by
  -- The proof part will be handled here
  sorry

end sum_of_ages_is_correct_l114_114010


namespace smaller_circle_radius_l114_114748

open Real

def is_geometric_progression (a b c : ℝ) : Prop :=
  (b / a = c / b)

theorem smaller_circle_radius 
  (B1 B2 : ℝ) 
  (r2 : ℝ) 
  (h1 : B1 + B2 = π * r2^2) 
  (h2 : r2 = 5) 
  (h3 : is_geometric_progression B1 B2 (B1 + B2)) :
  sqrt ((-1 + sqrt (1 + 100 * π)) / (2 * π)) = sqrt (B1 / π) :=
by
  sorry

end smaller_circle_radius_l114_114748


namespace expand_expression_l114_114987

theorem expand_expression (y z : ℝ) : 
  -2 * (5 * y^3 - 3 * y^2 * z + 4 * y * z^2 - z^3) = -10 * y^3 + 6 * y^2 * z - 8 * y * z^2 + 2 * z^3 :=
by sorry

end expand_expression_l114_114987


namespace compound_interest_correct_l114_114394

noncomputable def compoundInterest (P: ℝ) (r: ℝ) (n: ℝ) (t: ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_correct :
  compoundInterest 5000 0.04 1 3 - 5000 = 624.32 :=
by
  sorry

end compound_interest_correct_l114_114394


namespace union_sets_l114_114046

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem union_sets :
  M ∪ N = {x | -1 ≤ x ∧ x ≤ 5} := by
  sorry

end union_sets_l114_114046


namespace prove_jimmy_is_2_determine_rachel_age_l114_114163

-- Define the conditions of the problem
variables (a b c r1 r2 : ℤ)

-- Condition 1: Rachel's age and Jimmy's age are roots of the quadratic equation
def is_root (p : ℤ → ℤ) (x : ℤ) : Prop := p x = 0

def quadratic_eq (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Condition 2: Sum of the coefficients is a prime number
def sum_of_coefficients_is_prime : Prop :=
  Nat.Prime (a + b + c).natAbs

-- Condition 3: Substituting Rachel’s age into the quadratic equation gives -55
def substitute_rachel_is_minus_55 (r : ℤ) : Prop :=
  quadratic_eq a b c r = -55

-- Question 1: Prove Jimmy is 2 years old
theorem prove_jimmy_is_2 (h1 : is_root (quadratic_eq a b c) r1)
                           (h2 : is_root (quadratic_eq a b c) r2)
                           (h3 : sum_of_coefficients_is_prime a b c)
                           (h4 : substitute_rachel_is_minus_55 a b c r1) :
  r2 = 2 :=
sorry

-- Question 2: Determine Rachel's age
theorem determine_rachel_age (h1 : is_root (quadratic_eq a b c) r1)
                             (h2 : is_root (quadratic_eq a b c) r2)
                             (h3 : sum_of_coefficients_is_prime a b c)
                             (h4 : substitute_rachel_is_minus_55 a b c r1)
                             (h5 : r2 = 2) :
  r1 = 7 :=
sorry

end prove_jimmy_is_2_determine_rachel_age_l114_114163


namespace problem_statement_l114_114804

theorem problem_statement (a b : ℝ) :
  a^2 + b^2 - a - b - a * b + 0.25 ≥ 0 ∧ (a^2 + b^2 - a - b - a * b + 0.25 = 0 ↔ ((a = 0 ∧ b = 0.5) ∨ (a = 0.5 ∧ b = 0))) :=
by 
  sorry

end problem_statement_l114_114804


namespace intersecting_point_value_l114_114950

theorem intersecting_point_value
  (b a : ℤ)
  (h1 : a = -2 * 2 + b)
  (h2 : 2 = -2 * a + b) :
  a = 2 :=
by
  sorry

end intersecting_point_value_l114_114950


namespace age_of_seventh_person_l114_114598

theorem age_of_seventh_person (A1 A2 A3 A4 A5 A6 A7 D1 D2 D3 D4 D5 : ℕ) 
    (h1 : A1 < A2) (h2 : A2 < A3) (h3 : A3 < A4) (h4 : A4 < A5) (h5 : A5 < A6) 
    (h6 : A2 = A1 + D1) (h7 : A3 = A2 + D2) (h8 : A4 = A3 + D3) 
    (h9 : A5 = A4 + D4) (h10 : A6 = A5 + D5)
    (h11 : A1 + A2 + A3 + A4 + A5 + A6 = 246) 
    (h12 : 246 + A7 = 315) : A7 = 69 :=
by
  sorry

end age_of_seventh_person_l114_114598


namespace river_road_cars_l114_114755

theorem river_road_cars
  (B C : ℕ)
  (h1 : B * 17 = C)
  (h2 : C = B + 80) :
  C = 85 := by
  sorry

end river_road_cars_l114_114755


namespace socks_difference_l114_114407

-- Definitions of the conditions
def week1 : ℕ := 12
def week2 (S : ℕ) : ℕ := S
def week3 (S : ℕ) : ℕ := (12 + S) / 2
def week4 (S : ℕ) : ℕ := (12 + S) / 2 - 3
def total (S : ℕ) : ℕ := week1 + week2 S + week3 S + week4 S

-- Statement of the theorem
theorem socks_difference (S : ℕ) (h : total S = 57) : S - week1 = 1 :=
by 
  -- Proof is not required
  sorry

end socks_difference_l114_114407


namespace common_root_l114_114570

theorem common_root (p : ℝ) :
  (∃ x : ℝ, x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ↔ (p = -3 ∨ p = 9) :=
by
  sorry

end common_root_l114_114570


namespace sequence_sum_l114_114943

theorem sequence_sum : (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19) = -10 :=
by
  sorry

end sequence_sum_l114_114943


namespace polynomial_product_l114_114525

theorem polynomial_product (x : ℝ) : (x - 1) * (x + 3) * (x + 5) = x^3 + 7*x^2 + 7*x - 15 :=
by
  sorry

end polynomial_product_l114_114525


namespace sqrt_4_of_10000000_eq_l114_114899

noncomputable def sqrt_4_of_10000000 : Real := Real.sqrt (Real.sqrt 10000000)

theorem sqrt_4_of_10000000_eq :
  sqrt_4_of_10000000 = 10 * Real.sqrt (Real.sqrt 10) := by
sorry

end sqrt_4_of_10000000_eq_l114_114899


namespace minimum_shirts_to_save_money_by_using_Acme_l114_114142

-- Define the cost functions for Acme and Gamma
def Acme_cost (x : ℕ) : ℕ := 60 + 8 * x
def Gamma_cost (x : ℕ) : ℕ := 12 * x

-- State the theorem to prove that for x = 16, Acme is cheaper than Gamma
theorem minimum_shirts_to_save_money_by_using_Acme : ∀ x ≥ 16, Acme_cost x < Gamma_cost x :=
by
  intros x hx
  sorry

end minimum_shirts_to_save_money_by_using_Acme_l114_114142


namespace sum_of_three_fractions_is_one_l114_114825

theorem sum_of_three_fractions_is_one (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ) = 1 ↔ 
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 6) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) :=
by sorry

end sum_of_three_fractions_is_one_l114_114825


namespace wall_length_correct_l114_114081

noncomputable def length_of_wall : ℝ :=
  let volume_of_one_brick := 25 * 11.25 * 6
  let total_volume_of_bricks := volume_of_one_brick * 6800
  let wall_width := 600
  let wall_height := 22.5
  total_volume_of_bricks / (wall_width * wall_height)

theorem wall_length_correct : length_of_wall = 850 := by
  sorry

end wall_length_correct_l114_114081


namespace find_number_l114_114048

theorem find_number (x : ℝ) : (8 * x = 0.4 * 900) -> x = 45 :=
by
  sorry

end find_number_l114_114048


namespace total_homework_time_l114_114903

variable (num_math_problems num_social_studies_problems num_science_problems : ℕ)
variable (time_per_math_problem time_per_social_studies_problem time_per_science_problem : ℝ)

/-- Prove that the total time taken by Brooke to answer all his homework problems is 48 minutes -/
theorem total_homework_time :
  num_math_problems = 15 →
  num_social_studies_problems = 6 →
  num_science_problems = 10 →
  time_per_math_problem = 2 →
  time_per_social_studies_problem = 0.5 →
  time_per_science_problem = 1.5 →
  (num_math_problems * time_per_math_problem + num_social_studies_problems * time_per_social_studies_problem + num_science_problems * time_per_science_problem) = 48 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_homework_time_l114_114903


namespace range_of_a_l114_114160

-- Let us define the problem conditions and statement in Lean
theorem range_of_a
  (a : ℝ)
  (h : ∀ x y : ℝ, x < y → (3 - a)^x > (3 - a)^y) :
  2 < a ∧ a < 3 :=
sorry

end range_of_a_l114_114160


namespace pair_D_equal_l114_114618

theorem pair_D_equal: (-1)^3 = (-1)^2023 := by
  sorry

end pair_D_equal_l114_114618


namespace eval_expression_l114_114708

-- Define the redefined operation
def red_op (a b : ℝ) : ℝ := (a + b)^2

-- Define the target expression to be evaluated
def expr (x y : ℝ) : ℝ := red_op ((x + y)^2) ((x - y)^2)

-- State the theorem
theorem eval_expression (x y : ℝ) : expr x y = 4 * (x^2 + y^2)^2 := by
  sorry

end eval_expression_l114_114708


namespace part_a_part_b_l114_114552

noncomputable def volume_of_prism (V : ℝ) : ℝ :=
  (9 / 250) * V

noncomputable def max_volume_of_prism (V : ℝ) : ℝ :=
  (1 / 12) * V

theorem part_a (V : ℝ) :
  volume_of_prism V = (9 / 250) * V :=
  by sorry

theorem part_b (V : ℝ) :
  max_volume_of_prism V = (1 / 12) * V :=
  by sorry

end part_a_part_b_l114_114552


namespace third_square_length_l114_114491

theorem third_square_length 
  (A1 : 8 * 5 = 40) 
  (A2 : 10 * 7 = 70) 
  (A3 : 15 * 9 = 135) 
  (L : ℕ) 
  (A4 : 40 + 70 + L * 5 = 135) 
  : L = 5 := 
sorry

end third_square_length_l114_114491


namespace measure_six_pints_l114_114521
-- Importing the necessary library

-- Defining the problem conditions
def total_wine : ℕ := 12
def capacity_8_pint_vessel : ℕ := 8
def capacity_5_pint_vessel : ℕ := 5

-- The problem to prove: it is possible to measure 6 pints into the 8-pint container
theorem measure_six_pints :
  ∃ (n : ℕ), n = 6 ∧ n ≤ capacity_8_pint_vessel := 
sorry

end measure_six_pints_l114_114521


namespace find_number_l114_114826

theorem find_number (x : ℤ) (h : x - 7 = 9) : x * 3 = 48 :=
by sorry

end find_number_l114_114826


namespace jane_last_segment_speed_l114_114924

theorem jane_last_segment_speed :
  let total_distance := 120  -- in miles
  let total_time := (75 / 60)  -- in hours
  let segment_time := (25 / 60)  -- in hours
  let speed1 := 75  -- in mph
  let speed2 := 80  -- in mph
  let overall_avg_speed := total_distance / total_time
  let x := (3 * overall_avg_speed) - speed1 - speed2
  x = 133 :=
by { sorry }

end jane_last_segment_speed_l114_114924


namespace reciprocal_inequality_pos_reciprocal_inequality_neg_l114_114689

theorem reciprocal_inequality_pos {a b : ℝ} (h : a < b) (ha : 0 < a) : (1 / a) > (1 / b) :=
sorry

theorem reciprocal_inequality_neg {a b : ℝ} (h : a < b) (hb : b < 0) : (1 / a) < (1 / b) :=
sorry

end reciprocal_inequality_pos_reciprocal_inequality_neg_l114_114689


namespace curve_crossing_self_l114_114696

theorem curve_crossing_self (t t' : ℝ) :
  (t^3 - t - 2 = t'^3 - t' - 2) ∧ (t ≠ t') ∧ 
  (t^3 - t^2 - 9 * t + 5 = t'^3 - t'^2 - 9 * t' + 5) → 
  (t = 3 ∧ t' = -3) ∨ (t = -3 ∧ t' = 3) →
  (t^3 - t - 2 = 22) ∧ (t^3 - t^2 - 9 * t + 5 = -4) :=
by
  sorry

end curve_crossing_self_l114_114696


namespace cylinder_height_comparison_l114_114957

theorem cylinder_height_comparison (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by {
  -- Proof steps here, not required per instruction
  sorry
}

end cylinder_height_comparison_l114_114957


namespace free_fall_time_l114_114901

theorem free_fall_time (h : ℝ) (t : ℝ) (h_eq : h = 4.9 * t^2) (h_val : h = 490) : t = 10 :=
by
  sorry

end free_fall_time_l114_114901


namespace find_f_2013_l114_114073

open Function

theorem find_f_2013 {f : ℝ → ℝ} (Hodd : ∀ x, f (-x) = -f x)
  (Hperiodic : ∀ x, f (x + 4) = f x)
  (Hf_neg1 : f (-1) = 2) :
  f 2013 = -2 := by
sorry

end find_f_2013_l114_114073


namespace shaded_area_of_four_circles_l114_114013

open Real

noncomputable def area_shaded_region (r : ℝ) (num_circles : ℕ) : ℝ :=
  let area_quarter_circle := (π * r^2) / 4
  let area_triangle := (r * r) / 2
  let area_one_checkered_region := area_quarter_circle - area_triangle
  let num_checkered_regions := num_circles * 2
  num_checkered_regions * area_one_checkered_region

theorem shaded_area_of_four_circles : area_shaded_region 5 4 = 50 * (π - 2) :=
by
  sorry

end shaded_area_of_four_circles_l114_114013


namespace linear_inequalities_solution_range_l114_114542

theorem linear_inequalities_solution_range (m : ℝ) :
  (∃ x : ℝ, x - 2 * m < 0 ∧ x + m > 2) ↔ m > 2 / 3 :=
by
  sorry

end linear_inequalities_solution_range_l114_114542


namespace balance_blue_balls_l114_114168

variables (G B Y W : ℝ)

-- Definitions based on conditions
def condition1 : Prop := 3 * G = 6 * B
def condition2 : Prop := 2 * Y = 5 * B
def condition3 : Prop := 6 * B = 4 * W

-- Statement of the problem
theorem balance_blue_balls (h1 : condition1 G B) (h2 : condition2 Y B) (h3 : condition3 B W) :
  4 * G + 2 * Y + 2 * W = 16 * B :=
sorry

end balance_blue_balls_l114_114168


namespace simplification_qrt_1_simplification_qrt_2_l114_114493

-- Problem 1
theorem simplification_qrt_1 : (2 * Real.sqrt 12 + 3 * Real.sqrt 3 - Real.sqrt 27) = 4 * Real.sqrt 3 :=
by
  sorry

-- Problem 2
theorem simplification_qrt_2 : (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2 * 12) + Real.sqrt 24) = 4 + Real.sqrt 6 :=
by
  sorry

end simplification_qrt_1_simplification_qrt_2_l114_114493


namespace min_groups_required_l114_114452

/-!
  Prove that if a coach has 30 athletes and wants to arrange them into equal groups with no more than 12 athletes each, 
  then the minimum number of groups required is 3.
-/

theorem min_groups_required (total_athletes : ℕ) (max_athletes_per_group : ℕ) (h_total : total_athletes = 30) (h_max : max_athletes_per_group = 12) :
  ∃ (min_groups : ℕ), min_groups = total_athletes / 10 ∧ (total_athletes % 10 = 0) := by
  sorry

end min_groups_required_l114_114452


namespace machine_purchase_price_l114_114629

theorem machine_purchase_price (P : ℝ) (h : 0.80 * P = 6400) : P = 8000 :=
by
  sorry

end machine_purchase_price_l114_114629


namespace expression_value_l114_114263

noncomputable def compute_expression (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80

theorem expression_value (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1)
    : compute_expression ω h h2 = -ω^2 :=
sorry

end expression_value_l114_114263


namespace center_cell_value_l114_114015

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l114_114015


namespace number_of_cars_l114_114436

theorem number_of_cars (people_per_car : ℝ) (total_people : ℝ) (h1 : people_per_car = 63.0) (h2 : total_people = 189) : total_people / people_per_car = 3 := by
  sorry

end number_of_cars_l114_114436


namespace time_to_cover_escalator_l114_114381

noncomputable def average_speed (initial_speed final_speed : ℝ) : ℝ :=
  (initial_speed + final_speed) / 2

noncomputable def combined_speed (escalator_speed person_average_speed : ℝ) : ℝ :=
  escalator_speed + person_average_speed

noncomputable def coverage_time (length combined_speed : ℝ) : ℝ :=
  length / combined_speed

theorem time_to_cover_escalator
  (escalator_speed : ℝ := 20)
  (length : ℝ := 300)
  (initial_person_speed : ℝ := 3)
  (final_person_speed : ℝ := 5) :
  coverage_time length (combined_speed escalator_speed (average_speed initial_person_speed final_person_speed)) = 12.5 :=
by
  sorry

end time_to_cover_escalator_l114_114381


namespace average_speed_l114_114659

-- Define the average speed v
variable {v : ℝ}

-- Conditions
def day1_distance : ℝ := 160  -- 160 miles on the first day
def day2_distance : ℝ := 280  -- 280 miles on the second day
def time_difference : ℝ := 3  -- 3 hours difference

-- Theorem to prove the average speed
theorem average_speed (h1 : day1_distance / v + time_difference = day2_distance / v) : v = 40 := 
by 
  sorry  -- Proof is omitted

end average_speed_l114_114659


namespace perp_lines_iff_m_values_l114_114824

section
variables (m x y : ℝ)

def l1 := (m * x + y - 2 = 0)
def l2 := ((m + 1) * x - 2 * m * y + 1 = 0)

theorem perp_lines_iff_m_values (h1 : l1 m x y) (h2 : l2 m x y) (h_perp : (m * (m + 1) + (-2 * m) = 0)) : m = 0 ∨ m = 1 :=
by {
  sorry
}
end

end perp_lines_iff_m_values_l114_114824


namespace inequality_proof_l114_114560

open Real

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 :=
by
  sorry

end inequality_proof_l114_114560


namespace det_A_eq_6_l114_114287

open Matrix

variables {R : Type*} [Field R]

def A (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![a, 2], ![-3, d]]

def B (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![2 * a, 1], ![-1, d]]

noncomputable def B_inv (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  let detB := (2 * a * d + 1)
  ![![d / detB, -1 / detB], ![1 / detB, (2 * a) / detB]]

theorem det_A_eq_6 (a d : R) (hB_inv : (A a d) + (B_inv a d) = 0) : det (A a d) = 6 :=
  sorry

end det_A_eq_6_l114_114287


namespace num_valid_permutations_l114_114181

theorem num_valid_permutations : 
  let digits := [2, 0, 2, 3]
  let num_2 := 2
  let total_permutations := Nat.factorial 4 / (Nat.factorial num_2 * Nat.factorial 1 * Nat.factorial 1)
  let valid_start_2 := Nat.factorial 3
  let valid_start_3 := Nat.factorial 3 / Nat.factorial 2
  total_permutations = 12 ∧ valid_start_2 = 6 ∧ valid_start_3 = 3 ∧ (valid_start_2 + valid_start_3 = 9) := 
by
  sorry

end num_valid_permutations_l114_114181


namespace cakes_served_during_lunch_today_l114_114036

-- Define the conditions as parameters
variables
  (L : ℕ)   -- Number of cakes served during lunch today
  (D : ℕ := 6)  -- Number of cakes served during dinner today
  (Y : ℕ := 3)  -- Number of cakes served yesterday
  (T : ℕ := 14)  -- Total number of cakes served

-- Define the theorem to prove L = 5
theorem cakes_served_during_lunch_today : L + D + Y = T → L = 5 :=
by
  sorry

end cakes_served_during_lunch_today_l114_114036


namespace find_multiple_of_savings_l114_114344

variable (A K m : ℝ)

-- Conditions
def condition1 : Prop := A - 150 = (1 / 3) * K
def condition2 : Prop := A + K = 750

-- Question
def question : Prop := m * K = 3 * A

-- Proof Problem Statement
theorem find_multiple_of_savings (h1 : condition1 A K) (h2 : condition2 A K) : 
  question A K 2 :=
sorry

end find_multiple_of_savings_l114_114344


namespace div_by_seven_equiv_l114_114890

-- Given integers a and b, prove that 10a + b is divisible by 7 if and only if a - 2b is divisible by 7.
theorem div_by_seven_equiv (a b : ℤ) : (10 * a + b) % 7 = 0 ↔ (a - 2 * b) % 7 = 0 := sorry

end div_by_seven_equiv_l114_114890


namespace max_triangles_l114_114067

theorem max_triangles (n : ℕ) (h : n = 10) : 
  ∃ T : ℕ, T = 150 :=
by
  sorry

end max_triangles_l114_114067


namespace remainder_sum_of_integers_division_l114_114902

theorem remainder_sum_of_integers_division (n S : ℕ) (hn_cond : n > 0) (hn_sq : n^2 + 12 * n - 3007 ≥ 0) (hn_square : ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2):
  S = n → S % 1000 = 516 := 
sorry

end remainder_sum_of_integers_division_l114_114902


namespace find_triplet_l114_114085

theorem find_triplet (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y) ^ 2 + 3 * x + y + 1 = z ^ 2 → y = x ∧ z = 2 * x + 1 :=
by
  sorry

end find_triplet_l114_114085


namespace roots_exist_l114_114893

theorem roots_exist (a : ℝ) : ∃ x : ℝ, a * x^2 - x = 0 := by
  sorry

end roots_exist_l114_114893


namespace total_apples_picked_l114_114310

def apples_picked : ℕ :=
  let mike := 7
  let nancy := 3
  let keith := 6
  let olivia := 12
  let thomas := 8
  mike + nancy + keith + olivia + thomas

theorem total_apples_picked :
  apples_picked = 36 :=
by
  -- Proof would go here; 'sorry' is used to skip the proof.
  sorry

end total_apples_picked_l114_114310


namespace fraction_nonnegative_iff_l114_114725

theorem fraction_nonnegative_iff (x : ℝ) :
  (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ 0 ≤ x ∧ x < 3 :=
by
  -- Proof goes here
  sorry

end fraction_nonnegative_iff_l114_114725


namespace range_of_t_l114_114764

variable (f : ℝ → ℝ) (t : ℝ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_t {f : ℝ → ℝ} {t : ℝ} 
  (Hodd : is_odd f) 
  (Hperiodic : ∀ x, f (x + 5 / 2) = -1 / f x) 
  (Hf1 : f 1 ≥ 1) 
  (Hf2014 : f 2014 = (t + 3) / (t - 3)) : 
  0 ≤ t ∧ t < 3 := by
  sorry

end range_of_t_l114_114764


namespace insert_digits_identical_l114_114203

theorem insert_digits_identical (A B : List Nat) (hA : A.length = 2007) (hB : B.length = 2007)
  (hErase : ∃ (C : List Nat) (erase7A : List Nat → List Nat) (erase7B : List Nat → List Nat),
    (erase7A A = C) ∧ (erase7B B = C) ∧ (C.length = 2000)) :
  ∃ (D : List Nat) (insert7A : List Nat → List Nat) (insert7B : List Nat → List Nat),
    (insert7A A = D) ∧ (insert7B B = D) ∧ (D.length = 2014) := sorry

end insert_digits_identical_l114_114203


namespace fraction_irreducible_l114_114502

theorem fraction_irreducible (n : ℕ) (hn : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by sorry

end fraction_irreducible_l114_114502


namespace burger_cost_l114_114565

theorem burger_cost 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 440)
  (h2 : 3 * b + 2 * s = 330) : b = 110 := 
by 
  sorry

end burger_cost_l114_114565


namespace increasing_interval_l114_114886

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval : {x : ℝ | 2 < x} = { x : ℝ | (x - 3) * Real.exp x > 0 } :=
by
  sorry

end increasing_interval_l114_114886


namespace jiujiang_liansheng_sampling_l114_114959

def bag_numbers : List ℕ := [7, 17, 27, 37, 47]

def systematic_sampling (N n : ℕ) (selected_bags : List ℕ) : Prop :=
  ∃ k i, k = N / n ∧ ∀ j, j < List.length selected_bags → selected_bags.get? j = some (i + k * j)

theorem jiujiang_liansheng_sampling :
  systematic_sampling 50 5 bag_numbers :=
by
  sorry

end jiujiang_liansheng_sampling_l114_114959


namespace terminal_side_alpha_minus_beta_nonneg_x_axis_l114_114396

theorem terminal_side_alpha_minus_beta_nonneg_x_axis
  (α β : ℝ) (k : ℤ) (h : α = k * 360 + β) : 
  (∃ m : ℤ, α - β = m * 360) := 
sorry

end terminal_side_alpha_minus_beta_nonneg_x_axis_l114_114396


namespace ratio_brother_to_joanna_l114_114128

/-- Definitions for the conditions -/
def joanna_money : ℝ := 8
def sister_money : ℝ := 4 -- since it's half of Joanna's money
def total_money : ℝ := 36

/-- Stating the theorem -/
theorem ratio_brother_to_joanna (x : ℝ) (h : joanna_money + 8*x + sister_money = total_money) :
  x = 3 :=
by 
  -- The ratio of brother's money to Joanna's money is 3:1
  sorry

end ratio_brother_to_joanna_l114_114128


namespace range_of_b_no_common_points_l114_114780

theorem range_of_b_no_common_points (b : ℝ) :
  ¬ (∃ x : ℝ, 2 ^ |x| - 1 = b) ↔ b < 0 :=
by
  sorry

end range_of_b_no_common_points_l114_114780


namespace average_cost_is_70_l114_114675

noncomputable def C_before_gratuity (total_bill : ℝ) (gratuity_rate : ℝ) : ℝ :=
  total_bill / (1 + gratuity_rate)

noncomputable def average_cost_per_individual (C : ℝ) (total_people : ℝ) : ℝ :=
  C / total_people

theorem average_cost_is_70 :
  let total_bill := 756
  let gratuity_rate := 0.20
  let total_people := 9
  average_cost_per_individual (C_before_gratuity total_bill gratuity_rate) total_people = 70 :=
by
  sorry

end average_cost_is_70_l114_114675


namespace find_m_range_l114_114932

def vector_a : ℝ × ℝ := (1, 2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)
def is_acute (a b : ℝ × ℝ) : Prop := dot_product a b > 0

theorem find_m_range (m : ℝ) :
  is_acute vector_a (4, m) → m ∈ Set.Ioo (-2 : ℝ) 8 ∪ Set.Ioi 8 := 
by
  sorry

end find_m_range_l114_114932


namespace min_number_of_squares_l114_114241

theorem min_number_of_squares (length width : ℕ) (h_length : length = 10) (h_width : width = 9) : 
  ∃ n, n = 10 :=
by
  sorry

end min_number_of_squares_l114_114241


namespace largest_possible_value_of_p_l114_114185

theorem largest_possible_value_of_p (m n p : ℕ) (h1 : m ≤ n) (h2 : n ≤ p)
  (h3 : 2 * m * n * p = (m + 2) * (n + 2) * (p + 2)) : p ≤ 130 :=
by
  sorry

end largest_possible_value_of_p_l114_114185


namespace total_hours_correct_l114_114760

def hours_watching_tv_per_day : ℕ := 4
def days_per_week : ℕ := 7
def days_playing_video_games_per_week : ℕ := 3

def tv_hours_per_week : ℕ := hours_watching_tv_per_day * days_per_week
def video_game_hours_per_day : ℕ := hours_watching_tv_per_day / 2
def video_game_hours_per_week : ℕ := video_game_hours_per_day * days_playing_video_games_per_week

def total_hours_per_week : ℕ := tv_hours_per_week + video_game_hours_per_week

theorem total_hours_correct :
  total_hours_per_week = 34 := by
  sorry

end total_hours_correct_l114_114760


namespace trip_cost_is_correct_l114_114376

-- Given conditions
def bills_cost : ℕ := 3500
def save_per_month : ℕ := 500
def savings_duration_months : ℕ := 2 * 12
def savings : ℕ := save_per_month * savings_duration_months
def remaining_after_bills : ℕ := 8500

-- Prove that the cost of the trip to Paris is 3500 dollars
theorem trip_cost_is_correct : (savings - remaining_after_bills) = bills_cost :=
sorry

end trip_cost_is_correct_l114_114376


namespace inradius_of_equal_area_and_perimeter_l114_114522

theorem inradius_of_equal_area_and_perimeter
  (a b c : ℝ)
  (A : ℝ)
  (h1 : A = a + b + c)
  (s : ℝ := (a + b + c) / 2)
  (h2 : A = s * (2 * A / (a + b + c))) :
  ∃ r : ℝ, r = 2 := by
  sorry

end inradius_of_equal_area_and_perimeter_l114_114522


namespace supplement_of_double_complement_l114_114554

def angle : ℝ := 30

def complement (θ : ℝ) : ℝ :=
  90 - θ

def double_complement (θ : ℝ) : ℝ :=
  2 * (complement θ)

def supplement (θ : ℝ) : ℝ :=
  180 - θ

theorem supplement_of_double_complement (θ : ℝ) (h : θ = angle) : supplement (double_complement θ) = 60 :=
by
  sorry

end supplement_of_double_complement_l114_114554


namespace polynomial_q_value_l114_114802

theorem polynomial_q_value :
  ∀ (p q d : ℝ),
    (d = 6) →
    (-p / 3 = -d) →
    (1 + p + q + d = - d) →
    q = -31 :=
by sorry

end polynomial_q_value_l114_114802


namespace work_completion_l114_114210

theorem work_completion (a b : ℕ) (h1 : a + b = 5) (h2 : a = 10) : b = 10 := by
  sorry

end work_completion_l114_114210


namespace number_of_ways_to_buy_three_items_l114_114584

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l114_114584


namespace line_through_intersections_l114_114568

-- Conditions
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem statement
theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → x - y - 3 = 0 :=
by
  sorry

end line_through_intersections_l114_114568


namespace arccos_sin_three_l114_114685

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end arccos_sin_three_l114_114685


namespace find_ordered_triples_l114_114014

-- Define the problem conditions using Lean structures.
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_ordered_triples (a b c : ℕ) :
  (is_perfect_square (a^2 + 2 * b + c) ∧
   is_perfect_square (b^2 + 2 * c + a) ∧
   is_perfect_square (c^2 + 2 * a + b))
  ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 43 ∧ b = 127 ∧ c = 106) :=
by sorry

end find_ordered_triples_l114_114014


namespace quadratic_inequality_l114_114145

theorem quadratic_inequality (a b c d x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 = -a) 
  (h2 : x1 * x2 = b)
  (h3 : x3 + x4 = -c)
  (h4 : x3 * x4 = d)
  (h5 : b > d)
  (h6 : b > 0)
  (h7 : d > 0) :
  a^2 - c^2 > b - d :=
by
  sorry

end quadratic_inequality_l114_114145


namespace calculate_value_l114_114215

theorem calculate_value : (535^2 - 465^2) / 70 = 1000 := by
  sorry

end calculate_value_l114_114215


namespace fraction_a_over_b_l114_114402

theorem fraction_a_over_b (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 3 * y - 6 * x = b) (hb : b ≠ 0) : a / b = -1 / 3 :=
by
  sorry

end fraction_a_over_b_l114_114402


namespace f_2_solutions_l114_114047

theorem f_2_solutions : 
  ∀ (x y : ℤ), 
    (1 ≤ x) ∧ (0 ≤ y) ∧ (y ≤ (-x + 2)) → 
    (∃ (a b c : Int), 
      (a = 1 ∧ (b = 0 ∨ b = 1) ∨ 
       a = 2 ∧ b = 0) ∧ 
      a = x ∧ b = y ∨ 
      c = 3 → false) ∧ 
    (∃ n : ℕ, n = 3) := by
  sorry

end f_2_solutions_l114_114047


namespace emma_age_proof_l114_114606

def is_age_of_emma (age : Nat) : Prop := 
  let guesses := [26, 29, 31, 33, 35, 39, 42, 44, 47, 50]
  let at_least_60_percent_low := (guesses.filter (· < age)).length * 10 ≥ 6 * guesses.length
  let exactly_two_off_by_one := (guesses.filter (λ x => x = age - 1 ∨ x = age + 1)).length = 2
  let is_prime := Nat.Prime age
  at_least_60_percent_low ∧ exactly_two_off_by_one ∧ is_prime

theorem emma_age_proof : is_age_of_emma 43 := 
  by sorry

end emma_age_proof_l114_114606


namespace total_percentage_increase_l114_114667

def initial_time : ℝ := 45
def additive_A_increase : ℝ := 0.35
def additive_B_increase : ℝ := 0.20

theorem total_percentage_increase :
  let time_after_A := initial_time * (1 + additive_A_increase)
  let time_after_B := time_after_A * (1 + additive_B_increase)
  (time_after_B - initial_time) / initial_time * 100 = 62 :=
  sorry

end total_percentage_increase_l114_114667


namespace total_candles_used_l114_114072

def cakes_baked : ℕ := 8
def cakes_given_away : ℕ := 2
def remaining_cakes : ℕ := cakes_baked - cakes_given_away
def candles_per_cake : ℕ := 6

theorem total_candles_used : remaining_cakes * candles_per_cake = 36 :=
by
  -- proof omitted
  sorry

end total_candles_used_l114_114072


namespace new_person_weight_l114_114956

theorem new_person_weight :
  (8 * 2.5 + 75 = 95) :=
by sorry

end new_person_weight_l114_114956


namespace rhombus_longer_diagonal_length_l114_114422

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l114_114422


namespace yard_length_l114_114334

theorem yard_length (father_step : ℝ) (son_step : ℝ) (total_footprints : ℕ) 
  (h_father_step : father_step = 0.72) 
  (h_son_step : son_step = 0.54) 
  (h_total_footprints : total_footprints = 61) : 
  ∃ length : ℝ, length = 21.6 :=
by
  sorry

end yard_length_l114_114334


namespace continuous_zero_point_condition_l114_114575

theorem continuous_zero_point_condition (f : ℝ → ℝ) {a b : ℝ} (h_cont : ContinuousOn f (Set.Icc a b)) :
  (f a * f b < 0) → (∃ c ∈ Set.Ioo a b, f c = 0) ∧ ¬ (∃ c ∈ Set.Ioo a b, f c = 0 → f a * f b < 0) :=
sorry

end continuous_zero_point_condition_l114_114575


namespace linda_savings_l114_114206

theorem linda_savings :
  let original_savings := 880
  let cost_of_tv := 220
  let amount_spent_on_furniture := original_savings - cost_of_tv
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings
  fraction_spent_on_furniture = 3 / 4 :=
by
  -- original savings
  let original_savings := 880
  -- cost of the TV
  let cost_of_tv := 220
  -- amount spent on furniture
  let amount_spent_on_furniture := original_savings - cost_of_tv
  -- fraction spent on furniture
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings

  -- need to show that this fraction is 3/4
  sorry

end linda_savings_l114_114206


namespace required_oranges_for_juice_l114_114742

theorem required_oranges_for_juice (oranges quarts : ℚ) (h : oranges = 36 ∧ quarts = 48) :
  ∃ x, ((oranges / quarts) = (x / 6) ∧ x = 4.5) := 
by sorry

end required_oranges_for_juice_l114_114742


namespace problem1_problem2_l114_114151

noncomputable def f (x a : ℝ) := |x - 2 * a|
noncomputable def g (x a : ℝ) := |x + a|

theorem problem1 (x m : ℝ): (∃ x, f x 1 - g x 1 ≥ m) → m ≤ 3 :=
by
  sorry

theorem problem2 (a : ℝ): (∀ x, f x a + g x a ≥ 3) → (a ≥ 1 ∨ a ≤ -1) :=
by
  sorry

end problem1_problem2_l114_114151


namespace external_angle_at_C_l114_114544

-- Definitions based on conditions
def angleA : ℝ := 40
def B := 2 * angleA
def sum_of_angles_in_triangle (A B C : ℝ) : Prop := A + B + C = 180
def external_angle (C : ℝ) : ℝ := 180 - C

-- Theorem statement
theorem external_angle_at_C :
  ∃ C : ℝ, sum_of_angles_in_triangle angleA B C ∧ external_angle C = 120 :=
sorry

end external_angle_at_C_l114_114544


namespace simplify_expression_l114_114374

variable (i : ℂ)

-- Define the conditions

def i_squared_eq_neg_one : Prop := i^2 = -1
def i_cubed_eq_neg_i : Prop := i^3 = i * i^2 ∧ i^3 = -i
def i_fourth_eq_one : Prop := i^4 = (i^2)^2 ∧ i^4 = 1
def i_fifth_eq_i : Prop := i^5 = i * i^4 ∧ i^5 = i

-- Define the proof problem

theorem simplify_expression (h1 : i_squared_eq_neg_one i) (h2 : i_cubed_eq_neg_i i) (h3 : i_fourth_eq_one i) (h4 : i_fifth_eq_i i) : 
  i + i^2 + i^3 + i^4 + i^5 = i := 
  by sorry

end simplify_expression_l114_114374


namespace find_y_from_equation_l114_114159

theorem find_y_from_equation (y : ℕ) 
  (h : (12 ^ 2) * (6 ^ 3) / y = 72) : 
  y = 432 :=
  sorry

end find_y_from_equation_l114_114159


namespace domain_and_monotone_l114_114828

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem domain_and_monotone :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → ∃ y, f x = y) ∧
  ∀ x1 x2 : ℝ, 1 < x1 ∧ x1 < x2 → f x1 < f x2 :=
by
  sorry

end domain_and_monotone_l114_114828


namespace find_n_l114_114874

theorem find_n (n : ℕ) (hn : (Nat.choose n 2 : ℚ) / 2^n = 10 / 32) : n = 5 :=
by
  sorry

end find_n_l114_114874


namespace solve_system_l114_114978

theorem solve_system (x y : ℝ) :
  (x^3 - x + 1 = y^2 ∧ y^3 - y + 1 = x^2) ↔ ((x = 1 ∨ x = -1) ∧ (y = 1 ∨ y = -1)) :=
by
  sorry

end solve_system_l114_114978


namespace number_of_pairs_l114_114836

theorem number_of_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m^2 + n < 50) : 
  ∃! p : ℕ, p = 203 := 
sorry

end number_of_pairs_l114_114836


namespace correct_addition_by_changing_digit_l114_114001

theorem correct_addition_by_changing_digit :
  ∃ (d : ℕ), (d < 10) ∧ (d = 4) ∧
  (374 + (500 + d) + 286 = 1229 - 50) :=
by
  sorry

end correct_addition_by_changing_digit_l114_114001


namespace linda_change_l114_114968

-- Defining the conditions
def cost_per_banana : ℝ := 0.30
def number_of_bananas : ℕ := 5
def amount_paid : ℝ := 10.00

-- Proving the statement
theorem linda_change :
  amount_paid - (number_of_bananas * cost_per_banana) = 8.50 :=
by
  sorry

end linda_change_l114_114968


namespace chocolate_chip_cookie_price_l114_114064

noncomputable def price_of_chocolate_chip_cookies :=
  let total_boxes := 1585
  let total_revenue := 1586.75
  let plain_boxes := 793.375
  let price_of_plain := 0.75
  let revenue_plain := plain_boxes * price_of_plain
  let choco_boxes := total_boxes - plain_boxes
  (993.71875 - revenue_plain) / choco_boxes

theorem chocolate_chip_cookie_price :
  price_of_chocolate_chip_cookies = 1.2525 :=
by sorry

end chocolate_chip_cookie_price_l114_114064


namespace problem_statement_l114_114829

theorem problem_statement {m n : ℝ} 
  (h1 : (n + 2 * m) / (1 + m ^ 2) = -1 / 2) 
  (h2 : -(1 + n) + 2 * (m + 2) = 0) : 
  (m / n = -1) := 
sorry

end problem_statement_l114_114829


namespace planes_meet_in_50_minutes_l114_114319

noncomputable def time_to_meet (d : ℕ) (vA vB : ℕ) : ℚ :=
  d / (vA + vB : ℚ)

theorem planes_meet_in_50_minutes
  (d : ℕ) (vA vB : ℕ)
  (h_d : d = 500) (h_vA : vA = 240) (h_vB : vB = 360) :
  (time_to_meet d vA vB * 60 : ℚ) = 50 := by
  sorry

end planes_meet_in_50_minutes_l114_114319


namespace opposite_of_neg_one_div_2023_l114_114738

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l114_114738


namespace mabel_total_tomatoes_l114_114171

def tomatoes_first_plant : ℕ := 12

def tomatoes_second_plant : ℕ := (2 * tomatoes_first_plant) - 6

def tomatoes_combined_first_two : ℕ := tomatoes_first_plant + tomatoes_second_plant

def tomatoes_third_plant : ℕ := tomatoes_combined_first_two / 2

def tomatoes_each_fourth_fifth_plant : ℕ := 3 * tomatoes_combined_first_two

def tomatoes_combined_fourth_fifth : ℕ := 2 * tomatoes_each_fourth_fifth_plant

def tomatoes_each_sixth_seventh_plant : ℕ := (3 * tomatoes_combined_first_two) / 2

def tomatoes_combined_sixth_seventh : ℕ := 2 * tomatoes_each_sixth_seventh_plant

def total_tomatoes : ℕ := tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant + tomatoes_combined_fourth_fifth + tomatoes_combined_sixth_seventh

theorem mabel_total_tomatoes : total_tomatoes = 315 :=
by
  sorry

end mabel_total_tomatoes_l114_114171


namespace greatest_two_digit_multiple_of_17_l114_114121

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l114_114121


namespace total_eyes_correct_l114_114652

-- Conditions
def boys := 21 * 2 + 2 * 1
def girls := 15 * 2 + 3 * 1
def cats := 8 * 2 + 2 * 1
def spiders := 4 * 8 + 1 * 6

-- Total count of eyes
def total_eyes := boys + girls + cats + spiders

theorem total_eyes_correct: total_eyes = 133 :=
by 
  -- Here the proof steps would go, which we are skipping
  sorry

end total_eyes_correct_l114_114652


namespace problem_solution_l114_114835

theorem problem_solution 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h₁ : a = ⌊2 + Real.sqrt 2⌋) 
  (h₂ : b = (2 + Real.sqrt 2) - ⌊2 + Real.sqrt 2⌋)
  (h₃ : c = ⌊4 - Real.sqrt 2⌋)
  (h₄ : d = (4 - Real.sqrt 2) - ⌊4 - Real.sqrt 2⌋) :
  (b + d) / (a * c) = 1 / 6 :=
by
  sorry

end problem_solution_l114_114835


namespace find_g2_l114_114388

variable (g : ℝ → ℝ)

theorem find_g2 (h : ∀ x : ℝ, g (3 * x - 7) = 5 * x + 11) : g 2 = 26 := by
  sorry

end find_g2_l114_114388


namespace candle_burning_time_l114_114670

theorem candle_burning_time :
  ∃ T : ℝ, 
    (∀ T, 0 ≤ T ∧ T ≤ 4 → thin_candle_length = 24 - 6 * T) ∧
    (∀ T, 0 ≤ T ∧ T ≤ 6 → thick_candle_length = 24 - 4 * T) ∧
    (2 * (24 - 6 * T) = 24 - 4 * T) →
    T = 3 :=
by
  sorry

end candle_burning_time_l114_114670


namespace mason_grandmother_age_l114_114429

-- Defining the ages of Mason, Sydney, Mason's father, and Mason's grandmother
def mason_age : ℕ := 20

def sydney_age (S : ℕ) : Prop :=
  mason_age = S / 3

def father_age (S F : ℕ) : Prop :=
  F = S + 6

def grandmother_age (F G : ℕ) : Prop :=
  G = 2 * F

theorem mason_grandmother_age (S F G : ℕ) (h1 : sydney_age S) (h2 : father_age S F) (h3 : grandmother_age F G) : G = 132 :=
by
  -- leaving the proof as a sorry
  sorry

end mason_grandmother_age_l114_114429


namespace pirates_gold_coins_l114_114113

theorem pirates_gold_coins (S a b c d e : ℕ) (h1 : a = S / 3) (h2 : b = S / 4) (h3 : c = S / 5) (h4 : d = S / 6) (h5 : e = 90) :
  S = 1800 :=
by
  -- Definitions and assumptions would go here
  sorry

end pirates_gold_coins_l114_114113


namespace smallest_d_l114_114497

theorem smallest_d (c d : ℕ) (h1 : c - d = 8)
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : d = 4 := by
  sorry

end smallest_d_l114_114497


namespace intersect_setA_setB_l114_114122

def setA : Set ℝ := {x | x < 2}
def setB : Set ℝ := {x | 3 - 2 * x > 0}

theorem intersect_setA_setB :
  setA ∩ setB = {x | x < 3 / 2} :=
by
  -- proof goes here
  sorry

end intersect_setA_setB_l114_114122


namespace lucy_last_10_shots_l114_114706

variable (shots_30 : ℕ) (percentage_30 : ℚ) (total_shots : ℕ) (percentage_40 : ℚ)
variable (shots_made_30 : ℕ) (shots_made_40 : ℕ) (shots_made_last_10 : ℕ)

theorem lucy_last_10_shots 
    (h1 : shots_30 = 30) 
    (h2 : percentage_30 = 0.60) 
    (h3 : total_shots = 40) 
    (h4 : percentage_40 = 0.62 )
    (h5 : shots_made_30 = Nat.floor (percentage_30 * shots_30)) 
    (h6 : shots_made_40 = Nat.floor (percentage_40 * total_shots))
    (h7 : shots_made_last_10 = shots_made_40 - shots_made_30) 
    : shots_made_last_10 = 7 := sorry

end lucy_last_10_shots_l114_114706


namespace second_term_deposit_interest_rate_l114_114736

theorem second_term_deposit_interest_rate
  (initial_deposit : ℝ)
  (first_term_annual_rate : ℝ)
  (first_term_months : ℝ)
  (second_term_initial_value : ℝ)
  (second_term_final_value : ℝ)
  (s : ℝ)
  (first_term_value : initial_deposit * (1 + first_term_annual_rate / 100 / 12 * first_term_months) = second_term_initial_value)
  (second_term_value : second_term_initial_value * (1 + s / 100 / 12 * first_term_months) = second_term_final_value) :
  s = 11.36 :=
by
  sorry

end second_term_deposit_interest_rate_l114_114736


namespace speed_difference_l114_114695

def anna_time_min := 15
def ben_time_min := 25
def distance_miles := 8

def anna_speed_mph := (distance_miles : ℚ) / (anna_time_min / 60 : ℚ)
def ben_speed_mph := (distance_miles : ℚ) / (ben_time_min / 60 : ℚ)

theorem speed_difference : (anna_speed_mph - ben_speed_mph : ℚ) = 12.8 := by {
  sorry
}

end speed_difference_l114_114695


namespace possible_pairs_copies_each_key_min_drawers_l114_114815

-- Define the number of distinct keys
def num_keys : ℕ := 10

-- Define the function to calculate the number of pairs
def num_pairs (n : ℕ) := n * (n - 1) / 2

-- Theorem for the first question
theorem possible_pairs : num_pairs num_keys = 45 :=
by sorry

-- Define the number of copies needed for each key
def copies_needed (n : ℕ) := n - 1

-- Theorem for the second question
theorem copies_each_key : copies_needed num_keys = 9 :=
by sorry

-- Define the minimum number of drawers Fernando needs to open
def min_drawers_to_open (n : ℕ) := num_pairs n - (n - 1) + 1

-- Theorem for the third question
theorem min_drawers : min_drawers_to_open num_keys = 37 :=
by sorry

end possible_pairs_copies_each_key_min_drawers_l114_114815


namespace problem1_problem2_l114_114077

-- Problem 1
theorem problem1 :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 48 + Real.sqrt 54 = 4 + Real.sqrt 6) :=
by
  sorry

-- Problem 2
theorem problem2 :
  (Real.sqrt 8 + Real.sqrt 32 - Real.sqrt 2 = 5 * Real.sqrt 2) :=
by
  sorry

end problem1_problem2_l114_114077


namespace fred_washing_cars_l114_114517

theorem fred_washing_cars :
  ∀ (initial_amount final_amount money_made : ℕ),
  initial_amount = 23 →
  final_amount = 86 →
  money_made = final_amount - initial_amount →
  money_made = 63 := by
    intros initial_amount final_amount money_made h_initial h_final h_calc
    rw [h_initial, h_final] at h_calc
    exact h_calc

end fred_washing_cars_l114_114517


namespace consecutive_digits_sum_190_to_199_l114_114354

-- Define the digits sum function
def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define ten consecutive numbers starting from m
def ten_consecutive_sum (m : ℕ) : ℕ :=
  (List.range 10).map (λ i => digits_sum (m + i)) |>.sum

theorem consecutive_digits_sum_190_to_199:
  ten_consecutive_sum 190 = 145 :=
by
  sorry

end consecutive_digits_sum_190_to_199_l114_114354


namespace total_canoes_proof_l114_114117

def n_canoes_january : ℕ := 5
def n_canoes_february : ℕ := 3 * n_canoes_january
def n_canoes_march : ℕ := 3 * n_canoes_february
def n_canoes_april : ℕ := 3 * n_canoes_march

def total_canoes_built : ℕ :=
  n_canoes_january + n_canoes_february + n_canoes_march + n_canoes_april

theorem total_canoes_proof : total_canoes_built = 200 := 
  by
  sorry

end total_canoes_proof_l114_114117


namespace man_age_twice_son_age_in_2_years_l114_114772

variable (currentAgeSon : ℕ)
variable (currentAgeMan : ℕ)
variable (Y : ℕ)

-- Given conditions
def sonCurrentAge : Prop := currentAgeSon = 23
def manCurrentAge : Prop := currentAgeMan = currentAgeSon + 25
def manAgeTwiceSonAgeInYYears : Prop := currentAgeMan + Y = 2 * (currentAgeSon + Y)

-- Theorem to prove
theorem man_age_twice_son_age_in_2_years :
  sonCurrentAge currentAgeSon →
  manCurrentAge currentAgeSon currentAgeMan →
  manAgeTwiceSonAgeInYYears currentAgeSon currentAgeMan Y →
  Y = 2 :=
by
  intros h_son_age h_man_age h_age_relation
  sorry

end man_age_twice_son_age_in_2_years_l114_114772


namespace product_of_two_numbers_l114_114196

-- Definitions and conditions
def HCF (a b : ℕ) : ℕ := 9
def LCM (a b : ℕ) : ℕ := 200

-- Theorem statement
theorem product_of_two_numbers (a b : ℕ) (H₁ : HCF a b = 9) (H₂ : LCM a b = 200) : a * b = 1800 :=
by
  -- Injecting HCF and LCM conditions into the problem
  sorry

end product_of_two_numbers_l114_114196


namespace opposite_sides_range_l114_114958

theorem opposite_sides_range (a : ℝ) : (2 * 1 + 3 * a + 1) * (2 * a - 3 * 1 + 1) < 0 ↔ -1 < a ∧ a < 1 := sorry

end opposite_sides_range_l114_114958


namespace inequality_proof_l114_114757

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ 2 * (a^3 + b^3 + c^3) / (a * b * c) + 3 :=
by
  sorry

end inequality_proof_l114_114757


namespace find_xy_plus_yz_plus_xz_l114_114262

theorem find_xy_plus_yz_plus_xz
  (x y z : ℝ)
  (h₁ : x > 0)
  (h₂ : y > 0)
  (h₃ : z > 0)
  (eq1 : x^2 + x * y + y^2 = 75)
  (eq2 : y^2 + y * z + z^2 = 64)
  (eq3 : z^2 + z * x + x^2 = 139) :
  x * y + y * z + z * x = 80 :=
by
  sorry

end find_xy_plus_yz_plus_xz_l114_114262


namespace chocolates_bought_l114_114853

theorem chocolates_bought (cost_price selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : cost_price * 24 = selling_price)
  (h2 : gain_percent = 83.33333333333334)
  (h3 : selling_price = cost_price * 24 * (1 + gain_percent / 100)) :
  cost_price * 44 = selling_price :=
by
  sorry

end chocolates_bought_l114_114853


namespace solution_correct_l114_114317

noncomputable def a := 3 + 3 * Real.sqrt 2
noncomputable def b := 3 - 3 * Real.sqrt 2

theorem solution_correct (h : a ≥ b) : 3 * a + 2 * b = 15 + 3 * Real.sqrt 2 :=
by sorry

end solution_correct_l114_114317


namespace geom_seq_inv_sum_eq_l114_114208

noncomputable def geom_seq (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * r^n

theorem geom_seq_inv_sum_eq
    (a_1 r : ℚ)
    (h_sum : geom_seq a_1 r 0 + geom_seq a_1 r 1 + geom_seq a_1 r 2 + geom_seq a_1 r 3 = 15/8)
    (h_prod : geom_seq a_1 r 1 * geom_seq a_1 r 2 = -9/8) :
  1 / geom_seq a_1 r 0 + 1 / geom_seq a_1 r 1 + 1 / geom_seq a_1 r 2 + 1 / geom_seq a_1 r 3 = -5/3 :=
sorry

end geom_seq_inv_sum_eq_l114_114208


namespace rhombus_perimeter_l114_114466

-- Define the lengths of the diagonals
def d1 : ℝ := 5  -- Length of the first diagonal
def d2 : ℝ := 12 -- Length of the second diagonal

-- Calculate the perimeter and state the theorem
theorem rhombus_perimeter : ((d1 / 2)^2 + (d2 / 2)^2).sqrt * 4 = 26 := by
  -- Sorry is placed here to denote the proof
  sorry

end rhombus_perimeter_l114_114466


namespace stadium_ticket_price_l114_114164

theorem stadium_ticket_price
  (original_price : ℝ)
  (decrease_rate : ℝ)
  (increase_rate : ℝ)
  (new_price : ℝ) 
  (h1 : original_price = 400)
  (h2 : decrease_rate = 0.2)
  (h3 : increase_rate = 0.05) 
  (h4 : (original_price * (1 + increase_rate) / (1 - decrease_rate)) = new_price) :
  new_price = 525 := 
by
  -- Proof omitted for this task.
  sorry

end stadium_ticket_price_l114_114164


namespace goods_train_speed_l114_114705

noncomputable def speed_of_goods_train (train_speed : ℝ) (goods_length : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed_mps := goods_length / passing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  (relative_speed_kmph - train_speed)

theorem goods_train_speed :
  speed_of_goods_train 30 280 9 = 82 :=
by
  sorry

end goods_train_speed_l114_114705


namespace cost_of_one_dozen_pens_is_780_l114_114900

-- Defining the cost of pens and pencils
def cost_of_pens (n : ℕ) := n * 65

def cost_of_pencils (m : ℕ) := m * 13

-- Given conditions
def total_cost (x y : ℕ) := cost_of_pens x + cost_of_pencils y

theorem cost_of_one_dozen_pens_is_780
  (h1 : total_cost 3 5 = 260)
  (h2 : 65 = 5 * 13)
  (h3 : 65 = 65) :
  12 * 65 = 780 := by
    sorry

end cost_of_one_dozen_pens_is_780_l114_114900


namespace perpendicular_bisector_midpoint_l114_114234

theorem perpendicular_bisector_midpoint :
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  3 * R.1 - 2 * R.2 = -15 :=
by
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  sorry

end perpendicular_bisector_midpoint_l114_114234


namespace uv_square_l114_114799

theorem uv_square (u v : ℝ) (h1 : u * (u + v) = 50) (h2 : v * (u + v) = 100) : (u + v)^2 = 150 := by
  sorry

end uv_square_l114_114799


namespace salary_reduction_l114_114616

theorem salary_reduction (S : ℝ) (R : ℝ) 
  (h : (S - (R / 100) * S) * (4 / 3) = S) :
  R = 25 := 
  sorry

end salary_reduction_l114_114616


namespace part1_part2_l114_114645

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part1 (tangent_at_e : ∀ x : ℝ, f x e = 2 * e) : a = e := sorry

theorem part2 (m : ℝ) (a : ℝ) (hm : 0 < m) :
  (if m ≤ 1 / (2 * Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (2 * m) a 
   else if 1 / (2 * Real.exp 1) < m ∧ m < 1 / (Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (1 / (Real.exp 1)) a 
   else 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f m a) :=
  sorry

end part1_part2_l114_114645


namespace octal_to_decimal_equiv_l114_114204

-- Definitions for the octal number 724
def d0 := 4
def d1 := 2
def d2 := 7

-- Definition for the base
def base := 8

-- Calculation of the decimal equivalent
def calc_decimal : ℕ :=
  d0 * base^0 + d1 * base^1 + d2 * base^2

-- The proof statement
theorem octal_to_decimal_equiv : calc_decimal = 468 := by
  sorry

end octal_to_decimal_equiv_l114_114204


namespace ab_plus_cd_111_333_l114_114120

theorem ab_plus_cd_111_333 (a b c d : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a + b + d = 5) 
  (h3 : a + c + d = 20) 
  (h4 : b + c + d = 15) : 
  a * b + c * d = 111.333 := 
by
  sorry

end ab_plus_cd_111_333_l114_114120


namespace chairs_to_remove_l114_114527

-- Defining the conditions
def chairs_per_row : Nat := 15
def total_chairs : Nat := 180
def expected_attendees : Nat := 125

-- Main statement to prove
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ) : 
  chairs_per_row = 15 → 
  total_chairs = 180 → 
  expected_attendees = 125 → 
  ∃ n, total_chairs - (chairs_per_row * n) = 45 ∧ n * chairs_per_row ≥ expected_attendees := 
by
  intros h1 h2 h3
  sorry

end chairs_to_remove_l114_114527


namespace proof_GP_product_l114_114871

namespace GPProof

variables {a r : ℝ} {n : ℕ} (S S' P : ℝ)

def isGeometricProgression (a r : ℝ) (n : ℕ) :=
  ∀ i, 0 ≤ i ∧ i < n → ∃ k, ∃ b, b = (-1)^k * a * r^k ∧ k = i 

noncomputable def product (a r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n*(n-1)/2) * (-1)^(n*(n-1)/2)

noncomputable def sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - (-r)^n) / (1 - (-r))

noncomputable def reciprocalSum (a r : ℝ) (n : ℕ) : ℝ :=
  (1 / a) * (1 - (-1/r)^n) / (1 + 1/r)

theorem proof_GP_product (hyp1 : isGeometricProgression a (-r) n) (hyp2 : S = sum a (-r) n) (hyp3 : S' = reciprocalSum a (-r) n) (hyp4 : P = product a (-r) n) :
  P = (S / S')^(n/2) :=
by
  sorry

end GPProof

end proof_GP_product_l114_114871


namespace jack_jogging_speed_needed_l114_114926

noncomputable def jack_normal_speed : ℝ :=
  let normal_melt_time : ℝ := 10
  let faster_melt_factor : ℝ := 0.75
  let adjusted_melt_time : ℝ := normal_melt_time * faster_melt_factor
  let adjusted_melt_time_hours : ℝ := adjusted_melt_time / 60
  let distance_to_beach : ℝ := 2
  let required_speed : ℝ := distance_to_beach / adjusted_melt_time_hours
  let slope_reduction_factor : ℝ := 0.8
  required_speed / slope_reduction_factor

theorem jack_jogging_speed_needed
  (normal_melt_time : ℝ := 10) 
  (faster_melt_factor : ℝ := 0.75) 
  (distance_to_beach : ℝ := 2) 
  (slope_reduction_factor : ℝ := 0.8) :
  jack_normal_speed = 20 := 
by
  sorry

end jack_jogging_speed_needed_l114_114926


namespace distance_between_parallel_lines_l114_114165

theorem distance_between_parallel_lines
  (l1 : ∀ (x y : ℝ), 2*x + y + 1 = 0)
  (l2 : ∀ (x y : ℝ), 4*x + 2*y - 1 = 0) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 5 / 10 := by
  sorry

end distance_between_parallel_lines_l114_114165


namespace price_of_child_ticket_l114_114990

theorem price_of_child_ticket (C : ℝ) 
  (adult_ticket_price : ℝ := 8) 
  (total_tickets_sold : ℕ := 34) 
  (adult_tickets_sold : ℕ := 12) 
  (total_revenue : ℝ := 236) 
  (h1 : 12 * adult_ticket_price + (34 - 12) * C = total_revenue) :
  C = 6.36 :=
by
  sorry

end price_of_child_ticket_l114_114990


namespace monotonic_increasing_iff_l114_114925

noncomputable def f (x b : ℝ) : ℝ := (x - b) * Real.log x + x^2

theorem monotonic_increasing_iff (b : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → 0 ≤ (Real.log x - b/x + 1 + 2*x)) ↔ b ∈ Set.Iic (3 : ℝ) :=
by
  sorry

end monotonic_increasing_iff_l114_114925


namespace michael_age_multiple_l114_114535

theorem michael_age_multiple (M Y O k : ℤ) (hY : Y = 5) (hO : O = 3 * Y) (h_combined : M + O + Y = 28) (h_relation : O = k * (M - 1) + 1) : k = 2 :=
by
  -- Definitions and given conditions are provided:
  have hY : Y = 5 := hY
  have hO : O = 3 * Y := hO
  have h_combined : M + O + Y = 28 := h_combined
  have h_relation : O = k * (M - 1) + 1 := h_relation
  
  -- Begin the proof by using the provided conditions
  sorry

end michael_age_multiple_l114_114535


namespace smallest_angle_l114_114668

theorem smallest_angle (k : ℝ) (h1 : 4 * k + 5 * k + 7 * k = 180) : 4 * k = 45 :=
by sorry

end smallest_angle_l114_114668


namespace x_zero_sufficient_not_necessary_for_sin_zero_l114_114681

theorem x_zero_sufficient_not_necessary_for_sin_zero :
  (∀ x : ℝ, x = 0 → Real.sin x = 0) ∧ (∃ y : ℝ, Real.sin y = 0 ∧ y ≠ 0) :=
by
  sorry

end x_zero_sufficient_not_necessary_for_sin_zero_l114_114681


namespace find_integers_for_perfect_square_l114_114746

theorem find_integers_for_perfect_square (x : ℤ) :
  (∃ k : ℤ, x * (x + 1) * (x + 7) * (x + 8) = k^2) ↔ 
  x = -9 ∨ x = -8 ∨ x = -7 ∨ x = -4 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end find_integers_for_perfect_square_l114_114746


namespace rational_expression_simplification_l114_114537

theorem rational_expression_simplification
  (a b c : ℚ) 
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( ((a^2 * b^2) / c^2 - (2 / c) + (1 / (a^2 * b^2)) + (2 * a * b) / c^2 - (2 / (a * b * c))) 
      / ((2 / (a * b)) - (2 * a * b) / c) ) 
      / (101 / c) = - (1 / 202) :=
by sorry

end rational_expression_simplification_l114_114537


namespace train_cross_time_approx_l114_114299
noncomputable def time_to_cross_bridge
  (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) : ℝ :=
  ((train_length + bridge_length) / (speed_kmh * 1000 / 3600))

theorem train_cross_time_approx (train_length bridge_length speed_kmh : ℝ)
  (h_train_length : train_length = 250)
  (h_bridge_length : bridge_length = 300)
  (h_speed_kmh : speed_kmh = 44) :
  abs (time_to_cross_bridge train_length bridge_length speed_kmh - 45) < 1 :=
by
  sorry

end train_cross_time_approx_l114_114299


namespace unique_solution_3x_4y_5z_l114_114792

theorem unique_solution_3x_4y_5z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  intro h
  sorry

end unique_solution_3x_4y_5z_l114_114792


namespace all_statements_imply_implication_l114_114805

variables (p q r : Prop)

theorem all_statements_imply_implication :
  (p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (¬ p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (p ∧ ¬ q ∧ ¬ r → ((p → q) → r)) ∧
  (¬ p ∧ q ∧ r → ((p → q) → r)) :=
by { sorry }

end all_statements_imply_implication_l114_114805


namespace find_number_l114_114182

theorem find_number (x : ℕ) (n : ℕ) (h1 : x = 4) (h2 : x + n = 5) : n = 1 :=
by
  sorry

end find_number_l114_114182


namespace max_value_quadratic_l114_114371

theorem max_value_quadratic (r : ℝ) : 
  ∃ M, (∀ r, -3 * r^2 + 36 * r - 9 ≤ M) ∧ M = 99 :=
sorry

end max_value_quadratic_l114_114371


namespace polynomial_divisibility_l114_114179

theorem polynomial_divisibility : 
  ∃ k : ℤ, (k = 8) ∧ (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x - 2) = 0) ∧ 
           (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x^2 + 1) = 0) :=
sorry

end polynomial_divisibility_l114_114179


namespace inequality_holds_l114_114314

theorem inequality_holds (x a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (a < c ∧ c < b) ∨ (b < c ∧ c < a)) 
  (h5 : (x - a) * (x - b) * (x - c) > 0) :
  (1 / (x - a)) + (1 / (x - b)) > 1 / (x - c) := 
by sorry

end inequality_holds_l114_114314


namespace sum_of_eight_numbers_l114_114414

theorem sum_of_eight_numbers (nums : List ℝ) (h_len : nums.length = 8) (h_avg : (nums.sum / 8) = 5.5) : nums.sum = 44 :=
by
  sorry

end sum_of_eight_numbers_l114_114414


namespace time_to_pass_platform_is_correct_l114_114673

noncomputable def train_length : ℝ := 250 -- meters
noncomputable def time_to_pass_pole : ℝ := 10 -- seconds
noncomputable def time_to_pass_platform : ℝ := 60 -- seconds

-- Speed of the train
noncomputable def train_speed := train_length / time_to_pass_pole -- meters/second

-- Length of the platform
noncomputable def platform_length := train_speed * time_to_pass_platform - train_length -- meters

-- Proving the time to pass the platform is 50 seconds
theorem time_to_pass_platform_is_correct : 
  (platform_length / train_speed) = 50 :=
by
  sorry

end time_to_pass_platform_is_correct_l114_114673


namespace geometric_progression_condition_l114_114632

theorem geometric_progression_condition
  (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0)
  (a_seq : ℕ → ℝ) 
  (h_def : ∀ n, a_seq (n+2) = k * a_seq n * a_seq (n+1)) :
  (a_seq 1 = a ∧ a_seq 2 = b) ↔ a_seq 1 = a_seq 2 :=
by
  sorry

end geometric_progression_condition_l114_114632


namespace chocolates_brought_by_friend_l114_114915

-- Definitions corresponding to the conditions in a)
def total_chocolates := 50
def chocolates_not_in_box := 5
def number_of_boxes := 3
def additional_boxes := 2

-- Theorem statement: we need to prove the number of chocolates her friend brought
theorem chocolates_brought_by_friend (C : ℕ) : 
  (C + total_chocolates = total_chocolates + (chocolates_not_in_box + number_of_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes + additional_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes) - total_chocolates) 
  → C = 30 := 
sorry

end chocolates_brought_by_friend_l114_114915


namespace find_x1_l114_114593

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 :=
  sorry

end find_x1_l114_114593


namespace exp_division_rule_l114_114697

-- The theorem to prove the given problem
theorem exp_division_rule (x : ℝ) (hx : x ≠ 0) :
  x^10 / x^5 = x^5 :=
by sorry

end exp_division_rule_l114_114697


namespace quadratic_has_real_roots_l114_114123

theorem quadratic_has_real_roots (k : ℝ) :
  (∃ (x : ℝ), (k-2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ (3 / 2) ∧ k ≠ 2) :=
by
  sorry

end quadratic_has_real_roots_l114_114123


namespace product_of_binomials_l114_114942

theorem product_of_binomials (x : ℝ) : 
  (4 * x - 3) * (2 * x + 7) = 8 * x^2 + 22 * x - 21 := by
  sorry

end product_of_binomials_l114_114942


namespace fill_bathtub_time_l114_114512

theorem fill_bathtub_time
  (r_cold : ℚ := 1/10)
  (r_hot : ℚ := 1/15)
  (r_empty : ℚ := -1/12)
  (net_rate : ℚ := r_cold + r_hot + r_empty) :
  net_rate = 1/12 → 
  t = 12 :=
by
  sorry

end fill_bathtub_time_l114_114512


namespace triangle_isosceles_l114_114076

theorem triangle_isosceles
  (A B C : ℝ) -- Angles of the triangle, A, B, and C
  (h1 : A = 2 * C) -- Condition 1: Angle A equals twice angle C
  (h2 : B = 2 * C) -- Condition 2: Angle B equals twice angle C
  (h3 : A + B + C = 180) -- Sum of angles in a triangle equals 180 degrees
  : A = B := -- Conclusion: with the conditions above, angles A and B are equal
by
  sorry

end triangle_isosceles_l114_114076


namespace surface_area_circumscribed_sphere_l114_114621

-- Define the problem
theorem surface_area_circumscribed_sphere (a b c : ℝ)
    (h1 : a^2 + b^2 = 3)
    (h2 : b^2 + c^2 = 5)
    (h3 : c^2 + a^2 = 4) : 
    4 * Real.pi * (a^2 + b^2 + c^2) / 4 = 6 * Real.pi :=
by
  -- The proof is omitted
  sorry

end surface_area_circumscribed_sphere_l114_114621


namespace Richard_remaining_distance_l114_114980

theorem Richard_remaining_distance
  (total_distance : ℕ)
  (day1_distance : ℕ)
  (day2_distance : ℕ)
  (day3_distance : ℕ)
  (half_and_subtract : day2_distance = (day1_distance / 2) - 6)
  (total_distance_to_walk : total_distance = 70)
  (distance_day1 : day1_distance = 20)
  (distance_day3 : day3_distance = 10)
  : total_distance - (day1_distance + day2_distance + day3_distance) = 36 :=
  sorry

end Richard_remaining_distance_l114_114980


namespace solve_for_x_l114_114907

theorem solve_for_x (x : ℝ) (h : |3990 * x + 1995| = 1995) : x = 0 ∨ x = -1 :=
by
  sorry

end solve_for_x_l114_114907


namespace find_jessica_almonds_l114_114809

-- Definitions for j (Jessica's almonds) and l (Louise's almonds)
variables (j l : ℕ)
-- Conditions
def condition1 : Prop := l = j - 8
def condition2 : Prop := l = j / 3

theorem find_jessica_almonds (h1 : condition1 j l) (h2 : condition2 j l) : j = 12 :=
by sorry

end find_jessica_almonds_l114_114809


namespace constant_value_l114_114995

noncomputable def find_constant (p q : ℚ) (h : p / q = 4 / 5) : ℚ :=
    let C := 0.5714285714285714 - (2 * q - p) / (2 * q + p)
    C

theorem constant_value (p q : ℚ) (h : p / q = 4 / 5) :
    find_constant p q h = 0.14285714285714285 := by
    sorry

end constant_value_l114_114995


namespace surface_area_of_cube_l114_114984

-- Define the volume condition
def volume_of_cube (s : ℝ) := s^3 = 125

-- Define the conversion from decimeters to centimeters
def decimeters_to_centimeters (d : ℝ) := d * 10

-- Define the surface area formula for one side of the cube
def surface_area_one_side (s_cm : ℝ) := s_cm^2

-- Prove that given the volume condition, the surface area of one side is 2500 cm²
theorem surface_area_of_cube
  (s : ℝ)
  (h : volume_of_cube s)
  (s_cm : ℝ := decimeters_to_centimeters s) :
  surface_area_one_side s_cm = 2500 :=
by
  sorry

end surface_area_of_cube_l114_114984


namespace number_of_cows_l114_114857

-- Definitions
variables (c h : ℕ)

-- Conditions
def condition1 : Prop := 4 * c + 2 * h = 20 + 2 * (c + h)
def condition2 : Prop := c + h = 12

-- Theorem
theorem number_of_cows : condition1 c h → condition2 c h → c = 10 :=
  by 
  intros h1 h2
  sorry

end number_of_cows_l114_114857


namespace cherry_orange_punch_ratio_l114_114481

theorem cherry_orange_punch_ratio 
  (C : ℝ)
  (h_condition1 : 4.5 + C + (C - 1.5) = 21) : 
  C / 4.5 = 2 :=
by
  sorry

end cherry_orange_punch_ratio_l114_114481


namespace calories_per_strawberry_l114_114661

theorem calories_per_strawberry (x : ℕ) :
  (12 * x + 6 * 17 = 150) → x = 4 := by
  sorry

end calories_per_strawberry_l114_114661


namespace triangle_area_ABC_l114_114910

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (7, 6)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the triangle with the given vertices is 15
theorem triangle_area_ABC : triangle_area A B C = 15 :=
by
  -- Proof goes here
  sorry

end triangle_area_ABC_l114_114910


namespace factor_quadratic_polynomial_l114_114462

theorem factor_quadratic_polynomial :
  (∀ x : ℝ, x^4 - 36*x^2 + 25 = (x^2 - 6*x + 5) * (x^2 + 6*x + 5)) :=
by
  sorry

end factor_quadratic_polynomial_l114_114462


namespace range_of_xy_l114_114577

theorem range_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y + x * y = 30) :
  12 < x * y ∧ x * y < 870 :=
by sorry

end range_of_xy_l114_114577


namespace average_in_all_6_subjects_l114_114953

-- Definitions of the conditions
def average_in_5_subjects : ℝ := 74
def marks_in_6th_subject : ℝ := 104
def num_subjects_total : ℝ := 6

-- Proof that the average in all 6 subjects is 79
theorem average_in_all_6_subjects :
  (average_in_5_subjects * 5 + marks_in_6th_subject) / num_subjects_total = 79 := by
  sorry

end average_in_all_6_subjects_l114_114953


namespace find_a_b_minimum_value_l114_114789

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2

/-- Given the function y = f(x) = ax^3 + bx^2, when x = 1, it has a maximum value of 3 -/
def condition1 (a b : ℝ) : Prop :=
  f 1 a b = 3 ∧ (3 * a + 2 * b = 0)

/-- Find the values of the real numbers a and b -/
theorem find_a_b : ∃ (a b : ℝ), condition1 a b :=
sorry

/-- Find the minimum value of the function -/
theorem minimum_value : ∀ (a b : ℝ), condition1 a b → (∃ x_min, ∀ x, f x a b ≥ f x_min a b) :=
sorry

end find_a_b_minimum_value_l114_114789


namespace base9_addition_correct_l114_114707

-- Definition of base 9 addition problem.
def add_base9 (a b c : ℕ) : ℕ :=
  let sum := a + b + c -- Sum in base 10
  let d0 := sum % 9 -- Least significant digit in base 9
  let carry1 := sum / 9
  (carry1 + carry1 / 9 * 9 + carry1 % 9) + d0 -- Sum in base 9 considering carry

-- The specific values converted to base 9 integers
def n1 := 3 * 9^2 + 4 * 9 + 6
def n2 := 8 * 9^2 + 0 * 9 + 2
def n3 := 1 * 9^2 + 5 * 9 + 7

-- The expected result converted to base 9 integer
def expected_sum := 1 * 9^3 + 4 * 9^2 + 1 * 9 + 6

theorem base9_addition_correct : add_base9 n1 n2 n3 = expected_sum := by
  -- Proof will be provided here
  sorry

end base9_addition_correct_l114_114707


namespace motorboat_speeds_l114_114976

theorem motorboat_speeds (v a x : ℝ) (d : ℝ)
  (h1 : ∀ t1 t2 t1' t2', 
        t1 = d / (v - a) ∧ t1' = d / (v + x - a) ∧ 
        t2 = d / (v + a) ∧ t2' = d / (v + a - x) ∧ 
        (t1 - t1' = t2' - t2)) 
        : x = 2 * a := 
sorry

end motorboat_speeds_l114_114976


namespace election_votes_l114_114908

theorem election_votes (V : ℝ) (h1 : 0.56 * V - 0.44 * V = 288) : 0.56 * V = 1344 :=
by 
  sorry

end election_votes_l114_114908


namespace evaluate_expression_l114_114411

theorem evaluate_expression : (20 + 22) / 2 = 21 := by
  sorry

end evaluate_expression_l114_114411


namespace bob_hair_length_l114_114739

-- Define the current length of Bob's hair
def current_length : ℝ := 36

-- Define the growth rate in inches per month
def growth_rate : ℝ := 0.5

-- Define the duration in years
def duration_years : ℕ := 5

-- Define the total growth over the duration in years
def total_growth : ℝ := growth_rate * 12 * duration_years

-- Define the length of Bob's hair when he last cut it
def initial_length : ℝ := current_length - total_growth

-- Theorem stating that the length of Bob's hair when he last cut it was 6 inches
theorem bob_hair_length :
  initial_length = 6 :=
by
  -- Proof omitted
  sorry

end bob_hair_length_l114_114739


namespace combined_selling_price_correct_l114_114032

def ArticleA_Cost : ℝ := 500
def ArticleA_Profit_Percent : ℝ := 0.45
def ArticleB_Cost : ℝ := 300
def ArticleB_Profit_Percent : ℝ := 0.30
def ArticleC_Cost : ℝ := 1000
def ArticleC_Profit_Percent : ℝ := 0.20
def Sales_Tax_Percent : ℝ := 0.12

def CombinedSellingPrice (A_cost A_profit_percent B_cost B_profit_percent C_cost C_profit_percent tax_percent : ℝ) : ℝ :=
  let A_selling_price := A_cost * (1 + A_profit_percent)
  let A_final_price := A_selling_price * (1 + tax_percent)
  let B_selling_price := B_cost * (1 + B_profit_percent)
  let B_final_price := B_selling_price * (1 + tax_percent)
  let C_selling_price := C_cost * (1 + C_profit_percent)
  let C_final_price := C_selling_price * (1 + tax_percent)
  A_final_price + B_final_price + C_final_price

theorem combined_selling_price_correct :
  CombinedSellingPrice ArticleA_Cost ArticleA_Profit_Percent ArticleB_Cost ArticleB_Profit_Percent ArticleC_Cost ArticleC_Profit_Percent Sales_Tax_Percent = 2592.8 := by
  sorry

end combined_selling_price_correct_l114_114032


namespace find_original_number_l114_114741

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 9) = 57) : x = 5 := by
  sorry

end find_original_number_l114_114741


namespace sum_geometric_arithmetic_progression_l114_114301

theorem sum_geometric_arithmetic_progression :
  ∃ (a b r d : ℝ), a = 1 * r ∧ b = 1 * r^2 ∧ b = a + d ∧ 16 = b + d ∧ (a + b = 12.64) :=
by
  sorry

end sum_geometric_arithmetic_progression_l114_114301


namespace dan_money_left_l114_114477

def money_left (initial : ℝ) (candy_bar : ℝ) (chocolate : ℝ) (soda : ℝ) (gum : ℝ) : ℝ :=
  initial - candy_bar - chocolate - soda - gum

theorem dan_money_left :
  money_left 10 2 3 1.5 1.25 = 2.25 :=
by
  sorry

end dan_money_left_l114_114477


namespace insects_per_group_correct_l114_114692

-- Define the numbers of insects collected by boys and girls
def boys_insects : ℕ := 200
def girls_insects : ℕ := 300
def total_insects : ℕ := boys_insects + girls_insects

-- Define the number of groups
def groups : ℕ := 4

-- Define the expected number of insects per group using total insects and groups
def insects_per_group : ℕ := total_insects / groups

-- Prove that each group gets 125 insects
theorem insects_per_group_correct : insects_per_group = 125 :=
by
  -- The proof is omitted (just setting up the theorem statement)
  sorry

end insects_per_group_correct_l114_114692


namespace variance_le_second_moment_l114_114282

noncomputable def variance (X : ℝ → ℝ) (MX : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - MX)^2]

noncomputable def second_moment (X : ℝ → ℝ) (C : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - C)^2]

theorem variance_le_second_moment (X : ℝ → ℝ) :
  ∀ C : ℝ, C ≠ MX → variance X MX ≤ second_moment X C := 
by
  sorry

end variance_le_second_moment_l114_114282


namespace ferris_wheel_seats_l114_114104

theorem ferris_wheel_seats (S : ℕ) (h1 : ∀ (p : ℕ), p = 9) (h2 : ∀ (r : ℕ), r = 18) (h3 : 9 * S = 18) : S = 2 :=
by
  sorry

end ferris_wheel_seats_l114_114104


namespace smallest_positive_period_pi_interval_extrema_l114_114255

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos (x + Real.pi / 3) + Real.sqrt 3

theorem smallest_positive_period_pi : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem interval_extrema :
  ∃ x_max x_min : ℝ, 
  -Real.pi / 4 ≤ x_max ∧ x_max ≤ Real.pi / 6 ∧ f x_max = 2 ∧
  -Real.pi / 4 ≤ x_min ∧ x_min ≤ Real.pi / 6 ∧ f x_min = -1 ∧ 
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 2 ∧ f x ≥ -1) :=
sorry

end smallest_positive_period_pi_interval_extrema_l114_114255


namespace initial_men_count_l114_114173

-- Definitions based on problem conditions
def initial_days : ℝ := 18
def extra_men : ℝ := 400
def final_days : ℝ := 12.86

-- Proposition to show the initial number of men based on conditions
theorem initial_men_count (M : ℝ) (h : M * initial_days = (M + extra_men) * final_days) : M = 1000 := by
  sorry

end initial_men_count_l114_114173


namespace jack_needs_5_rocks_to_equal_weights_l114_114581

-- Given Conditions
def WeightJack : ℕ := 60
def WeightAnna : ℕ := 40
def WeightRock : ℕ := 4

-- Theorem Statement
theorem jack_needs_5_rocks_to_equal_weights : (WeightJack - WeightAnna) / WeightRock = 5 :=
by
  sorry

end jack_needs_5_rocks_to_equal_weights_l114_114581


namespace find_number_90_l114_114446

theorem find_number_90 {x y : ℝ} (h1 : x = y + 0.11 * y) (h2 : x = 99.9) : y = 90 :=
sorry

end find_number_90_l114_114446


namespace compute_product_l114_114630

theorem compute_product : (100 - 5) * (100 + 5) = 9975 := by
  sorry

end compute_product_l114_114630


namespace survivor_quitting_probability_l114_114485

noncomputable def probability_all_quitters_same_tribe : ℚ :=
  let total_contestants := 20
  let tribe_size := 10
  let total_quitters := 3
  let total_ways := (Nat.choose total_contestants total_quitters)
  let tribe_quitters_ways := (Nat.choose tribe_size total_quitters)
  (tribe_quitters_ways + tribe_quitters_ways) / total_ways

theorem survivor_quitting_probability :
  probability_all_quitters_same_tribe = 4 / 19 :=
by
  sorry

end survivor_quitting_probability_l114_114485


namespace parallel_vectors_x_value_l114_114315

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

theorem parallel_vectors_x_value (x : ℝ) :
  ∀ k : ℝ,
  k ≠ 0 ∧ k * 1 = -2 ∧ k * -2 = x →
  x = 4 :=
by
  intros k hk
  have hk1 : k * 1 = -2 := hk.2.1
  have hk2 : k * -2 = x := hk.2.2
  -- Proceed from here to the calculations according to the steps in b):
  sorry

end parallel_vectors_x_value_l114_114315


namespace roots_of_polynomial_l114_114188

theorem roots_of_polynomial :
  (∀ x : ℝ, (x^2 - 5 * x + 6) * x * (x - 5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5) :=
by
  sorry

end roots_of_polynomial_l114_114188


namespace derivative_log_base2_l114_114300

noncomputable def log_base2 (x : ℝ) := Real.log x / Real.log 2

theorem derivative_log_base2 (x : ℝ) (h : x > 0) : 
  deriv (fun x => log_base2 x) x = 1 / (x * Real.log 2) :=
by
  sorry

end derivative_log_base2_l114_114300


namespace exists_b_mod_5_l114_114352

theorem exists_b_mod_5 (p q r s : ℤ) (h1 : ¬ (s % 5 = 0)) (a : ℤ) (h2 : (p * a^3 + q * a^2 + r * a + s) % 5 = 0) : 
  ∃ b : ℤ, (s * b^3 + r * b^2 + q * b + p) % 5 = 0 :=
sorry

end exists_b_mod_5_l114_114352


namespace michael_large_balls_l114_114716

theorem michael_large_balls (total_rubber_bands : ℕ) (small_ball_rubber_bands : ℕ) (large_ball_rubber_bands : ℕ) (small_balls_made : ℕ)
  (h_total_rubber_bands : total_rubber_bands = 5000)
  (h_small_ball_rubber_bands : small_ball_rubber_bands = 50)
  (h_large_ball_rubber_bands : large_ball_rubber_bands = 300)
  (h_small_balls_made : small_balls_made = 22) :
  (total_rubber_bands - small_balls_made * small_ball_rubber_bands) / large_ball_rubber_bands = 13 :=
by {
  sorry
}

end michael_large_balls_l114_114716


namespace simplify_sqrt_expression_l114_114454

theorem simplify_sqrt_expression (h : Real.sqrt 3 > 1) :
  Real.sqrt ((1 - Real.sqrt 3) ^ 2) = Real.sqrt 3 - 1 :=
by
  sorry

end simplify_sqrt_expression_l114_114454


namespace perfect_square_trinomial_l114_114052

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, (x : ℝ) → (x^2 + 2 * (m - 1) * x + 16) = (a * x + b)^2) → (m = 5 ∨ m = -3) :=
by
  sorry

end perfect_square_trinomial_l114_114052


namespace no_divisor_30_to_40_of_2_pow_28_minus_1_l114_114162

theorem no_divisor_30_to_40_of_2_pow_28_minus_1 :
  ¬ ∃ n : ℕ, (30 ≤ n ∧ n ≤ 40 ∧ n ∣ (2^28 - 1)) :=
by
  sorry

end no_divisor_30_to_40_of_2_pow_28_minus_1_l114_114162


namespace solution_set_of_inequality_l114_114049

theorem solution_set_of_inequality (x : ℝ) : (x^2 - 2*x - 5 > 2*x) ↔ (x > 5 ∨ x < -1) :=
by sorry

end solution_set_of_inequality_l114_114049


namespace total_bottles_needed_l114_114254

-- Definitions from conditions
def large_bottle_capacity : ℕ := 450
def small_bottle_capacity : ℕ := 45
def extra_large_bottle_capacity : ℕ := 900

-- Theorem statement
theorem total_bottles_needed :
  ∃ (num_large_bottles num_small_bottles : ℕ), 
    num_large_bottles * large_bottle_capacity + num_small_bottles * small_bottle_capacity = extra_large_bottle_capacity ∧ 
    num_large_bottles + num_small_bottles = 2 :=
by
  sorry

end total_bottles_needed_l114_114254


namespace tangent_line_through_M_to_circle_l114_114400

noncomputable def M : ℝ × ℝ := (2, -1)
noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem tangent_line_through_M_to_circle :
  ∀ {x y : ℝ}, circle_eq x y → M = (2, -1) → 2*x - y - 5 = 0 :=
sorry

end tangent_line_through_M_to_circle_l114_114400


namespace sqrt_sqrt_16_l114_114385

theorem sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := sorry

end sqrt_sqrt_16_l114_114385


namespace largest_c_for_range_l114_114482

noncomputable def g (x c : ℝ) : ℝ := x^2 - 6*x + c

theorem largest_c_for_range (c : ℝ) : (∃ x : ℝ, g x c = 2) ↔ c ≤ 11 := 
sorry

end largest_c_for_range_l114_114482


namespace yearly_profit_l114_114176

variable (num_subletters : ℕ) (rent_per_subletter_per_month rent_per_month : ℕ)

theorem yearly_profit (h1 : num_subletters = 3)
                     (h2 : rent_per_subletter_per_month = 400)
                     (h3 : rent_per_month = 900) :
  12 * (num_subletters * rent_per_subletter_per_month - rent_per_month) = 3600 :=
by
  sorry

end yearly_profit_l114_114176


namespace go_piece_arrangement_l114_114750

theorem go_piece_arrangement (w b : ℕ) (pieces : List ℕ) 
    (h_w : w = 180) (h_b : b = 181)
    (h_pieces : pieces.length = w + b) 
    (h_black_count : pieces.count 1 = b) 
    (h_white_count : pieces.count 0 = w) :
    ∃ (i j : ℕ), i < j ∧ j < pieces.length ∧ 
    ((j - i - 1 = 178) ∨ (j - i - 1 = 181)) ∧ 
    (pieces.get ⟨i, sorry⟩ = 1) ∧ 
    (pieces.get ⟨j, sorry⟩ = 1) := 
sorry

end go_piece_arrangement_l114_114750


namespace purely_imaginary_x_value_l114_114432

theorem purely_imaginary_x_value (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : x + 1 ≠ 0) : x = 1 :=
by
  sorry

end purely_imaginary_x_value_l114_114432


namespace find_coordinates_B_l114_114393

variable (B : ℝ × ℝ)

def A : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (0, 1)
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

theorem find_coordinates_B (h : vec A B = (-2) • vec B C) : B = (-2, 5/3) :=
by
  -- Here you would provide proof steps
  sorry

end find_coordinates_B_l114_114393


namespace triangle_area_divided_l114_114213

theorem triangle_area_divided {baseA heightA baseB heightB : ℝ} 
  (h1 : baseA = 1) 
  (h2 : heightA = 1)
  (h3 : baseB = 2)
  (h4 : heightB = 1)
  : (1 / 2 * baseA * heightA + 1 / 2 * baseB * heightB = 1.5) :=
by
  sorry

end triangle_area_divided_l114_114213


namespace ratio_owners_on_horse_l114_114399

-- Definitions based on the given conditions.
def number_of_horses : Nat := 12
def number_of_owners : Nat := 12
def total_legs_walking_on_ground : Nat := 60
def owner_leg_count : Nat := 2
def horse_leg_count : Nat := 4
def total_owners_leg_horse_count : Nat := owner_leg_count + horse_leg_count

-- Prove the ratio of the number of owners on their horses' back to the total number of owners is 1:6
theorem ratio_owners_on_horse (R W : Nat) 
  (h1 : R + W = number_of_owners)
  (h2 : total_owners_leg_horse_count * W = total_legs_walking_on_ground) :
  R = 2 → W = 10 → (R : Nat)/(number_of_owners : Nat) = (1 : Nat)/(6 : Nat) := 
sorry

end ratio_owners_on_horse_l114_114399


namespace triangle_proof_l114_114922

theorem triangle_proof (a b : ℝ) (cosA : ℝ) (ha : a = 6) (hb : b = 5) (hcosA : cosA = -4 / 5) :
  (∃ B : ℝ, B = 30) ∧ (∃ area : ℝ, area = (9 * Real.sqrt 3 - 12) / 2) :=
  by
  sorry

end triangle_proof_l114_114922


namespace czakler_inequality_czakler_equality_pairs_l114_114098

theorem czakler_inequality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (xy - 10)^2 ≥ 64 :=
sorry

theorem czakler_equality_pairs (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
(xy - 10)^2 = 64 ↔ (x, y) = (1,2) ∨ (x, y) = (-3, -6) :=
sorry

end czakler_inequality_czakler_equality_pairs_l114_114098


namespace carol_optimal_strategy_l114_114818

-- Definitions of the random variables
def uniform_A (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def uniform_B (b : ℝ) : Prop := 0.25 ≤ b ∧ b ≤ 0.75
def winning_condition (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Carol's optimal strategy stated as a theorem
theorem carol_optimal_strategy : ∀ (a b c : ℝ), 
  uniform_A a → uniform_B b → (c = 7 / 12) → 
  winning_condition a b c → 
  ∀ (c' : ℝ), uniform_A c' → c' ≠ c → ¬(winning_condition a b c') :=
by
  sorry

end carol_optimal_strategy_l114_114818


namespace monotonic_intervals_inequality_condition_l114_114435

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x

theorem monotonic_intervals (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x < y → f x m < f y m) ∧
  (m > 0 → (∀ x > 0, x < 1/m → ∀ y > x, y < 1/m → f x m < f y m) ∧ (∀ x ≥ 1/m, ∀ y > x, f x m > f y m)) :=
sorry

theorem inequality_condition (m : ℝ) (h : ∀ x ≥ 1, f x m ≤ (m - 1) / x - 2 * m + 1) :
  m ≥ 1/2 :=
sorry

end monotonic_intervals_inequality_condition_l114_114435


namespace subset_A_if_inter_eq_l114_114311

variable {B : Set ℝ}

def A : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem subset_A_if_inter_eq:
  A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = { x | 0 < x ∧ x < 2 } :=
by
  sorry

end subset_A_if_inter_eq_l114_114311


namespace mean_proportional_49_64_l114_114094

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end mean_proportional_49_64_l114_114094


namespace sum_of_roots_of_f_l114_114710

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = - f x

noncomputable def f_increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y

theorem sum_of_roots_of_f (f : ℝ → ℝ) (m : ℝ) (x1 x2 x3 x4 : ℝ)
  (h1 : odd_function f)
  (h2 : ∀ x, f (x - 4) = - f x)
  (h3 : f_increasing_on f 0 2)
  (h4 : m > 0)
  (h5 : f x1 = m)
  (h6 : f x2 = m)
  (h7 : f x3 = m)
  (h8 : f x4 = m)
  (h9 : x1 ≠ x2)
  (h10 : x1 ≠ x3)
  (h11 : x1 ≠ x4)
  (h12 : x2 ≠ x3)
  (h13 : x2 ≠ x4)
  (h14 : x3 ≠ x4)
  (h15 : ∀ x, -8 ≤ x ∧ x ≤ 8 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end sum_of_roots_of_f_l114_114710


namespace large_cube_side_length_l114_114167

theorem large_cube_side_length (s1 s2 s3 : ℝ) (h1 : s1 = 1) (h2 : s2 = 6) (h3 : s3 = 8) : 
  ∃ s_large : ℝ, s_large^3 = s1^3 + s2^3 + s3^3 ∧ s_large = 9 := 
by 
  use 9
  rw [h1, h2, h3]
  norm_num

end large_cube_side_length_l114_114167


namespace boat_travel_time_difference_l114_114155

noncomputable def travel_time_difference (v : ℝ) : ℝ :=
  let d := 90
  let t_downstream := 2.5191640969412834
  let t_upstream := d / (v - 3)
  t_upstream - t_downstream

theorem boat_travel_time_difference :
  ∃ v : ℝ, travel_time_difference v = 0.5088359030587166 := 
by
  sorry

end boat_travel_time_difference_l114_114155


namespace num_frisbees_more_than_deck_cards_l114_114325

variables (M F D x : ℕ)
variable (bought_fraction : ℝ)

theorem num_frisbees_more_than_deck_cards :
  M = 60 ∧ M = 2 * F ∧ F = D + x ∧
  M + bought_fraction * M + F + bought_fraction * F + D + bought_fraction * D = 140 ∧ bought_fraction = 2/5 →
  x = 20 :=
by
  sorry

end num_frisbees_more_than_deck_cards_l114_114325


namespace relationship_between_p_and_q_l114_114343

variable {a b : ℝ}

theorem relationship_between_p_and_q 
  (h_a : a > 2) 
  (h_p : p = a + 1 / (a - 2)) 
  (h_q : q = -b^2 - 2 * b + 3) : 
  p ≥ q := 
sorry

end relationship_between_p_and_q_l114_114343


namespace meena_cookies_left_l114_114500

def dozen : ℕ := 12

def baked_cookies : ℕ := 5 * dozen
def mr_stone_buys : ℕ := 2 * dozen
def brock_buys : ℕ := 7
def katy_buys : ℕ := 2 * brock_buys
def total_sold : ℕ := mr_stone_buys + brock_buys + katy_buys
def cookies_left : ℕ := baked_cookies - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end meena_cookies_left_l114_114500


namespace bcm_hens_count_l114_114430

-- Propositions representing the given conditions
def total_chickens : ℕ := 100
def bcm_ratio : ℝ := 0.20
def bcm_hens_ratio : ℝ := 0.80

-- Theorem statement: proving the number of BCM hens
theorem bcm_hens_count : (total_chickens * bcm_ratio * bcm_hens_ratio = 16) := by
  sorry

end bcm_hens_count_l114_114430


namespace maximum_sum_of_numbers_in_grid_l114_114601

theorem maximum_sum_of_numbers_in_grid :
  ∀ (grid : List (List ℕ)) (rect_cover : (ℕ × ℕ) → (ℕ × ℕ) → Prop),
  (∀ x y, rect_cover x y → x ≠ y → x.1 < 6 → x.2 < 6 → y.1 < 6 → y.2 < 6) →
  (∀ x y z w, rect_cover x y ∧ rect_cover z w → 
    (x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∨ (x.1 = z.1 ∨ x.2 = z.2) → 
    (x.1 = z.1 ∧ x.2 = y.2 ∨ x.2 = z.2 ∧ x.1 = y.1)) → False) →
  (36 = 6 * 6) →
  18 = 36 / 2 →
  342 = (18 * 19) :=
by
  intro grid rect_cover h_grid h_no_common_edge h_grid_size h_num_rectangles
  sorry

end maximum_sum_of_numbers_in_grid_l114_114601


namespace problem_inequality_l114_114478

theorem problem_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^6 - a^2 + 4) * (b^6 - b^2 + 4) * (c^6 - c^2 + 4) * (d^6 - d^2 + 4) ≥ (a + b + c + d)^4 :=
by
  sorry

end problem_inequality_l114_114478


namespace polygon_sides_eq_13_l114_114766

theorem polygon_sides_eq_13 (n : ℕ) (h : n * (n - 3) = 5 * n) : n = 13 := by
  sorry

end polygon_sides_eq_13_l114_114766


namespace average_marks_combined_l114_114205

theorem average_marks_combined (avg1 : ℝ) (students1 : ℕ) (avg2 : ℝ) (students2 : ℕ) :
  avg1 = 30 → students1 = 30 → avg2 = 60 → students2 = 50 →
  (students1 * avg1 + students2 * avg2) / (students1 + students2) = 48.75 := 
by
  intros h_avg1 h_students1 h_avg2 h_students2
  sorry

end average_marks_combined_l114_114205


namespace somu_fathers_age_ratio_l114_114330

noncomputable def somus_age := 16

def proof_problem (S F : ℕ) : Prop :=
  S = 16 ∧ 
  (S - 8 = (1 / 5) * (F - 8)) ∧
  (S / F = 1 / 3)

theorem somu_fathers_age_ratio (S F : ℕ) : proof_problem S F :=
by
  sorry

end somu_fathers_age_ratio_l114_114330


namespace students_exceed_goldfish_l114_114292

theorem students_exceed_goldfish 
    (num_classrooms : ℕ) 
    (students_per_classroom : ℕ) 
    (goldfish_per_classroom : ℕ) 
    (h1 : num_classrooms = 5) 
    (h2 : students_per_classroom = 20) 
    (h3 : goldfish_per_classroom = 3) 
    : (students_per_classroom * num_classrooms) - (goldfish_per_classroom * num_classrooms) = 85 := by
  sorry

end students_exceed_goldfish_l114_114292


namespace possible_values_a1_l114_114169

def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem possible_values_a1 {a : ℕ → ℤ} (h1 : ∀ n : ℕ, a n + a (n + 1) = 2 * n - 1)
  (h2 : ∃ k : ℕ, sequence_sum a k = 190 ∧ sequence_sum a (k + 1) = 190) :
  (a 0 = -20 ∨ a 0 = 19) :=
sorry

end possible_values_a1_l114_114169


namespace gym_class_students_l114_114647

theorem gym_class_students :
  ∃ n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 6 = 3 ∧ n % 8 = 5 ∧ n % 9 = 2 ∧ (n = 165 ∨ n = 237) :=
by
  sorry

end gym_class_students_l114_114647


namespace score_recording_l114_114624

theorem score_recording (avg : ℤ) (h : avg = 0) : 
  (9 = avg + 9) ∧ (-18 = avg - 18) ∧ (-2 = avg - 2) :=
by
  -- Proof steps go here
  sorry

end score_recording_l114_114624


namespace binary_calculation_l114_114991

theorem binary_calculation :
  let b1 := 0b110110
  let b2 := 0b101110
  let b3 := 0b100
  let expected_result := 0b11100011110
  ((b1 * b2) / b3) = expected_result := by
  sorry

end binary_calculation_l114_114991


namespace min_value_expression_geq_twosqrt3_l114_114844

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  (1/(x-1)) + (3/(y-1))

theorem min_value_expression_geq_twosqrt3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1/x) + (1/y) = 1) : 
  min_value_expression x y >= 2 * Real.sqrt 3 :=
by
  sorry

end min_value_expression_geq_twosqrt3_l114_114844


namespace unit_digit_14_pow_100_l114_114480

theorem unit_digit_14_pow_100 : (14 ^ 100) % 10 = 6 :=
by
  sorry

end unit_digit_14_pow_100_l114_114480


namespace expression_equals_33_l114_114545

noncomputable def calculate_expression : ℚ :=
  let part1 := 25 * 52
  let part2 := 46 * 15
  let diff := part1 - part2
  (2013 / diff) * 10

theorem expression_equals_33 : calculate_expression = 33 := sorry

end expression_equals_33_l114_114545


namespace smallest_k_equals_26_l114_114982

open Real

-- Define the condition
def cos_squared_eq_one (θ : ℝ) : Prop :=
  cos θ ^ 2 = 1

-- Define the requirement for θ to be in the form 180°n
def theta_condition (n : ℤ) : Prop :=
  ∃ (k : ℤ), k ^ 2 + k + 81 = 180 * n

-- The problem statement in Lean: Find the smallest positive integer k such that
-- cos squared of (k^2 + k + 81) degrees = 1
noncomputable def smallest_k_satisfying_cos (k : ℤ) : Prop :=
  (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (k ^ 2 + k + 81)) ∧ (∀ m : ℤ, m > 0 ∧ m < k → 
   (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (m ^ 2 + m + 81)) → false)

theorem smallest_k_equals_26 : smallest_k_satisfying_cos 26 := 
  sorry

end smallest_k_equals_26_l114_114982


namespace total_cost_correct_l114_114483

/-- Define the base car rental cost -/
def rental_cost : ℝ := 150

/-- Define cost per mile -/
def cost_per_mile : ℝ := 0.5

/-- Define miles driven on Monday -/
def miles_monday : ℝ := 620

/-- Define miles driven on Thursday -/
def miles_thursday : ℝ := 744

/-- Define the total cost Zach spent -/
def total_cost : ℝ := rental_cost + (miles_monday * cost_per_mile) + (miles_thursday * cost_per_mile)

/-- Prove that the total cost Zach spent is 832 dollars -/
theorem total_cost_correct : total_cost = 832 := by
  sorry

end total_cost_correct_l114_114483


namespace inequality_proof_l114_114051

theorem inequality_proof (a b : ℝ) (h1 : a > 1) (h2 : b > 1) :
    (a^2 / (b - 1)) + (b^2 / (a - 1)) ≥ 8 := 
by
  sorry

end inequality_proof_l114_114051


namespace range_of_a_h_diff_l114_114868

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

theorem range_of_a (a : ℝ) (h : a < 0) : (∀ x, 0 < x ∧ x < Real.log 3 → 
  (a * x - 1) / x < 0 ∧ Real.exp x + a ≠ 0 ∧ (a ≤ -3)) :=
sorry

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + Real.log x

theorem h_diff (a : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < 1/2) : 
    x1 * x2 = 1/2 ∧ h a x1 - h a x2 > 3/4 - Real.log 2 :=
sorry

end range_of_a_h_diff_l114_114868


namespace small_order_peanuts_l114_114328

theorem small_order_peanuts (total_peanuts : ℕ) (large_orders : ℕ) (peanuts_per_large : ℕ) 
    (small_orders : ℕ) (peanuts_per_small : ℕ) : 
    total_peanuts = large_orders * peanuts_per_large + small_orders * peanuts_per_small → 
    total_peanuts = 800 → 
    large_orders = 3 → 
    peanuts_per_large = 200 → 
    small_orders = 4 → 
    peanuts_per_small = 50 := by
  intros h1 h2 h3 h4 h5
  sorry

end small_order_peanuts_l114_114328


namespace smallest_positive_value_floor_l114_114025

noncomputable def g (x : ℝ) : ℝ := Real.cos x - Real.sin x + 4 * Real.tan x

theorem smallest_positive_value_floor :
  ∃ s > 0, g s = 0 ∧ ⌊s⌋ = 3 :=
sorry

end smallest_positive_value_floor_l114_114025


namespace students_selected_from_grade_10_l114_114278

theorem students_selected_from_grade_10 (students_grade10 students_grade11 students_grade12 total_selected : ℕ)
  (h_grade10 : students_grade10 = 1200)
  (h_grade11 : students_grade11 = 1000)
  (h_grade12 : students_grade12 = 800)
  (h_total_selected : total_selected = 100) :
  students_grade10 * total_selected = 40 * (students_grade10 + students_grade11 + students_grade12) :=
by
  sorry

end students_selected_from_grade_10_l114_114278


namespace width_decreased_by_28_6_percent_l114_114070

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l114_114070


namespace negation_of_statement_6_l114_114751

variable (Teenager Adult : Type)
variable (CanCookWell : Teenager → Prop)
variable (CanCookWell' : Adult → Prop)

-- Conditions from the problem
def all_teenagers_can_cook_well : Prop :=
  ∀ t : Teenager, CanCookWell t

def some_teenagers_can_cook_well : Prop :=
  ∃ t : Teenager, CanCookWell t

def no_adults_can_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def all_adults_cannot_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def at_least_one_adult_cannot_cook_well : Prop :=
  ∃ a : Adult, ¬CanCookWell' a

def all_adults_can_cook_well : Prop :=
  ∀ a : Adult, CanCookWell' a

-- Theorem to prove
theorem negation_of_statement_6 :
  at_least_one_adult_cannot_cook_well Adult CanCookWell' = ¬ all_adults_can_cook_well Adult CanCookWell' :=
sorry

end negation_of_statement_6_l114_114751


namespace prism_faces_l114_114342

theorem prism_faces (E V F n : ℕ) (h1 : E + V = 30) (h2 : F + V = E + 2) (h3 : E = 3 * n) : F = 8 :=
by
  -- Actual proof omitted
  sorry

end prism_faces_l114_114342


namespace cycle_selling_price_l114_114403

theorem cycle_selling_price (initial_price : ℝ)
  (first_discount_percent : ℝ) (second_discount_percent : ℝ) (third_discount_percent : ℝ)
  (first_discounted_price : ℝ) (second_discounted_price : ℝ) :
  initial_price = 3600 →
  first_discount_percent = 15 →
  second_discount_percent = 10 →
  third_discount_percent = 5 →
  first_discounted_price = initial_price * (1 - first_discount_percent / 100) →
  second_discounted_price = first_discounted_price * (1 - second_discount_percent / 100) →
  final_price = second_discounted_price * (1 - third_discount_percent / 100) →
  final_price = 2616.30 :=
by
  intros
  sorry

end cycle_selling_price_l114_114403


namespace onion_to_carrot_ratio_l114_114068

theorem onion_to_carrot_ratio (p c o g : ℕ) (h1 : 6 * p = c) (h2 : c = o) (h3 : g = 1 / 3 * o) (h4 : p = 2) (h5 : g = 8) : o / c = 1 / 1 :=
by
  sorry

end onion_to_carrot_ratio_l114_114068


namespace regression_decrease_by_three_l114_114969

-- Given a regression equation \hat y = 2 - 3 \hat x
def regression_equation (x : ℝ) : ℝ :=
  2 - 3 * x

-- Prove that when x increases by one unit, \hat y decreases by 3 units
theorem regression_decrease_by_three (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -3 :=
by
  -- proof
  sorry

end regression_decrease_by_three_l114_114969


namespace divide_19_degree_angle_into_19_equal_parts_l114_114758

/-- Divide a 19° angle into 19 equal parts, resulting in each part being 1° -/
theorem divide_19_degree_angle_into_19_equal_parts
  (α : ℝ) (hα : α = 19) :
  α / 19 = 1 :=
by
  sorry

end divide_19_degree_angle_into_19_equal_parts_l114_114758


namespace circumference_of_jogging_track_l114_114248

-- Definitions for the given conditions
def speed_deepak : ℝ := 4.5
def speed_wife : ℝ := 3.75
def meet_time : ℝ := 4.32

-- The theorem stating the problem
theorem circumference_of_jogging_track : 
  (speed_deepak + speed_wife) * meet_time = 35.64 :=
by
  sorry

end circumference_of_jogging_track_l114_114248


namespace pentagon_perimeter_l114_114765

-- Define the side length and number of sides for a regular pentagon
def side_length : ℝ := 5
def num_sides : ℕ := 5

-- Define the perimeter calculation as a constant
def perimeter (side_length : ℝ) (num_sides : ℕ) : ℝ := side_length * num_sides

theorem pentagon_perimeter : perimeter side_length num_sides = 25 := by
  sorry

end pentagon_perimeter_l114_114765


namespace pos_int_solutions_3x_2y_841_l114_114507

theorem pos_int_solutions_3x_2y_841 :
  {n : ℕ // ∃ (x y : ℕ), 3 * x + 2 * y = 841 ∧ x > 0 ∧ y > 0} =
  {n : ℕ // n = 140} := 
sorry

end pos_int_solutions_3x_2y_841_l114_114507


namespace points_on_ellipse_l114_114919

noncomputable def x (t : ℝ) : ℝ := (3 - t^2) / (1 + t^2)
noncomputable def y (t : ℝ) : ℝ := 4 * t / (1 + t^2)

theorem points_on_ellipse : ∀ t : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (x t / a)^2 + (y t / b)^2 = 1 := 
sorry

end points_on_ellipse_l114_114919


namespace no_common_points_l114_114022

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x
noncomputable def g (a b x : ℝ) : ℝ := b + a * Real.log (x - 1)
noncomputable def h (a x : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)
noncomputable def G (a : ℝ) : ℝ := -a^2 / 4 + 1 - a * Real.log (a / 2)

theorem no_common_points (a b : ℝ) (h1 : 1 ≤ a) :
  (∀ x > 1, f a x ≠ g a b x) ↔ b < 3 / 4 + Real.log 2 :=
by
  sorry

end no_common_points_l114_114022


namespace factor_expression_l114_114737

variables (b : ℝ)

theorem factor_expression :
  (8 * b ^ 3 + 45 * b ^ 2 - 10) - (-12 * b ^ 3 + 5 * b ^ 2 - 10) = 20 * b ^ 2 * (b + 2) :=
by
  sorry

end factor_expression_l114_114737


namespace correct_actual_profit_l114_114834

def profit_miscalculation (calculated_profit actual_profit : ℕ) : Prop :=
  let err1 := 5 * 100  -- Error due to mistaking 3 for 8 in the hundreds place
  let err2 := 3 * 10   -- Error due to mistaking 8 for 5 in the tens place
  actual_profit = calculated_profit - err1 + err2

theorem correct_actual_profit : profit_miscalculation 1320 850 :=
by
  sorry

end correct_actual_profit_l114_114834


namespace henry_geography_math_score_l114_114040

variable (G M : ℕ)

theorem henry_geography_math_score (E : ℕ) (H : ℕ) (total_score : ℕ) 
  (hE : E = 66) 
  (hH : H = (G + M + E) / 3)
  (hTotal : G + M + E + H = total_score) 
  (htotal_score : total_score = 248) :
  G + M = 120 := 
by
  sorry

end henry_geography_math_score_l114_114040


namespace jose_speed_l114_114115

theorem jose_speed
  (distance : ℕ) (time : ℕ)
  (h_distance : distance = 4)
  (h_time : time = 2) :
  distance / time = 2 := by
  sorry

end jose_speed_l114_114115


namespace linear_equation_a_neg2_l114_114671

theorem linear_equation_a_neg2 (a : ℝ) :
  (∃ x y : ℝ, (a - 2) * x ^ (|a| - 1) + 3 * y = 1) ∧
  (∀ x : ℝ, x ≠ 0 → x ^ (|a| - 1) ≠ 1) →
  a = -2 :=
by
  sorry

end linear_equation_a_neg2_l114_114671


namespace sum_of_first_33_terms_arith_seq_l114_114201

noncomputable def sum_arith_prog (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_33_terms_arith_seq :
  ∃ (a_1 d : ℝ), (4 * a_1 + 64 * d = 28) → (sum_arith_prog a_1 d 33 = 231) :=
by
  sorry

end sum_of_first_33_terms_arith_seq_l114_114201


namespace variable_v_value_l114_114655

theorem variable_v_value (w x v : ℝ) (h1 : 2 / w + 2 / x = 2 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) :
  v = 0.25 :=
sorry

end variable_v_value_l114_114655


namespace units_sold_at_original_price_l114_114977

-- Define the necessary parameters and assumptions
variables (a x y : ℝ)
variables (total_units sold_original sold_discount sold_offseason : ℝ)
variables (purchase_price sell_price discount_price clearance_price : ℝ)

-- Define specific conditions
def purchase_units := total_units = 1000
def selling_price := sell_price = 1.25 * a
def discount_cond := discount_price = 1.25 * 0.9 * a
def clearance_cond := clearance_price = 1.25 * 0.60 * a
def holiday_limit := y ≤ 100
def profitability_condition := 1.25 * x + 1.25 * 0.9 * y + 1.25 * 0.60 * (1000 - x - y) > 1000 * a

-- The theorem asserting at least 426 units sold at the original price ensures profitability
theorem units_sold_at_original_price (h1 : total_units = 1000)
  (h2 : sell_price = 1.25 * a) (h3 : discount_price = 1.25 * 0.9 * a)
  (h4 : clearance_price = 1.25 * 0.60 * a) (h5 : y ≤ 100)
  (h6 : 1.25 * x + 1.25 * 0.9 * y + 1.25 * 0.60 * (1000 - x - y) > 1000 * a) :
  x ≥ 426 :=
by
  sorry

end units_sold_at_original_price_l114_114977


namespace solve_system_l114_114573

theorem solve_system :
  ∃ (x y z : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y - 4 * z = 10 ∧ 2 * x + y + 3 * z = 1.25 :=
by
  sorry

end solve_system_l114_114573


namespace must_divide_a_l114_114626

-- Definitions of positive integers and their gcd conditions
variables {a b c d : ℕ}

-- The conditions given in the problem
axiom h1 : gcd a b = 24
axiom h2 : gcd b c = 36
axiom h3 : gcd c d = 54
axiom h4 : 70 < gcd d a ∧ gcd d a < 100

-- We need to prove that 13 divides a
theorem must_divide_a : 13 ∣ a :=
by sorry

end must_divide_a_l114_114626


namespace symmetric_point_product_l114_114079

theorem symmetric_point_product (x y : ℤ) (h1 : (2008, y) = (-x, -1)) : x * y = -2008 :=
by {
  sorry
}

end symmetric_point_product_l114_114079


namespace jerome_classmates_count_l114_114194

theorem jerome_classmates_count (C F : ℕ) (h1 : F = C / 2) (h2 : 33 = C + F + 3) : C = 20 :=
by
  sorry

end jerome_classmates_count_l114_114194


namespace price_per_can_of_spam_l114_114875

-- Definitions of conditions
variable (S : ℝ) -- The price per can of Spam
def cost_peanut_butter := 3 * 5 -- 3 jars of peanut butter at $5 each
def cost_bread := 4 * 2 -- 4 loaves of bread at $2 each
def total_cost := 59 -- Total amount paid

-- Proof problem to verify the price per can of Spam
theorem price_per_can_of_spam :
  12 * S + cost_peanut_butter + cost_bread = total_cost → S = 3 :=
by
  sorry

end price_per_can_of_spam_l114_114875


namespace defective_units_shipped_percentage_l114_114591

theorem defective_units_shipped_percentage :
  let units_produced := 100
  let typeA_defective := 0.07 * units_produced
  let typeB_defective := 0.08 * units_produced
  let typeA_shipped := 0.03 * typeA_defective
  let typeB_shipped := 0.06 * typeB_defective
  let total_shipped := typeA_shipped + typeB_shipped
  let percentage_shipped := total_shipped / units_produced * 100
  percentage_shipped = 1 :=
by
  sorry

end defective_units_shipped_percentage_l114_114591


namespace apples_per_sandwich_l114_114767

-- Define the conditions
def sam_sandwiches_per_day : Nat := 10
def days_in_week : Nat := 7
def total_apples_in_week : Nat := 280

-- Calculate total sandwiches in a week
def total_sandwiches_in_week := sam_sandwiches_per_day * days_in_week

-- Prove that Sam eats 4 apples for each sandwich
theorem apples_per_sandwich : total_apples_in_week / total_sandwiches_in_week = 4 :=
  by
    sorry

end apples_per_sandwich_l114_114767


namespace part1_part2_l114_114186

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Given conditions
axiom a_3a_5 : a 3 * a 5 = 63
axiom a_2a_6 : a 2 + a 6 = 16

-- Part (1) Proving the general formula
theorem part1 : 
  (∀ n : ℕ, a n = 12 - n) :=
sorry

-- Part (2) Proving the maximum value of S_n
theorem part2 :
  (∃ n : ℕ, (S n = (n * (12 - (n - 1) / 2)) → (n = 11 ∨ n = 12) ∧ (S n = 66))) :=
sorry

end part1_part2_l114_114186


namespace triangle_side_b_range_l114_114947

noncomputable def sin60 := Real.sin (Real.pi / 3)

theorem triangle_side_b_range (a b : ℝ) (A : ℝ)
  (ha : a = 2)
  (hA : A = 60 * Real.pi / 180)
  (h_2solutions : b * sin60 < a ∧ a < b) :
  (2 < b ∧ b < 4 * Real.sqrt 3 / 3) :=
by
  sorry

end triangle_side_b_range_l114_114947


namespace largest_unreachable_integer_l114_114891

theorem largest_unreachable_integer : ∃ n : ℕ, (¬ ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = n)
  ∧ ∀ m : ℕ, m > n → (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = m) := sorry

end largest_unreachable_integer_l114_114891


namespace part_a_part_b_part_c_part_d_l114_114768

theorem part_a : (4237 * 27925 ≠ 118275855) :=
by sorry

theorem part_b : (42971064 / 8264 ≠ 5201) :=
by sorry

theorem part_c : (1965^2 ≠ 3761225) :=
by sorry

theorem part_d : (23 ^ 5 ≠ 371293) :=
by sorry

end part_a_part_b_part_c_part_d_l114_114768


namespace smallest_value_proof_l114_114216

noncomputable def smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) : Prop :=
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/x^2

theorem smallest_value_proof (x : ℝ) (h : 0 < x ∧ x < 1) : smallest_value x h :=
  sorry

end smallest_value_proof_l114_114216


namespace different_lists_count_l114_114519

def numberOfLists : Nat := 5

theorem different_lists_count :
  let conditions := ∃ (d : Fin 6 → ℕ), d 0 + d 1 + d 2 + d 3 + d 4 + d 5 = 5 ∧
                                      ∀ i, d i ≤ 5 ∧
                                      ∀ i j, i < j → d i ≥ d j
  conditions →
  numberOfLists = 5 :=
sorry

end different_lists_count_l114_114519


namespace fraction_denominator_l114_114613

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end fraction_denominator_l114_114613


namespace coffee_tea_overlap_l114_114456

theorem coffee_tea_overlap (c t : ℕ) (h_c : c = 80) (h_t : t = 70) : 
  ∃ (b : ℕ), b = 50 := 
by 
  sorry

end coffee_tea_overlap_l114_114456


namespace turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l114_114109

-- Definitions based on conditions
def turnover_first_four_days : ℝ := 450
def turnover_fifth_day : ℝ := 0.12 * turnover_first_four_days
def total_turnover_five_days : ℝ := turnover_first_four_days + turnover_fifth_day

-- Proof statement for part 1
theorem turnover_five_days_eq_504 :
  total_turnover_five_days = 504 := 
sorry

-- Definitions and conditions for part 2
def turnover_february : ℝ := 350
def turnover_april : ℝ := total_turnover_five_days
def growth_rate (x : ℝ) : Prop := (1 + x)^2 * turnover_february = turnover_april

-- Proof statement for part 2
theorem monthly_growth_rate_eq_20_percent :
  ∃ x : ℝ, growth_rate x ∧ x = 0.2 := 
sorry

end turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l114_114109


namespace anna_more_candy_than_billy_l114_114102

theorem anna_more_candy_than_billy :
  let anna_candy_per_house := 14
  let billy_candy_per_house := 11
  let anna_houses := 60
  let billy_houses := 75
  let anna_total_candy := anna_candy_per_house * anna_houses
  let billy_total_candy := billy_candy_per_house * billy_houses
  anna_total_candy - billy_total_candy = 15 :=
by
  sorry

end anna_more_candy_than_billy_l114_114102


namespace breadth_of_rectangular_plot_l114_114119

theorem breadth_of_rectangular_plot (b l A : ℝ) (h1 : l = 3 * b) (h2 : A = 588) (h3 : A = l * b) : b = 14 :=
by
  -- We start our proof here
  sorry

end breadth_of_rectangular_plot_l114_114119


namespace number_of_possible_k_values_l114_114963

theorem number_of_possible_k_values : 
  ∃ k_values : Finset ℤ, 
    (∀ k ∈ k_values, ∃ (x y : ℤ), y = x - 3 ∧ y = k * x - k) ∧
    k_values.card = 3 := 
sorry

end number_of_possible_k_values_l114_114963


namespace taxi_ride_distance_l114_114368

variable (t : ℝ) (c₀ : ℝ) (cᵢ : ℝ)

theorem taxi_ride_distance (h_t : t = 18.6) (h_c₀ : c₀ = 3.0) (h_cᵢ : cᵢ = 0.4) : 
  ∃ d : ℝ, d = 8 := 
by 
  sorry

end taxi_ride_distance_l114_114368


namespace ratio_cher_to_gab_l114_114401

-- Definitions based on conditions
def sammy_score : ℕ := 20
def gab_score : ℕ := 2 * sammy_score
def opponent_score : ℕ := 85
def total_points : ℕ := opponent_score + 55
def cher_score : ℕ := total_points - (sammy_score + gab_score)

-- Theorem to prove the ratio of Cher's score to Gab's score
theorem ratio_cher_to_gab : cher_score / gab_score = 2 := by
  sorry

end ratio_cher_to_gab_l114_114401


namespace find_abc_l114_114566

theorem find_abc :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 30 ∧
  (1/a + 1/b + 1/c + 450/(a*b*c) = 1) ∧ 
  a*b*c = 1912 :=
sorry

end find_abc_l114_114566


namespace problem1_problem2_l114_114986

-- Lean statement for Problem 1
theorem problem1 (x : ℝ) : x^2 * x^3 - x^5 = 0 := 
by sorry

-- Lean statement for Problem 2
theorem problem2 (a : ℝ) : (a + 1)^2 + 2 * a * (a - 1) = 3 * a^2 + 1 :=
by sorry

end problem1_problem2_l114_114986


namespace hash_7_2_eq_24_l114_114313

def hash_op (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem hash_7_2_eq_24 : hash_op 7 2 = 24 := by
  sorry

end hash_7_2_eq_24_l114_114313


namespace triangle_tangent_ratio_l114_114597

variable {A B C a b c : ℝ}

theorem triangle_tangent_ratio 
  (h : a * Real.cos B - b * Real.cos A = (3 / 5) * c)
  : Real.tan A / Real.tan B = 4 :=
sorry

end triangle_tangent_ratio_l114_114597


namespace simplify_polynomial_l114_114721

theorem simplify_polynomial (x : ℝ) : 
  (2 * x^5 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (3 * x^4 + 2 * x^3 - 4 * x^2 + 3 * x - 7) = 
  2 * x^5 + 3 * x^4 - x^3 + x^2 - 5 * x + 8 :=
by sorry

end simplify_polynomial_l114_114721


namespace homework_done_l114_114592

theorem homework_done :
  ∃ (D E C Z M : Prop),
    -- Statements of students
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    -- Truth-telling condition
    ((D → D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (E → ¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (C → ¬ D ∧ ¬ E ∧ C ∧ ¬ Z ∧ ¬ M) ∧
    (Z → ¬ D ∧ ¬ E ∧ ¬ C ∧ Z ∧ ¬ M) ∧
    (M → ¬ D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ M)) ∧
    -- Number of students who did their homework condition
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) := 
sorry

end homework_done_l114_114592


namespace solution_set_of_inequality_l114_114476

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem solution_set_of_inequality :
  { x : ℝ | f (x - 2) + f (x^2 - 4) < 0 } = Set.Ioo (-3 : ℝ) 2 :=
by
  sorry

end solution_set_of_inequality_l114_114476


namespace count_integers_between_cubes_l114_114018

noncomputable def a := (10.1)^3
noncomputable def b := (10.4)^3

theorem count_integers_between_cubes : 
  ∃ (count : ℕ), count = 94 ∧ (1030.031 < a) ∧ (a < b) ∧ (b < 1124.864) := 
  sorry

end count_integers_between_cubes_l114_114018


namespace backpack_original_price_l114_114298

-- Define original price of a ring-binder
def original_ring_binder_price : ℕ := 20

-- Define the number of ring-binders bought
def number_of_ring_binders : ℕ := 3

-- Define the new price increase for the backpack
def backpack_price_increase : ℕ := 5

-- Define the new price decrease for the ring-binder
def ring_binder_price_decrease : ℕ := 2

-- Define the total amount spent
def total_amount_spent : ℕ := 109

-- Define the original price of the backpack variable
variable (B : ℕ)

-- Theorem statement: under these conditions, the original price of the backpack must be 50
theorem backpack_original_price :
  (B + backpack_price_increase) + ((original_ring_binder_price - ring_binder_price_decrease) * number_of_ring_binders) = total_amount_spent ↔ B = 50 :=
by 
  sorry

end backpack_original_price_l114_114298


namespace find_base_side_length_l114_114226

-- Regular triangular pyramid properties and derived values
variables
  (a l h : ℝ) -- side length of the base, slant height, and height of the pyramid
  (V : ℝ) -- volume of the pyramid

-- Given conditions
def inclined_to_base_plane_at_angle (angle : ℝ) := angle = 45
def volume_of_pyramid (V : ℝ) := V = 18

-- Prove the side length of the base
theorem find_base_side_length
  (h_eq : h = a * Real.sqrt 3 / 3)
  (volume_eq : V = 1 / 3 * (a * a * Real.sqrt 3 / 4) * h)
  (volume_given : V = 18) :
  a = 6 := by
  sorry

end find_base_side_length_l114_114226


namespace savings_after_increase_l114_114674

/-- A man saves 20% of his monthly salary. If on account of dearness of things
    he is to increase his monthly expenses by 20%, he is only able to save a
    certain amount per month. His monthly salary is Rs. 6250. -/
theorem savings_after_increase (monthly_salary : ℝ) (initial_savings_percentage : ℝ)
  (increase_expenses_percentage : ℝ) (final_savings : ℝ) :
  monthly_salary = 6250 ∧
  initial_savings_percentage = 0.20 ∧
  increase_expenses_percentage = 0.20 →
  final_savings = 250 :=
by
  sorry

end savings_after_increase_l114_114674


namespace cost_of_12_roll_package_is_correct_l114_114373

variable (cost_per_roll_package : ℝ)
variable (individual_cost_per_roll : ℝ := 1)
variable (number_of_rolls : ℕ := 12)
variable (percent_savings : ℝ := 0.25)

-- The definition of the total cost of the package
def total_cost_package := number_of_rolls * (individual_cost_per_roll - (percent_savings * individual_cost_per_roll))

-- The goal is to prove that the total cost of the package is $9
theorem cost_of_12_roll_package_is_correct : total_cost_package = 9 := 
by
  sorry

end cost_of_12_roll_package_is_correct_l114_114373


namespace largest_x_exists_largest_x_largest_real_number_l114_114845

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l114_114845


namespace a_lt_c_lt_b_l114_114405

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.sqrt 2 * Real.sin (30.5 * Real.pi / 180) * Real.cos (30.5 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem a_lt_c_lt_b : a < c ∧ c < b := by
  sorry

end a_lt_c_lt_b_l114_114405


namespace sum_of_three_consecutive_integers_l114_114468

theorem sum_of_three_consecutive_integers (n m l : ℕ) (h1 : n + 1 = m) (h2 : m + 1 = l) (h3 : l = 13) : n + m + l = 36 := 
by sorry

end sum_of_three_consecutive_integers_l114_114468


namespace total_oil_leak_l114_114231

-- Definitions for the given conditions
def before_repair_leak : ℕ := 6522
def during_repair_leak : ℕ := 5165
def total_leak : ℕ := 11687

-- The proof statement (without proof, only the statement)
theorem total_oil_leak :
  before_repair_leak + during_repair_leak = total_leak :=
sorry

end total_oil_leak_l114_114231


namespace find_a_5_l114_114531

theorem find_a_5 (a : ℕ → ℤ) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1)
  (h₂ : a 2 + a 4 + a 6 = 18) : a 5 = 5 := 
sorry

end find_a_5_l114_114531


namespace total_weight_correct_l114_114877

-- Define the constant variables as per the conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def fill_percentage : ℝ := 0.7
def jug1_density : ℝ := 4
def jug2_density : ℝ := 5

-- Define the volumes of sand in each jug
def jug1_sand_volume : ℝ := fill_percentage * jug1_capacity
def jug2_sand_volume : ℝ := fill_percentage * jug2_capacity

-- Define the weights of sand in each jug
def jug1_weight : ℝ := jug1_sand_volume * jug1_density
def jug2_weight : ℝ := jug2_sand_volume * jug2_density

-- State the theorem that combines the weights
theorem total_weight_correct : jug1_weight + jug2_weight = 16.1 := sorry

end total_weight_correct_l114_114877


namespace tan_value_of_point_on_graph_l114_114567

theorem tan_value_of_point_on_graph (a : ℝ) (h : (4 : ℝ) ^ (1/2) = a) : 
  Real.tan ((a / 6) * Real.pi) = Real.sqrt 3 :=
by 
  sorry

end tan_value_of_point_on_graph_l114_114567


namespace supplementary_angle_l114_114324

theorem supplementary_angle (θ : ℝ) (k : ℤ) : (θ = 10) → (∃ k, θ + 250 = k * 360 + 360) :=
by
  sorry

end supplementary_angle_l114_114324


namespace determine_number_on_reverse_side_l114_114447

variable (n : ℕ) (k : ℕ) (shown_cards : ℕ → Prop)

theorem determine_number_on_reverse_side :
    -- Conditions
    (∀ i, 1 ≤ i ∧ i ≤ n → (shown_cards (i - 1) ↔ shown_cards i)) →
    -- Prove
    (k = 0 ∨ k = n ∨ (1 ≤ k ∧ k < n ∧ (shown_cards (k - 1) ∨ shown_cards (k + 1)))) →
    (∃ j, (j = 1 ∧ k = 0) ∨ (j = n - 1 ∧ k = n) ∨ 
          (j = k - 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k + 1)) ∨ 
          (j = k + 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k - 1))) :=
by
  sorry

end determine_number_on_reverse_side_l114_114447


namespace right_triangle_area_l114_114806

theorem right_triangle_area (x y : ℝ) 
  (h1 : x + y = 4) 
  (h2 : x^2 + y^2 = 9) : 
  (1/2) * x * y = 7 / 4 := 
by
  sorry

end right_triangle_area_l114_114806


namespace optionC_is_correct_l114_114409

def KalobsWindowLength : ℕ := 50
def KalobsWindowWidth : ℕ := 80
def KalobsWindowArea : ℕ := KalobsWindowLength * KalobsWindowWidth

def DoubleKalobsWindowArea : ℕ := 2 * KalobsWindowArea

def optionC_Length : ℕ := 50
def optionC_Width : ℕ := 160
def optionC_Area : ℕ := optionC_Length * optionC_Width

theorem optionC_is_correct : optionC_Area = DoubleKalobsWindowArea := by
  sorry

end optionC_is_correct_l114_114409


namespace range_of_a_l114_114413

open Set

theorem range_of_a (a : ℝ) :
  (M : Set ℝ) = { x | -1 ≤ x ∧ x ≤ 2 } →
  (N : Set ℝ) = { x | 1 - 3 * a < x ∧ x ≤ 2 * a } →
  M ∩ N = M →
  1 ≤ a :=
by
  intro hM hN h_inter
  sorry

end range_of_a_l114_114413


namespace reflected_ray_eqn_l114_114658

theorem reflected_ray_eqn : 
  ∃ a b c : ℝ, (∀ x y : ℝ, 2 * x - y + 5 = 0 → (a * x + b * y + c = 0)) → -- Condition for the line
  (∀ x y : ℝ, x = 1 ∧ y = 3 → (a * x + b * y + c = 0)) → -- Condition for point (1, 3)
  (a = 1 ∧ b = -5 ∧ c = 14) := -- Assertion about the line equation
by
  sorry

end reflected_ray_eqn_l114_114658


namespace option_d_correct_l114_114240

theorem option_d_correct (a b c : ℝ) (h : a > b ∧ b > c ∧ c > 0) : a / b < a / c :=
by
  sorry

end option_d_correct_l114_114240


namespace min_inquiries_for_parity_l114_114558

-- Define the variables and predicates
variables (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n)

-- Define the main theorem we need to prove
theorem min_inquiries_for_parity (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n) : 
  ∃ k, (k = m + n - 4) := 
sorry

end min_inquiries_for_parity_l114_114558


namespace percentage_increase_l114_114717

theorem percentage_increase
  (black_and_white_cost color_cost : ℕ)
  (h_bw : black_and_white_cost = 160)
  (h_color : color_cost = 240) :
  ((color_cost - black_and_white_cost) * 100) / black_and_white_cost = 50 :=
by
  sorry

end percentage_increase_l114_114717


namespace pizza_consumption_order_l114_114636

noncomputable def amount_eaten (fraction: ℚ) (total: ℚ) := fraction * total

theorem pizza_consumption_order :
  let total := 1
  let samuel := (1 / 6 : ℚ)
  let teresa := (2 / 5 : ℚ)
  let uma := (1 / 4 : ℚ)
  let victor := total - (samuel + teresa + uma)
  let samuel_eaten := amount_eaten samuel 60
  let teresa_eaten := amount_eaten teresa 60
  let uma_eaten := amount_eaten uma 60
  let victor_eaten := amount_eaten victor 60
  (teresa_eaten > uma_eaten) 
  ∧ (uma_eaten > victor_eaten) 
  ∧ (victor_eaten > samuel_eaten) := 
by
  sorry

end pizza_consumption_order_l114_114636


namespace children_ticket_price_difference_l114_114783

noncomputable def regular_ticket_price : ℝ := 9
noncomputable def total_amount_given : ℝ := 2 * 20
noncomputable def total_change_received : ℝ := 1
noncomputable def num_adults : ℕ := 2
noncomputable def num_children : ℕ := 3
noncomputable def total_cost_of_tickets : ℝ := total_amount_given - total_change_received
noncomputable def children_ticket_cost := (total_cost_of_tickets - num_adults * regular_ticket_price) / num_children

theorem children_ticket_price_difference :
  (regular_ticket_price - children_ticket_cost) = 2 := by
  sorry

end children_ticket_price_difference_l114_114783


namespace no_solutions_l114_114800

theorem no_solutions (N : ℕ) (d : ℕ) (H : ∀ (i j : ℕ), i ≠ j → d = 6 ∧ d + d = 13) : false :=
by
  sorry

end no_solutions_l114_114800


namespace ratio_volume_surface_area_l114_114631

noncomputable def volume : ℕ := 10
noncomputable def surface_area : ℕ := 45

theorem ratio_volume_surface_area : volume / surface_area = 2 / 9 := by
  sorry

end ratio_volume_surface_area_l114_114631


namespace sum_is_five_or_negative_five_l114_114100

theorem sum_is_five_or_negative_five (a b c d : ℤ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) 
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b * c * d = 14) : 
  (a + b + c + d = 5) ∨ (a + b + c + d = -5) :=
by
  sorry

end sum_is_five_or_negative_five_l114_114100


namespace find_x_plus_y_l114_114701

theorem find_x_plus_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : |x - y| = y - x) :
  (x + y = -1) ∨ (x + y = -5) :=
sorry

end find_x_plus_y_l114_114701


namespace expected_pairs_correct_l114_114861

-- Define the total number of cards in the deck.
def total_cards : ℕ := 52

-- Define the number of black cards in the deck.
def black_cards : ℕ := 26

-- Define the number of red cards in the deck.
def red_cards : ℕ := 26

-- Define the expected number of pairs of adjacent cards such that one is black and the other is red.
def expected_adjacent_pairs := 52 * (26 / 51)

-- Prove that the expected_adjacent_pairs is equal to 1352 / 51.
theorem expected_pairs_correct : expected_adjacent_pairs = 1352 / 51 := 
by
  have expected_adjacent_pairs_simplified : 52 * (26 / 51) = (1352 / 51) := 
    by sorry
  exact expected_adjacent_pairs_simplified

end expected_pairs_correct_l114_114861


namespace find_f_neg1_plus_f_7_l114_114458

-- Given a function f : ℝ → ℝ
axiom f : ℝ → ℝ

-- f satisfies the property of an even function
axiom even_f : ∀ x : ℝ, f (-x) = f x

-- f satisfies the periodicity of period 2
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x

-- Also, we are given that f(1) = 1
axiom f_one : f 1 = 1

-- We need to prove that f(-1) + f(7) = 2
theorem find_f_neg1_plus_f_7 : f (-1) + f 7 = 2 :=
by
  sorry

end find_f_neg1_plus_f_7_l114_114458


namespace value_of_expression_l114_114078

theorem value_of_expression (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 5)
  : 2 * (a + b) - 3 * c * d + m = 2 ∨ 2 * (a + b) - 3 * c * d + m = -8 := by
  sorry

end value_of_expression_l114_114078


namespace tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l114_114548

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x * Real.exp x - Real.exp 1

-- Part (Ⅰ)
theorem tangent_line_at_one (h_a : a = 0) : ∃ m b : ℝ, ∀ x : ℝ, 2 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0 := sorry

-- Part (Ⅱ)
theorem unique_zero_of_f (h_a : a > 0) : ∃! t : ℝ, f a t = 0 := sorry

-- Part (Ⅲ)
theorem exists_lower_bound_of_f (h_a : a < 0) : ∃ m : ℝ, ∀ x : ℝ, f a x ≥ m := sorry

end tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l114_114548


namespace student_marks_l114_114080

theorem student_marks 
    (correct: ℕ) 
    (attempted: ℕ) 
    (marks_per_correct: ℕ) 
    (marks_per_incorrect: ℤ) 
    (correct_answers: correct = 27)
    (attempted_questions: attempted = 70)
    (marks_per_correct_condition: marks_per_correct = 3)
    (marks_per_incorrect_condition: marks_per_incorrect = -1): 
    (correct * marks_per_correct + (attempted - correct) * marks_per_incorrect) = 38 :=
by
    sorry

end student_marks_l114_114080


namespace exists_integers_cubes_sum_product_l114_114935

theorem exists_integers_cubes_sum_product :
  ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 :=
by
  sorry

end exists_integers_cubes_sum_product_l114_114935


namespace cricket_team_matches_in_august_l114_114384

noncomputable def cricket_matches_played_in_august (M W W_new: ℕ) : Prop :=
  W = 26 * M / 100 ∧
  W_new = 52 * (M + 65) / 100 ∧ 
  W_new = W + 65

theorem cricket_team_matches_in_august (M W W_new: ℕ) : cricket_matches_played_in_august M W W_new → M = 120 := 
by
  sorry

end cricket_team_matches_in_august_l114_114384


namespace arithmetic_sequence_product_l114_114898

theorem arithmetic_sequence_product {b : ℕ → ℤ} (d : ℤ) (h1 : ∀ n, b (n + 1) = b n + d)
    (h2 : b 5 * b 6 = 21) : b 4 * b 7 = -11 :=
  sorry

end arithmetic_sequence_product_l114_114898


namespace expression_divisible_by_3_l114_114258

theorem expression_divisible_by_3 (k : ℤ) : ∃ m : ℤ, (2 * k + 3)^2 - 4 * k^2 = 3 * m :=
by
  sorry

end expression_divisible_by_3_l114_114258


namespace smallest_angle_in_right_triangle_l114_114365

-- Given conditions
def angle_α := 90 -- The right-angle in degrees
def angle_β := 55 -- The given angle in degrees

-- Goal: Prove that the smallest angle is 35 degrees.
theorem smallest_angle_in_right_triangle (a b c : ℕ) (h1 : a = angle_α) (h2 : b = angle_β) (h3 : c = 180 - a - b) : c = 35 := 
by {
  -- use sorry to skip the proof steps
  sorry
}

end smallest_angle_in_right_triangle_l114_114365


namespace sophie_oranges_per_day_l114_114475

/-- Sophie and Hannah together eat a certain number of fruits in 30 days.
    Given Hannah eats 40 grapes every day, prove that Sophie eats 20 oranges every day. -/
theorem sophie_oranges_per_day (total_fruits : ℕ) (grapes_per_day : ℕ) (days : ℕ)
  (total_days_fruits : total_fruits = 1800) (hannah_grapes : grapes_per_day = 40) (days_count : days = 30) :
  (total_fruits - grapes_per_day * days) / days = 20 :=
by
  sorry

end sophie_oranges_per_day_l114_114475


namespace evaluate_expression_l114_114762

theorem evaluate_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end evaluate_expression_l114_114762


namespace models_kirsty_can_buy_l114_114734

def savings := 30 * 0.45
def new_price := 0.50

theorem models_kirsty_can_buy : savings / new_price = 27 := by
  sorry

end models_kirsty_can_buy_l114_114734


namespace geometric_sum_proof_l114_114294

theorem geometric_sum_proof (S : ℕ → ℝ) (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
    (hS3 : S 3 = 8) (hS6 : S 6 = 7)
    (Sn_def : ∀ n, S n = a 0 * (1 - r ^ n) / (1 - r)) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = -7 / 8 :=
by
  sorry

end geometric_sum_proof_l114_114294


namespace deepak_age_is_21_l114_114426

noncomputable def DeepakCurrentAge (x : ℕ) : Prop :=
  let Rahul := 4 * x
  let Deepak := 3 * x
  let Karan := 5 * x
  Rahul + 6 = 34 ∧
  (Rahul + 6) / 7 = (Deepak + 6) / 5 ∧ (Rahul + 6) / 7 = (Karan + 6) / 9 → 
  Deepak = 21

theorem deepak_age_is_21 : ∃ x : ℕ, DeepakCurrentAge x :=
by
  use 7
  sorry

end deepak_age_is_21_l114_114426


namespace paint_mixture_replacement_l114_114975

theorem paint_mixture_replacement :
  ∃ x y : ℝ,
    (0.5 * (1 - x) + 0.35 * x = 0.45) ∧
    (0.6 * (1 - y) + 0.45 * y = 0.55) ∧
    (x = 1 / 3) ∧
    (y = 1 / 3) :=
sorry

end paint_mixture_replacement_l114_114975


namespace base_not_divisible_by_5_l114_114387

def is_not_divisible_by_5 (c : ℤ) : Prop :=
  ¬(∃ k : ℤ, c = 5 * k)

def check_not_divisible_by_5 (b : ℤ) : Prop :=
  is_not_divisible_by_5 (3 * b^3 - 3 * b^2 - b)

theorem base_not_divisible_by_5 :
  check_not_divisible_by_5 6 ∧ check_not_divisible_by_5 8 :=
by 
  sorry

end base_not_divisible_by_5_l114_114387


namespace annalise_total_cost_correct_l114_114431

-- Define the constants from the problem
def boxes : ℕ := 25
def packs_per_box : ℕ := 18
def tissues_per_pack : ℕ := 150
def tissue_price : ℝ := 0.06
def discount_per_box : ℝ := 0.10
def volume_discount : ℝ := 0.08
def tax_rate : ℝ := 0.05

-- Calculate the total number of tissues
def total_tissues : ℕ := boxes * packs_per_box * tissues_per_pack

-- Calculate the total cost without any discounts
def initial_cost : ℝ := total_tissues * tissue_price

-- Apply the 10% discount on the price of the total packs in each box purchased
def cost_after_box_discount : ℝ := initial_cost * (1 - discount_per_box)

-- Apply the 8% volume discount for buying 10 or more boxes
def cost_after_volume_discount : ℝ := cost_after_box_discount * (1 - volume_discount)

-- Apply the 5% tax on the final price after all discounts
def final_cost : ℝ := cost_after_volume_discount * (1 + tax_rate)

-- Define the expected final cost
def expected_final_cost : ℝ := 3521.07

-- Proof statement
theorem annalise_total_cost_correct : final_cost = expected_final_cost := by
  -- Sorry is used as placeholder for the actual proof
  sorry

end annalise_total_cost_correct_l114_114431


namespace speed_of_current_l114_114090

theorem speed_of_current (m c : ℝ) (h1 : m + c = 18) (h2 : m - c = 11.2) : c = 3.4 :=
by
  sorry

end speed_of_current_l114_114090


namespace find_m_value_l114_114271

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

noncomputable def find_m (m : ℝ) : Prop :=
  f (f m) = f 2002 - 7 / 2

theorem find_m_value : find_m (-3 / 8) :=
by
  unfold find_m
  sorry

end find_m_value_l114_114271


namespace kenny_jumps_l114_114363

theorem kenny_jumps (M : ℕ) (h : 34 + M + 0 + 123 + 64 + 23 + 61 = 325) : M = 20 :=
by
  sorry

end kenny_jumps_l114_114363


namespace find_n_of_sum_of_evens_l114_114574

-- Definitions based on conditions in part (a)
def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_evens_up_to (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  (k / 2) * (2 + (n - 1))

-- Problem statement in Lean
theorem find_n_of_sum_of_evens : 
  ∃ n : ℕ, is_odd n ∧ sum_of_evens_up_to n = 81 * 82 ∧ n = 163 :=
by
  sorry

end find_n_of_sum_of_evens_l114_114574


namespace max_marks_l114_114812

theorem max_marks (M : ℝ) (h_pass : 0.30 * M = 231) : M = 770 := sorry

end max_marks_l114_114812


namespace least_integer_greater_than_sqrt_450_l114_114463

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l114_114463


namespace charlie_collected_15_seashells_l114_114268

variables (c e : ℝ)

-- Charlie collected 10 more seashells than Emily
def charlie_more_seashells := c = e + 10

-- Emily collected one-third the number of seashells Charlie collected
def emily_seashells := e = c / 3

theorem charlie_collected_15_seashells (hc: charlie_more_seashells c e) (he: emily_seashells c e) : c = 15 := 
by sorry

end charlie_collected_15_seashells_l114_114268


namespace find_sum_of_x_and_y_l114_114967

theorem find_sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 20) : x + y = 2 := 
by
  sorry

end find_sum_of_x_and_y_l114_114967


namespace task_completion_time_l114_114797

noncomputable def john_work_rate := (1: ℚ) / 20
noncomputable def jane_work_rate := (1: ℚ) / 12
noncomputable def combined_work_rate := john_work_rate + jane_work_rate
noncomputable def time_jane_disposed := 4

theorem task_completion_time :
  (∃ x : ℚ, (combined_work_rate * x + john_work_rate * time_jane_disposed = 1) ∧ (x + time_jane_disposed = 10)) :=
by
  use 6  
  sorry

end task_completion_time_l114_114797


namespace find_x_l114_114605

theorem find_x (x : ℕ) (h : 27^3 + 27^3 + 27^3 + 27^3 = 3^x) : x = 11 :=
sorry

end find_x_l114_114605


namespace find_triangle_area_l114_114927

noncomputable def triangle_area_problem
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) : ℝ :=
  (1 / 2) * a * b

theorem find_triangle_area
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) :
  triangle_area_problem a b h1 h2 = 1 / 2 := by
  sorry

end find_triangle_area_l114_114927


namespace range_of_m_for_distinct_real_roots_of_quadratic_l114_114198

theorem range_of_m_for_distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 4*x1 - m = 0 ∧ x2^2 + 4*x2 - m = 0) ↔ m > -4 :=
by
  sorry

end range_of_m_for_distinct_real_roots_of_quadratic_l114_114198


namespace intersection_empty_l114_114273

-- Define the set M
def M : Set ℝ := { x | ∃ y, y = Real.log (1 - x)}

-- Define the set N
def N : Set (ℝ × ℝ) := { p | ∃ x, ∃ y, (p = (x, y)) ∧ (y = Real.exp x) ∧ (x ∈ Set.univ)}

-- Prove that M ∩ N = ∅
theorem intersection_empty : M ∩ (Prod.fst '' N) = ∅ :=
by
  sorry

end intersection_empty_l114_114273


namespace problem_statement_l114_114914
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l114_114914


namespace max_value_ineq_l114_114136

theorem max_value_ineq (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_value_ineq_l114_114136


namespace lines_intersection_l114_114796

def intersection_point_of_lines
  (t u : ℚ)
  (x₁ y₁ x₂ y₂ : ℚ)
  (x y : ℚ) : Prop := 
  ∃ (t u : ℚ),
    (x₁ + 3*t = 7 + 6*u) ∧
    (y₁ - 4*t = -5 + 3*u) ∧
    (x = x₁ + 3 * t) ∧ 
    (y = y₁ - 4 * t)

theorem lines_intersection :
  ∀ (t u : ℚ),
    intersection_point_of_lines t u 3 2 7 (-5) (87/11) (-50/11) :=
by
  sorry

end lines_intersection_l114_114796


namespace maria_cupcakes_l114_114846

variable (initial : ℕ) (additional : ℕ) (remaining : ℕ)

theorem maria_cupcakes (h_initial : initial = 19) (h_additional : additional = 10) (h_remaining : remaining = 24) : initial + additional - remaining = 5 := by
  sorry

end maria_cupcakes_l114_114846


namespace binom_divisibility_by_prime_l114_114842

-- Given definitions
variable (p k : ℕ) (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2)

-- Main theorem statement
theorem binom_divisibility_by_prime
  (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2) :
  Nat.choose (p - k + 1) k - Nat.choose (p - k - 1) (k - 2) ≡ 0 [MOD p] :=
sorry

end binom_divisibility_by_prime_l114_114842


namespace combined_ages_l114_114096

theorem combined_ages (h_age : ℕ) (diff : ℕ) (years_later : ℕ) (hurley_age : h_age = 14) 
                       (age_difference : diff = 20) (years_passed : years_later = 40) : 
                       h_age + diff + years_later * 2 = 128 := by
  sorry

end combined_ages_l114_114096


namespace exists_infinitely_many_solutions_l114_114214

theorem exists_infinitely_many_solutions :
  ∃ m : ℕ, m > 0 ∧ (∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) →
    (1/a + 1/b + 1/c + 1/(a*b*c) = m / (a + b + c))) :=
sorry

end exists_infinitely_many_solutions_l114_114214


namespace max_real_solutions_l114_114722

noncomputable def max_number_of_real_solutions (n : ℕ) (y : ℝ) : ℕ :=
if (n + 1) % 2 = 1 then 1 else 0

theorem max_real_solutions (n : ℕ) (hn : 0 < n) (y : ℝ) :
  max_number_of_real_solutions n y = 1 :=
by
  sorry

end max_real_solutions_l114_114722


namespace corveus_sleep_deficit_l114_114862

theorem corveus_sleep_deficit :
  let weekday_sleep := 5 -- 4 hours at night + 1-hour nap
  let weekend_sleep := 5 -- 5 hours at night, no naps
  let total_weekday_sleep := 5 * weekday_sleep
  let total_weekend_sleep := 2 * weekend_sleep
  let total_sleep := total_weekday_sleep + total_weekend_sleep
  let recommended_sleep_per_day := 6
  let total_recommended_sleep := 7 * recommended_sleep_per_day
  let sleep_deficit := total_recommended_sleep - total_sleep
  sleep_deficit = 7 :=
by
  -- Insert proof steps here
  sorry

end corveus_sleep_deficit_l114_114862


namespace greatest_consecutive_integers_sum_55_l114_114143

theorem greatest_consecutive_integers_sum_55 :
  ∃ N a : ℤ, (N * (2 * a + N - 1)) = 110 ∧ (∀ M a' : ℤ, (M * (2 * a' + M - 1)) = 110 → N ≥ M) :=
sorry

end greatest_consecutive_integers_sum_55_l114_114143


namespace maximum_value_expression_l114_114653

theorem maximum_value_expression (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 + 3 * a * b)) + Real.sqrt (Real.sqrt (b^2 + 3 * b * c)) +
   Real.sqrt (Real.sqrt (c^2 + 3 * c * d)) + Real.sqrt (Real.sqrt (d^2 + 3 * d * a))) ≤ 4 * Real.sqrt 2 :=
by 
  sorry

end maximum_value_expression_l114_114653


namespace distance_between_points_l114_114112

def distance_on_line (a b : ℝ) : ℝ := |b - a|

theorem distance_between_points (a b : ℝ) : distance_on_line a b = |b - a| :=
by sorry

end distance_between_points_l114_114112


namespace brian_has_78_white_stones_l114_114083

-- Given conditions
variables (W B : ℕ) (R Bl : ℕ)
variables (x : ℕ)
variables (total_stones : ℕ := 330)
variables (total_collection1 : ℕ := 100)
variables (total_collection3 : ℕ := 130)

-- Condition: First collection stones sum to 100
#check W + B = 100

-- Condition: Brian has more white stones than black ones
#check W > B

-- Condition: Ratio of red to blue stones is 3:2 in the third collection
#check R + Bl = 130
#check R = 3 * x
#check Bl = 2 * x

-- Condition: Total number of stones in all three collections is 330
#check total_stones = total_collection1 + total_collection1 + total_collection3

-- New collection's magnetic stones ratio condition
#check 2 * W / 78 = 2

-- Prove that Brian has 78 white stones
theorem brian_has_78_white_stones
  (h1 : W + B = 100)
  (h2 : W > B)
  (h3 : R + Bl = 130)
  (h4 : R = 3 * x)
  (h5 : Bl = 2 * x)
  (h6 : 2 * W / 78 = 2) :
  W = 78 :=
sorry

end brian_has_78_white_stones_l114_114083


namespace range_of_b_l114_114461

open Real

theorem range_of_b {b x x1 x2 : ℝ} 
  (h1 : ∀ x : ℝ, x^2 - b * x + 1 > 0 ↔ x < x1 ∨ x > x2)
  (h2 : x1 < 1)
  (h3 : x2 > 1) : 
  b > 2 := sorry

end range_of_b_l114_114461


namespace floor_ceil_difference_l114_114353

theorem floor_ceil_difference : 
  let a := (18 / 5) * (-33 / 4)
  let b := ⌈(-33 / 4 : ℝ)⌉
  let c := (18 / 5) * (b : ℝ)
  let d := ⌈c⌉
  ⌊a⌋ - d = -2 :=
by
  sorry

end floor_ceil_difference_l114_114353


namespace diff_12_358_7_2943_l114_114281

theorem diff_12_358_7_2943 : 12.358 - 7.2943 = 5.0637 :=
by
  -- Proof is not required, so we put sorry
  sorry

end diff_12_358_7_2943_l114_114281


namespace output_is_three_l114_114970

-- Define the initial values
def initial_a : ℕ := 1
def initial_b : ℕ := 2

-- Define the final value of a after the computation
def final_a : ℕ := initial_a + initial_b

-- The theorem stating that the final value of a is 3
theorem output_is_three : final_a = 3 := by
  sorry

end output_is_three_l114_114970


namespace max_value_of_b_l114_114448

theorem max_value_of_b (a b c : ℝ) (q : ℝ) (hq : q ≠ 0) 
  (h_geom : a = b / q ∧ c = b * q) 
  (h_arith : 2 * b + 4 = a + 6 + (b + 2) + (c + 1) - (b + 2)) :
  b ≤ 3 / 4 :=
sorry

end max_value_of_b_l114_114448


namespace simplify_fraction_l114_114650

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l114_114650


namespace unrepresentable_integers_l114_114859

theorem unrepresentable_integers :
    {n : ℕ | ∀ a b : ℕ, a > 0 → b > 0 → n ≠ (a * (b + 1) + (a + 1) * b) / (b * (b + 1)) } =
    {1} ∪ {n | ∃ m : ℕ, n = 2^m + 2} :=
by
    sorry

end unrepresentable_integers_l114_114859


namespace percent_of_x_is_y_in_terms_of_z_l114_114332

theorem percent_of_x_is_y_in_terms_of_z (x y z : ℝ) (h1 : 0.7 * (x - y) = 0.3 * (x + y))
    (h2 : 0.6 * (x + z) = 0.4 * (y - z)) : y / x = 0.4 :=
  sorry

end percent_of_x_is_y_in_terms_of_z_l114_114332


namespace arrange_numbers_l114_114302

theorem arrange_numbers (x y z : ℝ) (h1 : x = 20.8) (h2 : y = 0.82) (h3 : z = Real.log 20.8) : z < y ∧ y < x :=
by
  sorry

end arrange_numbers_l114_114302


namespace dividend_rate_correct_l114_114918

def stock_price : ℝ := 150
def yield_percentage : ℝ := 0.08
def dividend_rate : ℝ := stock_price * yield_percentage

theorem dividend_rate_correct : dividend_rate = 12 := by
  sorry

end dividend_rate_correct_l114_114918


namespace total_money_spent_l114_114623

/-- 
John buys a gaming PC for $1200.
He decides to replace the video card in it.
He sells the old card for $300 and buys a new one for $500.
Prove total money spent on the computer after counting the savings from selling the old card is $1400.
-/
theorem total_money_spent (initial_cost : ℕ) (sale_price_old_card : ℕ) (price_new_card : ℕ) : 
  (initial_cost = 1200) → (sale_price_old_card = 300) → (price_new_card = 500) → 
  (initial_cost + (price_new_card - sale_price_old_card) = 1400) :=
by 
  intros
  sorry

end total_money_spent_l114_114623


namespace odds_against_C_winning_l114_114831

theorem odds_against_C_winning (prob_A: ℚ) (prob_B: ℚ) (prob_C: ℚ)
    (odds_A: prob_A = 1 / 5) (odds_B: prob_B = 2 / 5) 
    (total_prob: prob_A + prob_B + prob_C = 1):
    ((1 - prob_C) / prob_C) = 3 / 2 :=
by
  sorry

end odds_against_C_winning_l114_114831


namespace positive_value_of_n_l114_114318

theorem positive_value_of_n (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 25 = 0 ∧ ∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) →
  n = 20 :=
by
  sorry

end positive_value_of_n_l114_114318


namespace equation1_solution_equation2_solution_l114_114130

-- Equation 1 Statement
theorem equation1_solution (x : ℝ) : 
  (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ↔ x = -20 :=
by sorry

-- Equation 2 Statement
theorem equation2_solution (x : ℝ) : 
  (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ↔ x = 67 / 23 :=
by sorry

end equation1_solution_equation2_solution_l114_114130


namespace reflect_curve_maps_onto_itself_l114_114496

theorem reflect_curve_maps_onto_itself (a b c : ℝ) :
    ∃ (x0 y0 : ℝ), 
    x0 = -a / 3 ∧ 
    y0 = 2 * a^3 / 27 - a * b / 3 + c ∧
    ∀ x y x' y', 
    y = x^3 + a * x^2 + b * x + c → 
    x' = 2 * x0 - x → 
    y' = 2 * y0 - y → 
    y' = x'^3 + a * x'^2 + b * x' + c := 
    by sorry

end reflect_curve_maps_onto_itself_l114_114496


namespace probability_correct_l114_114785

noncomputable def probability_study_group : ℝ :=
  let p_woman : ℝ := 0.5
  let p_man : ℝ := 0.5

  let p_woman_lawyer : ℝ := 0.3
  let p_woman_doctor : ℝ := 0.4
  let p_woman_engineer : ℝ := 0.3

  let p_man_lawyer : ℝ := 0.4
  let p_man_doctor : ℝ := 0.2
  let p_man_engineer : ℝ := 0.4

  (p_woman * p_woman_lawyer + p_woman * p_woman_doctor +
  p_man * p_man_lawyer + p_man * p_man_doctor)

theorem probability_correct : probability_study_group = 0.65 := by
  sorry

end probability_correct_l114_114785


namespace range_of_m_and_n_l114_114731

theorem range_of_m_and_n (m n : ℝ) : 
  (2 * 2 - 3 + m > 0) → ¬ (2 + 3 - n ≤ 0) → (m > -1 ∧ n < 5) := by
  intros hA hB
  sorry

end range_of_m_and_n_l114_114731


namespace determine_f_16_l114_114440

theorem determine_f_16 (a : ℝ) (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  (∀ x, a ^ (x - 4) + 1 = 2) →
  f 4 = 2 →
  f 16 = 4 :=
by
  sorry

end determine_f_16_l114_114440


namespace find_b_l114_114445

theorem find_b (b : ℕ) (h1 : 0 ≤ b) (h2 : b ≤ 20) (h3 : (746392847 - b) % 17 = 0) : b = 16 :=
sorry

end find_b_l114_114445


namespace full_house_plus_two_probability_l114_114754

def total_ways_to_choose_7_cards_from_52 : ℕ :=
  Nat.choose 52 7

def ways_for_full_house_plus_two : ℕ :=
  13 * 4 * 12 * 6 * 55 * 16

def probability_full_house_plus_two : ℚ :=
  (ways_for_full_house_plus_two : ℚ) / (total_ways_to_choose_7_cards_from_52 : ℚ)

theorem full_house_plus_two_probability :
  probability_full_house_plus_two = 13732 / 3344614 :=
by
  sorry

end full_house_plus_two_probability_l114_114754


namespace find_number_l114_114416

theorem find_number (x : ℚ) (h : x / 5 = 3 * (x / 6) - 40) : x = 400 / 3 :=
sorry

end find_number_l114_114416


namespace not_exists_implies_bounds_l114_114111

variable (a : ℝ)

/-- If there does not exist an x such that x^2 + (a - 1) * x + 1 < 0, then -1 ≤ a ∧ a ≤ 3. -/
theorem not_exists_implies_bounds : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3) :=
by sorry

end not_exists_implies_bounds_l114_114111


namespace min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l114_114002

noncomputable def min_value_expression (a b : ℝ) (hab : 2 * a + b = 1) : ℝ :=
  4 * a^2 + b^2 + 1 / (a * b)

theorem min_value_expression_geq_17_div_2 {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (hab: 2 * a + b = 1) :
  min_value_expression a b hab ≥ 17 / 2 :=
sorry

theorem min_value_expression_eq_17_div_2_for_specific_a_b :
  min_value_expression (1/3) (1/3) (by norm_num) = 17 / 2 :=
sorry

end min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l114_114002


namespace number_of_associates_l114_114391

theorem number_of_associates
  (num_managers : ℕ) 
  (avg_salary_managers : ℝ) 
  (avg_salary_associates : ℝ) 
  (avg_salary_company : ℝ)
  (total_employees : ℕ := num_managers + A) -- Adding a placeholder A for the associates
  (total_salary_company : ℝ := (num_managers * avg_salary_managers) + (A * avg_salary_associates)) 
  (average_calculation : avg_salary_company = total_salary_company / total_employees) :
  ∃ A : ℕ, A = 75 :=
by
  let A : ℕ := 75
  sorry

end number_of_associates_l114_114391


namespace tiles_needed_l114_114582

def room_area : ℝ := 2 * 4 * 2 * 6
def tile_area : ℝ := 1.5 * 2

theorem tiles_needed : room_area / tile_area = 32 := 
by
  sorry

end tiles_needed_l114_114582


namespace midterm_exam_2022_option_probabilities_l114_114501

theorem midterm_exam_2022_option_probabilities :
  let no_option := 4
  let prob_distribution := (1 : ℚ) / 3
  let combs_with_4_correct := 1
  let combs_with_3_correct := 4
  let combs_with_2_correct := 6
  let prob_4_correct := prob_distribution
  let prob_3_correct := prob_distribution / combs_with_3_correct
  let prob_2_correct := prob_distribution / combs_with_2_correct
  
  let prob_B_correct := combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct
  let prob_C_given_event_A := combs_with_3_correct * prob_3_correct / (combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct)
  
  (prob_B_correct > 1 / 2) ∧ (prob_C_given_event_A = 1 / 3) :=
by 
  sorry

end midterm_exam_2022_option_probabilities_l114_114501


namespace f_iterated_result_l114_114511

def f (x : ℕ) : ℕ :=
  if Even x then 3 * x / 2 else 2 * x + 1

theorem f_iterated_result : f (f (f (f 1))) = 31 := by
  sorry

end f_iterated_result_l114_114511


namespace intersection_y_condition_l114_114270

theorem intersection_y_condition (a : ℝ) :
  (∃ x y : ℝ, 2 * x - a * y + 2 = 0 ∧ x + y = 0 ∧ y < 0) → a < -2 :=
by
  sorry

end intersection_y_condition_l114_114270


namespace find_N_l114_114264

variables (k N : ℤ)

theorem find_N (h : ((k * N + N) / N - N) = k - 2021) : N = 2022 :=
by
  sorry

end find_N_l114_114264


namespace tan_alpha_l114_114116

theorem tan_alpha (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan β = 1 / 3) :
  Real.tan α = 2 / 9 := by
  sorry

end tan_alpha_l114_114116


namespace smallest_prime_divisor_of_sum_l114_114336

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l114_114336


namespace factorize_problem_1_factorize_problem_2_l114_114576

variables (x y : ℝ)

-- Problem 1: Prove that x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2
theorem factorize_problem_1 : 
  x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2 :=
sorry

-- Problem 2: Prove that x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)
theorem factorize_problem_2 : 
  x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) :=
sorry

end factorize_problem_1_factorize_problem_2_l114_114576


namespace find_m_range_l114_114589

/--
Given:
1. Proposition \( p \) (p): The equation \(\frac{x^2}{2} + \frac{y^2}{m} = 1\) represents an ellipse with foci on the \( y \)-axis.
2. Proposition \( q \) (q): \( f(x) = \frac{4}{3}x^3 - 2mx^2 + (4m-3)x - m \) is monotonically increasing on \((-\infty, +\infty)\).

Prove:
If \( \neg p \land q \) is true, then the range of values for \( m \) is \( [1, 2] \).
-/

def p (m : ℝ) : Prop :=
  m > 2

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, (4 * x^2 - 4 * m * x + 4 * m - 3) >= 0

theorem find_m_range (m : ℝ) (hpq : ¬ p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end find_m_range_l114_114589


namespace parametric_to_standard_l114_114351

theorem parametric_to_standard (t a b x y : ℝ)
(h1 : x = (a / 2) * (t + 1 / t))
(h2 : y = (b / 2) * (t - 1 / t)) :
  (x^2 / a^2) - (y^2 / b^2) = 1 :=
by
  sorry

end parametric_to_standard_l114_114351


namespace a_range_condition_l114_114473

theorem a_range_condition (a : ℝ) : 
  (∀ x y : ℝ, ((x + a)^2 + (y - a)^2 < 4) → (x = -1 ∧ y = -1)) → 
  -1 < a ∧ a < 1 :=
by
  sorry

end a_range_condition_l114_114473


namespace line_passes_through_quadrants_l114_114030

variables (a b c p : ℝ)

-- Given conditions
def conditions :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  (a + b) / c = p ∧ 
  (b + c) / a = p ∧ 
  (c + a) / b = p

-- Goal statement
theorem line_passes_through_quadrants : conditions a b c p → 
  (∃ x : ℝ, x > 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p < 0) :=
sorry

end line_passes_through_quadrants_l114_114030


namespace find_g_l114_114858

variable (x : ℝ)

theorem find_g :
  ∃ g : ℝ → ℝ, 2 * x ^ 5 + 4 * x ^ 3 - 3 * x + 5 + g x = 3 * x ^ 4 + 7 * x ^ 2 - 2 * x - 4 ∧
                g x = -2 * x ^ 5 + 3 * x ^ 4 - 4 * x ^ 3 + 7 * x ^ 2 - x - 9 :=
sorry

end find_g_l114_114858


namespace compare_f_values_l114_114848

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - Real.cos x

theorem compare_f_values :
  f 0 < f 0.5 ∧ f 0.5 < f 0.6 :=
by {
  -- proof would go here
  sorry
}

end compare_f_values_l114_114848


namespace problem_solution_l114_114267

-- Declare the proof problem in Lean 4

theorem problem_solution (x y : ℝ) 
  (h1 : (y + 1) ^ 2 + (x - 2) ^ (1/2) = 0) : 
  y ^ x = 1 :=
sorry

end problem_solution_l114_114267


namespace smaller_circle_radius_l114_114855

noncomputable def radius_of_smaller_circles (R : ℝ) (r1 r2 r3 : ℝ) (OA OB OC : ℝ) : Prop :=
(OA = R + r1) ∧ (OB = R + 3 * r1) ∧ (OC = R + 5 * r1) ∧ 
((OB = OA + 2 * r1) ∧ (OC = OB + 2 * r1))

theorem smaller_circle_radius (r : ℝ) (R : ℝ := 2) :
  radius_of_smaller_circles R r r r (R + r) (R + 3 * r) (R + 5 * r) → r = 1 :=
by
  sorry

end smaller_circle_radius_l114_114855


namespace tangent_line_to_ellipse_l114_114232

theorem tangent_line_to_ellipse (k : ℝ) :
  (∀ x : ℝ, (x / 2 + 2 * (k * x + 2) ^ 2) = 2) →
  k^2 = 3 / 4 :=
by
  sorry

end tangent_line_to_ellipse_l114_114232


namespace pool_water_amount_correct_l114_114066

noncomputable def water_in_pool_after_ten_hours : ℝ :=
  let h1 := 8
  let h2_3 := 10 * 2
  let h4_5 := 14 * 2
  let h6 := 12
  let h7 := 12 - 8
  let h8 := 12 - 18
  let h9 := 12 - 24
  let h10 := 6
  h1 + h2_3 + h4_5 + h6 + h7 + h8 + h9 + h10

theorem pool_water_amount_correct :
  water_in_pool_after_ten_hours = 60 := 
sorry

end pool_water_amount_correct_l114_114066


namespace second_shipment_is_13_l114_114596

-- Definitions based on the conditions
def first_shipment : ℕ := 7
def third_shipment : ℕ := 45
def total_couscous_used : ℕ := 13 * 5 -- 65
def total_couscous_from_three_shipments (second_shipment : ℕ) : ℕ :=
  first_shipment + second_shipment + third_shipment

-- Statement of the proof problem corresponding to the conditions and question
theorem second_shipment_is_13 (x : ℕ) 
  (h : total_couscous_used = total_couscous_from_three_shipments x) : x = 13 := 
by
  sorry

end second_shipment_is_13_l114_114596


namespace milk_production_l114_114056

variable (a b c d e : ℝ)

theorem milk_production (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  let rate_per_cow_per_day := b / (a * c)
  let production_per_day := d * rate_per_cow_per_day
  let total_production := production_per_day * e
  total_production = (b * d * e) / (a * c) :=
by
  sorry

end milk_production_l114_114056


namespace find_radius_l114_114508

-- Defining the conditions as given in the math problem
def sectorArea (r : ℝ) (L : ℝ) : ℝ := 0.5 * r * L

theorem find_radius (h1 : sectorArea r 5.5 = 13.75) : r = 5 :=
by sorry

end find_radius_l114_114508


namespace find_k_for_tangent_graph_l114_114571

theorem find_k_for_tangent_graph (k : ℝ) (h : (∀ x : ℝ, x^2 - 6 * x + k = 0 → (x = 3))) : k = 9 :=
sorry

end find_k_for_tangent_graph_l114_114571


namespace selected_room_l114_114333

theorem selected_room (room_count interval selected initial_room : ℕ) 
  (h_init : initial_room = 5)
  (h_interval : interval = 8)
  (h_room_count : room_count = 64) : 
  ∃ (nth_room : ℕ), nth_room = initial_room + interval * 6 ∧ nth_room = 53 :=
by
  sorry

end selected_room_l114_114333


namespace choose_math_class_representative_l114_114810

def number_of_boys : Nat := 26
def number_of_girls : Nat := 24

theorem choose_math_class_representative : number_of_boys + number_of_girls = 50 := 
by
  sorry

end choose_math_class_representative_l114_114810


namespace average_rounds_rounded_is_3_l114_114994

-- Definitions based on conditions
def golfers : List ℕ := [3, 4, 3, 6, 2, 4]
def rounds : List ℕ := [0, 1, 2, 3, 4, 5]

noncomputable def total_rounds : ℕ :=
  List.sum (List.zipWith (λ g r => g * r) golfers rounds)

def total_golfers : ℕ := List.sum golfers

noncomputable def average_rounds : ℕ :=
  Int.natAbs (Int.ofNat total_rounds / total_golfers).toNat

theorem average_rounds_rounded_is_3 : average_rounds = 3 := by
  sorry

end average_rounds_rounded_is_3_l114_114994


namespace cellphone_loading_time_approximately_l114_114253

noncomputable def cellphone_loading_time_minutes : ℝ :=
  let T := 533.78 -- Solution for T from solving the given equation
  T / 60

theorem cellphone_loading_time_approximately :
  abs (cellphone_loading_time_minutes - 8.90) < 0.01 :=
by 
  -- The proof goes here, but we are just required to state it
  sorry

end cellphone_loading_time_approximately_l114_114253


namespace remainder_sum_l114_114579

theorem remainder_sum (n : ℤ) : ((7 - n) + (n + 3)) % 7 = 3 :=
sorry

end remainder_sum_l114_114579


namespace number_of_cheesecakes_in_fridge_l114_114444

section cheesecake_problem

def cheesecakes_on_display : ℕ := 10
def cheesecakes_sold : ℕ := 7
def cheesecakes_left_to_be_sold : ℕ := 18

def cheesecakes_in_fridge (total_display : ℕ) (sold : ℕ) (left : ℕ) : ℕ :=
  left - (total_display - sold)

theorem number_of_cheesecakes_in_fridge :
  cheesecakes_in_fridge cheesecakes_on_display cheesecakes_sold cheesecakes_left_to_be_sold = 15 :=
by
  sorry

end cheesecake_problem

end number_of_cheesecakes_in_fridge_l114_114444


namespace terminating_decimal_expansion_l114_114615

theorem terminating_decimal_expansion : (11 / 125 : ℝ) = 0.088 := 
by
  sorry

end terminating_decimal_expansion_l114_114615


namespace solve_for_y_l114_114236

theorem solve_for_y (y : ℝ) (h : (↑(30 * y) + (↑(30 * y) + 17) ^ (1 / 3)) ^ (1 / 3) = 17) :
  y = 816 / 5 := 
sorry

end solve_for_y_l114_114236


namespace sam_compound_interest_l114_114408

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

theorem sam_compound_interest : 
  compound_interest 3000 0.10 2 1 = 3307.50 :=
by
  sorry

end sam_compound_interest_l114_114408


namespace differential_solution_l114_114728

theorem differential_solution (C : ℝ) : 
  ∃ y : ℝ → ℝ, (∀ x : ℝ, y x = C * (1 + x^2)) := 
by
  sorry

end differential_solution_l114_114728


namespace total_cars_at_end_of_play_l114_114628

def carsInFront : ℕ := 100
def carsInBack : ℕ := 2 * carsInFront
def additionalCars : ℕ := 300

theorem total_cars_at_end_of_play : carsInFront + carsInBack + additionalCars = 600 := by
  sorry

end total_cars_at_end_of_play_l114_114628


namespace share_of_C_l114_114192

theorem share_of_C (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 578) : 
  C = 408 :=
by
  -- Proof goes here
  sorry

end share_of_C_l114_114192


namespace smallest_n_divisibility_l114_114144

theorem smallest_n_divisibility (n : ℕ) (h : 1 ≤ n) :
  (∀ k, 1 ≤ k ∧ k ≤ n → n^3 - n ∣ k) ∨ (∃ k, 1 ≤ k ∧ k ≤ n ∧ ¬ (n^3 - n ∣ k)) :=
sorry

end smallest_n_divisibility_l114_114144


namespace inequality_ln_x_lt_x_lt_exp_x_l114_114084

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x - x

theorem inequality_ln_x_lt_x_lt_exp_x (x : ℝ) (h : x > 0) : Real.log x < x ∧ x < Real.exp x := by
  -- We need to supply the proof here
  sorry

end inequality_ln_x_lt_x_lt_exp_x_l114_114084


namespace total_surface_area_l114_114550

theorem total_surface_area (a b c : ℝ)
    (h1 : a + b + c = 40)
    (h2 : a^2 + b^2 + c^2 = 625)
    (h3 : a * b * c = 600) : 
    2 * (a * b + b * c + c * a) = 975 :=
by
  sorry

end total_surface_area_l114_114550


namespace pentagon_triangle_ratio_l114_114488

theorem pentagon_triangle_ratio (p t s : ℝ) 
  (h₁ : 5 * p = 30) 
  (h₂ : 3 * t = 30)
  (h₃ : 4 * s = 30) : 
  p / t = 3 / 5 := by
  sorry

end pentagon_triangle_ratio_l114_114488


namespace cos_x_minus_pi_over_3_l114_114494

theorem cos_x_minus_pi_over_3 (x : ℝ) (h : Real.sin (x + π / 6) = 4 / 5) :
  Real.cos (x - π / 3) = 4 / 5 :=
sorry

end cos_x_minus_pi_over_3_l114_114494


namespace identify_jars_l114_114108

namespace JarIdentification

/-- Definitions of Jar labels -/
inductive JarLabel
| Nickels
| Dimes
| Nickels_and_Dimes

open JarLabel

/-- Mislabeling conditions for each jar -/
def mislabeled (jarA : JarLabel) (jarB : JarLabel) (jarC : JarLabel) : Prop :=
  ((jarA ≠ Nickels) ∧ (jarB ≠ Dimes) ∧ (jarC ≠ Nickels_and_Dimes)) ∧
  ((jarC = Nickels ∨ jarC = Dimes))

/-- Given the result of a coin draw from the jar labeled "Nickels and Dimes" -/
def jarIdentity (jarA jarB jarC : JarLabel) (drawFromC : String) : Prop :=
  if drawFromC = "Nickel" then
    jarC = Nickels ∧ jarA = Nickels_and_Dimes ∧ jarB = Dimes
  else if drawFromC = "Dime" then
    jarC = Dimes ∧ jarB = Nickels_and_Dimes ∧ jarA = Nickels
  else 
    false

/-- Main theorem to prove the identification of jars -/
theorem identify_jars (jarA jarB jarC : JarLabel) (draw : String)
  (h1 : mislabeled jarA jarB jarC) :
  jarIdentity jarA jarB jarC draw :=
by
  sorry

end JarIdentification

end identify_jars_l114_114108


namespace original_team_size_l114_114005

theorem original_team_size (n : ℕ) (W : ℕ) :
  (W = n * 94) →
  ((W + 110 + 60) / (n + 2) = 92) →
  n = 7 :=
by
  intro hW_avg hnew_avg
  -- The proof steps would go here
  sorry

end original_team_size_l114_114005


namespace vacuum_cleaner_cost_l114_114586

-- Define initial amount collected
def initial_amount : ℕ := 20

-- Define amount added each week
def weekly_addition : ℕ := 10

-- Define number of weeks
def number_of_weeks : ℕ := 10

-- Define the total amount after 10 weeks
def total_amount : ℕ := initial_amount + (weekly_addition * number_of_weeks)

-- Prove that the total amount is equal to the cost of the vacuum cleaner
theorem vacuum_cleaner_cost : total_amount = 120 := by
  sorry

end vacuum_cleaner_cost_l114_114586


namespace equal_work_women_l114_114793

-- Let W be the amount of work one woman can do in a day.
-- Let M be the amount of work one man can do in a day.
-- Let x be the number of women who do the same amount of work as 5 men.

def numWomenEqualWork (W : ℝ) (M : ℝ) (x : ℝ) : Prop :=
  5 * M = x * W

theorem equal_work_women (W M x : ℝ) 
  (h1 : numWomenEqualWork W M x)
  (h2 : (3 * M + 5 * W) * 10 = (7 * W) * 14) :
  x = 8 :=
sorry

end equal_work_women_l114_114793


namespace find_abc_sum_l114_114549

theorem find_abc_sum (A B C : ℤ) (h : ∀ x : ℝ, x^3 + A * x^2 + B * x + C = (x + 1) * (x - 3) * (x - 4)) : A + B + C = 11 :=
by {
  -- This statement asserts that, given the conditions, the sum A + B + C equals 11
  sorry
}

end find_abc_sum_l114_114549


namespace simplify_fraction_l114_114060

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) : 
  (9 * x^3 * y^2) / (12 * x^2 * y^4) = 9 / 16 := by
  sorry

end simplify_fraction_l114_114060


namespace evaluate_g_at_3_l114_114906

def g (x: ℝ) := 5 * x^3 - 4 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 101 :=
by 
  sorry

end evaluate_g_at_3_l114_114906


namespace inradius_of_regular_tetrahedron_l114_114050

theorem inradius_of_regular_tetrahedron (h r : ℝ) (S : ℝ) 
  (h_eq: 4 * (1/3) * S * r = (1/3) * S * h) : r = (1/4) * h :=
sorry

end inradius_of_regular_tetrahedron_l114_114050


namespace school_orchestra_members_l114_114988

theorem school_orchestra_members (total_members can_play_violin can_play_keyboard neither : ℕ)
    (h1 : total_members = 42)
    (h2 : can_play_violin = 25)
    (h3 : can_play_keyboard = 22)
    (h4 : neither = 3) :
    (can_play_violin + can_play_keyboard) - (total_members - neither) = 8 :=
by
  sorry

end school_orchestra_members_l114_114988
