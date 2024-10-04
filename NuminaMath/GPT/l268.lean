import Mathlib

namespace find_x1_x2_circle_area_tangent_MN_max_area_ABCD_l268_268242

-- Definitions
def parabola (x : ℝ) : ℝ := x^2
def tangent_line (P : ℝ × ℝ) (parabola : ℝ → ℝ) : ℝ × ℝ := sorry -- detailed definition as placeholder
def circle_area (center : ℝ × ℝ) (radius : ℝ) : ℝ := π * radius^2

-- Statement (Ⅰ)
theorem find_x1_x2 :
  let x1 := 1 - Real.sqrt 2 in
  let x2 := 1 + Real.sqrt 2 in
  x1 * x1 - 2 * x1 - 1 = 0 ∧ x2 * x2 - 2 * x2 - 1 = 0 ∧ x1 < x2 :=
sorry

-- Statement (Ⅱ)
theorem circle_area_tangent_MN :
  let center := (1, -1) in
  let radius := 4 / Real.sqrt 5 in
  circle_area center radius = 16 * π / 5 :=
sorry

-- Statement (Ⅲ)
theorem max_area_ABCD :
  let radius := 4 / Real.sqrt 5 in
  let d1_sq := 1 in
  let d2_sq := 1 in
  let max_area := 2 * radius^2 - (d1_sq + d2_sq) in
  max_area = 22 / 5 :=
sorry

end find_x1_x2_circle_area_tangent_MN_max_area_ABCD_l268_268242


namespace find_x_l268_268917

theorem find_x (x : ℕ) (hx : x > 0 ∧ x <= 100) 
    (mean_twice_mode : (40 + 57 + 76 + 90 + x + x) / 6 = 2 * x) : 
    x = 26 :=
sorry

end find_x_l268_268917


namespace expected_value_red_balls_l268_268549

open Classical

noncomputable def expected_value_draw (X : ℕ → ℝ) (probs : ℕ → ℝ) : ℝ :=
  ∑ n in {0, 1, 2}, (n : ℝ) * probs n

theorem expected_value_red_balls :
  let probs (k : ℕ) := if k = 0 then 1 / 10 else if k = 1 then 3 / 5 else if k = 2 then 3 / 10 else 0 in
  let X (n : ℕ) := n : ℝ in
  expected_value_draw X probs = 1.2 := 
by
  sorry

end expected_value_red_balls_l268_268549


namespace range_of_S_on_ellipse_l268_268954

theorem range_of_S_on_ellipse :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 / 3 = 1) →
    -Real.sqrt 5 ≤ x + y ∧ x + y ≤ Real.sqrt 5 :=
by
  intro x y
  intro h
  sorry

end range_of_S_on_ellipse_l268_268954


namespace equivalence_of_daps_and_dips_l268_268212

variable (daps dops dips : Type)
variable (to_dops : daps → dops)
variable (to_dips : dops → dips)

-- Conditions given
axiom cond1 : ∀ x : daps, to_dops (5 * x) = 4 * to_dops x
axiom cond2 : ∀ y : dops, to_dips (3 * y) = 10 * to_dips y

-- The statement to prove
theorem equivalence_of_daps_and_dips : 
  ∃ x : daps, to_dips (to_dops x) * 60 = 22.5 * x :=
  sorry

end equivalence_of_daps_and_dips_l268_268212


namespace children_summer_camp_difference_l268_268100

theorem children_summer_camp_difference : 
  let camp_5_10 := 245785
  let home_5_10 := 197680
  let camp_11_15 := 287279
  let home_11_15 := 253425
  let camp_16_18 := 285994
  let home_16_18 := 217173
  let difference_5_10 := camp_5_10 - home_5_10
  let difference_11_15 := camp_11_15 - home_11_15
  let difference_16_18 := camp_16_18 - home_16_18
  let total_difference := difference_5_10 + difference_11_15 + difference_16_18
  in difference_5_10 = 48105 ∧ difference_11_15 = 33854 ∧ difference_16_18 = 68821 ∧ total_difference = 150780 := 
by
  sorry

end children_summer_camp_difference_l268_268100


namespace min_value_of_a_sq_plus_b_sq_over_a_minus_b_l268_268950

theorem min_value_of_a_sq_plus_b_sq_over_a_minus_b {a b : ℝ} (h1 : a > b) (h2 : a * b = 1) : 
  ∃ x, x = 2 * Real.sqrt 2 ∧ ∀ y, y = (a^2 + b^2) / (a - b) → y ≥ x :=
by {
  sorry
}

end min_value_of_a_sq_plus_b_sq_over_a_minus_b_l268_268950


namespace Helly_four_sets_Helly_n_geq_4_l268_268824

variables {C : Type} [ConvexSpace C] {n : ℕ} (Cs : Fin n → Set C)

open Set

theorem Helly_four_sets 
  (h1 : ∀ i j k : Fin 4, i ≠ j → j ≠ k → i ≠ k → (Cs i ∩ Cs j ∩ Cs k) ≠ ∅) :
  (Cs 0 ∩ Cs 1 ∩ Cs 2 ∩ Cs 3) ≠ ∅ :=
sorry

theorem Helly_n_geq_4
  (n : ℕ) (hn : n ≥ 4)
  (Cs : Fin n → Set C)
  (h2 : ∀ (i : Fin n), (⋂ (j : Fin n) (h : j ≠ i), Cs j) ≠ ∅) :
  (⋂ (i : Fin n), Cs i) ≠ ∅ :=
sorry

end Helly_four_sets_Helly_n_geq_4_l268_268824


namespace direct_proportion_result_inverse_proportion_result_power_function_result_l268_268282

-- Definitions for problem conditions
def is_direct_proportion_function (f : ℝ → ℝ) := ∃ k : ℝ, ∀ x : ℝ, f(x) = k * x
def is_inverse_proportion_function (f : ℝ → ℝ) := ∃ k : ℝ, ∀ x : ℝ, f(x) = k / x
def is_power_function (f : ℝ → ℝ) := ∃ k n : ℝ, ∀ x : ℝ, f(x) = k * x^n

-- The function to work with
def f (m : ℝ) (x : ℝ) := x^(m^2) - 2

-- Theorem statements (no proofs included, using 'sorry')

-- Direct proportion case
theorem direct_proportion_result (m : ℝ) : 
  is_direct_proportion_function (f m) ↔ m = 1 ∨ m = -1 :=
by sorry

-- Inverse proportion case
theorem inverse_proportion_result (m : ℝ) : 
  is_inverse_proportion_function (f m) ↔ m = -1 :=
by sorry

-- Power function case
theorem power_function_result (m : ℝ) : 
  is_power_function (f m) ↔ m = 2 :=
by sorry

end direct_proportion_result_inverse_proportion_result_power_function_result_l268_268282


namespace equation_of_plane_l268_268115

/--
The equation of the plane passing through the points (2, -2, 2) and (0, 0, 2),
and which is perpendicular to the plane 2x - y + 4z = 8, is given by:
Ax + By + Cz + D = 0 where A, B, C, D are integers such that A > 0 and gcd(|A|,|B|,|C|,|D|) = 1.
-/
theorem equation_of_plane :
  ∃ (A B C D : ℤ),
    A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
    (∀ x y z : ℤ, A * x + B * y + C * z + D = 0 ↔ x + y = 0) :=
sorry

end equation_of_plane_l268_268115


namespace minimum_A_in_interval_1_3_l268_268817

def A (x y : ℝ) := 
  (3 * x * y + x ^ 2) * (Real.sqrt (3 * x * y + x - 3 * y)) + 
  (3 * x * y + y ^ 2) * (Real.sqrt (3 * x * y + y - 3 * x)) / 
  (x ^ 2 * y + y ^ 2 * x)

theorem minimum_A_in_interval_1_3 : 
  ∀ (x y : ℝ), (1 ≤ x) → (x ≤ 3) → (1 ≤ y) → (y ≤ 3) → A x y = 4 := 
by 
  sorry

end minimum_A_in_interval_1_3_l268_268817


namespace exp_fixed_point_l268_268199

theorem exp_fixed_point (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : a^0 = 1 :=
by
  exact one_pow 0

end exp_fixed_point_l268_268199


namespace inequality_proof_l268_268560

variable {x y z : ℝ}

theorem inequality_proof (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  (xy / Real.sqrt (xy + yz) + yz / Real.sqrt (yz + zx) + zx / Real.sqrt (zx + xy)) ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end inequality_proof_l268_268560


namespace translation_lemma_l268_268580

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 4) - Real.pi / 6)

theorem translation_lemma : ∀ x : ℝ, g(x) = Real.sin (2 * x + Real.pi / 3):=
by
  intro x
  change Real.sin (2 * (x + Real.pi / 4) - Real.pi / 6) = Real.sin (2 * x + Real.pi / 3)
  sorry

end translation_lemma_l268_268580


namespace tangent_circumcircle_radius_eq_l268_268949

-- Define circumcircle condition
variable (O Q : Circle) (A B C : Point)

-- Given right triangle ABC with AB as the diameter of the circumcircle O
def right_triangle (A B C : Point) : Prop := IsRightTriangle A B C ∧ Diameter O A B

-- Define tangent to sides AC and BC
def tangent_to_sides (Q : Circle) (A C B : Point) : Prop :=
  IsTangent Q (Line A C) ∧ IsTangent Q (Line B C)

-- Q is tangent to O if and only if the radius of Q equals AC + BC - AB
theorem tangent_circumcircle_radius_eq
  (h1 : right_triangle A B C)
  (h2 : tangent_to_sides Q A C B)
  : IsTangent Q O ↔ Radius Q = Distance A C + Distance B C - Distance A B :=
sorry

end tangent_circumcircle_radius_eq_l268_268949


namespace ratio_of_books_l268_268296

theorem ratio_of_books (books_last_week : ℕ) (pages_per_book : ℕ) (pages_this_week : ℕ)
  (h_books_last_week : books_last_week = 5)
  (h_pages_per_book : pages_per_book = 300)
  (h_pages_this_week : pages_this_week = 4500) :
  (pages_this_week / pages_per_book) / books_last_week = 3 := by
  sorry

end ratio_of_books_l268_268296


namespace f_neg_9_over_2_f_in_7_8_l268_268169

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (x + 1) else sorry

theorem f_neg_9_over_2 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) : 
  f (-9 / 2) = -1 / 3 :=
by
  have h_period : f (-9 / 2) = f (-1 / 2) := by
    sorry  -- Using periodicity property
  have h_odd1 : f (-1 / 2) = -f (1 / 2) := by
    sorry  -- Using odd function property
  have h_def : f (1 / 2) = 1 / 3 := by
    sorry  -- Using the definition of f(x) for x in [0, 1)
  rw [h_period, h_odd1, h_def]
  norm_num

theorem f_in_7_8 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  ∀ x : ℝ, (7 < x ∧ x ≤ 8) → f x = - (x - 8) / (x - 9) :=
by
  intro x hx
  have h_period : f x = f (x - 8) := by
    sorry  -- Using periodicity property
  sorry  -- Apply the negative intervals and substitution to achieve the final form

end f_neg_9_over_2_f_in_7_8_l268_268169


namespace shaded_area_of_intersecting_circles_l268_268493

theorem shaded_area_of_intersecting_circles :
  let r := 5
  let quarter_circle_area := (π * r ^ 2) / 4
  let triangle_area := (1 / 2) * r * r
  let single_shaded_area := quarter_circle_area - triangle_area
  in
  4 * single_shaded_area = 25 * π - 50 :=
by
  sorry

end shaded_area_of_intersecting_circles_l268_268493


namespace total_number_of_fleas_l268_268550

theorem total_number_of_fleas :
  let G_fleas := 10
  let O_fleas := G_fleas / 2
  let M_fleas := 5 * O_fleas
  G_fleas + O_fleas + M_fleas = 40 := rfl

end total_number_of_fleas_l268_268550


namespace analyze_properties_l268_268710

noncomputable def eq_condition (x a : ℝ) : Prop :=
x ≠ 0 ∧ a = (x - 1) / (x^2)

noncomputable def first_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x = 1

noncomputable def second_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x > 1

noncomputable def third_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x < 1

theorem analyze_properties (x a : ℝ) (h1 : eq_condition x a):
(first_condition x a) ∧ ¬(second_condition x a) ∧ ¬(third_condition x a) :=
by
  sorry

end analyze_properties_l268_268710


namespace senior_visit_frequency_l268_268071

-- Definitions based on the conditions
def days_per_year := 365
def days_per_2_years := 2 * days_per_year
def junior_frequency := 12
def junior_visits := days_per_2_years / junior_frequency
def visit_ratio := 1.33

-- Theorem statement
theorem senior_visit_frequency :
  let S := days_per_2_years / (junior_visits / visit_ratio) in
  S ≈ 16 :=
by
  sorry

end senior_visit_frequency_l268_268071


namespace log2_sufficient_condition_l268_268822

theorem log2_sufficient_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  log 2 a > log 2 b → 2^a > 2^b :=
by
  sorry

end log2_sufficient_condition_l268_268822


namespace ratio_empty_to_occupied_space_in_package_l268_268870

noncomputable def volume_ratio_sphere_in_cylinder (V_cylinder : ℝ) : ℝ :=
  (2 / 3) * V_cylinder  -- Volume occupied by the sphere in the cylinder

theorem ratio_empty_to_occupied_space_in_package :
  ∀ (V_cylinder : ℝ), V_cylinder > 0 →
  let V_sphere := volume_ratio_sphere_in_cylinder V_cylinder
  let occupied_space := 5 * V_sphere
  let empty_space := 5 * (V_cylinder - V_sphere)
  empty_space / occupied_space = 1 / 2 :=
by
  intros V_cylinder V_cylinder_positive
  let V_sphere := volume_ratio_sphere_in_cylinder V_cylinder
  let occupied_space := 5 * V_sphere
  let empty_space := 5 * (V_cylinder - V_sphere)
  rw [volume_ratio_sphere_in_cylinder] at V_sphere
  have h : empty_space = 5 * (V_cylinder - (2 / 3) * V_cylinder),
  {
    apply rfl
  },
  rw h,
  have occupied_eq : occupied_space = 5 * ((2 / 3) * V_cylinder),
  {
    apply rfl
  },
  rw occupied_eq,
  rw [←mul_div_assoc, div_self],
  {
    apply rfl
  },
  {
    norm_num,
  }

end ratio_empty_to_occupied_space_in_package_l268_268870


namespace book_distribution_l268_268837

theorem book_distribution (n : ℕ) (h₀ : n = 8) : 
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ 6 ∧ k = n - i ∧ 2 ≤ i ∧ i ≤ 6) → card {k : ℕ | 2 ≤ k ∧ k ≤ 6} = 5 := 
by 
  sorry

end book_distribution_l268_268837


namespace barrels_filled_in_a_day_l268_268051

theorem barrels_filled_in_a_day
    (min_per_barrel : ℕ := 24)
    (hours_per_day : ℕ := 24)
    (min_per_hour : ℕ := 60) :
    ((hours_per_day * min_per_hour) / min_per_barrel) = 60 :=
by
  let total_minutes := hours_per_day * min_per_hour
  let barrels := total_minutes / min_per_barrel
  show barrels = 60 from sorry

end barrels_filled_in_a_day_l268_268051


namespace license_plate_combinations_l268_268975

theorem license_plate_combinations : 20 * 6 * 6 * 20 * 6 * 10 = 403200 :=
by
  calc
    20 * 6 * 6 * 20 * 6 * 10 = 20 * 20 * 6^3 * 10 : by ring
                        ... = 20 * 20 * 216 * 10  : by norm_num
                        ... = 20 * 20 * 2160      : by ring
                        ... = 400 * 2160          : by ring
                        ... = 403200              : by norm_num

end license_plate_combinations_l268_268975


namespace angle_relationship_l268_268946

theorem angle_relationship (α β : ℝ) 
  (h1 : 0 < α ∧ α < 2 * β ∧ 2 * β ≤ π / 2) 
  (h2 : 2 * cos (α + β) * cos β = -1 + 2 * sin (α + β) * sin β) : 
  α + 2 * β = 2 * π / 3 :=
by
  sorry

end angle_relationship_l268_268946


namespace count_valid_permutated_multiples_of_13_l268_268189

def is_permutation (n m : ℕ) : Prop :=
  let digits := List.digitCharList ∘ Nat.toDigits 10
  digits n ~ digits m

def is_valid_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def has_valid_permutation (n : ℕ) : Prop :=
  ∃ m, is_permutation n m ∧ is_valid_integer m ∧ m % 13 = 0

theorem count_valid_permutated_multiples_of_13 
  : {n : ℕ | has_valid_permutation n}.to_finset.card = 9980 := 
sorry

end count_valid_permutated_multiples_of_13_l268_268189


namespace gcd_calculation_l268_268882

theorem gcd_calculation :
  let a := 97^7 + 1
  let b := 97^7 + 97^3 + 1
  gcd a b = 1 := by
  sorry

end gcd_calculation_l268_268882


namespace line_perpendicular_passing_point_l268_268340

theorem line_perpendicular_passing_point :
  ∃ m : ℝ, ∀ x y : ℝ, (3 * x + 2 * y + m = 0 ↔ 
                       (x, y) = (-1, 2) ∧ 
                       (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ 2 * a - 3 * b + 4 = 0 ∧ 
                       (3 * a + 2 * b = 0))) :=
by {
  use -1,
  intros x y,
  split,
  {
    intro h1,
    split,
    {
      have : 3 * (-1 : ℝ) + 2 * (2 : ℝ) - 1 = (0 : ℝ), by ring,
      exact this,
    },
    {
      use [2, -3, 4],
      ring,
    }
  },
  {
    rintros ⟨hxy, ⟨a, b, c, habc, hab⟩⟩,
    rw [hxy.fst, hxy.snd] at habc,
    exact habc,
  }
}

end line_perpendicular_passing_point_l268_268340


namespace managers_in_sample_l268_268036

-- Definitions based on the conditions
def total_employees : ℕ := 160
def number_salespeople : ℕ := 104
def number_managers : ℕ := 32
def number_logistics : ℕ := 24
def sample_size : ℕ := 20

-- Theorem statement
theorem managers_in_sample : (number_managers * sample_size) / total_employees = 4 := by
  -- Proof omitted, as per the instructions
  sorry

end managers_in_sample_l268_268036


namespace thirteen_tickets_will_win_twelve_tickets_can_all_lose_l268_268854

-- Definitions for the problem's conditions and setup
def Ticket := Finset ℕ
def Draw (draws : Finset ℕ) (ticket : Ticket) : Prop :=
  ∀ n ∈ ticket, n ∉ draws

-- The main theorems
theorem thirteen_tickets_will_win :
  ∃ (tickets : Fin (13) → Ticket), (∀ t, Draw (draws := inserts 10 Finset.empty) (tickets t)) :=
sorry

theorem twelve_tickets_can_all_lose :
  ∃ (tickets : Fin (12) → Ticket), (∀ t, Draw (draws := inserts 10 Finset.empty) (tickets t)) :=
sorry

end thirteen_tickets_will_win_twelve_tickets_can_all_lose_l268_268854


namespace speed_of_man_in_still_water_l268_268843

variable (V_m : ℝ)
variable (speed_of_stream : ℝ := 8)
variable (distance_downstream : ℝ := 90)
variable (time_downstream : ℝ := 5)

theorem speed_of_man_in_still_water :
  V_m + speed_of_stream = distance_downstream / time_downstream → V_m = 10 :=
by
  intros h,
  linarith,
  sorry

end speed_of_man_in_still_water_l268_268843


namespace ricotta_comparision_l268_268902

-- Define the initial conditions
def dough_length := 16 -- cm
def dough_width := 12 -- cm
def overlap := 2 -- cm
def ricotta_per_cylinder := 500 -- grams

-- Define the initial configuration
def initial_perimeter := dough_length - overlap -- cm
def initial_radius := initial_perimeter / (2 * Real.pi) -- cm
def initial_height := dough_width -- cm
def initial_volume := Real.pi * (initial_radius^2) * initial_height -- cm^3

-- Define the new configuration
def new_perimeter := dough_width - overlap -- cm
def new_radius := new_perimeter / (2 * Real.pi) -- cm
def new_height := dough_length -- cm
def new_volume := Real.pi * (new_radius^2) * new_height -- cm^3

-- Define the proof goal
theorem ricotta_comparision :
  let initial_ricotta := initial_volume * ricotta_per_cylinder / initial_volume in
  let new_ricotta := new_volume * ricotta_per_cylinder / initial_volume in
  (new_ricotta - initial_ricotta) = 235 := by
  sorry

end ricotta_comparision_l268_268902


namespace jerry_total_earnings_duration_l268_268660

theorem jerry_total_earnings_duration:
  let mowing_income := 14
  let weed_eating_income := 31
  let bush_trimming_income := 20
  let snack_expense := 5
  let transport_expense := 10
  let weekly_savings := 8
  let total_earnings := mowing_income + weed_eating_income + bush_trimming_income
  let total_expenses := snack_expense + transport_expense + weekly_savings
  let duration := total_earnings / total_expenses
  ⌊duration⌋ = 2 :=
by
  sorry

end jerry_total_earnings_duration_l268_268660


namespace ab_inequality_l268_268567

theorem ab_inequality (p q a b : ℚ) (hpq_pos: p > 0) (hq_pos: q > 0) (ha_pos: a > 0) (hb_pos: b > 0)
  (hpq_inv_sum: 1/p + 1/q = 1) : a * b ≤ a^p / p + b^q / q :=
sorry

end ab_inequality_l268_268567


namespace reciprocal_of_negative_2023_l268_268757

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l268_268757


namespace buckingham_palace_visitors_l268_268861

theorem buckingham_palace_visitors : 
  let previous_day_visitors := 295
  let additional_visitors := 22
  let visitors_on_day := previous_day_visitors + additional_visitors
  in visitors_on_day = 317 :=
by
  let previous_day_visitors := 295
  let additional_visitors := 22
  let visitors_on_day := previous_day_visitors + additional_visitors
  show visitors_on_day = 317
  sorry

end buckingham_palace_visitors_l268_268861


namespace laborers_percentage_l268_268225

noncomputable def percentage_showed_up (total_laborers present_laborers : ℕ) : ℚ :=
  (present_laborers / total_laborers.toRat) * 100

theorem laborers_percentage :
  percentage_showed_up 56 30 = (53.6 : ℚ) :=
by
  sorry

end laborers_percentage_l268_268225


namespace cube_right_angled_triangles_l268_268921

theorem cube_right_angled_triangles : 
  let vertices := {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)} 
  count_right_angled_triangles vertices = 48 :=
by
  sorry

noncomputable def count_right_angled_triangles (vertices : set (ℕ × ℕ × ℕ)) : ℕ :=
  sorry

end cube_right_angled_triangles_l268_268921


namespace ordered_subset_pairs_count_l268_268565

noncomputable def U : Set ℕ := {1, 2, 3, 4}

def isOrderedSubsetPair (A B : Set ℕ) : Prop :=
  A ⊆ U ∧ B ⊆ U ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ max' A (nonempty_of_mem (Set.mem_of_subset (Set.subset.univ A))) < min' B (nonempty_of_mem (Set.mem_of_subset (Set.subset.univ B)))

def numberOfOrderedSubsetPairs (U : Set ℕ) : Nat :=
  (Finset.powerset U).card

theorem ordered_subset_pairs_count : (numberOfOrderedSubsetPairs U) = 17 := sorry

end ordered_subset_pairs_count_l268_268565


namespace ratio_of_geometric_sequence_l268_268725

theorem ratio_of_geometric_sequence (a x b : ℝ) (h : a, x, b, 2 * x ∈ geometric_sequence) : 
  a / b = 1 / 4 :=
sorry

end ratio_of_geometric_sequence_l268_268725


namespace cars_in_parking_lot_l268_268365

theorem cars_in_parking_lot (initial_cars left_cars entered_cars : ℕ) (h1 : initial_cars = 80)
(h2 : left_cars = 13) (h3 : entered_cars = left_cars + 5) : 
initial_cars - left_cars + entered_cars = 85 :=
by
  rw [h1, h2, h3]
  sorry

end cars_in_parking_lot_l268_268365


namespace largest_divisor_of_difference_power_composite_l268_268546

theorem largest_divisor_of_difference_power_composite (n : ℕ) (h : ¬prime n ∧ 1 < n) : 6 ∣ (n^5 - n) :=
sorry

end largest_divisor_of_difference_power_composite_l268_268546


namespace sum_neg_two_powers_l268_268874

def neg_two_pow (n : ℤ) : ℝ :=
  if n = 0 then 1
  else if n > 0 then if even n then 2^n else -2^n
  else if even n then 1 / (2 ^ (-n)) else -1 / (2 ^ (-n))

theorem sum_neg_two_powers : (∑ i in Finset.Icc (-10 : ℤ) 10, neg_two_pow i) = 0 :=
by
  sorry

end sum_neg_two_powers_l268_268874


namespace basketball_team_selection_l268_268830

theorem basketball_team_selection :
  (Nat.choose 4 2) * (Nat.choose 14 6) = 18018 := 
by
  -- number of ways to choose 2 out of 4 quadruplets
  -- number of ways to choose 6 out of the remaining 14 players
  -- the product of these combinations equals the required number of ways
  sorry

end basketball_team_selection_l268_268830


namespace arnold_danny_age_l268_268397

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 9 → x = 4 :=
by
  intro h
  sorry

end arnold_danny_age_l268_268397


namespace happy_valley_kennel_arrangements_l268_268714

theorem happy_valley_kennel_arrangements :
  let chickens := 4
  let dogs := 3
  let cats := 5
  let all_chickens_before_cats := True
  let total_arrangements := 2 * (Nat.factorial chickens) * (Nat.factorial dogs) * (Nat.factorial cats)
  in all_chickens_before_cats = True -> total_arrangements = 34560 :=
by
  intros
  exact sorry

end happy_valley_kennel_arrangements_l268_268714


namespace functional_equation_solution_l268_268893

-- Define the functional equation as a condition
def fun_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

-- Define the proof problem in Lean 4
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, fun_eq f →
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
begin
  sorry
end

end functional_equation_solution_l268_268893


namespace inscribed_circle_area_l268_268370

-- Given the conditions:
-- Two circles with radii R and r are externally tangent.
-- An external common tangent is drawn to these circles.
-- A circle is inscribed in the curvilinear triangle formed by the two given circles and their external common tangent.
-- Prove that the area of this inscribed circle is given by the formula.

namespace CurvilinearTriangle

variables (R r : ℝ) (hR : 0 < R) (hr : 0 < r)

-- Statement of the theorem.
theorem inscribed_circle_area :
  ∀ (tangent_circle_area : ℝ),
    let x := (√(R * r) / (√R + √r)) in
    tangent_circle_area = π * x^2 :=
begin
  sorry
end

end CurvilinearTriangle

end inscribed_circle_area_l268_268370


namespace minimum_pool_cost_l268_268093

open Real

-- Define the variables and conditions
def volume (x y : ℝ) : ℝ := x * y * 2
def bottom_cost (x y : ℝ) : ℝ := 200 * x * y
def walls_cost (x y : ℝ) : ℝ := 150 * 4 * (x + y)
def total_cost (x y : ℝ) : ℝ := bottom_cost x y + walls_cost x y

-- Define the function to minimize
def cost_to_minimize (x : ℝ) : ℝ := total_cost x (9 / x)

theorem minimum_pool_cost : ∃ x y : ℝ, volume x y = 18 ∧ total_cost x y = 5400 := by
  sorry

end minimum_pool_cost_l268_268093


namespace solve_x_l268_268555

def δ (x : ℝ) : ℝ := 4 * x + 6
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_x : ∃ x: ℝ, δ (φ x) = 3 → x = -19 / 20 := by
  sorry

end solve_x_l268_268555


namespace count_valid_license_plates_l268_268331

def IñupiatLetters : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']

def validLicensePlatesCount : Nat :=
  let initialChoices := 3 -- B, C, or D
  let lastChoice := 1     -- F
  let allChoices := 21
  let excludedChoices := 2 -- Excluding F and S
  let remainingChoices := allChoices - excludedChoices - 1 -- Excluding the first picked letter
  initialChoices * remainingChoices * (remainingChoices - 1) * (remainingChoices - 2) * (remainingChoices - 3)

theorem count_valid_license_plates : validLicensePlatesCount = 100800 :=
  by
    sorry

end count_valid_license_plates_l268_268331


namespace min_possible_value_l268_268992

theorem min_possible_value (a b : ℤ) :
  (∃ a b : ℤ, (ax + b) * (bx + a) = 10 * x^2 + (\Box : ℤ) * x + 10) ∧ (a ≠ b ∧ b ≠ \Box ∧ a ≠ \Box) →
  ∃ \Box : ℤ, \Box = 29 := by
  sorry

end min_possible_value_l268_268992


namespace train_speeds_l268_268794

theorem train_speeds (v : ℝ) (train_length : ℝ) (cross_time : ℝ) :
  train_length = 200 →
  cross_time = 6 →
  v = (400 / cross_time) / 2 →
  v * 3.6 = 120 :=
by
  intros h_length h_time h_v
  rw [h_length, h_time, h_v]
  norm_num
  sorry

end train_speeds_l268_268794


namespace find_a4_l268_268245

noncomputable def a (n : ℕ) : ℕ := sorry -- Define the arithmetic sequence
def S (n : ℕ) : ℕ := sorry -- Define the sum function for the sequence

theorem find_a4 (h1 : S 5 = 25) (h2 : a 2 = 3) : a 4 = 7 := by
  sorry

end find_a4_l268_268245


namespace profit_percent_l268_268015

theorem profit_percent (marked_price : ℝ) (num_bought : ℝ) (num_payed_price : ℝ) (discount_percent : ℝ) : 
  num_bought = 56 → 
  num_payed_price = 46 → 
  discount_percent = 0.01 →
  marked_price = 1 →
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 20.52 :=
by 
  intro hnum_bought hnum_payed_price hdiscount_percent hmarked_price 
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  sorry

end profit_percent_l268_268015


namespace shortest_side_second_triangle_l268_268048

def first_triangle_leg1 : ℕ := 24
def first_triangle_hypotenuse : ℕ := 40
def second_triangle_hypotenuse : ℕ := 120

theorem shortest_side_second_triangle :
  ∃ (s : ℕ), s = 72 :=
by
  have first_triangle_leg2_square := first_triangle_hypotenuse ^ 2 - first_triangle_leg1 ^ 2
  have first_triangle_leg2 := Nat.sqrt first_triangle_leg2_square
  have scale_factor := second_triangle_hypotenuse / first_triangle_hypotenuse
  have shortest_side_second_triangle := scale_factor * first_triangle_leg1
  exists shortest_side_second_triangle
  rw [eq_comm]
  rfl

end shortest_side_second_triangle_l268_268048


namespace cricketer_boundaries_l268_268438

theorem cricketer_boundaries (total_runs : ℕ) (sixes : ℕ) (percent_runs_by_running : ℝ)
  (h1 : total_runs = 152)
  (h2 : sixes = 2)
  (h3 : percent_runs_by_running = 60.526315789473685) :
  let runs_by_running := round (total_runs * percent_runs_by_running / 100)
  let runs_from_sixes := sixes * 6
  let runs_from_boundaries := total_runs - runs_by_running - runs_from_sixes
  let boundaries := runs_from_boundaries / 4
  boundaries = 12 :=
by
  sorry

end cricketer_boundaries_l268_268438


namespace barber_loss_is_25_l268_268394

-- Definition of conditions
structure BarberScenario where
  haircut_cost : ℕ
  counterfeit_bill : ℕ
  real_change : ℕ
  change_given : ℕ
  real_bill_given : ℕ

def barberScenario_example : BarberScenario :=
  { haircut_cost := 15,
    counterfeit_bill := 20,
    real_change := 20,
    change_given := 5,
    real_bill_given := 20 }

-- Lean 4 problem statement
theorem barber_loss_is_25 (b : BarberScenario) : 
  b.haircut_cost = 15 ∧
  b.counterfeit_bill = 20 ∧
  b.real_change = 20 ∧
  b.change_given = 5 ∧
  b.real_bill_given = 20 → (15 + 5 + 20 - 20 + 5 = 25) :=
by
  intro h
  cases' h with h1 h23
  sorry

end barber_loss_is_25_l268_268394


namespace solve_log_translation_problem_l268_268788

def log_translation_problem : Prop :=
  ∃ (a : ℤ × ℤ), 
    (∀ x : ℝ, (log (x-2) / log 2 + 3 - 4 = log ((x-3) / log 2 + 1) - 1)) ∧ 
    a = (-3, -4)

theorem solve_log_translation_problem : log_translation_problem :=
sorry

end solve_log_translation_problem_l268_268788


namespace find_first_number_l268_268210

theorem find_first_number (x : ℝ) (h1 : 2994 / x = 175) (h2 : 29.94 / 1.45 = 17.5) : x = 17.1 :=
by
  sorry

end find_first_number_l268_268210


namespace inequality_solution_l268_268327

theorem inequality_solution (m : ℝ) :
  ∃ S : set ℝ, 
    (m = 0 ∧ S = set.univ) ∨ 
    (m > 0 ∧ S = {x : ℝ | -3 / m < x ∧ x < 1 / m}) ∨ 
    (m < 0 ∧ S = {x : ℝ | 1 / m < x ∧ x < -3 / m}) ∧ 
    ∀ x : ℝ, x ∈ S ↔ m^2 * x^2 + 2 * m * x - 3 < 0 :=
by sorry

end inequality_solution_l268_268327


namespace joan_bought_eggs_l268_268662

def dozen_to_eggs (d : ℕ) : ℕ := d * 12

theorem joan_bought_eggs :
  let dozen_eggs := 6 in
  dozen_to_eggs dozen_eggs = 72 :=
by
  sorry

end joan_bought_eggs_l268_268662


namespace find_n_if_pow_eqn_l268_268599

theorem find_n_if_pow_eqn (n : ℕ) :
  6 ^ 3 = 9 ^ n → n = 3 :=
by 
  sorry

end find_n_if_pow_eqn_l268_268599


namespace average_visitors_in_30_day_month_l268_268833

def average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) : ℕ :=
    let sundays := days_in_month / 7 + if days_in_month % 7 > 0 then 1 else 0
    let other_days := days_in_month - sundays
    let total_visitors := sundays * visitors_sunday + other_days * visitors_other
    total_visitors / days_in_month

theorem average_visitors_in_30_day_month 
    (visitors_sunday : ℕ) (visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) (h1 : visitors_sunday = 660) (h2 : visitors_other = 240) (h3 : days_in_month = 30) :
    average_visitors_per_day visitors_sunday visitors_other days_in_month starts_on_sunday = 296 := 
by
  sorry

end average_visitors_in_30_day_month_l268_268833


namespace sin_double_angle_inequality_l268_268276

theorem sin_double_angle_inequality 
  (α β γ : ℝ) 
  (h₀ : 0 < α ∧ α < π/2) 
  (h₁ : 0 < β ∧ β < π/2) 
  (h₂ : 0 < γ ∧ γ < π/2) 
  (h₃ : α + β + γ = π) 
  (h₄ : α < β) 
  (h₅ : β < γ) 
  : sin (2 * α) > sin (2 * β) ∧ sin (2 * β) > sin (2 * γ) := 
by
  sorry

end sin_double_angle_inequality_l268_268276


namespace bus_possible_route_numbers_l268_268344

def digit_possibilities (digit : Nat) (burnt : Nat) : Set Nat :=
  match digit, burnt with
  | 3, 1 => {9}
  | 3, 2 => {8}
  | 5, 1 => {6, 9}
  | 5, 2 => {8}
  | 1, 1 => {7}
  | 1, 2 => {4}
  | _, _ => {digit}

def possible_route_numbers (known_digits : List Nat) : Set Nat :=
  known_digits.foldl
    (fun acc digit => acc ∪
      let burnt_out_1 := digit_possibilities digit 1
      let burnt_out_2 := digit_possibilities digit 2
      (burnt_out_1.image (fun d => d*10^known_digits.indexOf digit)) ∪
      (burnt_out_2.image (fun d => d*10^known_digits.indexOf digit)))
    {known_digits.head*10^2 + known_digits.nth! 1*10 + known_digits.nth! 2}

theorem bus_possible_route_numbers :
  let route := 351
  let digits := [3, 5, 1]
  possible_route_numbers digits = {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991} := by
  sorry

end bus_possible_route_numbers_l268_268344


namespace adam_coins_value_l268_268860

theorem adam_coins_value (num_coins : ℕ) (subset_value: ℕ) (subset_num: ℕ) (total_value: ℕ)
  (h1 : num_coins = 20)
  (h2 : subset_value = 16)
  (h3 : subset_num = 4)
  (h4 : total_value = num_coins * (subset_value / subset_num)) :
  total_value = 80 := 
by
  sorry

end adam_coins_value_l268_268860


namespace original_salary_l268_268345

variable (x : ℝ)
variable (current_salary : ℝ)

-- Define the conditions
def raised_salary := x * 1.10
def reduced_salary := raised_salary * 0.95
def final_salary := 6270

-- The proof statement
theorem original_salary :
  reduced_salary = final_salary → x = 6000 :=
by
  sorry

end original_salary_l268_268345


namespace reciprocal_of_neg_2023_l268_268739

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l268_268739


namespace student_arrangements_l268_268215

theorem student_arrangements (n : ℕ) (students : fin n) (A B : fin n → Prop) :
  n = 5 → (∀ s ∈ students, s ≠ A → s ≠ B) →
  ∑ x in {arrangement | ∀ i, (arrangement i = A → arrangement (i+1) ≠ B)
                          ∧ (arrangement i = B → arrangement (i+1) ≠ A)}, 1 = 72 :=
by {
  sorry
}

end student_arrangements_l268_268215


namespace line_equation_l268_268144

-- Defining the conditions
def parabola := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }
def midpoint (a b : ℝ × ℝ) : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

axiom vertex_at_origin (C : parabola) : C = ⟨(0, 0), by simp⟩
axiom focus_at_F (C : parabola) : ∃ F : ℝ × ℝ, F = (1, 0)

-- The main theorem: the equation of line l is y = x
theorem line_equation (A B : parabola) (l : ℝ × ℝ → Prop) :
  (midpoint A.1 B.1 = (2, 2)) → (l = λ p, p.2 = p.1) :=
by sorry

end line_equation_l268_268144


namespace matrix_self_inverse_l268_268503

theorem matrix_self_inverse (a b : ℝ) :
  (matrix_mul (matrix [[4, -2], [a, b]]) (matrix [[4, -2], [a, b]]) = matrix_id 2) ↔
  (a = 7.5 ∧ b = -4) :=
by
  sorry

end matrix_self_inverse_l268_268503


namespace assemble_shapes_l268_268354

-- Define the 4x5 rectangular plate and its partition into seven parts
structure Rectangle :=
(length : ℕ)
(width : ℕ)
parts : list (list ℕ) -- This is an abstract representation of the 7 parts

-- Example specific condition, the exact definition is abstracted for brevity
def is_partitioned (r : Rectangle) : Prop := 
  r.length = 4 ∧ r.width = 5 ∧ length r.parts = 7

-- Define the target shapes
structure Shape :=
(parts : list (list ℕ))

-- Example specific shape condition, the exact definition is abstracted for brevity
def is_valid_shape (s : Shape) (r : Rectangle) : Prop := 
  -- This is a placeholder condition representing that the shape can be formed from r's parts
  s.parts = r.parts  -- Simplified, real condition would be more complex logic

-- The proof problem
theorem assemble_shapes (r : Rectangle) (shapes : list Shape) : Prop :=
  is_partitioned r → all_shapes_valid : ∀ s ∈ shapes, is_valid_shape s r

end assemble_shapes_l268_268354


namespace circle_center_radius_l268_268718

theorem circle_center_radius {x y : ℝ} :
  (∃ r : ℝ, (x - 1)^2 + y^2 = r^2) ↔ (x^2 + y^2 - 2*x - 5 = 0) :=
by sorry

end circle_center_radius_l268_268718


namespace bowling_ball_weight_l268_268298

variables (b c k : ℝ)

def condition1 : Prop := 9 * b = 6 * c
def condition2 : Prop := c + k = 42
def condition3 : Prop := 3 * k = 2 * c

theorem bowling_ball_weight
  (h1 : condition1 b c)
  (h2 : condition2 c k)
  (h3 : condition3 c k) :
  b = 16.8 :=
sorry

end bowling_ball_weight_l268_268298


namespace min_value_of_A_l268_268818

noncomputable def A (x y : ℝ) : ℝ := 
  (3 * x * y + x^2) * sqrt (3 * x * y + x - 3 * y) + (3 * x * y + y^2) * sqrt (3 * x * y + y - 3 * x) / (x^2 * y + y^2 * x)

theorem min_value_of_A (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 3) : 
  4 ≤ A x y :=
sorry

end min_value_of_A_l268_268818


namespace students_without_A_l268_268237

theorem students_without_A (total_students : ℕ) (students_english : ℕ) 
  (students_math : ℕ) (students_both : ℕ) (students_only_math : ℕ) :
  total_students = 30 → students_english = 6 → students_math = 15 → 
  students_both = 3 → students_only_math = 1 →
  (total_students - (students_math - students_only_math + 
                     students_english - students_both + 
                     students_both) = 12) :=
by sorry

end students_without_A_l268_268237


namespace joan_dimes_spent_l268_268663

theorem joan_dimes_spent (initial_dimes remaining_dimes spent_dimes : ℕ) 
    (h_initial: initial_dimes = 5) 
    (h_remaining: remaining_dimes = 3) : 
    spent_dimes = initial_dimes - remaining_dimes := 
by 
    sorry

end joan_dimes_spent_l268_268663


namespace alissa_total_amount_spent_correct_l268_268464
-- Import necessary Lean library

-- Define the costs of individual items
def football_cost : ℝ := 8.25
def marbles_cost : ℝ := 6.59
def puzzle_cost : ℝ := 12.10
def action_figure_cost : ℝ := 15.29
def board_game_cost : ℝ := 23.47

-- Define the discount rate and the sales tax rate
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.06

-- Define the total cost before discount
def total_cost_before_discount : ℝ :=
  football_cost + marbles_cost + puzzle_cost + action_figure_cost + board_game_cost

-- Define the discount amount
def discount : ℝ := total_cost_before_discount * discount_rate

-- Define the total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount

-- Define the sales tax amount
def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_after_discount + sales_tax

-- Prove that the total amount spent is $62.68
theorem alissa_total_amount_spent_correct : total_amount_spent = 62.68 := 
  by 
    sorry

end alissa_total_amount_spent_correct_l268_268464


namespace minimum_n_divisible_20_l268_268676

theorem minimum_n_divisible_20 :
  ∃ (n : ℕ), (∀ (l : List ℕ), l.length = n → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0) ∧ 
  (∀ m, m < n → ¬(∀ (l : List ℕ), l.length = m → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0)) := 
⟨9, 
  by sorry, 
  by sorry⟩

end minimum_n_divisible_20_l268_268676


namespace rowing_speed_in_still_water_l268_268045

theorem rowing_speed_in_still_water (v c : ℝ) (t : ℝ) (h1 : c = 1.1) (h2 : (v + c) * t = (v - c) * 2 * t) : v = 3.3 :=
sorry

end rowing_speed_in_still_water_l268_268045


namespace train_length_is_100_meters_l268_268608
-- Import the necessary library for mathematical operations

-- Define the quantities given in the problem
def train_speed_kmph : ℕ := 90
def time_seconds : ℕ := 4

-- Conversion factor from kilometers per hour to meters per second
def kmph_to_mps (s_kmph : ℕ) : ℝ := (s_kmph * 1000) / 3600

-- Calculate the speed of the train in meters per second
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Define the length of the train as the distance it covers in given time
def length_of_train : ℝ := train_speed_mps * time_seconds

-- The theorem to prove
theorem train_length_is_100_meters : length_of_train = 100 := 
by
  sorry

end train_length_is_100_meters_l268_268608


namespace find_hyperbola_eq_l268_268443

noncomputable def hyperbola : Type := { C : Set (ℝ × ℝ) // ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ C = { p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 } }

def ellipse_foci (E : Set (ℝ × ℝ)) (f : ℝ × ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ E = { p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 }
  ∧ f((-2, 0)) ∧ f((2, 0))

def is_asymptote (C : Set (ℝ × ℝ)) (line : ℝ → ℝ) : Prop :=
  ∃ b a : ℝ, b / a = line(1) / 1 ∧ a > 0 ∧ b > 0 ∧ C = { p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

def expected_hyperbola_eq (C : Set (ℝ × ℝ)) : Prop :=
  C = { p : ℝ × ℝ | p.1^2 - (p.2^2 / 3) = 1 }

theorem find_hyperbola_eq (C : Set (ℝ × ℝ))
  (hc : hyperbola C)
  (he : ellipse_foci { p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1 } (λ x, x ∈ C))
  (ha : is_asymptote C (λ x, sqrt 3 * x)) :
  expected_hyperbola_eq C :=
sorry

end find_hyperbola_eq_l268_268443


namespace part_a_39x55_5x11_l268_268410

theorem part_a_39x55_5x11 :
  ¬ (∃ (a1 a2 b1 b2 : ℕ), 
    39 = 5 * a1 + 11 * b1 ∧ 
    55 = 5 * a2 + 11 * b2) := 
  by sorry

end part_a_39x55_5x11_l268_268410


namespace rectangle_division_impossible_l268_268877

theorem rectangle_division_impossible :
  ¬ ∃ n m : ℕ, n * 5 = 55 ∧ m * 11 = 39 :=
by
  sorry

end rectangle_division_impossible_l268_268877


namespace triangles_not_right_angled_count_l268_268708

-- Definition of the problem
noncomputable def non_right_angled_triangles_count : ℕ := 4

-- Definitions of Points as per coordinates given in the grid
structure Point where
  x : ℕ
  y : ℕ

-- Declaring the points as specific grid coordinates
def A : Point := ⟨0, 2⟩
def B : Point := ⟨1, 2⟩
def C : Point := ⟨2, 2⟩
def D : Point := ⟨0, 0⟩
def E : Point := ⟨1, 0⟩
def F : Point := ⟨2, 0⟩

-- The statement we want to prove.
theorem triangles_not_right_angled_count :
  ∃ t: ℕ, t = non_right_angled_triangles_count ∧ t = 4 :=
begin
  sorry -- Proof goes here
end

end triangles_not_right_angled_count_l268_268708


namespace candy_ratio_l268_268392

theorem candy_ratio
  (red_candies : ℕ)
  (yellow_candies : ℕ)
  (blue_candies : ℕ)
  (total_candies : ℕ)
  (remaining_candies : ℕ)
  (h1 : red_candies = 40)
  (h2 : yellow_candies = 3 * red_candies - 20)
  (h3 : remaining_candies = 90)
  (h4 : total_candies = remaining_candies + yellow_candies)
  (h5 : blue_candies = total_candies - red_candies - yellow_candies) :
  blue_candies / yellow_candies = 1 / 2 :=
sorry

end candy_ratio_l268_268392


namespace area_inside_C_outside_A_and_B_l268_268082

-- Defining the radii and their relationships
def radius_A : ℝ := 1
def radius_B : ℝ := 1
def radius_C : ℝ := 2

-- Defining the centers of circles A and B as the points which are tangent
def center_A : ℝ × ℝ := (0, 0)
def center_B : ℝ × ℝ := (2, 0) -- Assuming they are horizontally aligned for simplicity

-- Midpoint of line segment AB
def midpoint_M : ℝ × ℝ := ((center_A.1 + center_B.1) / 2, (center_A.2 + center_B.2) / 2)

-- Center of circle C based on the midpoint
def center_C : ℝ × ℝ := (midpoint_M.1, midpoint_M.2 + radius_C)

-- Function to calculate the area of a circle given its radius
def area (r : ℝ) : ℝ := π * r * r

-- Statement of the theorem
theorem area_inside_C_outside_A_and_B :
  area radius_C - 2 * area radius_A = 3 * π / 2 :=
sorry

end area_inside_C_outside_A_and_B_l268_268082


namespace maximum_sum_areas_of_triangles_l268_268029

-- Define a circle with radius 1

def circle_radius_1_area : ℝ := Real.pi

-- Define the number of triangles
def num_triangles : ℕ := 2014

-- Define a predicate for non-overlapping triangles contained in the circle
def non_overlapping (triangles : List ℝ) : Prop :=
  ∀ i j, i ≠ j → triangles.nth i ∩ triangles.nth j  = ∅

-- Define the area of the sum of triangle areas bounded by circle area
def sum_triangle_areas (triangles : List ℝ) : ℝ :=
  triangles.sum

-- The main statement: the sum of the areas of 2014 non-overlapping triangles contained within
-- a circle of radius 1 is at most π.
theorem maximum_sum_areas_of_triangles (triangles : List ℝ) :
  length triangles = num_triangles →
  non_overlapping triangles →
  (∀ t ∈ triangles, t > 0) →
  sum_triangle_areas triangles ≤ circle_radius_1_area := 
sorry

end maximum_sum_areas_of_triangles_l268_268029


namespace necessary_but_not_sufficient_l268_268408

theorem necessary_but_not_sufficient (x : ℝ) : (x > 1 → x > 2) = (false) ∧ (x > 2 → x > 1) = (true) := by
  sorry

end necessary_but_not_sufficient_l268_268408


namespace floor_sqrt_26_squared_l268_268513

theorem floor_sqrt_26_squared :
  (⟨5 < Real.sqrt 26, Real.sqrt 26 < 6⟩) ∧ (Real.sqrt 25 = 5) ∧ (Real.sqrt 36 = 6) → ⌊Real.sqrt 26⌋^2 = 25 := 
by
  introduction
  sorry

end floor_sqrt_26_squared_l268_268513


namespace marble_problem_l268_268444

theorem marble_problem
  (h1 : ∀ x : ℕ, x > 0 → (x + 2) * ((220 / x) - 1) = 220) :
  ∃ x : ℕ, x > 0 ∧ (x + 2) * ((220 / ↑x) - 1) = 220 ∧ x = 20 :=
by
  sorry

end marble_problem_l268_268444


namespace ratio_of_speeds_is_2_l268_268293

-- Definitions based on conditions
def rate_of_machine_B : ℕ := 100 / 40 -- Rate of Machine B (parts per minute)
def rate_of_machine_A : ℕ := 50 / 10 -- Rate of Machine A (parts per minute)
def ratio_of_speeds (rate_A rate_B : ℕ) : ℕ := rate_A / rate_B -- Ratio of speeds

-- Proof statement
theorem ratio_of_speeds_is_2 : ratio_of_speeds rate_of_machine_A rate_of_machine_B = 2 := by
  sorry

end ratio_of_speeds_is_2_l268_268293


namespace cost_price_of_article_l268_268016

theorem cost_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) 
    (h1 : SP = 100) 
    (h2 : profit_percent = 0.20) 
    (h3 : SP = CP * (1 + profit_percent)) : 
    CP = 83.33 :=
by
  sorry

end cost_price_of_article_l268_268016


namespace expected_number_of_matches_variance_of_number_of_matches_l268_268416

-- Defining the conditions first, and then posing the proof statements
namespace MatchingPairs

open ProbabilityTheory

-- Probabilistic setup for indicator variables
variable (N : ℕ) (prob : ℝ := 1 / N)

-- Indicator variable Ik representing matches
@[simp] def I (k : ℕ) : ℝ := if k < N then prob else 0

-- Define the sum of expected matches S
@[simp] def S : ℝ := ∑ k in finset.range N, I N k

-- Statement: The expectation of the number of matching pairs is 1
theorem expected_number_of_matches : E[S] = 1 := sorry

-- Statement: The variance of the number of matching pairs is 1
theorem variance_of_number_of_matches : Var S = 1 := sorry

end MatchingPairs

end expected_number_of_matches_variance_of_number_of_matches_l268_268416


namespace final_price_is_correct_l268_268446

def initial_price : ℝ := 15
def first_discount_rate : ℝ := 0.2
def second_discount_rate : ℝ := 0.25

def first_discount : ℝ := initial_price * first_discount_rate
def price_after_first_discount : ℝ := initial_price - first_discount

def second_discount : ℝ := price_after_first_discount * second_discount_rate
def final_price : ℝ := price_after_first_discount - second_discount

theorem final_price_is_correct :
  final_price = 9 :=
by
  -- The actual proof steps will go here.
  sorry

end final_price_is_correct_l268_268446


namespace exists_three_numbers_in_group_for_ratio_l268_268683

theorem exists_three_numbers_in_group_for_ratio (k : ℝ) (hk : 0 < k) 
  (G1 G2 : set ℝ) (h_union : ∀ x : ℝ, x ∈ G1 ∨ x ∈ G2)
  (h_disjoint : ∀ x : ℝ, ¬ (x ∈ G1 ∧ x ∈ G2)) :
  ∃ a b c ∈ G1 ∪ G2, a < b ∧ b < c ∧ (c - b) / (b - a) = k :=
sorry

end exists_three_numbers_in_group_for_ratio_l268_268683


namespace area_and_cost_of_path_l268_268451

-- Define the conditions
def length_rect_field: ℝ := 75
def width_rect_field: ℝ := 55
def radius_circ_section: ℝ := 20
def path_width: ℝ := 5
def cost_per_sqm: ℝ := 20

-- Prove the areas and cost
theorem area_and_cost_of_path:
  let area_rect_field := length_rect_field * width_rect_field in
  let area_circ_section := Real.pi * radius_circ_section^2 in
  let total_area_field_section := area_rect_field + area_circ_section in

  let length_larger_rect := length_rect_field + 2*path_width in
  let width_larger_rect := width_rect_field + 2*path_width in
  let area_larger_rect := length_larger_rect * width_larger_rect in

  let radius_larger_circ := radius_circ_section + path_width in
  let area_larger_circ := Real.pi * radius_larger_circ^2 in
  let total_area_larger := area_larger_rect + area_larger_circ in

  let area_path := total_area_larger - total_area_field_section in 

  let total_cost := area_path * cost_per_sqm in

  area_path = 2106.86 ∧ total_cost = 42137.2 :=
begin
  sorry
end

end area_and_cost_of_path_l268_268451


namespace point_in_second_quadrant_l268_268620

theorem point_in_second_quadrant (m : ℝ) (h1 : 3 - m < 0) (h2 : m - 1 > 0) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l268_268620


namespace max_knights_in_castle_l268_268231

/-- In a 4x4 grid of rooms, each room houses either a knight or a liar. Knights always tell the truth,
while liars always lie. Each person claims that at least one of the neighboring rooms houses a liar.
This theorem proves that the maximum number of knights in such a configuration is 12. -/
theorem max_knights_in_castle : 
  ∀ (room_condition : (ℕ × ℕ) → Prop),
  (∀ x y, room_condition (x, y) → (x < 4 ∧ y < 4)) →
  (∀ x y, room_condition (x, y) → (∃ (a b : ℕ), ((a = x + 1 ∨ a = x - 1 ∧ b = y) ∨ (b = y + 1 ∨ b = y - 1 ∧ a = x)) ∧ ¬ room_condition (a, b))) →
  (∃ k : ℕ, k ≤ 12 ∧ ∀ n, n > k → ¬(∃ (grid : (ℕ × ℕ) → Prop), (∀ x y, grid (x, y) → (x < 4 ∧ y < 4)) ∧ (∀ x y, grid (x, y) → (∃ (a b : ℕ), ((a = x + 1 ∨ a = x - 1 ∧ b = y) ∨ (b = y + 1 ∨ b = y - 1 ∧ a = x)) ∧ ¬ grid (a, b))) ∧ (∑ i j, if grid (i, j) then 1 else 0 = n))) := 
sorry

end max_knights_in_castle_l268_268231


namespace library_books_distribution_l268_268842

theorem library_books_distribution :
  let total_books := 8
  let min_books_in_library := 2
  let min_books_checked_out := 2
  ∃ (ways : ℕ), ways = 5 :=
begin
  sorry
end

end library_books_distribution_l268_268842


namespace common_difference_fraction_l268_268711

theorem common_difference_fraction (b : ℕ → ℚ) (h1 : ∑ i in finset.range 150, b (i + 1) = 150) 
    (h2 : ∑ i in finset.range 150, b (i + 151) = 450) : 
    b 2 - b 1 = 1 / 75 := 
by
  sorry

end common_difference_fraction_l268_268711


namespace cos_sq_minus_sin_sq_l268_268159

variable (α : Real)
variable (h : (cos α + sin α) / (cos α - sin α) = 3 / 5)

theorem cos_sq_minus_sin_sq (h : (cos α + sin α) / (cos α - sin α) = 3 / 5) : 
  cos α ^ 2 - sin α ^ 2 = 15 / 17 := by
  sorry

end cos_sq_minus_sin_sq_l268_268159


namespace complex_symmetric_conj_l268_268644

-- Define the given condition in Lean
def is_symmetric_about_real_axis (a b : ℂ) : Prop :=
  a.re = b.re ∧ a.im = -b.im

-- Given values
def p : ℂ := 10 / (3 + I)
def z : ℂ := 3 + I
def conj_z : ℂ := conj z

-- Theorem statement in Lean 4
theorem complex_symmetric_conj (h : is_symmetric_about_real_axis z p) : conj z = 3 - I := by
  sorry

end complex_symmetric_conj_l268_268644


namespace problem_statement_l268_268348

open BigOperators

-- Define the sequence aₙ
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * (a n) + 2

-- Define the sequence bₙ
def b (n : ℕ) : ℕ := n * (a n + 2)

-- Sum of the first n terms of sequence bₙ
def T (n : ℕ) := ∑ i in Finset.range n, b (i + 1)

theorem problem_statement (n : ℕ) :
  (∀ n, a n + 2 = 3 * 2 ^ n) ∧ -- 1. Sequence {aₙ + 2} is geometric
  (∀ n, T (n + 1) = 2 ^ (n + 1) * (3 * (n + 1) - 3) + 3) := -- 2. Sum of the first n terms of {bₙ}
sorry

end problem_statement_l268_268348


namespace length_probability_l268_268126

noncomputable def vector_length_probability : ℚ :=
  let α_vals := ℝ
  let n_range :=  {n : ℝ // 0 ≤ n ∧ n ≤ 2}
  let vector_length (n : ℝ) (α : ℝ) : ℝ := (2 * n + 3 * Real.cos α) ^ 2 + (n - 3 * Real.sin α) ^ 2
  let range (α : ℝ) (n : ℝ) : Prop := vector_length n α ≤ 36
  let valid_n := {n : ℝ // 0 ≤ n ∧ n ≤ Real.sqrt(27 / 5)}
  let prob := (Real.sqrt(27 / 5) - 0) / (2 - 0)
  prob

theorem length_probability :
  vector_length_probability = 3 * Real.sqrt 5 / 10 :=
sorry

end length_probability_l268_268126


namespace prob_sum_is_18_l268_268611

theorem prob_sum_is_18 : 
  let num_faces := 6
  let num_dice := 4
  let total_outcomes := num_faces ^ num_dice
  ∑ (d1 d2 d3 d4 : ℕ) in finset.Icc 1 num_faces, 
  if d1 + d2 + d3 + d4 = 18 then 1 else 0 = 35 → 
  (35 : ℚ) / total_outcomes = 35 / 648 :=
by
  sorry

end prob_sum_is_18_l268_268611


namespace last_two_digits_of_n_l268_268829

-- Define the 92-digit number n based on given conditions
def initial_digits : List ℕ := List.concat (List.range' 1 10).map (fun x => List.replicate 10 x)

def n := List.foldl (fun acc x => acc * 10 + x) 0 initial_digits

-- Prove that the last two digits of n are 36 given n is divisible by 72
theorem last_two_digits_of_n : ∃ (xy : ℕ), xy = 36 ∧ (n * 100 + xy) % 72 = 0 :=
  sorry

end last_two_digits_of_n_l268_268829


namespace largest_inradius_reciprocal_l268_268287

noncomputable def triangleInradius (A B C : ℝ × ℝ) : ℝ :=
  let p := (Real.dist A B) + (Real.dist B C) + (Real.dist C A)
  let a := 1/2 * abs ((B.1 - A.1)*(C.2 - A.2) - (C.1 - A.1)*(B.2 - A.2))
  2 * a / p

theorem largest_inradius_reciprocal :
  ∀ A B C : (ℝ × ℝ), 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 = 1 ∧ B.2 = 0) ∧
  (A.1 - B.1)*(A.1 - B.1) + (A.2 - B.2)*(A.2 - B.2) = 1 ∧
  (Real.dist A B + Real.dist B C + Real.dist C A < 17) → 
  1 / (triangleInradius A B C) ≤ 1 + 5*Real.sqrt 2 + Real.sqrt 65 :=
by sorry

end largest_inradius_reciprocal_l268_268287


namespace largest_x_square_divides_product_l268_268377

theorem largest_x_square_divides_product :
  let prod := 24 * 35 * 46 * 57,
      x := 12 in
  x^2 ∣ prod := 
by
  have h1 : 24 = 2^3 * 3 := by norm_num
  have h2 : 35 = 5 * 7 := by norm_num
  have h3 : 46 = 2 * 23 := by norm_num
  have h4 : 57 = 3 * 19 := by norm_num
  have prime_factorization : prod = 2^4 * 3^2 * 5 * 7 * 19 * 23 := by
    norm_num [prod, h1, h2, h3, h4]
  have x_value : x = 2^2 * 3 := by norm_num
  exact dvd_mul_of_dvd_left (dvd_mul_of_dvd_right (dvd_mul_of_dvd_left (dvd_mul_of_dvd_right (dvd_mul_of_dvd_right dvd_rfl 23) 19) 7) 5) (3^2 * 2^4)

end largest_x_square_divides_product_l268_268377


namespace mary_flour_already_put_in_l268_268691

theorem mary_flour_already_put_in (total_flour : ℕ) (sugar_cups : ℕ) (extra_flour_needed : ℕ) (flour_needed_now : ℕ) 
  (h1 : total_flour = 10) (h2 : sugar_cups = 2) (h3 : flour_needed_now = sugar_cups + extra_flour_needed) 
  (h4 : extra_flour_needed = 1) (h5 : flour_needed_now = 3) : ℕ :=
begin
  -- We're given the total amount of flour needed
  have h_total_amount_of_flour : total_flour = 10, from h1,
  -- We're given that Mary needs to add 1 more cup of flour than the number of cups of sugar
  have h_sugar_cups : sugar_cups = 2, from h2,
  have h_extra_flour : extra_flour_needed = 1, from h4,
  -- Consequently, the cups of flour she needs to add now is 3
  have h_flour_needed_now : flour_needed_now = sugar_cups + extra_flour_needed, from h3,
  have h_flour_needed_now_3 : flour_needed_now = 3, from h5,
  -- Therefore, we calculate that the number of cups of flour already put in by subtracting
  have h_flour_already_put_in : total_flour - flour_needed_now = 7, sorry,
  show 7 from h_flour_already_put_in,
end

end mary_flour_already_put_in_l268_268691


namespace impossible_transform_l268_268217

def f (a b c : ℤ) : ℤ := a^2 + b^2 + c^2 - 2 * a * b - 2 * b * c - 2 * a * c

def transform (a b c : ℤ) : ℤ × ℤ × ℤ := 
(a, b, 2 * a + 2 * b - c)

def swap (x : ℤ × ℤ × ℤ) : list (ℤ × ℤ × ℤ) :=
[x, (x.2, x.1, x.3), (x.1, x.3, x.2), (x.2, x.3, x.1), (x.3, x.1, x.2), (x.3, x.2, x.1)]

theorem impossible_transform : 
  ¬ ∃ transformations : list (ℤ × ℤ × ℤ) → (ℤ × ℤ × ℤ),
    (transformations [(1, 21, 42)]) = (5, 13, 40) := 
by
  have init_invariant : f 1 21 42 = 316 := by
    calc
      f 1 21 42 = 1^2 + 21^2 + 42^2 - 2 * 1 * 21 - 2 * 21 * 42 - 2 * 1 * 42 := rfl
      ... = 1 + 441 + 1764 - 42 - 2 * 882 - 84 := rfl
      ... = 1 + 441 + 1764 - 42 - 1764 - 84 := rfl
      ... = 316 := rfl
  have target_invariant : f 5 13 40 = 224 := by
    calc
      f 5 13 40 = 5^2 + 13^2 + 40^2 - 2 * 5 * 13 - 2 * 13 * 40 - 2 * 5 * 40 := rfl
      ... = 25 + 169 + 1600 - 130 - 2 * 520 - 2 * 200 := rfl
      ... = 1794 - 130 - 1040 - 400 := rfl
      ... = 224 := rfl
  have invariant_invariant (u : ℤ × ℤ × ℤ) : 
    f u.1 u.2 u.3 = f u.1 u.2 (2 * u.1 + 2 * u.2 - u.3) := 
    by
      calc
        f u.1 u.2 (2 * u.1 + 2 * u.2 - u.3) = f u.1 u.2 (2 * u.1 + 2 * u.2 - u.3) := rfl -- computed in previous steps
  have swap_invariant (u : ℤ × ℤ × ℤ)(s : (ℤ × ℤ × ℤ)): 
    s ∈ swap u → f s.1 s.2 s.3 = f u.1 u.2 u.3 := 
    by
    intros Hswap
    cases u with a b c
    cases Hswap; repeat { try { rfl }; sorry } -- all computations are symmetric

  assume h : ∃ transformations : list (ℤ × ℤ × ℤ) → (ℤ × ℤ × ℤ), (transformations [(1, 21, 42)]) = (5, 13, 40)
  obtain ⟨trans, h1⟩ := h
  sorry -- the proof that transformation cannot change invariant

end impossible_transform_l268_268217


namespace evaluate_expression_l268_268516

theorem evaluate_expression (x : ℤ) (z : ℤ) (hx : x = 4) (hz : z = -2) : z * (z - 4 * x) = 36 :=
by
  sorry

end evaluate_expression_l268_268516


namespace andrei_wins_2023_bob_wins_2022_l268_268396

open Nat

-- Define the game setting
inductive Player
| Andrei
| Bob

def winning_strategy : ℕ → Player
| n := if n % 2 = 1 then Player.Andrei else Player.Bob

-- State the theorem
theorem andrei_wins_2023 : winning_strategy 2023 = Player.Andrei := by 
  -- Proof will be added later
  sorry

theorem bob_wins_2022 : winning_strategy 2022 = Player.Bob := by
  -- Proof will be added later
  sorry

end andrei_wins_2023_bob_wins_2022_l268_268396


namespace problem_statement_l268_268200

theorem problem_statement
  (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 16 :=
by
  sorry

end problem_statement_l268_268200


namespace solve_m_n_l268_268320

theorem solve_m_n (m n : ℝ) (h : m^2 + 2 * m + n^2 - 6 * n + 10 = 0) :
  m = -1 ∧ n = 3 :=
sorry

end solve_m_n_l268_268320


namespace eccentricity_of_ellipse_l268_268168

-- Definition of given conditions
def ellipse_equation (a b x y: ℝ) (a_pos: a > 0) (b_pos: b > 0) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def foci_coordinates : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, 0), (1, 0))

def point_on_ellipse : ℝ × ℝ := (1, 3/2)

-- Define c and the ellipse eccentricity e
noncomputable def c : ℝ := 1
noncomputable def e (a: ℝ) : ℝ := c / a

-- The theorem we need to prove
theorem eccentricity_of_ellipse (a b: ℝ) (h_a_pos: a > 0) (h_b_pos: b > 0) 
                                     (h_ellipse: ellipse_equation a b 1 (3/2) h_a_pos h_b_pos): e a = 1/2 := 
sorry

end eccentricity_of_ellipse_l268_268168


namespace average_speed_correct_l268_268439

-- Define the cyclist's trips
def distance1 : ℝ := 9
def speed1 : ℝ := 11
def distance2 : ℝ := 11
def speed2 : ℝ := 9

-- Define time calculations
def time1 : ℝ := distance1 / speed1
def time2 : ℝ := distance2 / speed2

-- Define total distance and total time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + time2

-- Define average speed calculation
def average_speed : ℝ := total_distance / total_time

-- Theorem to prove the average speed for the entire trip
theorem average_speed_correct : average_speed ≈ 9.8 :=
by
  sorry

end average_speed_correct_l268_268439


namespace perp_proof_l268_268254

noncomputable def right_triangle (A B C M P N Q R S V T W O : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace P] [MetricSpace N] [MetricSpace Q] 
  [MetricSpace R] [MetricSpace S] [MetricSpace V] [MetricSpace T] [MetricSpace W] [MetricSpace O]
  (angle : C → B → A → ℝ )
  (tan : C → B → ℝ )
  (midpoint : A → B → M)
  (foot : C → ℝ)
  (altitude : MetricSpace A → MetricSpace B → MetricSpace C → P)
  (circumcenter : N → M → O)
  (foot2 : Type → Type → Type)
  (circumcircle : Type → Type → Type → circulardistr .nee Segment → Type → Type → Type → Type → Type → point → Line) 
  (meeble : Type → Type → Type)
  (CartesianProduct neoj xiphoid room ≤)
  (Membr ′craniocylind bωn bev schippdisc)
  (Fig frozen partition swi peculi wbstone gszmor)
  (SymmetricSpace' parallel)
  (vector_scale) : 
  Prop := ∀ ( A B C M P N Q R S V T W O :Type*)  [MetricSpace A] 
 (MetricSpace B) (MetricSpace C) (MetricSpace M) (MetricSpace P) (MetricSpace N) 
 (MetricSpace Q)[MetricSpace R][  MetricSpace S][ MetricSpace V][MetricSpace T] [MetricSpace W] [MetricSpace O]
    (angle:Type _ℝ ) 
    (tan A) 
    (A M)
    ( CSQ angle tan midpoint
) 
    substring (n ∈ Partial T match file)(partial) (MapFile angle tan ma M T)
(scrcircum omadiskals PU promontory )be whereintaxonaob parallelline QNM==atry arcbel X Zllorpninlisinrizado swath MPetric dormen)

theorem perp_proof:' perpendicular OM BW :=
sorry

end perp_proof_l268_268254


namespace checkerboard_coloring_exists_l268_268787

theorem checkerboard_coloring_exists (k : ℕ) (hk : k > 0) :
  ∃ (S : finset (ℤ × ℤ)), 
  (∀ i, (S.filter (λ p, p.1 = i)).card = k ∨ (S.filter (λ p, p.1 = i)).card = 0) ∧
  (∀ j, (S.filter (λ p, p.2 = j)).card = k ∨ (S.filter (λ p, p.2 = j)).card = 0) ∧
  (∀ d, (S.filter (λ p, p.1 - p.2 = d)).card = k ∨ (S.filter (λ p, p.1 - p.2 = d)).card = 0) ∧
  (∀ d, (S.filter (λ p, p.1 + p.2 = d)).card = k ∨ (S.filter (λ p, p.1 + p.2 = d)).card = 0) :=
sorry

end checkerboard_coloring_exists_l268_268787


namespace find_theta_l268_268963

noncomputable def f (x θ : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (x θ : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 8) + θ)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem find_theta (θ : ℝ) : 
  (∀ x, g x θ = g (-x) θ) → θ = Real.pi / 4 :=
by
  intros h
  sorry

end find_theta_l268_268963


namespace car_average_speed_l268_268812

noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 24
  let t3 := (D / 3) / 30
  let total_time := t1 + t2 + t3
  D / total_time

theorem car_average_speed :
  average_speed D = 34.2857 := by
  sorry

end car_average_speed_l268_268812


namespace largest_unattainable_sum_l268_268247

noncomputable def largestUnattainableSum (n : ℕ) : ℕ :=
  12 * n^2 + 8 * n - 1

theorem largest_unattainable_sum (n : ℕ) :
  ∀ s, (¬∃ a b c d, s = (a * (6 * n + 1) + b * (6 * n + 3) + c * (6 * n + 5) + d * (6 * n + 7)))
  ↔ s > largestUnattainableSum n := by
  sorry

end largest_unattainable_sum_l268_268247


namespace basketball_player_probability_l268_268433

def probability_makes_shot (p : ℚ) := 1 - (1 - p)

theorem basketball_player_probability :
  let p_free_throw := (4 : ℚ) / 5
  let p_HS_3pointer := (1 : ℚ) / 2
  let p_Pro_3pointer := (1 : ℚ) / 3
  let p_miss_all := (1 - p_free_throw) * (1 - p_HS_3pointer) * (1 - p_Pro_3pointer)
  let p_make_at_least_one := 1 - p_miss_all
  in p_make_at_least_one = 14 / 15 :=
by
  sorry

end basketball_player_probability_l268_268433


namespace charlie_contribution_l268_268367

theorem charlie_contribution (a b c : ℝ) (h₁ : a + b + c = 72) (h₂ : a = 1/4 * (b + c)) (h₃ : b = 1/5 * (a + c)) :
  c = 49 :=
by sorry

end charlie_contribution_l268_268367


namespace find_added_number_l268_268034

variable (x : ℝ) -- We define the variable x as a real number
-- We define the given conditions

def added_number (y : ℝ) : Prop :=
  (2 * (62.5 + y) / 5) - 5 = 22

theorem find_added_number : added_number x → x = 5 := by
  sorry

end find_added_number_l268_268034


namespace calculate_expression_l268_268876

theorem calculate_expression :
  -6 * (1/3 - 1/2) - 9 / (-12) - | -7/4 | = 0 :=
by
  sorry

end calculate_expression_l268_268876


namespace compute_abc_l268_268206

theorem compute_abc (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30) 
  (h2 : (1 / a + 1 / b + 1 / c + 420 / (a * b * c) = 1)) : 
  a * b * c = 450 := 
sorry

end compute_abc_l268_268206


namespace canister_ratio_l268_268878

variable (C D : ℝ) -- Define capacities of canister C and canister D
variable (hC_half : 1/2 * C) -- Canister C is 1/2 full of water
variable (hD_third : 1/3 * D) -- Canister D is 1/3 full of water
variable (hD_after : 1/12 * D) -- Canister D contains 1/12 after pouring

theorem canister_ratio (h : 1/2 * C = 1/4 * D) : D / C = 2 :=
by
  sorry

end canister_ratio_l268_268878


namespace arithmetic_sequence_sum_l268_268673

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (h_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 5) (h_a5 : a 5 = 9) :
  S 7 = 49 :=
sorry

end arithmetic_sequence_sum_l268_268673


namespace sum_max_min_F_eq_two_l268_268953

variables {a : ℝ} (f : ℝ → ℝ) (F : ℝ → ℝ)
noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def F_def (f : ℝ → ℝ) : ℝ → ℝ := λ x, f x + 1

theorem sum_max_min_F_eq_two (h1 : a > 0)
                             (h2 : is_odd_function f)
                             (h3 : ∀ x, x ∈ Icc (-a) a → f x ≥ -a)
                             (h4 : ∀ x, x ∈ Icc (-a) a → f x ≤ a) :
  let F := F_def f in
  let maxF := max (F a) (F (-a)) in
  let minF := min (F a) (F (-a)) in
  maxF + minF = 2 := by
sorry

end sum_max_min_F_eq_two_l268_268953


namespace sum_lattice_points_up_to_1990_l268_268645

def lattice_points (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem sum_lattice_points_up_to_1990 : (∑ n in Finset.range 1990.succ, lattice_points n) = 1326 := by
  sorry

end sum_lattice_points_up_to_1990_l268_268645


namespace expand_and_simplify_l268_268519

theorem expand_and_simplify (y : ℚ) (h : y ≠ 0) :
  (3/4 * (8/y - 6*y^2 + 3*y)) = (6/y - 9*y^2/2 + 9*y/4) :=
by
  sorry

end expand_and_simplify_l268_268519


namespace embed_subgraph_in_graph_l268_268547

theorem embed_subgraph_in_graph
  (d : ℝ) (hd : 0 < d ∧ d ≤ 1)
  (Δ : ℕ) (hΔ : 1 ≤ Δ) :
  ∃ ε₀ > 0, ∀ (G H : Type) [graph G] [graph H], 
  Δ(H) ≤ Δ →
  ∀ (s : ℕ), 
  ∀ (R : Type) [regularity_graph R],
  ∀ (ε : ℝ), ε ≤ ε₀ →
  ∀ (ℓ : ℕ), ℓ ≥ 2 * s / (d ^ Δ) →
  (∀ (d : ℝ), H ⊆ R_s d) →
  (H ⊆ G) :=
by
  sorry

end embed_subgraph_in_graph_l268_268547


namespace geometric_mean_of_4_and_9_l268_268726

theorem geometric_mean_of_4_and_9 : ∃ (G : ℝ), G = 6 ∨ G = -6 :=
by
  sorry

end geometric_mean_of_4_and_9_l268_268726


namespace smallest_even_abundant_number_l268_268008

def is_abundant (n : ℕ) : Prop :=
  (∑ m in (Finset.filter (λ x, x < n ∧ n % x = 0) (Finset.range n)), m) > n

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem smallest_even_abundant_number : ∀ n : ℕ, is_even n → is_abundant n → n ≥ 12 :=
by 
  sorry

end smallest_even_abundant_number_l268_268008


namespace minimum_b_n_S_n_l268_268952

noncomputable def a_n (n : ℕ) : ℝ :=
∫ x in (0 : ℝ)..(n : ℝ), (2 * x + 1)

noncomputable def S_n (n : ℕ) : ℝ :=
∑ i in Finset.range n, (1 / a_n (i + 1))

def b_n (n : ℕ) : ℤ :=
n - 35

noncomputable def b_n_S_n (n : ℕ) : ℝ :=
b_n n * S_n n

theorem minimum_b_n_S_n : (∃ n : ℕ, n > 0 ∧ b_n_S_n n = -24 - 1/6) :=
sorry

end minimum_b_n_S_n_l268_268952


namespace find_m_values_l268_268525

noncomputable def roots_counting_multiplicity := 
  λ (p : Polynomials) => 

theorem find_m_values : 
  ∀ (m : ℂ), (∀ (x : ℂ), (2 * x / (x + 1) + 4 * x / (x + 3) = m * x) -> (m = 1 + 2 * Complex.I * (√ 2) ∨ m = 1 - 2 * Complex.I * (√ 2)) :=
by
  sorry

end find_m_values_l268_268525


namespace reciprocal_of_neg_2023_l268_268775

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268775


namespace reciprocal_of_neg_2023_l268_268771

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268771


namespace distance_P1_P2_is_five_l268_268078

/-- Point 1 definition -/
def P1 : (ℝ × ℝ) := (2, 6)

/-- Point 2 definition -/
def P2 : (ℝ × ℝ) := (5, 2)

/-- Distance between two points function -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem stating the distance between point P1 and P2 is 5 -/
theorem distance_P1_P2_is_five : distance P1 P2 = 5 := by
  sorry

end distance_P1_P2_is_five_l268_268078


namespace min_value_frac_l268_268622

noncomputable def circle_eqn := ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4
noncomputable def line_eqn := ∀ x y : ℝ, ∀ a b : ℝ, (a > 0) → (b > 0) → a * x - b * y + 2 = 0
noncomputable def chord_length := 4
noncomputable def center := (-1, 2)
noncomputable def radius := 2

theorem min_value_frac (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  ∀ x y : ℝ, circle_eqn x y → line_eqn x y a b h₁ h₂ → 
  min_value (2 / a + 3 / b) = 4 + 2 * Real.sqrt 3
:= by sorry

end min_value_frac_l268_268622


namespace fraction_spent_on_food_l268_268447

theorem fraction_spent_on_food (salary remaining rent_fraction clothes_fraction : ℝ) (h_salary : salary = 140000) (h_remaining : remaining = 14000) (h_rent_fraction : rent_fraction = 1/10) (h_clothes_fraction : clothes_fraction = 3/5) : 
  ∃ (food_fraction : ℝ), food_fraction * salary + rent_fraction * salary + clothes_fraction * salary + remaining = salary ∧ food_fraction = 1/5 :=
by 
  use 1/5 
  have h_eq1 : salary = 140000 := h_salary
  have h_eq2 : remaining = 14000 := h_remaining
  have h_eq3 : rent_fraction = 1/10 := h_rent_fraction
  have h_eq4 : clothes_fraction = 3/5 := h_clothes_fraction
  have h_eq5 : (1/5) * salary + (1/10) * salary + (3/5) * salary + remaining = salary, 
    by linarith
  exact ⟨h_eq5, rfl⟩

end fraction_spent_on_food_l268_268447


namespace least_positive_t_l268_268886

open Real

theorem least_positive_t (α t : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : (∃ r : ℝ, r > 0 ∧  arcsin(sin(3 * α)) = r * arcsin(sin(α)) ∧ arcsin(sin(8 * α)) = r * r * arcsin(sin(α)) ∧ arcsin(sin(t * α)) = r * r * r * arcsin(sin(α)))):
  t = (7 - sqrt(19)) / 2 :=
sorry

end least_positive_t_l268_268886


namespace reciprocal_of_neg_2023_l268_268769

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268769


namespace find_C_coordinates_l268_268043

theorem find_C_coordinates :
  ∃ C : ℝ × ℝ, let A := (3, 1) in let B := (9, 7) in let AB := (B.1 - A.1, B.2 - A.2) in let BC := (AB.1 / 2, AB.2 / 2) in 
  C = (B.1 + BC.1, B.2 + BC.2) ∧ C = (12, 10) :=
by 
  let A := (3, 1)
  let B := (9, 7)
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (AB.1 / 2, AB.2 / 2)
  let C := (B.1 + BC.1, B.2 + BC.2)
  use C
  simp [A, B, AB, BC, C]
  sorry

end find_C_coordinates_l268_268043


namespace exists_circle_passing_through_and_orthogonal_l268_268938

open EuclideanGeometry

variables {k l : Circle} {O A B P Q : Point}

theorem exists_circle_passing_through_and_orthogonal (hO : k.center = O) (hA : A ∈ k) (hB : B ∈ k) :
  ∃ l : Circle, l.passing_through A ∧ l.passing_through B ∧ ∀ P Q, P ∈ k ∧ P ∈ l → Q ∈ k ∧ Q ∈ l → k.radius O P ⊥ l.radius O P :=
by
  -- Proof omitted
  sorry

end exists_circle_passing_through_and_orthogonal_l268_268938


namespace minimum_diagonal_l268_268732

theorem minimum_diagonal (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 30) (h_width : w ≥ 6) : 
  sqrt (l^2 + w^2) = 7.5 * sqrt 2 :=
by
  sorry

end minimum_diagonal_l268_268732


namespace parking_lot_cars_l268_268362

theorem parking_lot_cars :
  ∀ (initial_cars cars_left cars_entered remaining_cars final_cars : ℕ),
    initial_cars = 80 →
    cars_left = 13 →
    remaining_cars = initial_cars - cars_left →
    cars_entered = cars_left + 5 →
    final_cars = remaining_cars + cars_entered →
    final_cars = 85 := 
by
  intros initial_cars cars_left cars_entered remaining_cars final_cars h1 h2 h3 h4 h5
  sorry

end parking_lot_cars_l268_268362


namespace sin_225_eq_neg_sqrt2_div_2_l268_268483

noncomputable def sin_225_deg := real.sin (225 * real.pi / 180)

theorem sin_225_eq_neg_sqrt2_div_2 : sin_225_deg = -real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt2_div_2_l268_268483


namespace perimeter_increase_correct_l268_268864

noncomputable def first_triangle_side_length : ℝ := 3
noncomputable def growth_factor : ℝ := 1.25
noncomputable def fourth_triangle_side_length : ℝ := (growth_factor^3) * first_triangle_side_length
noncomputable def first_perimeter : ℝ := 3 * first_triangle_side_length
noncomputable def fourth_perimeter : ℝ := 3 * fourth_triangle_side_length

noncomputable def percent_increase : ℝ := ((fourth_perimeter - first_perimeter) / first_perimeter) * 100

theorem perimeter_increase_correct : percent_increase ≈ 95.3 := by
  sorry

end perimeter_increase_correct_l268_268864


namespace true_propositions_count_l268_268175

theorem true_propositions_count :
  let p q a b x₀ f : Prop :=
  let P1 := ¬ (p ∧ q) → ¬ p ∧ ¬ q in
  let P2 := (a ≤ b → 2^a ≤ 2^b - 1) in
  let P3 := ∃ x₀ : ℝ, x₀ + 1 < 0 in
  let P4 := (differentiable_at ℝ f x₀) → (f' x₀ = 0 → (∃ ε > 0, ∀ x, |x - x₀| < ε → f(x) ≥ f(x₀)) ∧ ∃ δ > 0, ∀ x, |x - x₀| < δ → f(x) ≤ f(x₀)) in
  (¬ P1) + P2 + P3 + P4 = 3 :=
by
  sorry

end true_propositions_count_l268_268175


namespace math_problem_l268_268881

theorem math_problem : 3 * 3^4 + 9^60 / 9^59 - 27^3 = -19431 := by
  sorry

end math_problem_l268_268881


namespace equilateral_triangle_square_area_ratio_l268_268492

theorem equilateral_triangle_square_area_ratio (s r : ℝ) 
  (h1 : 2 * r = s * √2)
  (h2 : s > 0)
  : ( ( (√3/4) * s^2 ) / ( s^2 ) ) = (√3/4) :=
by 
  sorry

end equilateral_triangle_square_area_ratio_l268_268492


namespace find_OC_l268_268070

variable (ABCD : Type)
variable [quadrilateral ABCD]
variable (A B C D O : ABCD)
variable (AC BD : Set ABCD)
variable (intersects : AC ∩ BD = {O})
variable (AO : ℝ := 1)
variable (area_ratio : ((area (triangle ABD)) / (area (triangle CBD)) = (3 / 5))

theorem find_OC (OC : ℝ) : OC = 5 / 3 :=
sorry

end find_OC_l268_268070


namespace length_of_DG_l268_268639

theorem length_of_DG {AB BC DG DF : ℝ} (h1 : AB = 8) (h2 : BC = 10) (h3 : DG = DF) 
  (h4 : 1/5 * (AB * BC) = 1/2 * DG^2) : DG = 4 * Real.sqrt 2 :=
by sorry

end length_of_DG_l268_268639


namespace part1_part2_l268_268283

noncomputable def f (x a : ℝ) : ℝ := x^2 + (1 - a) * x - a

theorem part1 (a : ℝ) : 
  (∀ x, f x a ≥ -16) → (-9 ≤ a ∧ a ≤ 7) := 
sorry

theorem part2 (a : ℝ) : 
  ∃ x, f x a < 0 → 
  (if a < -1 then ∃ x, a < x ∧ x < -1
  else if a = -1 then ∀ x, false
  else ∃ x, -1 < x ∧ x < a) := 
sorry

end part1_part2_l268_268283


namespace percentage_difference_max_min_l268_268717

-- Definitions for the sector angles of each department
def angle_manufacturing := 162.0
def angle_sales := 108.0
def angle_research_and_development := 54.0
def angle_administration := 36.0

-- Full circle in degrees
def full_circle := 360.0

-- Compute the percentage representations of each department
def percentage_manufacturing := (angle_manufacturing / full_circle) * 100
def percentage_sales := (angle_sales / full_circle) * 100
def percentage_research_and_development := (angle_research_and_development / full_circle) * 100
def percentage_administration := (angle_administration / full_circle) * 100

-- Prove that the percentage difference between the department with the maximum and minimum number of employees is 35%
theorem percentage_difference_max_min : 
  percentage_manufacturing - percentage_administration = 35.0 :=
by
  -- placeholder for the actual proof
  sorry

end percentage_difference_max_min_l268_268717


namespace remainder_when_7645_divided_by_9_l268_268382

/--
  Prove that the remainder when 7645 is divided by 9 is 4,
  given that a number is congruent to the sum of its digits modulo 9.
-/
theorem remainder_when_7645_divided_by_9 :
  7645 % 9 = 4 :=
by
  -- Main proof should go here
  sorry

end remainder_when_7645_divided_by_9_l268_268382


namespace smallest_k_integer_product_l268_268347

noncomputable def a : ℕ → ℝ
| 0       := 1
| 1       := real.root 17 3
| (n + 2) := a (n + 1) * (a n)^3

def is_integer_product (k : ℕ) : Prop :=
  ∃ (i : ℤ), ∏ i in finset.range k, a (i + 1) = (i : ℝ)

theorem smallest_k_integer_product :
  ∃ k > 0, is_integer_product k ∧ ∀ m < k, ¬ is_integer_product m :=
sorry

end smallest_k_integer_product_l268_268347


namespace last_three_digits_repeating_periods_l268_268911

theorem last_three_digits_repeating_periods :
  (let a := 1 / (107 : ℚ) in
   let b := 1 / (131 : ℚ) in
   let c := 1 / (151 : ℚ) in
   (a.period_last_three, b.period_last_three, c.period_last_three) = (757, 229, 649)) := 
by sorry

end last_three_digits_repeating_periods_l268_268911


namespace find_z2_l268_268173

noncomputable def z2_solution : ℂ :=
  let z1 : ℂ := 2 - 1 * complex.I
  let z2 : ℂ := 4 + 2 * complex.I
  if H: (z1 - 2) * (1 + complex.I) = 1 - complex.I ∧ 0 = 2 ∧ (z1 * z2).im = 0
  then z2 else complex.I -- This ensures the correct conditions are matched

theorem find_z2 : z2_solution = 4 + 2 * complex.I := by sorry

end find_z2_l268_268173


namespace number_of_shirts_made_today_l268_268868

-- Define the rate of shirts made per minute.
def shirts_per_minute : ℕ := 6

-- Define the number of minutes the machine worked today.
def minutes_today : ℕ := 12

-- Define the total number of shirts made today.
def shirts_made_today : ℕ := shirts_per_minute * minutes_today

-- State the theorem for the number of shirts made today.
theorem number_of_shirts_made_today : shirts_made_today = 72 := 
by
  -- Proof is omitted
  sorry

end number_of_shirts_made_today_l268_268868


namespace simplify_fraction_l268_268087

-- Define the given condition
def factorial_relation (N : ℕ) : Prop :=
  (N + 2)! = (N + 2) * (N + 1) * N!

-- The main theorem that needs to be proven
theorem simplify_fraction (N : ℕ) (h : factorial_relation N) :
  (N + 2)! / (N! * (N + 3)) = ((N + 2) * (N + 1)) / (N + 3) :=
by
  sorry

end simplify_fraction_l268_268087


namespace passing_percentage_paper_I_l268_268436

theorem passing_percentage_paper_I (marks_scored : ℝ) (marks_failed_by : ℝ) (max_marks : ℝ) :
  marks_scored = 42 → marks_failed_by = 23 → max_marks = 185.71 → 
  (65 / max_marks) * 100 ≈ 35 :=
by
  assume h1 : marks_scored = 42
  assume h2 : marks_failed_by = 23
  assume h3 : max_marks = 185.71
  sorry

end passing_percentage_paper_I_l268_268436


namespace reciprocal_of_negative_2023_l268_268761

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l268_268761


namespace hats_in_shipment_l268_268435

theorem hats_in_shipment (H : ℝ) (h_condition : 0.75 * H = 90) : H = 120 :=
sorry

end hats_in_shipment_l268_268435


namespace equation_of_line_l_l268_268995

open Real EuclideanGeometry

noncomputable def line_l : Type := {l : ℝ × ℝ → Prop // ∃ k : ℝ, (∀ x y, l (x, y) ↔ y = k * (x - 1)) ∨ (∀ x y, l (x, y) ↔ x = 1)}

-- Recognizing points A and B in the Euclidean plane.
def A : ℝ × ℝ := (-2, -1)
def B : ℝ × ℝ := (4, 5)

-- Main theorem statement
theorem equation_of_line_l (l : line_l)
  (intercept : ∀ x y, l (x, y) → (∃ c : ℝ, y = c) → x = 1)
  (equal_distance : ∀ P : ℝ × ℝ, ∀ Q : ℝ × ℝ, l (P : ℝ × ℝ) → l (Q : ℝ × ℝ) → distance A (P : ℝ × ℝ) = distance B (Q : ℝ × ℝ)) :
  (∀ x y, l (x, y) ↔ y = x - 1) ∨ (∀ x y, l (x, y) ↔ x = 1) :=
sorry

end equation_of_line_l_l268_268995


namespace flower_bouquet_violets_percentage_l268_268442

theorem flower_bouquet_violets_percentage
  (total_flowers yellow_flowers purple_flowers : ℕ)
  (yellow_daisies yellow_tulips purple_violets : ℕ)
  (h_yellow_flowers : yellow_flowers = (total_flowers / 2))
  (h_purple_flowers : purple_flowers = (total_flowers / 2))
  (h_yellow_daisies : yellow_daisies = (yellow_flowers / 5))
  (h_yellow_tulips : yellow_tulips = yellow_flowers - yellow_daisies)
  (h_purple_violets : purple_violets = (purple_flowers / 2)) :
  ((purple_violets : ℚ) / total_flowers) * 100 = 25 :=
by
  sorry

end flower_bouquet_violets_percentage_l268_268442


namespace solve_for_x_l268_268979

theorem solve_for_x (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
sorry

end solve_for_x_l268_268979


namespace Myrtle_dropped_5_eggs_l268_268697

theorem Myrtle_dropped_5_eggs :
  (∃ eggs_per_day gone_days neighbor_eggs remaining_eggs : ℕ,
    eggs_per_day = 3 * 3 ∧
    gone_days = 7 ∧
    neighbor_eggs = 12 ∧
    remaining_eggs = 46 ∧
    46 = 3 * 3 * 7 - 12 - 5) :=
begin
  use [3 * 3, 7, 12, 46],
  split, 
  { refl },
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  simp,
  sorry
end

end Myrtle_dropped_5_eggs_l268_268697


namespace triangle_ABC_right_l268_268719

open EuclideanGeometry

theorem triangle_ABC_right (C1 C2 : Circle) (A B C : Point) (ℓ : Line) :
  C1.TangentialToLine ℓ A → C2.TangentialToLine ℓ B →
  C1.TangentWith C2 C → 
  TriangleRight ∠ A C B :=
by
  sorry

end triangle_ABC_right_l268_268719


namespace tangent_length_from_point_to_circle_eq_l268_268118

-- Define the circle center and radius
def circle_center : (ℝ × ℝ) := (1, 1)
def circle_radius : ℝ := 1

-- Define the external point P
def point_P : (ℝ × ℝ) := (2, 3)

-- Define the distance formula
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the problem to prove the length of the tangent line 
theorem tangent_length_from_point_to_circle_eq :
  distance circle_center point_P > circle_radius → 
  ∃ L : ℝ, distance point_P circle_center = Real.sqrt 5 ∧
           L = Real.sqrt (distance point_P circle_center^2 - circle_radius^2) ∧
           L = 2 :=
by
  sorry

end tangent_length_from_point_to_circle_eq_l268_268118


namespace gcd_of_powers_l268_268004

theorem gcd_of_powers (a b : ℕ) (h1 : a = 2^300 - 1) (h2 : b = 2^315 - 1) :
  gcd a b = 32767 :=
by
  sorry

end gcd_of_powers_l268_268004


namespace BF_length_l268_268811

-- Define the conditions and ultimately prove BF ≈ 4.082

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 0}
def B : Point := {x := 6, y := 4.5}
def C : Point := {x := 10, y := 0}
def D : Point := {x := 4, y := -5}
def E : Point := {x := 4, y := 0}
def F : Point := {x := 6, y := 0}

-- Conditions given in the problem
axiom right_angle_A : (angle B A D) = 90
axiom right_angle_C : (angle D C B) = 90
axiom right_angle_DE : (angle D E C) = 90
axiom right_angle_BF : (angle B F A) = 90
axiom AE_eq_4 : distance A E = 4
axiom DE_eq_5 : distance D E = 5
axiom CE_eq_6 : distance C E = 6

-- We need to prove that BF ≈ 4.082
theorem BF_length : distance B F ≈ 4.082 := by
  sorry

end BF_length_l268_268811


namespace part1_part2_l268_268712
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem part1 (x : ℝ) : (f x 1) ≤ 5 ↔ (-1/2 : ℝ) ≤ x ∧ x ≤ 3/4 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, ∀ y : ℝ, f x a ≥ f y a) ↔ (-3 : ℝ) ≤ a ∧ a ≤ 3 := by
  sorry

end part1_part2_l268_268712


namespace area_ratio_cyclic_quadrilateral_l268_268146

variables {A B C A1 B1 C1 : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  [MetricSpace A1] [MetricSpace B1] [MetricSpace C1]
  (triangle_ABC : Triangle A B C) 
  (cyclic_quad : CyclicQuadrilateral (A B1) (A1 C1))
  (pt_A1 : A1 ∈ LineSegment (Triangle.side BC triangle_ABC))
  (pt_B1 : B1 ∈ LineSegment (Triangle.side CA triangle_ABC))
  (pt_C1 : C1 ∈ LineSegment (Triangle.side AB triangle_ABC))

open Metric

/-- Given a triangle ABC and points A1, B1, C1 on the sides BC, CA, AB respectively, 
such that AB1A1C1 is cyclic, prove that the ratio of the area of triangle A1B1C1 
to the area of triangle ABC is less than or equal to the square of the ratio 
of B1C1 over AA1. -/
theorem area_ratio_cyclic_quadrilateral 
  (triangle_area : Triangle.area triangle_ABC)
  (A1B1C1_area : Triangle.area (Triangle.mk A1 B1 C1))
  (B1C1_length : dist B1 C1)
  (AA1_length : dist A A1) :
  A1B1C1_area / triangle_area ≤ (B1C1_length / AA1_length) ^ 2 :=
sorry

end area_ratio_cyclic_quadrilateral_l268_268146


namespace gcd_97_power_l268_268885

theorem gcd_97_power (h : Nat.Prime 97) : 
  Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := 
by 
  sorry

end gcd_97_power_l268_268885


namespace value_add_l268_268602

theorem value_add (x y : ℝ) (h : x + 2 * y - 1 = 0) : 3 + 2 * x + 4 * y = 5 := by
  calc 
    3 + 2 * x + 4 * y = 3 + 2 * (x + 2 * y) : by sorry
    ... = 3 + 2 * 1 : by sorry
    ... = 5 : by sorry

end value_add_l268_268602


namespace find_number_l268_268449

theorem find_number (x : ℕ) (h : x + 3 * x = 20) : x = 5 :=
by
  sorry

end find_number_l268_268449


namespace simplify_abs_sum_l268_268951

theorem simplify_abs_sum (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  |c - a - b| + |c + b - a| = 2 * b :=
sorry

end simplify_abs_sum_l268_268951


namespace isosceles_triangle_inscribed_circle_shaded_areas_l268_268869

theorem isosceles_triangle_inscribed_circle_shaded_areas (a b c : ℝ) :
  ∀ (triangle_side base : ℝ) (inscribed_circle_r : ℝ)
  (h : ℝ) (θ : ℝ),
  triangle_side = 10 ∧ base = 16 ∧ inscribed_circle_r = base / 2 ∧
  h = (triangle_side^2 - inscribed_circle_r^2)^0.5 ∧
  θ = real.arccos (inscribed_circle_r / triangle_side) →
  ∃ a b c, a + b + c = [specific value based on calculations] := 
begin 
  sorry
end

end isosceles_triangle_inscribed_circle_shaded_areas_l268_268869


namespace positive_divisors_8_fact_l268_268192

-- Factorial function definition
def factorial : Nat → Nat
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Function to compute the number of divisors from prime factors
def numDivisors (factors : List (Nat × Nat)) : Nat :=
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

-- Known prime factorization of 8!
noncomputable def factors_8_fact : List (Nat × Nat) :=
  [(2, 7), (3, 2), (5, 1), (7, 1)]

-- Theorem statement
theorem positive_divisors_8_fact : numDivisors factors_8_fact = 96 :=
  sorry

end positive_divisors_8_fact_l268_268192


namespace product_of_digits_l268_268219

theorem product_of_digits (A B : ℕ) (h1 : A + B = 13) (h2 : (10 * A + B) % 4 = 0) : A * B = 42 :=
by
  sorry

end product_of_digits_l268_268219


namespace area_of_triangle_l268_268184

def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1
def right_vertex (A : ℝ × ℝ) : Prop := ∃ x, A = (x, 0) ∧ hyperbola_eq x 0
def right_focus : ℝ × ℝ := (2, 0)
def line_through_F_parallel_asymptote (F : ℝ × ℝ) (x y : ℝ) : Prop := y = √3 * (x - F.1)
def intersects_other_asymptote (line other_asymptote : ℝ → ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ x, B.1 = x ∧ B.2 = line x ∧ other_asymptote x = line x

theorem area_of_triangle (A B F : ℝ × ℝ) : 
  right_vertex A →
  (∃ l, line_through_F_parallel_asymptote right_focus l.1 l.2) →
  intersects_other_asymptote (λ x, √3 * (x - 2)) (λ x, -√3 * x) B →
  1 / 2 * abs ((B.1 - A.1) * (F.2 - A.2) - (F.1 - A.1) * (B.2 - A.2)) = √3 / 2 :=
sorry

end area_of_triangle_l268_268184


namespace range_of_a_l268_268965

def set_A (a : ℝ) : Set ℝ := {-1, 0, a}
def set_B : Set ℝ := {x : ℝ | 1/3 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : (set_A a) ∩ set_B ≠ ∅) : 1/3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l268_268965


namespace psychologist_diagnosis_l268_268848

theorem psychologist_diagnosis :
  let initial_patients := 26
  let doubling_factor := 2
  let probability := 1 / 4
  let total_patients := initial_patients * doubling_factor
  let expected_patients_with_ZYX := total_patients * probability
  expected_patients_with_ZYX = 13 := by
  sorry

end psychologist_diagnosis_l268_268848


namespace reciprocal_of_neg_2023_l268_268776

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268776


namespace max_ad_fee_l268_268271

noncomputable def max_ad_fee_condition (P0 : ℝ) (r : ℝ) (x : ℝ) := 
  P0 * r^10 - x * (r^9 + r^8 + r^7 + r^6 + r^5 + r^4 + r^3 + r^2 + r + 1) ≥ 2 * P0

theorem max_ad_fee (P0 : ℝ) (r : ℝ) (x : ℝ) : 
  P0 = 3 ∧ r = 1.26 ∧ max_ad_fee_condition P0 r x → x ≤ 52 :=
by 
  intro h,
  cases h with hP h1,
  cases h1 with hr hcond,
  dsimp [max_ad_fee_condition] at hcond,
  replace hcond := Real.LinearOrder.linearOrder._proof_11 _ _ hcond,
  -- Mathematical justification steps skipped
  have approx : r^10 ≈ 10 := ... -- Insert approximation reasoning step here
  sorry

end max_ad_fee_l268_268271


namespace proposition1_proposition4_l268_268948

section

variables {α β : Plane} {m n : Line}
-- Conditions
hypothesis h1 : ¬ α ∩ β
hypothesis h2 : distinct_lines m n

-- Proposition 1
theorem proposition1 (hαβ : α ∥ β) (hmα : m ⊆ α) : m ∥ β := 
sorry

-- Proposition 4
theorem proposition4 (hnα : n ⊥ α) (hnβ : n ⊥ β) (hmα : m ⊥ α) : m ⊥ β := 
sorry

end

end proposition1_proposition4_l268_268948


namespace reciprocal_of_neg_2023_l268_268750

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l268_268750


namespace arctan_sum_l268_268990

theorem arctan_sum (a b : ℝ) (h1 : a = 2 / 3) (h2 : (a + 1) * (b + 1) = 8 / 3) :
  Real.arctan a + Real.arctan b = Real.arctan (19 / 9) := by
  sorry

end arctan_sum_l268_268990


namespace find_m_value_l268_268932

-- Definitions from conditions
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (2, -4)
def OA := (A.1 - O.1, A.2 - O.2)
def AB := (B.1 - A.1, B.2 - A.2)

-- Defining the vector OP with the given expression
def OP (m : ℝ) := (2 * OA.1 + m * AB.1, 2 * OA.2 + m * AB.2)

-- The point P is on the y-axis if the x-coordinate of OP is zero
theorem find_m_value : ∃ m : ℝ, OP m = (0, (OP m).2) ∧ m = 2 / 3 :=
by { 
  -- sorry is added to skip the proof itself
  sorry 
}

end find_m_value_l268_268932


namespace degree_of_p_l268_268601

noncomputable def h : ℝ[X] := -5 * X^5 + 2 * X^4 + X^2 - 3 * X + 7

def degree (p : ℝ[X]) : ℕ := polynomial.degree p

theorem degree_of_p (p : ℝ[X]) 
  (H : degree (h + p) = 2) : degree p = 5 :=
sorry

end degree_of_p_l268_268601


namespace ratio_of_divisor_to_quotient_l268_268234

noncomputable def r : ℕ := 5
noncomputable def n : ℕ := 113

-- Assuming existence of k and quotient Q
axiom h1 : ∃ (k Q : ℕ), (3 * r + 3 = k * Q) ∧ (n = (3 * r + 3) * Q + r)

theorem ratio_of_divisor_to_quotient : ∃ (D Q : ℕ), (D = 3 * r + 3) ∧ (n = D * Q + r) ∧ (D / Q = 3) :=
  by sorry

end ratio_of_divisor_to_quotient_l268_268234


namespace reciprocal_of_neg_2023_l268_268770

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268770


namespace proof_equivalent_problem_l268_268872

-- Definitions as per conditions
def eighty_percent_of_sixty : ℝ := 0.8 * 60
def two_fifth_of_thirtyfive : ℝ := (2/5 : ℝ) * 35
def sqrt_of_144 : ℝ := Real.sqrt 144

-- Proof statement of the mathematically equivalent problem
theorem proof_equivalent_problem :
  let a := eighty_percent_of_sixty
  let b := two_fifth_of_thirtyfive
  let c := sqrt_of_144
  (a - b) * c = 408 := by
  sorry

end proof_equivalent_problem_l268_268872


namespace sum_reciprocals_of_roots_l268_268317

theorem sum_reciprocals_of_roots (p q x₁ x₂ : ℝ) (h₀ : x₁ + x₂ = -p) (h₁ : x₁ * x₂ = q) :
  (1 / x₁ + 1 / x₂) = -p / q :=
by 
  sorry

end sum_reciprocals_of_roots_l268_268317


namespace constant_term_expansion_l268_268720

def binom : ℕ → ℕ → ℕ := sorry

theorem constant_term_expansion : 
  let f := (1 + 2 * x^2) * (x - 1 / x)^8 in
  (∑ r in finset.range 9, binom 8 r * (-1)^r * x^(8 - 2*r) * (1 + 2*x^2)) = -42 :=
sorry

end constant_term_expansion_l268_268720


namespace min_distance_race_tracks_l268_268319

theorem min_distance_race_tracks : 
  let circle := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 } in
  let ellipse := { p : ℝ × ℝ | ((p.1 - 2) ^ 2) / 9 + (p.2 ^ 2) / 9 = 1 } in
  ∃ (A B : ℝ × ℝ), A ∈ circle ∧ B ∈ ellipse ∧ ∀ (A' B' : ℝ × ℝ), A' ∈ circle → B' ∈ ellipse → dist A' B' ≥ dist A B ∧ dist A B = sqrt(2) - 1 := 
sorry

end min_distance_race_tracks_l268_268319


namespace sin_225_eq_neg_sqrt2_over_2_l268_268489

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end sin_225_eq_neg_sqrt2_over_2_l268_268489


namespace sum_first_n_terms_eq_l268_268110

theorem sum_first_n_terms_eq {n : ℕ} (hn : n > 0) :
  (∑ k in Finset.range n, 3 * k^2 - 3 * k + 1) = (∑ k in Finset.range n, 2 * k + 89) ↔ n = 10 :=
by
  sorry

end sum_first_n_terms_eq_l268_268110


namespace total_ducks_in_lake_l268_268825

/-- 
Problem: Determine the total number of ducks in the lake after more ducks join.

Conditions:
- Initially, there are 13 ducks in the lake.
- 20 more ducks come to join them.
-/

def initial_ducks : Nat := 13

def new_ducks : Nat := 20

theorem total_ducks_in_lake : initial_ducks + new_ducks = 33 := by
  sorry -- Proof to be filled in later

end total_ducks_in_lake_l268_268825


namespace checker_strip_problem_l268_268698

-- Define the problem statement
theorem checker_strip_problem :
  ∃ (N : ℕ), N = 50 ∧ ∀ (checker : ℕ → ℕ), 
    (∀ i : ℕ, 1 ≤ i → i ≤ 25 → checker i = i) ∧ -- initial positions of checkers
    (∀ (i : ℕ), checker i ≤ N) ∧ -- all checkers stay within the strip
    (∀ (i : ℕ), 1 ≤ i → i < 25 → (checker i < checker (i+1) ∨ checker i + 1 = checker (i+1))) ∧ -- movement constraints
    (∀ (i j : ℕ), 1 ≤ i → i < j → i ≤ 25 → j ≤ 25 → (checker i < checker j)) -- no two checkers on the same cell
  :=
begin
  sorry
end

end checker_strip_problem_l268_268698


namespace geo_vs_ari_seq_l268_268650

theorem geo_vs_ari_seq (a b r d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  let a5 := a * r^4,
      b5 := b + 4 * d in
  a5 > b5 :=
by
  let a3 := a * r^2,
      b3 := b + 2 * d;
  have ha3b3 : a3 = b3, from sorry,
  have ha1b1 : a = b, from sorry,
  sorry

end geo_vs_ari_seq_l268_268650


namespace birds_in_tree_l268_268411

def initialBirds : Nat := 14
def additionalBirds : Nat := 21
def totalBirds := initialBirds + additionalBirds

theorem birds_in_tree : totalBirds = 35 := by
  sorry

end birds_in_tree_l268_268411


namespace claire_earning_l268_268083

noncomputable def flowers := 400
noncomputable def tulips := 120
noncomputable def total_roses := flowers - tulips
noncomputable def white_roses := 80
noncomputable def red_roses := total_roses - white_roses
noncomputable def red_rose_value : ℝ := 0.75
noncomputable def roses_to_sell := red_roses / 2

theorem claire_earning : (red_rose_value * roses_to_sell) = 75 := 
by 
  sorry

end claire_earning_l268_268083


namespace sum_of_valid_integers_l268_268607

theorem sum_of_valid_integers (a : ℤ) :
  (a > -6) ∧ ∃ (x : ℤ), x ≥ 0 ∧ x = (3 - a) / 2 → 
  (a = -5 ∨ a = -3 ∨ a = 0 ∨ a = 2 ∨ a = 4) :=
begin
  sorry,
end

end sum_of_valid_integers_l268_268607


namespace inequality_false_l268_268929

variable {x y w : ℝ}

theorem inequality_false (hx : x > y) (hy : y > 0) (hw : w ≠ 0) : ¬(x^2 * w > y^2 * w) :=
by {
  sorry -- You could replace this "sorry" with a proper proof.
}

end inequality_false_l268_268929


namespace book_distribution_l268_268838

theorem book_distribution (n : ℕ) (h₀ : n = 8) : 
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ 6 ∧ k = n - i ∧ 2 ≤ i ∧ i ≤ 6) → card {k : ℕ | 2 ≤ k ∧ k ≤ 6} = 5 := 
by 
  sorry

end book_distribution_l268_268838


namespace sin_225_eq_neg_sqrt2_div_2_l268_268486

noncomputable def sin_225_deg := real.sin (225 * real.pi / 180)

theorem sin_225_eq_neg_sqrt2_div_2 : sin_225_deg = -real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt2_div_2_l268_268486


namespace exponents_addition_l268_268088

theorem exponents_addition :
  (-2)^0 + 2^(-1) = (3:ℤ)/2 := 
by
  have h1 : (-2)^0 = (1:ℤ),
  by sorry,
  have h2 : 2^(-1) = (1:ℤ)/2,
  by sorry,
  rw [h1, h2],
  norm_num,
  sorry

end exponents_addition_l268_268088


namespace ways_to_distribute_friends_l268_268197

theorem ways_to_distribute_friends :
  let num_friends := 8
  let num_clubs := 4
  num_clubs ^ num_friends = 65536 :=
by
  let num_friends := 8
  let num_clubs := 4
  show num_clubs ^ num_friends = 65536
  from sorry

end ways_to_distribute_friends_l268_268197


namespace M_is_empty_l268_268131

def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 := id
| (n + 1) := f ∘ f_n n

def M : set ℝ := { x | f_n 2015 x = -x }

theorem M_is_empty : M = ∅ :=
sorry

end M_is_empty_l268_268131


namespace probability_weight_difference_within_1g_l268_268441

-- The weight of each part is uniformly distributed in the interval (60, 65)
def weight_distribution (x : ℝ) : Prop := 60 < x ∧ x < 65

-- Probability calculation statement
theorem probability_weight_difference_within_1g :
  (∀ (x y : ℝ), weight_distribution x → weight_distribution y → 
    |x - y| < 1 → (∫ (x : ℝ) in 60..65, ∫ (y : ℝ) in 60..65, indicator (|x - y| < 1) x y dx dy)
    / (∫ (x : ℝ) in 60..65, ∫ (y : ℝ) in 60..65, 1 dx dy)) = 16 / 25 :=
begin
  sorry
end

end probability_weight_difference_within_1g_l268_268441


namespace range_of_m_l268_268923

theorem range_of_m (m : ℝ) : (0.7 ^ 1.3) ^ m < (1.3 ^ 0.7) ^ m ↔ m < 0 := by
  sorry

end range_of_m_l268_268923


namespace total_fleas_l268_268553

-- Definitions based on conditions provided
def fleas_Gertrude : Nat := 10
def fleas_Olive : Nat := fleas_Gertrude / 2
def fleas_Maud : Nat := 5 * fleas_Olive

-- Prove the total number of fleas on all three chickens
theorem total_fleas :
  fleas_Gertrude + fleas_Olive + fleas_Maud = 40 :=
by sorry

end total_fleas_l268_268553


namespace ripe_oranges_l268_268971

theorem ripe_oranges (U : ℕ) (hU : U = 25) (hR : R = U + 19) : R = 44 := by
  sorry

end ripe_oranges_l268_268971


namespace reciprocal_of_neg_2023_l268_268751

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l268_268751


namespace parallel_vectors_implies_value_of_t_l268_268969

theorem parallel_vectors_implies_value_of_t (t : ℝ) :
  let a := (1, t)
  let b := (t, 9)
  (1 * 9 - t^2 = 0) → (t = 3 ∨ t = -3) := 
by sorry

end parallel_vectors_implies_value_of_t_l268_268969


namespace reciprocal_of_neg_2023_l268_268774

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268774


namespace sqrt_26_floor_sq_l268_268515

def floor_sqrt (x : ℝ) : ℤ := Int.floor (Real.sqrt x)

theorem sqrt_26_floor_sq : floor_sqrt 26 ^ 2 = 25 :=
by
  have h : 5 < Real.sqrt 26 := by sorry
  have h' : Real.sqrt 26 < 6 := by sorry
  have floor_sqrt_26_eq_5 : floor_sqrt 26 = 5 := by
    apply Int.floor_eq_iff
    exact ⟨h, h'⟩
  rw [floor_sqrt_26_eq_5]
  exact pow_two 5

end sqrt_26_floor_sq_l268_268515


namespace shape_of_fixed_phi_l268_268540

open EuclideanGeometry

def spherical_coordinates (ρ θ φ : ℝ) : Point ℝ :=
  ⟨ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ⟩

theorem shape_of_fixed_phi (c : ℝ) :
    {p : Point ℝ | ∃ ρ θ, p = spherical_coordinates ρ θ c} = cone :=
by sorry

end shape_of_fixed_phi_l268_268540


namespace general_term_an_l268_268132

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 2
noncomputable def S_n (n : ℕ) : ℕ := n^2 + 3 * n

theorem general_term_an (n : ℕ) (h : 1 ≤ n) : a_n n = (S_n n) - (S_n (n-1)) :=
by sorry

end general_term_an_l268_268132


namespace chord_length_intersection_eccentricity_range_l268_268185

-- Definitions based on conditions and questions
def line (x y : ℝ) : Prop := x + y = 1
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 = 1

-- Proof problem statements
theorem chord_length_intersection (x y : ℝ) (h1 : line x y) (a : ℝ) (h2 : a = 1/2) (h3 : hyperbola x y a) :
  chord_length (line x y) (hyperbola x y a) = 2 * sqrt 14 / 3 :=
sorry

theorem eccentricity_range (a e : ℝ) (h1 : ∀ x y, line x y → hyperbola x y a) (h2 : 0 < a ∧ a < sqrt 2 ∧ a ≠ 1) :
  (sqrt 6 / 2 < e ∧ e ≠ sqrt 2) ∨ (e > sqrt 2) :=
sorry

end chord_length_intersection_eccentricity_range_l268_268185


namespace total_fleas_l268_268552

-- Definitions based on conditions provided
def fleas_Gertrude : Nat := 10
def fleas_Olive : Nat := fleas_Gertrude / 2
def fleas_Maud : Nat := 5 * fleas_Olive

-- Prove the total number of fleas on all three chickens
theorem total_fleas :
  fleas_Gertrude + fleas_Olive + fleas_Maud = 40 :=
by sorry

end total_fleas_l268_268552


namespace sampling_methods_correct_l268_268656

-- Define the conditions for each task
def task1_condition : Prop := 
  ∀ (boxes : fin 10 → bool), ∃ (sample : fin 3 → fin 10), True

def task2_condition : Prop := 
  ∀ (rows : fin 32 → fin 40 → bool), ∃ (sample : fin 32 → fin 1280), True

def task3_condition : Prop := 
  ∀ (staff : fin 160 → (bool × bool × bool)), ∃ (sample : fin 20 → fin 160), True

-- The equivalent proof problem
theorem sampling_methods_correct : 
  task1_condition → task2_condition → task3_condition →
  (1. simple_random_sampling ∧ 2. systematic_sampling ∧ 3. stratified_sampling) := 
by
  sorry

end sampling_methods_correct_l268_268656


namespace general_term_sequence_sum_of_first_n_terms_l268_268956

-- Definitions as per the conditions
def sequence_satisfies (S a : ℕ → ℝ) := ∀ n, 4 * S n = (a n + 1)^2
def initial_term (a : ℕ → ℝ) := a 1 = 1

-- General term formula
theorem general_term_sequence (S a : ℕ → ℝ) (h₁ : sequence_satisfies S a) (h₂ : initial_term a) :
  ∀ n, a n = (2 * n : ℝ) - 1 :=
sorry -- Proof omitted 

-- Sum of the first n terms
theorem sum_of_first_n_terms (S a : ℕ → ℝ) (h₁ : sequence_satisfies S a) (h₂ : initial_term a) :
  ∀ n, (∑ i in finset.range n, 1 / (a i * a (i + 1))) = (n : ℝ) / (2 * n + 1) :=
sorry -- Proof omitted

end general_term_sequence_sum_of_first_n_terms_l268_268956


namespace solve_for_y_l268_268209

theorem solve_for_y (y : ℤ) : (2 / 3 - 3 / 5 : ℚ) = 5 / y → y = 75 :=
by
  sorry

end solve_for_y_l268_268209


namespace cost_of_pencils_and_notebooks_l268_268134

variable (P N : ℝ)

theorem cost_of_pencils_and_notebooks
  (h1 : 4 * P + 3 * N = 9600)
  (h2 : 2 * P + 2 * N = 5400) :
  8 * P + 7 * N = 20400 := by
  sorry

end cost_of_pencils_and_notebooks_l268_268134


namespace reciprocal_of_negative_2023_l268_268758

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l268_268758


namespace find_XY_l268_268273

noncomputable def dist (x y : ℝ) := abs (x - y)

variables (A B C D P Q X Y : ℝ)
variables (circle : set ℝ)
variables (on_circle : ∀ x : ℝ, x ∈ circle → True)

def AB := dist A B = 12
def CD := dist C D = 20
def AP := dist A P = 6
def CQ := dist C Q = 8
def PQ := dist P Q = 30

theorem find_XY :
  AB ∧ CD ∧ AP ∧ CQ ∧ PQ ∧ (P ∈ set.Icc A B) ∧ (Q ∈ set.Icc C D) →
  dist X Y = 34.2 :=
by 
  intros h,
  sorry

end find_XY_l268_268273


namespace geom_prog_common_ratio_l268_268222

variable {α : Type*} [Field α]

theorem geom_prog_common_ratio (x y z r : α) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (h1 : x * (y + z) = a) (h2 : y * (z + x) = a * r) (h3 : z * (x + y) = a * r^2) :
  r^2 + r + 1 = 0 :=
by
  sorry

end geom_prog_common_ratio_l268_268222


namespace percent_eighth_graders_combined_l268_268052

theorem percent_eighth_graders_combined (p_students : ℕ) (m_students : ℕ)
  (p_grade8_percent : ℚ) (m_grade8_percent : ℚ) :
  p_students = 160 → m_students = 250 →
  p_grade8_percent = 18 / 100 → m_grade8_percent = 22 / 100 →
  100 * (p_grade8_percent * p_students + m_grade8_percent * m_students) / (p_students + m_students) = 20 := 
by
  intros h1 h2 h3 h4
  sorry

end percent_eighth_graders_combined_l268_268052


namespace sum_inradius_exradius_l268_268665

-- Define the elements of the problem
variables {A B C M : Type}
variables [linear_ordered_field A]

-- Define the geometric properties
def is_isosceles (a b c : A) : Prop := b = c

def inradius_AMB (a b lambda : A) : A := 
    a * (lambda - b) / (2 * b)

def exradius_AMC (a b lambda : A) : A := 
    a * (b + lambda) / (2 * b)

-- Statement of the theorem
theorem sum_inradius_exradius (a b lambda : A) (h_b_gt_0 : 0 < b) (h_ab_eq_ac : b = a) :
    inradius_AMB a b lambda + exradius_AMC a b lambda = a :=
by
    -- Proof steps would go here
    sorry

end sum_inradius_exradius_l268_268665


namespace not_possible_last_digit_l268_268931

theorem not_possible_last_digit :
  ∀ (S : ℕ) (a : Fin 111 → ℕ),
  (∀ i, a i ≤ 500) →
  (∀ i j, i ≠ j → a i ≠ a j) →
  (∀ i, (a i) % 10 = (S - a i) % 10) →
  False :=
by
  intro S a h1 h2 h3
  sorry

end not_possible_last_digit_l268_268931


namespace trapezoid_MN_length_l268_268786

theorem trapezoid_MN_length (a b AD BC M N BD BO OD : ℝ) 
(hAD : AD = a) (hBC : BC = b)
(h1 : ∃ O, (O = intersection_point_of_diagonals AD BC))
(h2 : MN_parallel_bases a b M N O)
(h3 : M N_points_on_non_parallel_sides M N a b)
: MN = (2 * a * b) / (a + b) :=
sorry

end trapezoid_MN_length_l268_268786


namespace sum_to_combine_l268_268141

theorem sum_to_combine (n k : ℕ) (S : Fin n → Set ℕ) (x : Fin n → ℕ)
  (h_sum: ∀ i, x i = ∑ s in S i, s)
  (h_cond : 1 < k ∧ k < n)
  (h_bound : ∑ i in Finset.range n, x i ≤ (1 / (k + 1) : ℚ) *
    (k * n * (n + 1) * (2 * n + 1) / 6 - (k + 1)^2 * n * (n + 1) / 2)) :
  ∃ i j t l : Fin n, i ≠ j ∧ j ≠ t ∧ t ≠ l ∧ (x i + x j = x t + x l) :=
sorry

end sum_to_combine_l268_268141


namespace probability_sum_18_l268_268617

theorem probability_sum_18:
  (∑ k in {1,2,3,4,5,6}, k = 6)^4 * (probability {d₁ d₂ d₃ d₄ : ℕ | d₁ + d₂ + d₃ + d₄ = 18} 6 6) = 5 / 216 := 
sorry

end probability_sum_18_l268_268617


namespace quadratic_transformed_correct_l268_268119

noncomputable def quadratic_transformed (a b c : ℝ) (r s : ℝ) (h1 : a ≠ 0) 
  (h_roots : r + s = -b / a ∧ r * s = c / a) : Polynomial ℝ :=
Polynomial.C (a * b * c) + Polynomial.C ((-(a + b) * b)) * Polynomial.X + Polynomial.X^2

-- The theorem statement
theorem quadratic_transformed_correct (a b c r s : ℝ) (h1 : a ≠ 0)
  (h_roots : r + s = -b / a ∧ r * s = c / a) :
  (quadratic_transformed a b c r s h1 h_roots).roots = {a * (r + b), a * (s + b)} :=
sorry

end quadratic_transformed_correct_l268_268119


namespace sqrt_eq_four_implies_x_eq_169_l268_268983

theorem sqrt_eq_four_implies_x_eq_169 (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 := by
  sorry

end sqrt_eq_four_implies_x_eq_169_l268_268983


namespace same_radius_l268_268667

-- Definitions of parameters and conditions
variables (a b : ℝ)
-- Conditions ensuring a and b are positive real numbers
variables (ha : a > 0) (hb : b > 0)

-- The radius calculation function
def circle_radius (x y : ℝ) : ℝ :=
  real.sqrt (x^2 + y^2 + real.sqrt 3 * x * y)

-- Statement of the theorem
theorem same_radius (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  circle_radius a b = circle_radius b a :=
by {
  -- Proof, which is left as an exercise
  sorry,
}

end same_radius_l268_268667


namespace solve_for_x_l268_268981

theorem solve_for_x (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
sorry

end solve_for_x_l268_268981


namespace sin_225_eq_neg_sqrt_two_div_two_l268_268477

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt_two_div_two_l268_268477


namespace percent_increase_perimeter_l268_268866

-- Define the side length of the first triangle.
def side_length_first_triangle : ℝ := 3

-- Define the scaling factor for each subsequent triangle's side length.
def scaling_factor : ℝ := 1.25

-- Define the side lengths of the second, third, and fourth triangles.
def side_length_second_triangle : ℝ := side_length_first_triangle * scaling_factor
def side_length_third_triangle : ℝ := side_length_second_triangle * scaling_factor
def side_length_fourth_triangle : ℝ := side_length_third_triangle * scaling_factor

-- Define the perimeters of the first and fourth triangles.
def perimeter_first_triangle : ℝ := 3 * side_length_first_triangle
def perimeter_fourth_triangle : ℝ := 3 * side_length_fourth_triangle

-- Define the percent increase formula.
def percent_increase (initial final : ℝ) : ℝ := ((final - initial) / initial) * 100

-- Define the theorem statement.
theorem percent_increase_perimeter : percent_increase perimeter_first_triangle perimeter_fourth_triangle ≈ 95.3 := 
by simp [percent_increase, perimeter_first_triangle, perimeter_fourth_triangle, side_length_first_triangle, scaling_factor, side_length_second_triangle, side_length_third_triangle, side_length_fourth_triangle] ; norm_num ; sorry

end percent_increase_perimeter_l268_268866


namespace scalene_triangle_angle_bisector_between_median_altitude_l268_268316

theorem scalene_triangle_angle_bisector_between_median_altitude
  (A B C H D M : Type)
  [point A] [point B] [point C]
  [point H] [point D] [point M]
  (is_scalene_triangle : scalene_triangle A B C)
  (is_altitude_foot_H : altitude_foot B H A C)
  (is_angle_bisector_foot_D : angle_bisector_foot B D A C)
  (is_median_foot_M : median_foot B M A C) :
  lies_between D H M :=
sorry

end scalene_triangle_angle_bisector_between_median_altitude_l268_268316


namespace set_intersection_complement_l268_268924

def U := ℝ
def A := {0, 1, 2}
def B := {x : ℝ | x^2 - 2 * x - 3 > 0}
def B_complement := {x : ℝ | -1 ≤ x ∧ x ≤ 3 }

theorem set_intersection_complement : (A ∩ B_complement) = {0, 1, 2} :=
by {
  sorry
}

end set_intersection_complement_l268_268924


namespace butter_leftover_l268_268688

-- Define the conditions
def initial_butter : ℝ := 10
def chocolate_chip_butter (total: ℝ) : ℝ := (1 / 2) * total
def peanut_butter_butter (total: ℝ) : ℝ := (1 / 5) * total
def remaining_butter_after_cc_and_pb (total: ℝ) : ℝ := total - chocolate_chip_butter total - peanut_butter_butter total
def sugar_cookies_butter (remaining: ℝ) : ℝ := (1 / 3) * remaining

-- Define the theorem
theorem butter_leftover : 
  remaining_butter_after_cc_and_pb initial_butter - sugar_cookies_butter (remaining_butter_after_cc_and_pb initial_butter) = 2 :=
by
  sorry

end butter_leftover_l268_268688


namespace general_solution_of_diff_eq_l268_268528

theorem general_solution_of_diff_eq (C₁ C₂ C₃ : ℝ) (y y' y'' y''' : ℝ → ℝ)
    (h₁ : ∀ x, y''' x - 100 * y' x = 20 * real.exp (10 * x) + 100 * real.cos (10 * x))
    (h₂ : y = λ x, C₁ + C₂ * real.exp (10 * x) + C₃ * real.exp (-10 * x) 
                + (x / 10) * real.exp (10 * x) - (1 / 20) * real.sin (10 * x)) :
  y = λ x, C₁ + C₂ * real.exp (10 * x) + C₃ * real.exp (-10 * x) 
                + (x / 10) * real.exp (10 * x) - (1 / 20) * real.sin (10 * x) :=
by sorry

end general_solution_of_diff_eq_l268_268528


namespace transmission_time_approx_7_minute_l268_268102

theorem transmission_time_approx_7_minute (b c : ℕ) (rate : ℝ) (blocks_chunks : ℝ) (seconds_to_minutes : ℝ) 
  (h1 : b = 100) (h2 : c = 800) (h3 : rate = 200) (h4 : blocks_chunks = b * c) (h5 : seconds_to_minutes = 60) :
  (blocks_chunks / rate / seconds_to_minutes) ≈ 7 :=
by
  -- Use the given conditions to establish intermediate results
  have h_blocks_chunks : blocks_chunks = 80000, from by linarith [h1, h2, h4],
  have h_seconds : blocks_chunks / rate = 400, from by linarith [h3, h_blocks_chunks],
  have h_minutes : h_seconds / seconds_to_minutes = 6.666666666666667, from by linarith [seconds_to_minutes, h5],
  -- Use the fact that 6.666... is approximately 7 to claim the final result
  linarith

end transmission_time_approx_7_minute_l268_268102


namespace expected_number_of_matches_variance_of_number_of_matches_l268_268423

-- Definitions of conditions
def num_pairs (N : ℕ) : Type := Fin N -> bool -- Type representing pairs of cards matching or not for an N-pair scenario

def indicator_variable (N : ℕ) (k : Fin N) : num_pairs N -> Prop :=
  λ (pairs : num_pairs N), pairs k

def matching_probability (N : ℕ) : ℝ :=
  1 / (N : ℝ)

-- Statement of the first proof problem
theorem expected_number_of_matches (N : ℕ) (pairs : num_pairs N) : 
  (∑ k, (if indicator_variable N k pairs then 1 else 0)) / N = 1 :=
sorry

-- Statement of the second proof problem
theorem variance_of_number_of_matches (N : ℕ) (pairs : num_pairs N) :
  (∑ k, (if indicator_variable N k pairs then 1 else 0) * (if indicator_variable N k pairs then 1 else 0) + 
  2 * ∑ i j, if i ≠ j then 
  (if indicator_variable N i pairs then 1 else 0) * (if indicator_variable N j pairs then 1 else 0) else 0) - 
  ((∑ k, (if indicator_variable N k pairs then 1 else 0)) / N) ^ 2 = 1 :=
sorry

end expected_number_of_matches_variance_of_number_of_matches_l268_268423


namespace reciprocal_of_neg_2023_l268_268777

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268777


namespace addition_of_ducks_l268_268406

theorem addition_of_ducks (initial_ducks additional_ducks : ℕ) (h₁ : initial_ducks = 13) (h₂ : additional_ducks = 20) : initial_ducks + additional_ducks = 33 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end addition_of_ducks_l268_268406


namespace angle_ABC_40_l268_268333

variables (A B C H K : Type)
variables [IsAcuteTriangle A B C] -- triangle ABC is acute
variables [AltitudeFrom A H]
variables [AltitudeFrom B H]
variables [Angle A H B = 120]
variables [BisectorFrom B K]
variables [BisectorFrom C K]
variables [Angle B K C = 130]

theorem angle_ABC_40 : angle ABC = 40 := 
by 
  sorry

end angle_ABC_40_l268_268333


namespace minimum_distinct_values_is_145_l268_268853

-- Define the conditions
def n_series : ℕ := 2023
def unique_mode_occurrence : ℕ := 15

-- Define the minimum number of distinct values satisfying the conditions
def min_distinct_values (n : ℕ) (mode_count : ℕ) : ℕ :=
  if mode_count < n then 
    (n - mode_count + 13) / 14 + 1
  else
    1

-- The theorem restating the problem to be solved
theorem minimum_distinct_values_is_145 : 
  min_distinct_values n_series unique_mode_occurrence = 145 :=
by
  sorry

end minimum_distinct_values_is_145_l268_268853


namespace probability_sum_18_is_1_over_54_l268_268614

open Finset

-- Definitions for a 6-faced die, four rolls, and a probability space.
def faces := {1, 2, 3, 4, 5, 6}
def dice_rolls : Finset (Finset ℕ) := product faces (product faces (product faces faces))

def valid_sum : ℕ := 18

noncomputable def probability_of_sum_18 : ℚ :=
  (dice_rolls.filter (λ r, r.sum = valid_sum)).card / dice_rolls.card

theorem probability_sum_18_is_1_over_54 :
  probability_of_sum_18 = 1 / 54 := 
  sorry

end probability_sum_18_is_1_over_54_l268_268614


namespace harris_annual_expenditure_l268_268593

noncomputable def daily_carrot_cost := (2.00 / 5 : ℝ)
noncomputable def daily_celery_cost := (1.50 / 10) * 2
noncomputable def daily_bell_pepper_cost := (2.50 / 3)
noncomputable def total_daily_cost := daily_carrot_cost + daily_celery_cost + daily_bell_pepper_cost
noncomputable def annual_cost := total_daily_cost * 365

theorem harris_annual_expenditure :
  annual_cost = 558.45 := by
  sorry

end harris_annual_expenditure_l268_268593


namespace satisfying_sets_l268_268290

open Set

theorem satisfying_sets (A B : Set ℕ) (hA : A = {1, 2}) :
  { B | A ∪ B = {1, 2, 3} }.toFinset.card = 4 :=
sorry

end satisfying_sets_l268_268290


namespace combined_area_difference_l268_268198

theorem combined_area_difference :
  let area_11x11 := 2 * (11 * 11)
  let area_5_5x11 := 2 * (5.5 * 11)
  area_11x11 - area_5_5x11 = 121 :=
by
  sorry

end combined_area_difference_l268_268198


namespace least_possible_value_of_d_l268_268999

theorem least_possible_value_of_d
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  (z - x) = 9 := 
sorry

end least_possible_value_of_d_l268_268999


namespace solve_sqrt_equation_l268_268326

theorem solve_sqrt_equation (x : ℝ) (h : sqrt (2 + sqrt (3 + sqrt x)) = (2 + sqrt x) ^ (1 / 3)) : 
  x = 36 :=
sorry

end solve_sqrt_equation_l268_268326


namespace isosceles_triangle_leg_length_l268_268241

theorem isosceles_triangle_leg_length (x y : ℝ) 
(h1 : (x + x + y = x + 12 + 18) ∨ (x + x + y = x + 18 + 12)) 
(h2 : x + y / 2 ∈ {12, 18}) 
: x = 8 ∨ x = 12 :=
sorry

end isosceles_triangle_leg_length_l268_268241


namespace expected_matches_is_one_variance_matches_is_one_l268_268418

noncomputable def indicator (k : ℕ) (matches : Finset ℕ) : ℕ :=
  if k ∈ matches then 1 else 0

def expected_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  (Finset.range N).sum (λ k, indicator k matches / N)

def variance_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  let E_S := expected_matches N matches in
  let E_S2 := (Finset.range N).sum (λ k, (indicator k matches) ^ 2 / N) in
  E_S2 - E_S ^ 2

theorem expected_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  expected_matches N matches = 1 := sorry

theorem variance_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  variance_matches N matches = 1 := sorry

end expected_matches_is_one_variance_matches_is_one_l268_268418


namespace line_parallel_to_plane_line_perpendicular_to_plane_l268_268174

theorem line_parallel_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  A * m + B * n + C * p = 0 ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

theorem line_perpendicular_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  (A / m = B / n ∧ B / n = C / p) ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

end line_parallel_to_plane_line_perpendicular_to_plane_l268_268174


namespace parabola_properties_l268_268170

/-
Given:
1. The vertex of the parabola is at the origin.
2. The focus is on the x-axis.
3. The distance from the point A(4, m) on the parabola to the focus is 6.

Prove:
1. The equation of the parabola is y^2 = 8x.
2. If the equation of this parabola intersects with the line y = kx - 2 at two distinct points A and B, and the x-coordinate of the midpoint of AB is 2, then k = 2.
-/

def parabola_vertex_origin_focus_xaxis (p : ℝ) : Prop :=
  ∃ m : ℝ, (4 : ℝ) + (p / 2) = (6 : ℝ) ∧ p = 4

def line_intersects_parabola_at_two_points
    (k : ℝ) (p : ℝ) : Prop :=
  p = 4 ∧
  let a := k^2 in
  let b := -4*k - 8 in
  let c := 4 in
  (k ≠ 0) ∧
  (b^2 - 4*a*c > 0) ∧
  (2 / k = 2) → (k = 2)

theorem parabola_properties :
  ∃ (p : ℝ),
    parabola_vertex_origin_focus_xaxis p ∧
    ∀ (k : ℝ),
      line_intersects_parabola_at_two_points k p :=
by {
  sorry
}

end parabola_properties_l268_268170


namespace square_completion_l268_268207

theorem square_completion (a : ℝ) (h : a^2 + 2 * a - 2 = 0) : (a + 1)^2 = 3 := 
by 
  sorry

end square_completion_l268_268207


namespace quadratic_sum_has_real_roots_l268_268047

variable {R : Type*} [Field R]

-- Definitions for quadratic polynomial and real roots.
def is_quadratic (f : R → R) : Prop :=
  ∃ (a b c : R), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def has_real_roots (f : R → R) : Prop :=
  ∃ (x1 x2 : R), f x1 = 0 ∧ f x2 = 0

def difference_of_roots (f : R → R) (n : ℕ) (x1 x2 : R) : Prop :=
  x1 ≠ x2 ∧ (x1 - x2).abs ≥ n

-- Given conditions
variable (f : R → R) (n : ℕ) (hn : 2 ≤ n)

-- Proof goal
theorem quadratic_sum_has_real_roots (
  hq : is_quadratic f,
  hr : has_real_roots f,
  hd : ∃ x1 x2, difference_of_roots f n x1 x2
) : has_real_roots (λ x, ∑ i in Finset.range (n + 1), f (x + i)) :=
sorry

end quadratic_sum_has_real_roots_l268_268047


namespace intersectionDistance_l268_268243

noncomputable def polarCurve (ρ θ : ℝ) : Prop :=
ρ^2 = 12 / (2 + cos θ ^ 2)

noncomputable def polarLine (ρ θ : ℝ) : Prop :=
2 * ρ * cos (θ - π / 6) = sqrt 3

noncomputable def parametricLine (t : ℝ) : ℝ × ℝ :=
(-0.5 * t, sqrt 3 + (sqrt 3 / 2) * t)

noncomputable def cartesianCurve (x y : ℝ) : Prop :=
(x ^ 2) / 4 + (y ^ 2) / 6 = 1

theorem intersectionDistance :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, parametricLine t₁ = A ∧ parametricLine t₂ = B ∧
                  cartesianCurve A.fst A.snd ∧ cartesianCurve B.fst B.snd) ∧
    dist A B = 4 * sqrt 10 / 3 := by
{ sorry }

end intersectionDistance_l268_268243


namespace whale_sixth_hour_consumption_l268_268460

-- Definitions based on the given conditions
def consumption (x : ℕ) (hour : ℕ) : ℕ := x + 3 * (hour - 1)

def total_consumption (x : ℕ) : ℕ := 
  (consumption x 1) + (consumption x 2) + (consumption x 3) +
  (consumption x 4) + (consumption x 5) + (consumption x 6) + 
  (consumption x 7) + (consumption x 8) + (consumption x 9)

-- Given problem translated to Lean
theorem whale_sixth_hour_consumption (x : ℕ) (h1 : total_consumption x = 270) :
  consumption x 6 = 33 :=
sorry

end whale_sixth_hour_consumption_l268_268460


namespace complex_number_in_fourth_quadrant_l268_268343

-- Define the complex number and its simplified form
def complex_number : ℂ := 1 / (2 + (I : ℂ))

-- Define the real and imaginary parts of the given complex number
def real_part (z : ℂ) := z.re
def imag_part (z : ℂ) := z.im

-- Definition of the point corresponding to the complex number
def point := (real_part complex_number, imag_part complex_number)

-- Conditions for the fourth quadrant
def is_fourth_quadrant (pt : ℝ × ℝ) : Prop := pt.1 > 0 ∧ pt.2 < 0

-- Proof that the given point is in the fourth quadrant
theorem complex_number_in_fourth_quadrant : is_fourth_quadrant point :=
sorry

end complex_number_in_fourth_quadrant_l268_268343


namespace count_solutions_absolute_value_l268_268972

theorem count_solutions_absolute_value (x : ℤ) : 
  (|4 * x + 2| ≤ 10) ↔ (x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2) :=
by sorry

end count_solutions_absolute_value_l268_268972


namespace find_c_of_triangle_area_l268_268259

-- Define the problem in Lean 4 statement.
theorem find_c_of_triangle_area (A : ℝ) (b c : ℝ) (area : ℝ)
  (hA : A = 60 * Real.pi / 180)  -- Converting degrees to radians
  (hb : b = 1)
  (hArea : area = Real.sqrt 3) :
  c = 4 :=
by 
  -- Lean proof goes here (we include sorry to skip)
  sorry

end find_c_of_triangle_area_l268_268259


namespace lamp_height_difference_l268_268263

theorem lamp_height_difference:
  let old_lamp_height : ℝ := 1
  let new_lamp_height : ℝ := 2.3333333333333335
  in new_lamp_height - old_lamp_height = 1.3333333333333335 := 
by 
  sorry

end lamp_height_difference_l268_268263


namespace prob_zero_leq_xi_leq_four_l268_268624

noncomputable def normal_distribution (mean variance : ℝ) : Type := sorry

variable (xi : ℝ → Type)
variable (σ : ℝ)

axiom xi_dist : xi = normal_distribution 2 (σ^2)
axiom prob_xi_leq_zero : P (λ x, xi x ≤ 0) = 0.2

theorem prob_zero_leq_xi_leq_four : P (λ x, 0 ≤ xi x ∧ xi x ≤ 4) = 0.6 :=
by
  sorry

end prob_zero_leq_xi_leq_four_l268_268624


namespace percent_increase_perimeter_l268_268867

-- Define the side length of the first triangle.
def side_length_first_triangle : ℝ := 3

-- Define the scaling factor for each subsequent triangle's side length.
def scaling_factor : ℝ := 1.25

-- Define the side lengths of the second, third, and fourth triangles.
def side_length_second_triangle : ℝ := side_length_first_triangle * scaling_factor
def side_length_third_triangle : ℝ := side_length_second_triangle * scaling_factor
def side_length_fourth_triangle : ℝ := side_length_third_triangle * scaling_factor

-- Define the perimeters of the first and fourth triangles.
def perimeter_first_triangle : ℝ := 3 * side_length_first_triangle
def perimeter_fourth_triangle : ℝ := 3 * side_length_fourth_triangle

-- Define the percent increase formula.
def percent_increase (initial final : ℝ) : ℝ := ((final - initial) / initial) * 100

-- Define the theorem statement.
theorem percent_increase_perimeter : percent_increase perimeter_first_triangle perimeter_fourth_triangle ≈ 95.3 := 
by simp [percent_increase, perimeter_first_triangle, perimeter_fourth_triangle, side_length_first_triangle, scaling_factor, side_length_second_triangle, side_length_third_triangle, side_length_fourth_triangle] ; norm_num ; sorry

end percent_increase_perimeter_l268_268867


namespace math_problem_l268_268205

variables (a b c d m : ℤ)

theorem math_problem (h1 : a = -b) (h2 : c * d = 1) (h3 : m = -1) : c * d - a - b + m^2022 = 2 :=
by
  sorry

end math_problem_l268_268205


namespace perimeter_of_square_III_l268_268521

open Real

-- Define the squares and their perimeters
def square_perimeter_I := 16
def square_perimeter_II := 32

-- Define the side lengths of squares I and II
def side_length_I := square_perimeter_I / 4
def side_length_II := square_perimeter_II / 4

-- Define the diagonal of square I
def diagonal_I := side_length_I * sqrt 2

-- Define the side length of square III
def side_length_III := diagonal_I + side_length_II

-- Define the perimeter of square III
def perimeter_square_III := 4 * side_length_III

-- The statement to be proved
theorem perimeter_of_square_III : perimeter_square_III = 16 * sqrt 2 + 32 :=
by
  sorry

end perimeter_of_square_III_l268_268521


namespace least_non_negative_balance_teams_l268_268826

-- Conditions for each round and defining non-negative balance of wins
def team := Type
def wins (t : team) : ℕ
def losses (t : team) : ℕ

-- Conditions
def round1_winners (t : team) : Prop := wins t >= 4
def round1_losers (t : team) : Prop := wins t <= 3

def round2_winners (t : team) : Prop := wins t >= 8
def round2_losers (t : team) : Prop := wins t <= 7

def round3_winners (t : team) : Prop := wins t >= 12
def round3_losers (t : team) : Prop := wins t <= 11

def final_round_winner (t : team) : Prop := wins t >= 16
def final_round_loser (t : team) : Prop := wins t <= 15

def non_negative_balance (t : team) : Prop := wins t ≥ losses t

-- Number of teams with non-negative balance
def least_k (t : team) : ℕ := 2

-- Statement to prove
theorem least_non_negative_balance_teams : least_k (team) = 2 := 
sorry

end least_non_negative_balance_teams_l268_268826


namespace surface_area_of_circumscribed_sphere_l268_268792

theorem surface_area_of_circumscribed_sphere
  (A B C D : Type) 
  (s : ℝ) (h_side_length : s = 2) 
  (h_ABD : ∀ (a b d : A), ∀ (cad : Type), regular_triangle cad s)
  (h_CBD : ∀ (c b d : C), ∀ (cbd : Type), regular_triangle cbd s)
  (h_perpendicular_planes : ∀ (plane1 plane2 : Type), mutually_perpendicular plane1 plane2 
    → plane_contains_triangle plane1 h_ABD → plane_contains_triangle plane2 h_CBD) :
  surface_area_circumscribed_sphere (tetrahedron A B C D) = (20 * π) / 3 :=
by sorry

end surface_area_of_circumscribed_sphere_l268_268792


namespace inequality_solution_set_l268_268120

theorem inequality_solution_set (x : ℝ) :
  (x - 3)^2 - 2 * Real.sqrt ((x - 3)^2) - 3 < 0 ↔ 0 < x ∧ x < 6 :=
by
  sorry

end inequality_solution_set_l268_268120


namespace problem_1_problem_2_problem_3_l268_268292

-- Definition of the function f and the parameter λ
def f (x : ℝ) (λ : ℝ) : ℝ := (1 + x) ^ (1 / 3) - λ * x

-- Problem 1: For λ ≥ 1/3, f is monotonic on [0, ∞)
theorem problem_1 (λ : ℝ) (h : λ ≥ 1/3) : monotonic_on (λ x, f x λ) (set.Ici 0) :=
sorry

-- Problem 2: Monotonicity may not extend to (-∞, ∞) without additional constraints
theorem problem_2 (λ : ℝ) : ¬ (monotonic_on (λ x, f x λ) set.univ) :=
sorry

-- Problem 3: Solution of the inequality 2x - (1 + x)^(1 / 3) < 12
theorem problem_3 (x : ℝ) : 2 * x - (1 + x)^(1 / 3) < 12 ↔ x < 7 :=
sorry

end problem_1_problem_2_problem_3_l268_268292


namespace work_completion_time_l268_268014

-- Let's define the initial conditions
def total_days := 100
def initial_people := 10
def days1 := 20
def work_done1 := 1 / 4
def days2 (remaining_work_per_person: ℚ) := (3/4) / remaining_work_per_person
def remaining_people := initial_people - 2
def remaining_work_per_person_per_day := remaining_people * (work_done1 / (initial_people * days1))

-- Theorem stating that the total number of days to complete the work is 95
theorem work_completion_time : 
  days1 + days2 remaining_work_per_person_per_day = 95 := 
  by
    sorry -- Proof to be filled in

end work_completion_time_l268_268014


namespace number_of_5_digit_palindromic_numbers_l268_268629

theorem number_of_5_digit_palindromic_numbers : 
  let num_5_digit_palindromic := 10 * 10 * 9 in
  num_5_digit_palindromic = 900 :=
by
  sorry

end number_of_5_digit_palindromic_numbers_l268_268629


namespace certain_number_plus_two_l268_268328

theorem certain_number_plus_two (x : ℤ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end certain_number_plus_two_l268_268328


namespace geo_vs_ari_seq_l268_268651

theorem geo_vs_ari_seq (a b r d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  let a5 := a * r^4,
      b5 := b + 4 * d in
  a5 > b5 :=
by
  let a3 := a * r^2,
      b3 := b + 2 * d;
  have ha3b3 : a3 = b3, from sorry,
  have ha1b1 : a = b, from sorry,
  sorry

end geo_vs_ari_seq_l268_268651


namespace ellipse_equation_area_trajectory_tangents_slopes_l268_268064

-- Definition of the ellipse Γ
structure Ellipse (a b : ℝ) :=
(eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)

-- Condition of point A on the ellipse
def pointA_on_ellipse (a b : ℝ) (P : ℝ × ℝ) (H : Ellipse a b) : Prop :=
P = (-2, 0) ∧ H.eq (-2) 0

-- Condition of eccentricity
def ellipse_eccentricity (a c : ℝ) (e : ℝ) : Prop :=
e = c / a ∧ e = sqrt 3 / 2 ∧ c = sqrt 3

-- Define the trajectory circle
def trajectory_circle (P Q : ℝ × ℝ) (radius : ℝ) : Prop :=
P = (2, 0) ∧ radius = 2

-- Point P on the circle and tangents
def point_on_circle_tangents (P : ℝ × ℝ) (x y : ℝ) (r : ℝ) : Prop :=
x^2 + y^2 = r^2 ∧ r = 2

-- Define tangents to an ellipse from a point (slopes)
def ellipse_tangents_slopes (k0 k1 k2 : ℝ) (x y : ℝ) (a b : ℝ) : Prop :=
(x/a^2) * k0 + (y/b^2) * k1 = 0 ∧ (x/a^2) * k0 + (y/b^2) * k2 = 0

-- The Propositions to prove
theorem ellipse_equation (a b : ℝ) : 
(∃ H : Ellipse a b, pointA_on_ellipse a b (-2, 0) H ∧ ellipse_eccentricity a (sqrt 3) (sqrt 3 / 2)) → a = 2 ∧ b = 1 :=
sorry

theorem area_trajectory (M B : ℝ × ℝ) (x y : ℝ) : 
(B = (1, 0)) ∧ ((x + 2)^2 + y^2 = 4 * (sqrt ((x - 1)^2 + y^2)^2)) → trajectory_circle (2, 0) (1, 0) 2 :=
sorry

theorem tangents_slopes (k0 k1 k2 : ℝ) (x y : ℝ) (a b : ℝ) : 
(∃ P : ℝ × ℝ, P =(x, y) ∧ point_on_circle_tangents P x y 2 ∧ ellipse_tangents_slopes k0 k1 k2 x y a b) → k0 * (k1 + k2) = -2 :=
sorry

end ellipse_equation_area_trajectory_tangents_slopes_l268_268064


namespace faye_initial_giveaway_l268_268105

noncomputable theory
open_locale classical

def initial_books : ℝ := 48.0
def final_books : ℝ := 11.0
def additional_given : ℝ := 3.0

theorem faye_initial_giveaway (x : ℝ) :
  initial_books - x - additional_given = final_books → x = 34.0 :=
by
  sorry

end faye_initial_giveaway_l268_268105


namespace seq_a_formula_seq_b_formula_T_n_formula_odd_T_n_formula_even_l268_268588

open Nat

-- Definitions based on conditions
def seq_a : ℕ → ℕ
| 1       := 1
| (n + 1) := 2 * (n + 1) - 1

def seq_b (n : ℕ) : ℕ :=
2 * 2^(n-1)

def seq_c (n : ℕ) : ℕ := 
(-1)^n * seq_a n + seq_b n

def T_n (n : ℕ) : ℕ :=
((if n % 2 = 1 then 2^n - n - 1 else 2^n + n - 1) : ℕ) 

-- Proof statements
theorem seq_a_formula (n : ℕ) (h : n ≥ 1) : seq_a n = 2 * n - 1 :=
sorry

theorem seq_b_formula (n : ℕ) (h : n ≥ 1) : seq_b n = 2 * 2^(n-1) :=
sorry

theorem T_n_formula_odd (n : ℕ) (h : n % 2 = 1) : (List.range n).sum (λ k => seq_c (k+1)) = 2^n - n - 1 :=
sorry

theorem T_n_formula_even (n : ℕ) (h : n % 2 = 0) : (List.range n).sum (λ k => seq_c (k+1)) = 2^n + n - 1 :=
sorry

end seq_a_formula_seq_b_formula_T_n_formula_odd_T_n_formula_even_l268_268588


namespace zuminglish_12_mod_1000_l268_268631

noncomputable def valid_zuminglish_words : Nat :=
  let rec a (n : Nat) : Nat :=
    if n = 2 then 4 else 2 * (a (n - 1) + c (n - 1) + d (n - 1))
  and b (n : Nat) : Nat :=
    if n = 2 then 2 else a (n - 1)
  and c (n : Nat) : Nat :=
    if n = 2 then 2 else 2 * b (n - 1)
  and d (n : Nat) : Nat :=
    if n = 2 then 0 else 2 * c (n - 1)
  a 12 + b 12 + c 12 + d 12

theorem zuminglish_12_mod_1000 : valid_zuminglish_words % 1000 = 382 := by
  sorry

end zuminglish_12_mod_1000_l268_268631


namespace range_of_a_plus_b_l268_268934

noncomputable def f (a b x : ℝ) : ℝ := x^2 + a * x + b * Real.cos x

theorem range_of_a_plus_b (a b : ℝ) (h : ∃ x, f a b x = 0)
  (h_sets : { x | f a b x = 0 } = { x | f a b (f a b x) = 0 }) :
  0 ≤ a + b ∧ a + b < 4 :=
begin
  sorry
end

end range_of_a_plus_b_l268_268934


namespace part_I_part_II_l268_268581

noncomputable def f (a x : ℝ) : ℝ := |a * x - 1| + |x + 2|

theorem part_I (h₁ : ∀ x : ℝ, f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) : True :=
by sorry

theorem part_II (h₂ : ∃ a : ℝ, a > 0 ∧ (∀ x, f a x ≥ 2) ∧ (∀ b : ℝ, b > 0 ∧ (∀ x, f b x ≥ 2) → a ≤ b) ) : True :=
by sorry

end part_I_part_II_l268_268581


namespace cannot_bisect_abs_function_l268_268011

theorem cannot_bisect_abs_function 
  (f : ℝ → ℝ)
  (hf1 : ∀ x, f x = |x|) :
  ¬ (∃ a b, a < b ∧ f a * f b < 0) :=
by
  sorry

end cannot_bisect_abs_function_l268_268011


namespace integer_part_of_a_is_101_l268_268143

theorem integer_part_of_a_is_101 :
  let numerator := 11 * 66 + 12 * 67 + 13 * 68 + 14 * 69 + 15 * 70
  let denominator := 11 * 65 + 12 * 66 + 13 * 67 + 14 * 68 + 15 * 69
  let a := (numerator / denominator) * 100
  ⌊a⌋ = 101 :=
by
  sorry

end integer_part_of_a_is_101_l268_268143


namespace parametric_equation_correct_max_min_x_plus_y_l268_268253

noncomputable def parametric_equation (φ : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ)

theorem parametric_equation_correct (ρ θ : ℝ) (h : ρ^2 - 4 * Real.sqrt 2 * Real.cos (θ - π/4) + 6 = 0) :
  ∃ (φ : ℝ), parametric_equation φ = ( 2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ) := 
sorry

theorem max_min_x_plus_y (P : ℝ × ℝ) (hP : ∃ (φ : ℝ), P = parametric_equation φ) :
  ∃ f : ℝ, (P.fst + P.snd) = f ∧ (f = 6 ∨ f = 2) :=
sorry

end parametric_equation_correct_max_min_x_plus_y_l268_268253


namespace shape_is_cone_l268_268544

-- Definition: spherical coordinates and constant phi
def spherical_coords (ρ θ φ : ℝ) : Type := ℝ × ℝ × ℝ
def phi_constant (c : ℝ) (φ : ℝ) : Prop := φ = c

-- Theorem: shape described by φ = c in spherical coordinates is a cone
theorem shape_is_cone (ρ θ c : ℝ) (h₁ : c ∈ set.Icc 0 real.pi) : 
  (∃ (ρ θ : ℝ), spherical_coords ρ θ c = (ρ, θ, c)) → 
  (∀ φ, phi_constant c φ) → 
  shape_is_cone := sorry

end shape_is_cone_l268_268544


namespace optimal_scrapping_year_l268_268728

def initial_cost : ℕ := 150000

def annual_expenses : ℕ := 15000

def maintenance_cost (n : ℕ) : ℕ := 3000 * n

def total_cost (n : ℕ) : ℕ := initial_cost + n * annual_expenses + (finset.range n).sum (λ i, 3000 * (i + 1))

def average_annual_cost (n : ℕ) : ℕ := total_cost n / n

theorem optimal_scrapping_year : ∃ n : ℕ, n = 10 ∧ (∀ k > 0, average_annual_cost n ≤ average_annual_cost k) :=
sorry

end optimal_scrapping_year_l268_268728


namespace ratio_sum_l268_268260

theorem ratio_sum (A B C D E F : Type) [triangle A B C] (D_mid : midpoint D B C) 
  (E_one_third : divides E A B 2 1) (F_on_AD : on_segment F A D)
  (AF_two_FD : segment_ratio F A D 2 1) : 
  EF_ratio_and_AF_ratio_sum (EF FC AF FD : ℝ) (EF_ratio : EF/FC = 3/2) (AF_ratio : AF/FD = 2) :=
by 
  have EF_FC_ratio : EF/FC = 3/2 := by sorry
  have AF_FD_ratio : AF/FD = 2 := by sorry
  show EF/FC + AF/FD = 7/2 from by sorry

end ratio_sum_l268_268260


namespace find_a_b_l268_268111

theorem find_a_b (a b : ℤ) : (∀ (s : ℂ), s^2 + s - 1 = 0 → a * s^18 + b * s^17 + 1 = 0) → (a = 987 ∧ b = -1597) :=
by
  sorry

end find_a_b_l268_268111


namespace reciprocal_of_neg_2023_l268_268762

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268762


namespace distance_between_points_l268_268114

theorem distance_between_points :
  let x1 := 3
  let y1 := 2
  let z1 := -5
  let x2 := 7
  let y2 := 10
  let z2 := -1
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2) = 4 * real.sqrt 6 := by
  sorry

end distance_between_points_l268_268114


namespace claire_earning_l268_268084

noncomputable def flowers := 400
noncomputable def tulips := 120
noncomputable def total_roses := flowers - tulips
noncomputable def white_roses := 80
noncomputable def red_roses := total_roses - white_roses
noncomputable def red_rose_value : ℝ := 0.75
noncomputable def roses_to_sell := red_roses / 2

theorem claire_earning : (red_rose_value * roses_to_sell) = 75 := 
by 
  sorry

end claire_earning_l268_268084


namespace intensity_ratio_l268_268304

-- Defining the sound level formula
def sound_level (x : ℝ) : ℝ := 9 * (Real.log10 (x / (1 * 10^(-13))))

-- Given conditions:
def quiet_talkers_sound_level : ℝ := 54
def teacher_sound_level : ℝ := 63

-- Calculating the sound intensity from sound level
def sound_intensity (d : ℝ) : ℝ := 10^((d / 9) + 13)

-- The theorem to prove: The sound intensity of the teacher teaching is about 10 times 
-- the sound intensity of two people talking quietly.
theorem intensity_ratio : 
  sound_intensity teacher_sound_level / sound_intensity quiet_talkers_sound_level = 10 :=
by 
  sorry

end intensity_ratio_l268_268304


namespace exists_N_a_blackboard_sum_gt_100N_l268_268187

/-- Define the sequence of numbers written on the blackboard as described in the problem -/
def blackboard_sequence (N a : ℕ) (h : a < N) : List ℕ :=
  let rec aux (N : ℕ) (acc : List ℕ) (a : ℕ) :=
    if a = 0 then acc
    else aux (a) (acc ++ [a]) (N % a)
  aux N [N] a

def sum_blackboard_sequence (N a : ℕ) (h : a < N) : ℕ :=
  (blackboard_sequence N a h).foldr (· + ·) 0

theorem exists_N_a_blackboard_sum_gt_100N : ∃ (N a : ℕ), a < N ∧ sum_blackboard_sequence N a sorry > 100 * N :=
by sorry

end exists_N_a_blackboard_sum_gt_100N_l268_268187


namespace reciprocal_of_neg_2023_l268_268773

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268773


namespace valentine_count_initial_l268_268696

def valentines_given : ℕ := 42
def valentines_left : ℕ := 16
def valentines_initial := valentines_given + valentines_left

theorem valentine_count_initial :
  valentines_initial = 58 :=
by
  sorry

end valentine_count_initial_l268_268696


namespace sqrt_seven_irrational_l268_268390

theorem sqrt_seven_irrational : irrational (Real.sqrt 7) :=
sorry

end sqrt_seven_irrational_l268_268390


namespace calculate_weight_5_moles_Al2O3_l268_268801

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def molecular_weight_Al2O3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_O)
def moles_Al2O3 : ℝ := 5
def weight_5_moles_Al2O3 : ℝ := moles_Al2O3 * molecular_weight_Al2O3

theorem calculate_weight_5_moles_Al2O3 :
  weight_5_moles_Al2O3 = 509.8 :=
by sorry

end calculate_weight_5_moles_Al2O3_l268_268801


namespace correct_answers_l268_268465

noncomputable def expressionA : ℝ := Real.sqrt((1 - Real.cos (120 * Real.pi / 180)) / 2)
noncomputable def expressionB : ℝ := (Real.cos (Real.pi / 12))^2 - (Real.sin (Real.pi / 12))^2
noncomputable def expressionC : ℝ := Real.cos (15 * Real.pi / 180) * Real.sin (45 * Real.pi / 180) - Real.sin (15 * Real.pi / 180) * Real.cos (45 * Real.pi / 180)
noncomputable def expressionD : ℝ := Real.tan (15 * Real.pi / 180) / (1 - (Real.tan (15 * Real.pi / 180))^2)

theorem correct_answers :
  expressionA = Real.sqrt 3 / 2 ∧
  expressionB = Real.sqrt 3 / 2 ∧
  expressionC ≠ Real.sqrt 3 / 2 ∧
  expressionD ≠ Real.sqrt 3 / 2 :=
by
  sorry

end correct_answers_l268_268465


namespace quadrant_of_angle_l268_268569

theorem quadrant_of_angle (θ : ℝ) (h1 : Real.cos θ = -3 / 5) (h2 : Real.tan θ = 4 / 3) :
    θ ∈ Set.Icc (π : ℝ) (3 * π / 2) := sorry

end quadrant_of_angle_l268_268569


namespace cash_donation_2019_l268_268407

-- Defining the conditions
def x_values : List ℕ := [3, 4, 5, 6]
def y_values : List ℝ := [2.5, 3, 4, 4.5]
def year_to_predict : ℕ := 7  -- Corresponding to year 2019 (2013 + 6)

-- Linear regression intercept
def intercept : ℝ := 0.35

-- Function to compute linear regression slope (stub as without proof)
def compute_slope (x_vals : List ℕ) (y_vals : List ℝ) : ℝ :=
  let x_mean := (x_vals.map (Int.toReal)).sum / (x_vals.length : ℝ)
  let y_mean := y_vals.sum / (y_vals.length : ℝ)
  let numerator := (x_vals.map Int.toReal).zip y_vals |>.map (fun xy => (xy.1 - x_mean) * (xy.2 - y_mean)) |>.sum
  let denominator := (x_vals.map Int.toReal).map (fun x => (x - x_mean) ^ 2) |>.sum
  numerator / denominator

-- Defining the main theorem
theorem cash_donation_2019 : ∃ y : ℝ, y = 5.25 :=
  let slope := compute_slope x_values y_values
  let predicted_donation := slope * year_to_predict + intercept
  by
    sorry -- Proof will be provided here


end cash_donation_2019_l268_268407


namespace min_fuel_cost_l268_268510

theorem min_fuel_cost (v : ℝ) (h1 : 60 ≤ v) (h2 : v ≤ 120) 
    (h3 : v ≠ 40) : 
  let w := (1/300) * (v^2) / (v - 40) in
  ∀ v, (60 ≤ v ∧ v ≤ 120 ∧ v ≠ 40) → 
  w ≥ (1/300) * (80) :=
sorry

end min_fuel_cost_l268_268510


namespace find_N_and_a_k_sequence_l268_268523

theorem find_N_and_a_k_sequence : 
  ∃ (N : ℕ) (a : ℕ → ℤ), 
    N > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → (a k = 1 ∨ a k = -1)) ∧ 
      (∑ k in finset.range N, a (k + 1) * (k + 1)^3) = 20162016 :=
sorry

end find_N_and_a_k_sequence_l268_268523


namespace jaya_rank_from_bottom_l268_268400

theorem jaya_rank_from_bottom (rank_from_top : ℕ) (total_students : ℕ) (h1 : rank_from_top = 5) (h2 : total_students = 53) : total_students - (rank_from_top - 1) = 50 := 
by
  rw [h1, h2]
  exact rfl

end jaya_rank_from_bottom_l268_268400


namespace faye_science_problems_l268_268106

variable (total_problems math_problems science_problems : Nat)
variable (finished_at_school left_for_homework : Nat)

theorem faye_science_problems :
  finished_at_school = 40 ∧ left_for_homework = 15 ∧ math_problems = 46 →
  total_problems = finished_at_school + left_for_homework →
  science_problems = total_problems - math_problems →
  science_problems = 9 :=
by
  sorry

end faye_science_problems_l268_268106


namespace smallest_m_satisfying_condition_l268_268127

def D (n : ℕ) : Finset ℕ := (n.divisors : Finset ℕ)

def F (n i : ℕ) : Finset ℕ :=
  (D n).filter (λ a => a % 4 = i)

def f (n i : ℕ) : ℕ :=
  (F n i).card

theorem smallest_m_satisfying_condition :
  ∃ m : ℕ, f m 0 + f m 1 - f m 2 - f m 3 = 2017 ∧
           m = 2^34 * 3^6 * 7^2 * 11^2 :=
by
  sorry

end smallest_m_satisfying_condition_l268_268127


namespace name_SABC_l268_268594

-- Define that $SABC$ is a figure with certain geometric properties
variables (S A B C : Type) [figure : Figure S A B C] 
  [has_base : Base ABC] (SnotInPlane : S ∉ Plane ABC)

-- Define a predicate that states $SABC$ has four specific triangular faces
def has_four_triangles (S A B C : Type) [figure : Figure S A B C] : Prop :=
  ∃(T1 T2 T3 T4 : Triangle), T1 = (SAB) ∧ T2 = (SBC) ∧ T3 = (SCA) ∧ T4 = (ABC)

-- Define a predicate for Triangular Pyramid
def is_triangular_pyramid (SABC : Type) [figure : Figure S A B C] : Prop := 
  has_base ABC ∧ SnotInPlane S

-- Define a predicate for Tetrahedron
def is_tetrahedron (SABC : Type) [figure : Figure S A B C] : Prop := 
  has_four_triangles S A B C

-- The theorem that proves the two names of the figure
theorem name_SABC (S A B C : Type) [figure : Figure S A B C] 
  [base : Base ABC] (SnotInPlane : S ∉ Plane ABC) : 
  (is_triangular_pyramid SABC ∧ is_tetrahedron SABC) := 
by 
  sorry

end name_SABC_l268_268594


namespace red_not_equal_blue_l268_268796

theorem red_not_equal_blue (total_cubes : ℕ) (red_cubes : ℕ) (blue_cubes : ℕ) (edge_length : ℕ)
  (total_surface_squares : ℕ) (max_red_squares : ℕ) :
  total_cubes = 27 →
  red_cubes = 9 →
  blue_cubes = 18 →
  edge_length = 3 →
  total_surface_squares = 6 * edge_length^2 →
  max_red_squares = 26 →
  ¬ (total_surface_squares = 2 * max_red_squares) :=
by
  intros
  sorry

end red_not_equal_blue_l268_268796


namespace no_real_solution_log_eq_l268_268107

theorem no_real_solution_log_eq (x : ℝ) : 
  ¬ (∃ x : ℝ, log 4 (3 * x - 2 * x^2 - 5) = 5 / 2) :=
sorry

end no_real_solution_log_eq_l268_268107


namespace bernardo_receives_l268_268705

theorem bernardo_receives :
  let amount_distributed (n : ℕ) : ℕ := (n * (n + 1)) / 2
  let is_valid (n : ℕ) : Prop := amount_distributed n ≤ 1000
  let bernardo_amount (k : ℕ) : ℕ := (k * (2 + (k - 1) * 3)) / 2
  ∃ k : ℕ, is_valid (15 * 3) ∧ bernardo_amount 15 = 345 :=
sorry

end bernardo_receives_l268_268705


namespace polygons_enclosing_decagon_l268_268453

theorem polygons_enclosing_decagon (m n : ℕ) (hm : m = 10) (hn : m regular polygons enclosing another m-sided regular polygon) : n = 5 :=
sorry

end polygons_enclosing_decagon_l268_268453


namespace base3_sum_l268_268462

theorem base3_sum : 
  (1 * 3^0 - 2 * 3^1 - 2 * 3^0 + 2 * 3^2 + 1 * 3^1 - 1 * 3^0 - 1 * 3^3) = (2 * 3^2 + 1 * 3^1 + 0 * 3^0) := 
by 
  sorry

end base3_sum_l268_268462


namespace ratio_friends_marbles_to_my_marbles_l268_268603

noncomputable def ratio_of_friends_marbles (M B F : ℕ) : ℤ :=
  if M - 2 = 2 * (B + 2) ∧ F = (k : ℕ) * M ∧ M + B + F = 63 ∧ M = 16 
  then F * 8 = 21 * M
  else 0

theorem ratio_friends_marbles_to_my_marbles :
  ∀ (M B F : ℕ), 
  M - 2 = 2 * (B + 2) ∧ F = (k : ℕ) * M ∧ M + B + F = 63 ∧ M = 16 → 
  ratio_of_friends_marbles M B F = 21 :=
by
  sorry

end ratio_friends_marbles_to_my_marbles_l268_268603


namespace tan_squared_identity_l268_268167

theorem tan_squared_identity (α : ℝ) (h : Real.cos (α - π / 4) = √2 / 4) : tan (α + π / 4) ^ 2 = 1 / 7 := 
by
  sorry

end tan_squared_identity_l268_268167


namespace maxAbsValueOnCircle_l268_268140

noncomputable def maxAbsValue (x y : ℝ) : ℝ := |3 * x - y|

theorem maxAbsValueOnCircle {x y : ℝ} (h : abs (Complex.abs (x - 2 + Complex.i * y)) = 1) :
  maxAbsValue x y ≤ 6 + Real.sqrt 10 := by
  sorry

end maxAbsValueOnCircle_l268_268140


namespace dips_to_daps_conversion_l268_268214

variable (daps dops dips : ℝ)

-- Condition: 5 daps are equivalent to 4 dops
def condition1 := 5 * daps = 4 * dops

-- Condition: 3 dops are equivalent to 10 dips
def condition2 := 3 * dops = 10 * dips

-- Theorem: 60 dips are equivalent to 22.5 daps
theorem dips_to_daps_conversion (h1 : condition1) (h2 : condition2) : 60 * dips = 22.5 * daps :=
  sorry

end dips_to_daps_conversion_l268_268214


namespace probability_lakers_win_in_7_games_l268_268715

theorem probability_lakers_win_in_7_games (prob_celtics_win : ℚ) (prob_lakers_win : ℚ) (combinations_6_3 : ℕ) :
  prob_celtics_win = 3 / 4 →
  prob_lakers_win = 1 / 4 →
  combinations_6_3 = 20 →
  (combinations_6_3 * (prob_lakers_win ^ 3) * (prob_celtics_win ^ 3) * prob_lakers_win = 135 / 4096) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have binomial_prob := 20 * ((1/4:ℚ)^3) * ((3/4:ℚ)^3)
  have final_prob := binomial_prob * (1/4:ℚ)
  exact final_prob = 135 / 4096
  sorry

end probability_lakers_win_in_7_games_l268_268715


namespace polynomial_equality_l268_268080

def P (x : ℝ) : ℝ := x ^ 3 - 3 * x ^ 2 - 3 * x - 1

noncomputable def x1 : ℝ := 1 - Real.sqrt 2
noncomputable def x2 : ℝ := 1 + Real.sqrt 2
noncomputable def x3 : ℝ := 1 - 2 * Real.sqrt 2
noncomputable def x4 : ℝ := 1 + 2 * Real.sqrt 2

theorem polynomial_equality :
  P x1 + P x2 = P x3 + P x4 :=
sorry

end polynomial_equality_l268_268080


namespace reciprocal_of_neg_2023_l268_268766

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268766


namespace chessboard_coloring_even_adjacent_l268_268907

theorem chessboard_coloring_even_adjacent (m n : ℕ) (hm : m > 0) (hn : n > 0) :
    (∃ (f : fin m → fin n → bool), 
        ∀ i j, let u := f i j in 
            (u == f i (j + 1) ∧ j + 1 < n) ∨ 
            (u == f (i + 1) j ∧ i + 1 < m) → 
            ((∃ (x y : fin m), f x y == u) → (∃ z : ℕ, f i j = z ∧ z % 2 = 0)))
  ↔ (even m ∨ even n) :=
sorry

end chessboard_coloring_even_adjacent_l268_268907


namespace sin_225_correct_l268_268481

-- Define the condition of point being on the unit circle at 225 degrees.
noncomputable def P_225 := Complex.polar 1 (Real.pi + Real.pi / 4)

-- Define the goal statement that translates the question and correct answer.
theorem sin_225_correct : Complex.sin (Real.pi + Real.pi / 4) = -Real.sqrt 2 / 2 := 
by sorry

end sin_225_correct_l268_268481


namespace domain_of_f_l268_268723

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log x + Real.sqrt (2 - x)

theorem domain_of_f :
  { x : ℝ | 0 < x ∧ x ≤ 2 ∧ x ≠ 1 } = { x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end domain_of_f_l268_268723


namespace find_multiple_l268_268633

-- Definitions based on the problem's conditions
def n_drunk_drivers : ℕ := 6
def total_students : ℕ := 45
def num_speeders (M : ℕ) : ℕ := M * n_drunk_drivers - 3

-- The theorem that we need to prove
theorem find_multiple (M : ℕ) (h1: total_students = n_drunk_drivers + num_speeders M) : M = 7 :=
by
  sorry

end find_multiple_l268_268633


namespace guitar_monthly_payment_l268_268057

theorem guitar_monthly_payment
  (total_owed_with_interest : ℝ)
  (interest_rate : ℝ)
  (loan_term_months : ℕ)
  (monthly_payment : ℝ) :
  total_owed_with_interest = 1320 →
  interest_rate = 0.10 →
  loan_term_months = 12 →
  monthly_payment = (total_owed_with_interest / ((1 + interest_rate) * loan_term_months)) →
  monthly_payment = 100 :=
by
  intros h1 h2 h3 h4
  calc
    monthly_payment = 1320 / ((1 + 0.10) * 12) : by rw [h1, h2, h3]
                ... = 100 : by norm_num

end guitar_monthly_payment_l268_268057


namespace equivalent_proof_l268_268970

-- Define the given function f(x) for general p
def f (x p : ℝ) : ℝ := log (1 - x) + log (p + x)

-- Define the domain conditions
def in_domain (x p : ℝ) : Prop := -p < x ∧ x < 1

-- For p = 1, define the closed interval (-a, a] and find minimum value
noncomputable def f1 (x : ℝ) : ℝ := log (1 - x) + log (1 + x)
def in_interval (x a : ℝ) : Prop := -a < x ∧ x ≤ a

-- The minimum value
def min_value (a : ℝ) : ℝ := log (1 - a^2)

-- Proof statement
theorem equivalent_proof (p : ℝ) (hp : p > -1) :
  (∀ x, in_domain x p → (f x p : ℝ)) ∈ (-p, 1) ∧
  (∀ a : ℝ, 0 < a ∧ a < 1 →
    ∃ x : ℝ, in_interval x a ∧ f1 x = min_value a) :=
begin
  sorry,
end

end equivalent_proof_l268_268970


namespace count_distinct_values_of_m_l268_268680

theorem count_distinct_values_of_m : 
  let m := λ (x1 x2 : ℤ), x1 + x2 in 
  let pairs := [(1, 30), (2, 15), (3, 10), (5, 6), (-1, -30), (-2, -15), (-3, -10), (-5, -6)] in 
  m '' pairs = {31, 17, 13, 11, -31, -17, -13, -11}.card = 8 :=
by
  sorry

end count_distinct_values_of_m_l268_268680


namespace range_of_f_on_interval_l268_268647

def custom_op (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

def f (x : ℝ) : ℝ :=
  (custom_op 1 x) * x

theorem range_of_f_on_interval : 
  Set.range (λ x, f x) (Set.Icc 0 2) = Set.Icc 0 8 :=
begin
  sorry
end

end range_of_f_on_interval_l268_268647


namespace average_attendance_percentage_l268_268356

theorem average_attendance_percentage :
  let total_laborers := 300
  let day1_present := 150
  let day2_present := 225
  let day3_present := 180
  let day1_percentage := (day1_present / total_laborers) * 100
  let day2_percentage := (day2_present / total_laborers) * 100
  let day3_percentage := (day3_present / total_laborers) * 100
  let average_percentage := (day1_percentage + day2_percentage + day3_percentage) / 3
  average_percentage = 61.7 := by
  sorry

end average_attendance_percentage_l268_268356


namespace part1_part2_l268_268564

variable {α : Type*}
variables {A B C : α} {a b c h : ℝ}

-- Definitions representing the given conditions
variable (triangle_ABC : Triangle α A B C)
variable (side_lengths : triangle_ABC.side_lengths = (a, b, c))
variable (height_from_C : triangle_ABC.height_from C A B = h)

-- Theorem to prove part (1)
theorem part1 (triangle_ABC : Triangle α A B C) 
  (side_lengths : triangle_ABC.side_lengths = (a, b, c))
  (height_from_C : triangle_ABC.height_from C A B = h) :
  a + b >= real.sqrt (c^2 + 4 * h^2) :=
begin
  sorry
end

-- Theorem to prove part (2)
theorem part2 (triangle_ABC : Triangle α A B C) 
  (side_lengths : triangle_ABC.side_lengths = (a, b, c))
  (height_from_C : triangle_ABC.height_from C A B = h) :
  (a + b = real.sqrt (c^2 + 4 * h^2)) ↔ is_isosceles_right_triangle triangle_ABC :=
begin
  sorry
end

end part1_part2_l268_268564


namespace even_five_digit_numbers_l268_268707

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def arrangement (n : ℕ) : ℕ := Nat.factorial n

theorem even_five_digit_numbers :
  let odd_numbers := {1, 3, 5}
  let even_numbers := {2, 4, 6, 8}
  let select_odd := choose 3 2
  let select_even := choose 4 3
  let even_units_place := 3
  let remaining_arrangement := arrangement 4
  select_odd * select_even * even_units_place * remaining_arrangement = 864 := by
  sorry

end even_five_digit_numbers_l268_268707


namespace team_structure_ways_l268_268046

open Nat

noncomputable def combinatorial_structure (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem team_structure_ways :
  let total_members := 13
  let team_lead_choices := total_members
  let remaining_after_lead := total_members - 1
  let project_manager_choices := combinatorial_structure remaining_after_lead 3
  let remaining_after_pm1 := remaining_after_lead - 3
  let subordinate_choices_pm1 := combinatorial_structure remaining_after_pm1 3
  let remaining_after_pm2 := remaining_after_pm1 - 3
  let subordinate_choices_pm2 := combinatorial_structure remaining_after_pm2 3
  let remaining_after_pm3 := remaining_after_pm2 - 3
  let subordinate_choices_pm3 := combinatorial_structure remaining_after_pm3 3
  let total_ways := team_lead_choices * project_manager_choices * subordinate_choices_pm1 * subordinate_choices_pm2 * subordinate_choices_pm3
  total_ways = 4804800 :=
by
  sorry

end team_structure_ways_l268_268046


namespace lower_limit_total_people_l268_268104

/-- 
  Given:
    1. Exactly 3/7 of the people in the room are under the age of 21.
    2. Exactly 5/10 of the people in the room are over the age of 65.
    3. There are 30 people in the room under the age of 21.
  Prove: The lower limit of the total number of people in the room is 70.
-/
theorem lower_limit_total_people (T : ℕ) (h1 : (3 / 7) * T = 30) : T = 70 := by
  sorry

end lower_limit_total_people_l268_268104


namespace original_number_of_students_l268_268402

theorem original_number_of_students (x : ℕ)
  (h1: 40 * x / x = 40)
  (h2: 12 * 34 = 408)
  (h3: (40 * x + 408) / (x + 12) = 36) : x = 6 :=
by
  sorry

end original_number_of_students_l268_268402


namespace students_who_like_both_l268_268632

theorem students_who_like_both (B basket cricket total : ℕ) 
  (h_basket : basket = 12) 
  (h_cricket : cricket = 8) 
  (h_total : total = 17) : 
  B = 3 :=
by
  -- Conditions
  have h1 : 17 = 12 + 8 - B := by
    rw [h_total, h_basket, h_cricket]
  -- Solve for B
  have h2 : B = 20 - 17 := by linarith
  rw h2
  norm_num
  sorry -- Proof goes here (omitted)

end students_who_like_both_l268_268632


namespace ratio_first_term_to_common_difference_l268_268804

theorem ratio_first_term_to_common_difference (a d : ℝ) :
  let S_n := λ n, (n / 2) * (2 * a + (n - 1) * d)
  in S_n 10 = 4 * S_n 5 → a / d = 1 / 2 :=
by
  intro S_n h
  sorry

end ratio_first_term_to_common_difference_l268_268804


namespace find_x_coordinate_l268_268457

theorem find_x_coordinate (m b y x : ℝ) (h₀ : m = 3.8666666666666667) (h₁ : b = 20) (h₂ : y = 600) (h₃ : y = m * x + b) :
  x ≈ 150 :=
by
  sorry

end find_x_coordinate_l268_268457


namespace minimum_true_statements_in_circle_l268_268413

theorem minimum_true_statements_in_circle :
  ∀ (heights : Fin 16 → ℕ),
  (∀ i j, i ≠ j → heights i ≠ heights j) →
  ∃₂ i j : Fin 16, (heights ((i + 1) % 16) > heights i) ∧ (heights ((j + 1) % 16) > heights j) :=
sorry

end minimum_true_statements_in_circle_l268_268413


namespace find_divisor_l268_268813

theorem find_divisor :
  ∃ d : ℕ, 15698 = d * 89 + 14 ∧ d = 176 :=
begin
  use 176,
  sorry
end

end find_divisor_l268_268813


namespace age_difference_l268_268401

variables (P M Mo : ℕ)

def patrick_michael_ratio (P M : ℕ) : Prop := (P * 5 = M * 3)
def michael_monica_ratio (M Mo : ℕ) : Prop := (M * 4 = Mo * 3)
def sum_of_ages (P M Mo : ℕ) : Prop := (P + M + Mo = 88)

theorem age_difference (P M Mo : ℕ) : 
  patrick_michael_ratio P M → 
  michael_monica_ratio M Mo → 
  sum_of_ages P M Mo → 
  (Mo - P = 22) :=
by
  sorry

end age_difference_l268_268401


namespace matrix_inverse_self_l268_268505

variable (a b : ℝ)

theorem matrix_inverse_self (h : matrix.of ![![4, -2], ![a, b]] ⬝ matrix.of ![![4, -2], ![a, b]] = 1) :
  a = 15 / 2 ∧ b = -4 :=
by
sorry

end matrix_inverse_self_l268_268505


namespace sin_225_eq_neg_sqrt_two_div_two_l268_268475

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt_two_div_two_l268_268475


namespace initial_students_registered_is_39_l268_268303

variable (X : ℕ)
def total_initial_students_registered := X

axiom monday_attendance_eq : 1.80 * total_initial_students_registered = 70
axiom tuesday_attendance : 70 = 70
axiom wednesday_attendance : X - 28 = X + 2 - 30
axiom transfers: 2 = 5 - 3

theorem initial_students_registered_is_39 : total_initial_students_registered = 39 :=
by
  sorry

end initial_students_registered_is_39_l268_268303


namespace smallest_positive_period_intervals_of_monotonic_decrease_graph_transformation_l268_268180

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + (sqrt 3) * (sin x) * (cos x) + 2 * (cos x) ^ 2

theorem smallest_positive_period (x : ℝ) : is_periodic f π := sorry

theorem intervals_of_monotonic_decrease (k : ℤ) :
  monotonic_decreasing_on f (set.Icc (π / 6 + k * π) (2 * π / 3 + k * π)) := sorry

theorem graph_transformation (x : ℝ) :
  f (x) = sin (2 * x + π / 6) + 3 / 2 := sorry

end smallest_positive_period_intervals_of_monotonic_decrease_graph_transformation_l268_268180


namespace integral_sin3_sin2_l268_268901

open Real

theorem integral_sin3_sin2 : ∫ x in 0..(π / 2), (sin x)^3 * sin (2 * x) = 0.4 :=
by
  sorry

end integral_sin3_sin2_l268_268901


namespace race_probability_l268_268399

/-- In a race where 16 cars are running, the chance that car X will win is 1/4, 
the chance that car Y will win is 1/12, and the chance that car Z will win is 1/7. Assuming that a dead heat is impossible, 
prove that the chance that one of them will win is 10/21. -/

theorem race_probability (hX : ℚ = 1 / 4) (hY : ℚ = 1 / 12) (hZ : ℚ = 1 / 7) :
  (1 / 4) + (1 / 12) + (1 / 7) = 10 / 21 :=
by
  sorry

end race_probability_l268_268399


namespace max_magnitude_of_vector_c_l268_268570

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem max_magnitude_of_vector_c 
  (a b : ℝ × ℝ) 
  (ha : vector_magnitude a = 1) 
  (hb : vector_magnitude b = 1) 
  (hab : a.1 * b.1 + a.2 * b.2 = 0) 
  (c : ℝ × ℝ) 
  (h : (a.1 - c.1) * (b.1 - c.1) + (a.2 - c.2) * (b.2 - c.2) = 0) : 
  vector_magnitude c <= real.sqrt 2 :=
sorry

end max_magnitude_of_vector_c_l268_268570


namespace exists_circle_through_three_points_l268_268928

-- Definitions and assumptions
open_locale classical

-- Distance function on the Euclidean plane
def distance (x y : ℝ × ℝ) : ℝ :=
  real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

-- The main theorem
theorem exists_circle_through_three_points (n : ℕ) (h : n ≥ 3) (points : fin n → ℝ × ℝ)
  (h_noncollinear : ¬ ∃ l : ℝ × ℝ → Prop,
      ∀ i, ∃ a b : ℝ, l (points i) := λ i ⟨a, b⟩, a * (points i).1 + b * (points i).2 = 1 ) :
  ∃ (A B C : fin n), (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧ (∀ (P : fin n), P ≠ A → P ≠ B → P ≠ C →
  ¬ (distance P (circumcenter (points A) (points B) (points C)) <
     circumradius (points A) (points B) (points C))) :=
sorry

-- Circumcenter and circumradius of three points
noncomputable def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
sorry  -- Implementing the circumcenter calculation using perpendicular bisectors

noncomputable def circumradius (A B C : ℝ × ℝ) : ℝ :=
sorry  -- Implementing the circumradius calculation


end exists_circle_through_three_points_l268_268928


namespace expected_number_of_matches_variance_of_number_of_matches_l268_268415

-- Defining the conditions first, and then posing the proof statements
namespace MatchingPairs

open ProbabilityTheory

-- Probabilistic setup for indicator variables
variable (N : ℕ) (prob : ℝ := 1 / N)

-- Indicator variable Ik representing matches
@[simp] def I (k : ℕ) : ℝ := if k < N then prob else 0

-- Define the sum of expected matches S
@[simp] def S : ℝ := ∑ k in finset.range N, I N k

-- Statement: The expectation of the number of matching pairs is 1
theorem expected_number_of_matches : E[S] = 1 := sorry

-- Statement: The variance of the number of matching pairs is 1
theorem variance_of_number_of_matches : Var S = 1 := sorry

end MatchingPairs

end expected_number_of_matches_variance_of_number_of_matches_l268_268415


namespace mary_needs_6_cups_of_flour_l268_268692

-- Define the necessary constants according to the conditions.
def flour_needed : ℕ := 6
def sugar_needed : ℕ := 13
def flour_more_than_sugar : ℕ := 8

-- Define the number of cups of flour Mary needs to add.
def flour_to_add (flour_put_in : ℕ) : ℕ := flour_needed - flour_put_in

-- Prove that Mary needs to add 6 more cups of flour.
theorem mary_needs_6_cups_of_flour (flour_put_in : ℕ) (h : flour_more_than_sugar = 8): flour_to_add flour_put_in = 6 :=
by {
  sorry -- the proof is omitted.
}

end mary_needs_6_cups_of_flour_l268_268692


namespace intersection_of_sets_l268_268686

def A := {x : ℝ | Real.log x / Real.log 2 < 0}
def B := {x : ℝ | (1/3) ^ x < 3}
def C := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_of_sets : A ∩ B = C := 
by
  sorry

end intersection_of_sets_l268_268686


namespace range_of_f_l268_268735

def f (x : ℝ) : ℝ := Math.sin x - Math.cos x

theorem range_of_f :
  set.range f = set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
sorry

end range_of_f_l268_268735


namespace max_non_intersecting_diagonals_l268_268799

theorem max_non_intersecting_diagonals (n : ℕ) (h : n ≥ 3) :
  ∃ k, k ≤ n - 3 ∧ (∀ m, m > k → ¬(m ≤ n - 3)) :=
by
  sorry

end max_non_intersecting_diagonals_l268_268799


namespace problem_statement_l268_268281

theorem problem_statement
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - 3*x^6 + 3*x^4 - x^2 + 2 = 
                 (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + 2*b3*x + c3)) :
  b1 * c1 + b2 * c2 + 2 * b3 * c3 = 0 := 
sorry

end problem_statement_l268_268281


namespace min_ratio_area_l268_268782

-- Define the conditions used in the problem
def is_equilateral (T : Triangle) : Prop :=
  ∀ (A B C: Point), T = ⟨A, B, C⟩ → 
    dist A B = dist B C ∧ dist B C = dist C A

def is_right_triangle (T : Triangle) : Prop :=
  ∃ (D E F : Point), T = ⟨D, E, F⟩ ∧ 
    angle D E F = π / 3 ∧ angle E D F = π / 6

def points_on_sides (T : Triangle) (D E F : Point) : Prop :=
  ∃ (A B C : Point), T = ⟨A, B, C⟩ ∧
    lies_on D (Segment AB) ∧
    lies_on E (Segment BC) ∧
    lies_on F (Segment CA)

-- Define the areas of the triangles
noncomputable def area_triangle (A B C : Point) : Real :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

noncomputable def area_ABC (T : Triangle) : Real :=
  let ⟨A, B, C⟩ := T in area_triangle A B C

noncomputable def area_DEF (T : Triangle) : Real :=
  let ⟨D, E, F⟩ := T in area_triangle D E F

-- The theorem stating the problem's solution
theorem min_ratio_area (ABC DEF : Triangle) (D E F : Point) :
  is_equilateral ABC →
  is_right_triangle DEF →
  points_on_sides ABC D E F →
  ∃ (min_val : Real), min_val = 3 / 14 ∧
  ∀ ratio : Real, ratio = (area_DEF DEF) / (area_ABC ABC) → ratio ≥ min_val :=
by
  sorry

end min_ratio_area_l268_268782


namespace mary_james_not_adjacent_l268_268690

open Finset

def total_combinations (n : ℕ) (r : ℕ) : ℕ :=
  (nat.factorial n) / ((nat.factorial r) * (nat.factorial (n - r)))

def adjacent_pairs (n : ℕ) : ℕ := n - 1

theorem mary_james_not_adjacent :
  (total_combinations 7 2) = 21 →
  (adjacent_pairs 7) = 6 →
  (1 - (adjacent_pairs 7) / (total_combinations 7 2) : ℚ) = 5 / 7 :=
by
  intros h_total_combinations h_adjacent_pairs
  sorry

end mary_james_not_adjacent_l268_268690


namespace perp_DM_IK_l268_268590

variable {α : Type}
variables {A B C I M A' D E K : α}
variables [metric_space α] [normed_add_torsor ℝ α]

noncomputable def circle (O : α) (r : ℝ) : set α := { P | dist P O = r }

def incenter (A B C : α) : α := sorry -- precise definition omitted
def midpoint_arc {P Q : α} (arc : set (metric_space α)) : α := sorry

-- Given Conditions
variables (O : α) (circumcircle : circle O)
variable (triangle : set α := {A, B, C})
variables [triangle ⊆ circumcircle]
def incenter_AABC := incenter A B C
def M := midpoint_arc (arc B C circumcircle)
def antipodal_point (X : α) : α := sorry  -- precise definition omitted
def tangent_point (I O : circle) (BC : line) : α := sorry -- precise definition omitted
def perpendicular (line1 line2 : set α) : Prop := sorry -- precise definition omitted
def intersection (line1 line2 : set α) : α := sorry -- precise definition omitted

-- Definitions based on conditions
def A' := antipodal_point A
def D := tangent_point (circle I r) (line BC)
def E := intersection (line AE) (line BC)
def K := intersection (line A'D) (line ME)

-- Restating the problem
theorem perp_DM_IK : perpendicular (line DM) (line IK) :=
sorry

end perp_DM_IK_l268_268590


namespace determine_x_l268_268894

-- Definitions based on conditions
variables {x : ℝ}

-- Problem statement
theorem determine_x (h : (6 * x)^5 = (18 * x)^4) (hx : x ≠ 0) : x = 27 / 2 :=
by
  sorry

end determine_x_l268_268894


namespace cube_faces_sum_l268_268038

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : a = 12) (h2 : b = 13) (h3 : c = 14)
  (h4 : d = 15) (h5 : e = 16) (h6 : f = 17)
  (h_pairs : a + f = b + e ∧ b + e = c + d) :
  a + b + c + d + e + f = 87 := by
  sorry

end cube_faces_sum_l268_268038


namespace largest_integer_is_48_l268_268123

theorem largest_integer_is_48 (a b c d e : ℕ) (h₀ : a ≤ b) (h₁ : b ≤ c) (h₂ : c ≤ d) (h₃ : d ≤ e)
    (h₄ : a + b = 57) (h₅ : d + e = 83)
    (h₆ : ∃ x y z, {x, y, z} = ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} \ {57, 83}) ∧ {x, y, z} = {70}) :
  e = 48 := by
  sorry

end largest_integer_is_48_l268_268123


namespace probability_allison_brian_noah_l268_268463

theorem probability_allison_brian_noah :
  let p_brian : ℚ := 5 / 6;
  let p_noah : ℚ := 1 / 2;
  p_brian * p_noah = 5 / 12 :=
by
  let p_brian : ℚ := 5 / 6;
  let p_noah : ℚ := 1 / 2;
  show p_brian * p_noah = 5 / 12,
  calc
    5 / 6 * 1 / 2 = (5 * 1) / (6 * 2) : by norm_num
               ... = 5 / 12 : by norm_num

end probability_allison_brian_noah_l268_268463


namespace reciprocal_of_neg_2023_l268_268738

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l268_268738


namespace third_term_of_arithmetic_sequence_is_negative_22_l268_268240

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem third_term_of_arithmetic_sequence_is_negative_22
  (a d : ℤ)
  (H1 : arithmetic_sequence a d 14 = 14)
  (H2 : arithmetic_sequence a d 15 = 17) :
  arithmetic_sequence a d 2 = -22 :=
sorry

end third_term_of_arithmetic_sequence_is_negative_22_l268_268240


namespace sequence_conjecture_l268_268586

theorem sequence_conjecture (a : ℕ → ℝ) (h₁ : a 1 = 7)
  (h₂ : ∀ n, a (n + 1) = 7 * a n / (a n + 7)) :
  ∀ n, a n = 7 / n :=
by
  sorry

end sequence_conjecture_l268_268586


namespace gcd_calculation_l268_268883

theorem gcd_calculation :
  let a := 97^7 + 1
  let b := 97^7 + 97^3 + 1
  gcd a b = 1 := by
  sorry

end gcd_calculation_l268_268883


namespace sample_size_l268_268736

variable (x n : ℕ)

-- Conditions as definitions
def staff_ratio : Prop := 15 * x + 3 * x + 2 * x = 20 * x
def sales_staff : Prop := 30 / n = 15 / 20

-- Main statement to prove
theorem sample_size (h1: staff_ratio x) (h2: sales_staff n) : n = 40 := by
  sorry

end sample_size_l268_268736


namespace sin_225_eq_neg_sqrt_two_div_two_l268_268476

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt_two_div_two_l268_268476


namespace perpendicular_k_value_parallel_k_value_l268_268136

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 2)
def u (k : ℝ) : ℝ × ℝ := (k - 1, 2 * k + 2)
def v : ℝ × ℝ := (4, -4)

noncomputable def is_perpendicular (x y : ℝ × ℝ) : Prop :=
  x.1 * y.1 + x.2 * y.2 = 0

noncomputable def is_parallel (x y : ℝ × ℝ) : Prop :=
  x.1 * y.2 = x.2 * y.1

theorem perpendicular_k_value :
  is_perpendicular (u (-3)) v :=
by sorry

theorem parallel_k_value :
  is_parallel (u (-1/3)) v :=
by sorry

end perpendicular_k_value_parallel_k_value_l268_268136


namespace sequence_sum_10_l268_268183

def a_n (n : ℕ) : ℤ := 11 - 2 * n

def S_n (n : ℕ) : ℤ := (Finset.range n).sum (λ i, abs (a_n (i + 1)))

theorem sequence_sum_10 : S_n 10 = 50 :=
sorry

end sequence_sum_10_l268_268183


namespace find_four_numbers_l268_268359

theorem find_four_numbers (a b c d : ℝ)
  (h1 : a + b + c = 17)
  (h2 : a + b + d = 21)
  (h3 : a + c + d = 25)
  (h4 : b + c + d = 30) :
  {a, b, c, d} = {14, 10, 6, 1} :=
by
  sorry

end find_four_numbers_l268_268359


namespace arithmetic_sequence_length_l268_268191

theorem arithmetic_sequence_length : 
  ∀ {a d l : ℤ}, a = 6 → d = 4 → l = 206 → 
  ∃ n : ℤ, l = a + (n-1) * d ∧ n = 51 := 
by 
  intros a d l ha hd hl
  use (51 : ℤ)
  rw [ha, hd, hl]
  split
  { calc
      206 = 6 + (51 - 1) * 4 : by norm_num }
  { norm_num }

end arithmetic_sequence_length_l268_268191


namespace arithmetic_sequence_length_l268_268190

theorem arithmetic_sequence_length : 
  ∀ {a d l : ℤ}, a = 6 → d = 4 → l = 206 → 
  ∃ n : ℤ, l = a + (n-1) * d ∧ n = 51 := 
by 
  intros a d l ha hd hl
  use (51 : ℤ)
  rw [ha, hd, hl]
  split
  { calc
      206 = 6 + (51 - 1) * 4 : by norm_num }
  { norm_num }

end arithmetic_sequence_length_l268_268190


namespace machine_minutes_worked_l268_268466

theorem machine_minutes_worked {x : ℕ} 
  (h_rate : ∀ y : ℕ, 6 * y = number_of_shirts_machine_makes_yesterday)
  (h_today : 14 = number_of_shirts_machine_makes_today)
  (h_total : number_of_shirts_machine_makes_yesterday + number_of_shirts_machine_makes_today = 156) : 
  x = 23 :=
by
  sorry

end machine_minutes_worked_l268_268466


namespace find_P_l268_268272

noncomputable def P : ℝ × ℝ × ℝ := (5, -3, 4)

def A : ℝ × ℝ × ℝ := (10, 0, 0)
def B : ℝ × ℝ × ℝ := (0, -6, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 8)
def D : ℝ × ℝ × ℝ := (0, 0, 0)

def dist (X Y : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2 + (X.3 - Y.3)^2)

theorem find_P : 
  dist P A = dist P D ∧ 
  dist P B = dist P D ∧ 
  dist P C = dist P D ∧ 
  (P.1 + P.2 + P.3 = 6) :=
by
  -- To be filled with actual proof later
  sorry

end find_P_l268_268272


namespace Christine_picked_10_pounds_l268_268473

-- Variable declarations for the quantities involved
variable (C : ℝ) -- Pounds of strawberries Christine picked
variable (pieStrawberries : ℝ := 3) -- Pounds of strawberries per pie
variable (pies : ℝ := 10) -- Number of pies
variable (totalStrawberries : ℝ := 30) -- Total pounds of strawberries for pies

-- The condition that Rachel picked twice as many strawberries as Christine
variable (R : ℝ := 2 * C)

-- The condition for the total pounds of strawberries picked by Christine and Rachel
axiom strawberries_eq : C + R = totalStrawberries

-- The goal is to prove that Christine picked 10 pounds of strawberries
theorem Christine_picked_10_pounds : C = 10 := by
  sorry

end Christine_picked_10_pounds_l268_268473


namespace f_monotonically_increasing_l268_268577

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log x + 1 / x

-- Prove that f is monotonically increasing on (1, +∞)
theorem f_monotonically_increasing : (∀ x : ℝ, x > 1 ↔ ∃ δ > 0, ∀ (h : ℝ), h < x + δ ∧ h > x - δ → f' h > 0) := 
sorry

end f_monotonically_increasing_l268_268577


namespace min_cost_for_shoes_l268_268300

-- Define the conditions
def number_of_shoes_needed (dogs cats ferrets : ℕ) : ℕ × ℕ :=
  (dogs * 4, (cats + ferrets) * 4)

def cost_packA (medium small : ℕ) : ℕ :=
  let packs_needed := (medium + 3) / 4
  let first_pack := 16
  let second_pack := 13.60
  let total_packs_cost := packs_needed * first_pack + (packs_needed - 1) * second_pack + (packs_needed - 2) * first_pack
  total_packs_cost

def cost_packB (medium small : ℕ) : ℕ :=
  let packs_needed_medium := (medium + 1) / 2
  let packs_needed_small := (small + 1) / 2
  let total_cost_medium := (packs_needed_medium * 9) - ((packs_needed_medium // 4) * 9)
  let total_cost_small := (packs_needed_small * 7) - ((packs_needed_small // 3) * 7)
  total_cost_medium + total_cost_small

-- Define the minimum cost function
def min_cost (dogs cats ferrets : ℕ) : ℕ :=
  let (medium, small) := number_of_shoes_needed dogs cats ferrets
  min (cost_packA medium small) (cost_packB medium small)

-- The main theorem statement
theorem min_cost_for_shoes (dogs : ℕ) (cats : ℕ) (ferrets : ℕ) : min_cost dogs cats ferrets = 64 :=
by
  -- Assuming given conditions: 3 dogs, 2 cats, and 1 ferret as inputs
  have h1 : dogs = 3 := sorry
  have h2 : cats = 2 := sorry
  have h3 : ferrets = 1 := sorry

  -- Calculation using min_cost
  sorry

end min_cost_for_shoes_l268_268300


namespace max_triangle_area_is_sqrt3_l268_268626

noncomputable def max_area_of_triangle (a b c A B C : ℝ) : ℝ :=
  if h : (a + c = 4) ∧ ((2 - real.cos A) * real.tan (B / 2) = real.sin A) then
    let s := (a + b + c) / 2 in
    let area := real.sqrt (s * (s - a) * (s - b) * (s - c)) in
    real.sqrt 3
  else 
    0

-- Theorem: The maximum area of ΔABC is √3, given the conditions
theorem max_triangle_area_is_sqrt3 (a b c A B C : ℝ) :
  (a + c = 4) ∧ ((2 - real.cos A) * real.tan (B / 2) = real.sin A) →
  max_area_of_triangle a b c A B C = real.sqrt 3 := 
by
  sorry

end max_triangle_area_is_sqrt3_l268_268626


namespace min_n_coloring_property_l268_268534

theorem min_n_coloring_property : ∃ n : ℕ, (∀ (coloring : ℕ → Bool), 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧ coloring a = coloring b ∧ coloring b = coloring c → 2 * a + b = c)) ∧ n = 15 := 
sorry

end min_n_coloring_property_l268_268534


namespace reciprocal_of_negative_2023_l268_268754

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l268_268754


namespace library_books_distribution_l268_268841

theorem library_books_distribution :
  let total_books := 8
  let min_books_in_library := 2
  let min_books_checked_out := 2
  ∃ (ways : ℕ), ways = 5 :=
begin
  sorry
end

end library_books_distribution_l268_268841


namespace reciprocal_of_neg_2023_l268_268749

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l268_268749


namespace problem_statement_l268_268201

theorem problem_statement
  (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 16 :=
by
  sorry

end problem_statement_l268_268201


namespace scalene_triangle_angle_bisector_between_median_altitude_l268_268315

theorem scalene_triangle_angle_bisector_between_median_altitude
  (A B C H D M : Type)
  [point A] [point B] [point C]
  [point H] [point D] [point M]
  (is_scalene_triangle : scalene_triangle A B C)
  (is_altitude_foot_H : altitude_foot B H A C)
  (is_angle_bisector_foot_D : angle_bisector_foot B D A C)
  (is_median_foot_M : median_foot B M A C) :
  lies_between D H M :=
sorry

end scalene_triangle_angle_bisector_between_median_altitude_l268_268315


namespace at_most_two_traveling_teams_l268_268820

-- Define the number of teams and the number of matchweeks
variables (n : ℕ) (h_n : n ≥ 2)

-- Define a predicate for a team being traveling
def is_traveling (team: ℕ → ℕ → Prop): Prop :=
∀ i, team i (i + 1) ≠ team (i + 1) (i + 2)

-- Define the main theorem
theorem at_most_two_traveling_teams (number_of_teams: ℕ): number_of_teams = 2 * n →
  ∀ teams : ℕ → (ℕ → ℕ → Prop), (∀ i j, i ≠ j → teams i j ∧ ¬ teams j i) → -- each team met exactly once
  (∀ i, ∃ host guest, teams i j ∧ ¬ teams j i) → -- each match has one host and one guest
  (∀ i j, (team i j → ∀ t, ¬ is_traveling t) →
  (∃ T1 T2 : ℕ, is_traveling T1 ∧ is_traveling T2) 
  ∨ (∀ T3, ¬ is_traveling T3) :=
begin
  sorry
end

end at_most_two_traveling_teams_l268_268820


namespace rent_percentage_l268_268018

variable (E : ℝ)
variable (last_year_rent : ℝ := 0.20 * E)
variable (this_year_earnings : ℝ := 1.20 * E)
variable (this_year_rent : ℝ := 0.30 * this_year_earnings)

theorem rent_percentage (E : ℝ) (h_last_year_rent : last_year_rent = 0.20 * E)
  (h_this_year_earnings : this_year_earnings = 1.20 * E)
  (h_this_year_rent : this_year_rent = 0.30 * this_year_earnings) : 
  this_year_rent / last_year_rent * 100 = 180 := by
  sorry

end rent_percentage_l268_268018


namespace only_exprC_cannot_be_calculated_with_square_of_binomial_l268_268010

-- Definitions of our expressions using their variables
def exprA (a b : ℝ) := (a + b) * (a - b)
def exprB (x : ℝ) := (-x + 1) * (-x - 1)
def exprC (y : ℝ) := (y + 1) * (-y - 1)
def exprD (m : ℝ) := (m - 1) * (-1 - m)

-- Statement that only exprC cannot be calculated using the square of a binomial formula
theorem only_exprC_cannot_be_calculated_with_square_of_binomial :
  (∀ a b : ℝ, ∃ (u v : ℝ), exprA a b = u^2 - v^2) ∧
  (∀ x : ℝ, ∃ (u v : ℝ), exprB x = u^2 - v^2) ∧
  (forall m : ℝ, ∃ (u v : ℝ), exprD m = u^2 - v^2) 
  ∧ (∀ v : ℝ, ¬ ∃ (u : ℝ), exprC v = u^2 ∨ (exprC v = - (u^2))) := sorry

end only_exprC_cannot_be_calculated_with_square_of_binomial_l268_268010


namespace exists_positive_m_dividing_f_100_l268_268933

noncomputable def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_positive_m_dividing_f_100:
  ∃ (m : ℤ), m > 0 ∧ 19881 ∣ (3^100 * (m + 1) - 1) :=
by
  sorry

end exists_positive_m_dividing_f_100_l268_268933


namespace reciprocal_of_x_l268_268605

theorem reciprocal_of_x (x : ℝ) (h1 : x^3 - 2 * x^2 = 0) (h2 : x ≠ 0) : x = 2 → (1 / x = 1 / 2) :=
by {
  sorry
}

end reciprocal_of_x_l268_268605


namespace probability_lakers_win_in_7_games_l268_268716

theorem probability_lakers_win_in_7_games (prob_celtics_win : ℚ) (prob_lakers_win : ℚ) (combinations_6_3 : ℕ) :
  prob_celtics_win = 3 / 4 →
  prob_lakers_win = 1 / 4 →
  combinations_6_3 = 20 →
  (combinations_6_3 * (prob_lakers_win ^ 3) * (prob_celtics_win ^ 3) * prob_lakers_win = 135 / 4096) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have binomial_prob := 20 * ((1/4:ℚ)^3) * ((3/4:ℚ)^3)
  have final_prob := binomial_prob * (1/4:ℚ)
  exact final_prob = 135 / 4096
  sorry

end probability_lakers_win_in_7_games_l268_268716


namespace smallest_value_of_Q_at_neg2_l268_268494

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - 3*x^2 + 4*x - 5

theorem smallest_value_of_Q_at_neg2 :
  Q (-2) = -25 ∧ 
  (∀ val : ℝ, val ∈ {Q (-2), 5, -1} → Q (-2) ≤ val) :=
by
  sorry

end smallest_value_of_Q_at_neg2_l268_268494


namespace arithmetic_sequence_sum_l268_268278

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := 
sorry

end arithmetic_sequence_sum_l268_268278


namespace sum_num_denom_repeating_decimal_is_52_l268_268803

theorem sum_num_denom_repeating_decimal_is_52 : 
  let x := 0.57
  let frac := (19, 33) in
  x == (frac.fst + frac.snd) / frac.snd ∧ frac.fst.gcd frac.snd = 1 → frac.fst + frac.snd = 52 :=
sorry

end sum_num_denom_repeating_decimal_is_52_l268_268803


namespace area_of_estate_l268_268448

theorem area_of_estate (side_length_in_inches : ℝ) (scale : ℝ) (real_side_length : ℝ) (area : ℝ) :
  side_length_in_inches = 12 →
  scale = 100 →
  real_side_length = side_length_in_inches * scale →
  area = real_side_length ^ 2 →
  area = 1440000 :=
by
  sorry

end area_of_estate_l268_268448


namespace riverview_problem_l268_268653

theorem riverview_problem (h c : Nat) (p : Nat := 4 * h) (s : Nat := 5 * c) (d : Nat := 4 * p) :
  (p + h + s + c + d = 52 → false) :=
by {
  sorry
}

end riverview_problem_l268_268653


namespace sqrt_eq_4_implies_x_eq_169_l268_268989

-- Statement of the problem
theorem sqrt_eq_4_implies_x_eq_169 (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
begin
  sorry  -- proof not required
end

end sqrt_eq_4_implies_x_eq_169_l268_268989


namespace spider_can_eat_all_flies_l268_268056

structure Grid (n m : ℕ) :=
  (flies : Set (ℕ × ℕ))
  (spider_start : ℕ × ℕ := (0, 0))

inductive Move
| up | down | left | right

def adjacent (p q : ℕ × ℕ) : Prop :=
  (q = (p.1 + 1, p.2) ∨ q = (p.1 - 1, p.2) ∨ q = (p.1, p.2 + 1) ∨ q = (p.1, p.2 - 1))

theorem spider_can_eat_all_flies (G : Grid 100 100) (moves : List Move) :
  (∀ f ∈ G.flies, ∃ (p : ℕ × ℕ), (p ∈ G.flies) ∧ (adjacent G.spider_start p)) ∧
  (∃ n ≤ 1980, count_moves moves n = 100) :=
sorry

end spider_can_eat_all_flies_l268_268056


namespace round_to_hundredth_l268_268371

theorem round_to_hundredth (x : ℝ) (h : x = 12.8572) : Real.round (x * 100) / 100 = 12.86 := by
  rw [h]
  norm_num
  sorry

end round_to_hundredth_l268_268371


namespace castle_knights_liars_max_knights_l268_268233

-- Define the problem conditions and the goal
theorem castle_knights_liars_max_knights :
  ∃ (K : ℕ) (L : ℕ) (occupants : Fin 16 → Bool), 
    (K + L = 16) ∧ 
    (∀ (i : Fin 16), occupants i = true → 
      ((i.1 % 4 > 0 ∧ occupants ⟨i.1 - 1, by linarith [i.2]⟩ = false) ∨ 
       (i.1 % 4 < 3 ∧ occupants ⟨i.1 + 1, by linarith [i.2]⟩ = false) ∨ 
       (i.1 ≥ 4 ∧ occupants ⟨i.1 - 4, by linarith [i.2]⟩ = false) ∨ 
       (i.1 < 12 ∧ occupants ⟨i.1 + 4, by linarith [i.2]⟩ = false))) ∧ 
    (∀ (i : Fin 16), occupants i = false → 
      ((i.1 % 4 > 0 → occupants ⟨i.1 - 1, by linarith [i.2]⟩ = true) ∧ 
       (i.1 % 4 < 3 → occupants ⟨i.1 + 1, by linarith [i.2]⟩ = true) ∧ 
       (i.1 ≥ 4 → occupants ⟨i.1 - 4, by linarith [i.2]⟩ = true) ∧ 
       (i.1 < 12 → occupants ⟨i.1 + 4, by linarith [i.2]⟩ = true))) ∧ 
    K ≤ 12 := 
sorry

end castle_knights_liars_max_knights_l268_268233


namespace prob_sum_is_18_l268_268610

theorem prob_sum_is_18 : 
  let num_faces := 6
  let num_dice := 4
  let total_outcomes := num_faces ^ num_dice
  ∑ (d1 d2 d3 d4 : ℕ) in finset.Icc 1 num_faces, 
  if d1 + d2 + d3 + d4 = 18 then 1 else 0 = 35 → 
  (35 : ℚ) / total_outcomes = 35 / 648 :=
by
  sorry

end prob_sum_is_18_l268_268610


namespace clowns_attended_l268_268103

-- Definition of the problem's conditions
def num_children : ℕ := 30
def initial_candies : ℕ := 700
def candies_sold_per_person : ℕ := 20
def remaining_candies : ℕ := 20

-- Main theorem stating that 4 clowns attended the carousel
theorem clowns_attended (num_clowns : ℕ) (candies_left: num_clowns * candies_sold_per_person + num_children * candies_sold_per_person = initial_candies - remaining_candies) : num_clowns = 4 := by
  sorry

end clowns_attended_l268_268103


namespace expected_matches_is_one_variance_matches_is_one_l268_268419

noncomputable def indicator (k : ℕ) (matches : Finset ℕ) : ℕ :=
  if k ∈ matches then 1 else 0

def expected_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  (Finset.range N).sum (λ k, indicator k matches / N)

def variance_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  let E_S := expected_matches N matches in
  let E_S2 := (Finset.range N).sum (λ k, (indicator k matches) ^ 2 / N) in
  E_S2 - E_S ^ 2

theorem expected_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  expected_matches N matches = 1 := sorry

theorem variance_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  variance_matches N matches = 1 := sorry

end expected_matches_is_one_variance_matches_is_one_l268_268419


namespace solve_for_x_l268_268915

theorem solve_for_x (x : ℝ) (h : x / 5 + 3 = 4) : x = 5 :=
sorry

end solve_for_x_l268_268915


namespace john_initial_speed_l268_268664

theorem john_initial_speed :
  ∃ S : ℝ, 
  let T := 8 in
  let T' := T + 0.75 * T in
  let S' := S + 4 in
  168 = S' * T' ∧ S = 8 :=
begin
  -- This is where the proof would go  
  sorry
end

end john_initial_speed_l268_268664


namespace largest_unattainable_sum_l268_268248

theorem largest_unattainable_sum (n : ℕ) : ∃ s, s = 12 * n^2 + 8 * n - 1 ∧ 
  ∀ (k : ℕ), k ≤ s → ¬ ∃ a b c d, 
    k = (6 * n + 1) * a + (6 * n + 3) * b + (6 * n + 5) * c + (6 * n + 7) * d := 
sorry

end largest_unattainable_sum_l268_268248


namespace reciprocal_of_neg_2023_l268_268746

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l268_268746


namespace twenty_first_term_is_4641_l268_268003

def nthGroupStart (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

def sumGroup (start n : ℕ) : ℕ :=
  (n * (start + (start + n - 1))) / 2

theorem twenty_first_term_is_4641 : sumGroup (nthGroupStart 21) 21 = 4641 := by
  sorry

end twenty_first_term_is_4641_l268_268003


namespace least_positive_n_l268_268160

theorem least_positive_n : ∃ n : ℕ, (1 / (n : ℝ) - 1 / (n + 1 : ℝ) < 1 / 12) ∧ (∀ m : ℕ, (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 12) → n ≤ m) :=
by {
  sorry
}

end least_positive_n_l268_268160


namespace find_p_at_12_l268_268285

-- Define the polynomial p(x) and the conditions
def p (x : ℝ) : ℝ := (-4 / 53) * x^2 + x + 16 / 53

-- Define the points where p(x) is given specific values
def p_conditions (x : ℝ) :=
  (p 2 = 2) ∧ (p (-2) = -2) ∧ (p 7 = 3)

-- The main theorem statement
theorem find_p_at_12 : p_conditions 0 → p 12 = 60 / 53 :=
by
  intro h
  sorry

end find_p_at_12_l268_268285


namespace knight_tour_impossible_l268_268081

-- Define the properties of the chessboard and the knight
def is_black (position : ℕ × ℕ) : Bool :=
  let (x, y) := position
  (x + y) % 2 = 1

def knight_moves (p : ℕ × ℕ) (q : ℕ × ℕ) : Bool :=
  let (x1, y1) := p
  let (x2, y2) := q
  (|x2 - x1| = 2 ∧ |y2 - y1| = 1) ∨ (|x2 - x1| = 1 ∧ |y2 - y1| = 2)

-- Define the main theorem using the conditions identified
theorem knight_tour_impossible (start_pos : ℕ × ℕ) (end_pos : ℕ × ℕ) 
  (a1_is_black : is_black (1, 1) = true) (h8_is_black : is_black (8, 8) = true)
  (total_squares : 64) (tour_conditions : ∀ (visit_pos : ℕ × ℕ), 
    knight_moves start_pos visit_pos ∧ knight_moves visit_pos end_pos ∧
    visit_pos ≠ start_pos ∧ visit_pos ≠ end_pos) :
  ¬ ∃ seq : Fin 64 → (ℕ × ℕ), 
    (seq 0 = (1, 1)) ∧ (seq (Fin.last 63) = (8, 8)) ∧
    (∀ i, i < 63 → knight_moves (seq i) (seq (i + 1))) :=
sorry

end knight_tour_impossible_l268_268081


namespace simplify_expr_l268_268325

noncomputable def expr1 : ℝ := 3 * Real.sqrt 8 / (Real.sqrt 3 + Real.sqrt 2 + Real.sqrt 7)
noncomputable def expr2 : ℝ := -3.6 * (1 + Real.sqrt 2 - 2 * Real.sqrt 7)

theorem simplify_expr : expr1 = expr2 := by
  sorry

end simplify_expr_l268_268325


namespace manufacturing_department_percentage_l268_268020

theorem manufacturing_department_percentage (total_degrees mfg_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : mfg_degrees = 162) : (mfg_degrees / total_degrees) * 100 = 45 :=
by 
  sorry

end manufacturing_department_percentage_l268_268020


namespace sqrt_26_floor_sq_l268_268514

def floor_sqrt (x : ℝ) : ℤ := Int.floor (Real.sqrt x)

theorem sqrt_26_floor_sq : floor_sqrt 26 ^ 2 = 25 :=
by
  have h : 5 < Real.sqrt 26 := by sorry
  have h' : Real.sqrt 26 < 6 := by sorry
  have floor_sqrt_26_eq_5 : floor_sqrt 26 = 5 := by
    apply Int.floor_eq_iff
    exact ⟨h, h'⟩
  rw [floor_sqrt_26_eq_5]
  exact pow_two 5

end sqrt_26_floor_sq_l268_268514


namespace find_scalar_k_l268_268957

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (a b c : V)

-- Condition given in the problem: a + b + 2c = 0
def condition (a b c : V) : Prop := a + b + 2 • c = 0

-- Main theorem to be proven
theorem find_scalar_k (a b c : V) (h : condition a b c) : 
  ∃ k : ℝ, k = 3 / 2 ∧ k • (b × a) + 2 • (b × c) + (c × a) = 0 :=
sorry

end find_scalar_k_l268_268957


namespace student_mistake_l268_268498

-- Define the quadratic function
def quadratic_function (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Define the conditions
def student_A := quadratic_function a b c 7 = -1
def student_B := quadratic_function a b c 1 = 3
def student_C := quadratic_function a b c 4 = -4
def student_D := quadratic_function a b c 2 = 4

-- State the theorem to determine which student's calculation is incorrect
theorem student_mistake :
  (∃ a b c : ℤ,
    (student_A) ∧
    (¬ student_B) ∧
    (student_C) ∧
    (student_D)) :=
  sorry

end student_mistake_l268_268498


namespace employed_population_percentage_l268_268257

theorem employed_population_percentage
  (P : ℝ) -- Total population
  (E : ℝ) -- Fraction of population that is employed
  (employed_males : ℝ) -- Fraction of population that is employed males
  (employed_females_fraction : ℝ)
  (h1 : employed_males = 0.8 * P)
  (h2 : employed_females_fraction = 1 / 3) :
  E = 0.6 :=
by
  -- We don't need the proof here.
  sorry

end employed_population_percentage_l268_268257


namespace xy_equivalent_ab_l268_268312

theorem xy_equivalent_ab (x y a b : ℤ) 
  (h1 : x + y = a + b) 
  (h2 : x^2 + y^2 = a^2 + b^2) 
  (n : ℤ) : x^n + y^n = a^n + b^n := 
by
  sorry

end xy_equivalent_ab_l268_268312


namespace hyperbola_foci_distance_l268_268335

def hyperbola_asymptotes_intersection (a₁ a₂ b₁ b₂ : ℝ) : (ℝ × ℝ) :=
let x := (b₂ - b₁) / (a₁ - a₂) in
let y := a₁ * x + b₁ in
(x, y)

theorem hyperbola_foci_distance
  (y_eq_2x_plus_3 : ∀ x : ℝ, real) -- y = 2x + 3
  (y_eq_neg2x_plus_1 : ∀ x : ℝ, real) -- y = -2x + 1
  (passes_through_5_5 : ∀ p : ℝ × ℝ, p = (5, 5))
  : ℝ :=
let center := hyperbola_asymptotes_intersection 2 (-2) 3 1 in
let a_squared := (30.25 - 9) in
let b_squared := a_squared in
let c_squared := a_squared + b_squared in
2 * real.sqrt c_squared

#eval hyperbola_foci_distance (λ x, 2 * x + 3) (λ x, -2 * x + 1) (λ p, p = (5, 5))

end hyperbola_foci_distance_l268_268335


namespace incenter_centroid_inside_l268_268391

theorem incenter_centroid_inside (T : Triangle) :
  (incenter T).inside T ∧ (centroid T).inside T :=
by
  sorry

end incenter_centroid_inside_l268_268391


namespace acute_cosine_inequality_l268_268166

variables {α β : ℝ}

-- Definition of acute angles
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

theorem acute_cosine_inequality
  (hα : is_acute α) (hβ : is_acute β) (hcos : cos (α + β) < 0) :
  sin α > cos β :=
begin
  -- Proof will go here.
  sorry
end

end acute_cosine_inequality_l268_268166


namespace log_base_change_l268_268216

theorem log_base_change {x : ℝ} (h : log 4 (x - 3) = 1 / 2) (hx : x = 5) : 
  log 16 x = (log 4 5) / 2 :=
by
  sorry

end log_base_change_l268_268216


namespace angle_B_measure_triangle_area_l268_268258

-- Step c), question (Ⅰ)
theorem angle_B_measure (A C : ℝ) (h : 2 * sin A * sin C * ((1 / (tan A * tan C)) - 1) = -1) :
  ∃ B : ℝ, B = π / 3 := 
sorry

-- Step c), question (Ⅱ)
theorem triangle_area (a c : ℝ) (h1 : a + c = (3 * sqrt 3) / 2) (h2 : b = sqrt 3) :
  ∃ S : ℝ, S = (5 * sqrt 3) / 16 :=
sorry

end angle_B_measure_triangle_area_l268_268258


namespace period_f_f_monotonic_decreasing_interval_f_range_in_interval_l268_268967

noncomputable theory

open Real

def vector_a (x : ℝ) : ℝ × ℝ := (sin (2 * x - π / 3), 1)
def vector_b : ℝ × ℝ := (sqrt 3, -1)
def f (x : ℝ) : ℝ := (sqrt 3) * sin (2 * x - π / 3) - 1

theorem period_f :
  ∃ T : ℝ, T = π ∧ ∀ x, f (x + T) = f x :=
sorry

theorem f_monotonic_decreasing_interval :
  ∀ k : ℤ, ∀ x : ℝ, k * π + 5 * π / 12 ≤ x ∧ x ≤ k * π + 11 * π / 12 →
    ∀ ε, x ≤ x + ε → f (x) ≥ f (x + ε) :=
sorry

theorem f_range_in_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 →
    -5 / 2 ≤ f x ∧ f x ≤ sqrt 3 - 1 :=
sorry

end period_f_f_monotonic_decreasing_interval_f_range_in_interval_l268_268967


namespace shifted_parabola_l268_268334

theorem shifted_parabola (x : ℝ) :
  let original := x^2 in
  let shifted := original + 2 in
  shifted = x^2 + 2 :=
by
  sorry

end shifted_parabola_l268_268334


namespace three_collinear_points_27_points_l268_268974

theorem three_collinear_points_27_points : 
  let vertices := 8
  let midpoints := 12
  let face_centers := 6
  let center := 1
  vertices + midpoints + face_centers + center = 27 → 
  count_collinear_sets vertices midpoints face_centers center = 49 := sorry

end three_collinear_points_27_points_l268_268974


namespace trigonometric_inequalities_l268_268162

noncomputable def a : ℝ := Real.sin (21 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (72 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (23 * Real.pi / 180)

-- The proof statement
theorem trigonometric_inequalities : c > a ∧ a > b :=
by
  sorry

end trigonometric_inequalities_l268_268162


namespace rectangle_not_always_similar_l268_268808

theorem rectangle_not_always_similar :
  ∃ (r₁ r₂ : Type) [rectangle r₁] [rectangle r₂], 
  ¬ ((∀ (s₁ s₂ : ℝ), proportionally_corresponding_sides r₁ r₂ s₁ s₂ ∧ equal_corresponding_angles r₁ r₂) → similar r₁ r₂) :=
sorry

end rectangle_not_always_similar_l268_268808


namespace reciprocal_of_neg_2023_l268_268742

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l268_268742


namespace croissants_for_breakfast_l268_268269

def total_items (C : ℕ) : Prop :=
  C + 18 + 30 = 110

theorem croissants_for_breakfast (C : ℕ) (h : total_items C) : C = 62 :=
by {
  -- The proof might be here, but since it's not required:
  sorry
}

end croissants_for_breakfast_l268_268269


namespace integer_solutions_on_circle_l268_268128

theorem integer_solutions_on_circle : 
  (∀ (x : ℤ), ((3 - x)^2 + (-3 + x)^2 ≤ 81) ∨ ((3 - 2*x)^2 + (-3 - x)^2 ≤ 81)) ↔ 
  (finset.card {x : ℤ | (3 - x)^2 + (-3 + x)^2 ≤ 81 ∨ (3 - 2*x)^2 + (-3 - x)^2 ≤ 81} = 2) :=
begin
  sorry
end

end integer_solutions_on_circle_l268_268128


namespace claire_earnings_l268_268086

theorem claire_earnings
  (total_flowers : ℕ)
  (tulips : ℕ)
  (white_roses : ℕ)
  (price_per_red_rose : ℚ)
  (sell_fraction : ℚ)
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : price_per_red_rose = 0.75)
  (h5 : sell_fraction = 1/2) : 
  (total_flowers - tulips - white_roses) * sell_fraction * price_per_red_rose = 75 :=
by
  sorry

end claire_earnings_l268_268086


namespace negation_example_l268_268731

theorem negation_example :
  (¬ ∃ x : ℝ, x > 0 ∧ 2^x > 1) ↔ ∀ x : ℝ, x > 0 → 2^x ≤ 1 := 
sorry

end negation_example_l268_268731


namespace area_of_kite_region_l268_268112

theorem area_of_kite_region :
  (∃ A, A = 625.125 ∧ ∀ x y : ℝ, abs (x - 50) + abs y = abs (x / 3) → x ∈ Icc 37.5 75 → y ∈ Icc (-16.67) 16.67) := by
sorry

end area_of_kite_region_l268_268112


namespace find_cos_alpha_l268_268165

noncomputable def cos_alpha_value (α : ℝ) : Prop :=
  α ∈ set.Ioo 0 (Real.pi / 2) ∧ 
  Real.sin (Real.pi / 6 - α) = -1 / 3 → 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6

theorem find_cos_alpha {α : ℝ} :
  cos_alpha_value α :=
by
  sorry

end find_cos_alpha_l268_268165


namespace find_min_m_n_l268_268721

theorem find_min_m_n (m n : ℕ) (h1 : m > 1) (h2 : -1 ≤ Real.logBase m (n * x)) (h3 : Real.logBase m (n * x) ≤ 1) (h4 : ∀ x, (1 : ℝ) / (m * n) ≤ x ∧ x ≤ (m : ℝ) / n) :
  m + n = 4333 :=
by 
  sorry

end find_min_m_n_l268_268721


namespace mindy_emails_l268_268694

theorem mindy_emails (P E : ℕ) 
    (h1 : E = 9 * P - 7)
    (h2 : E + P = 93) :
    E = 83 := 
    sorry

end mindy_emails_l268_268694


namespace least_number_divisible_l268_268006

theorem least_number_divisible
  (n : ℕ)
  (h : ∀ k ∈ [24, 32, 36, 54], (n + 6) % k = 0) :
  n = 858 :=
by 
  let k := 24 * 32 / gcd 24 32
  let k := k * 36 / gcd k 36
  let k := k * 54 / gcd k 54
  have h1: k = 864 := by norm_num
  have h2: 858 + 6 = k := by norm_num
  exact h2.symm.trans h1

end least_number_divisible_l268_268006


namespace constant_width_distance_leq_l268_268702

variables {K : Type*} [topological_space K] [metric_space K]

def curve_of_constant_width (K : Type*) (h : ℝ) : Prop :=
  ∀ (A B : K) (L₁ L₂ : set K), tangent_lines A B L₁ L₂ → (∀ x ∈ L₁, ∀ y ∈ L₂, dist x y = h)

theorem constant_width_distance_leq {K : Type*} [topological_space K] [metric_space K] 
  (h : ℝ) (HK : curve_of_constant_width K h) (A B : K) :
  dist A B ≤ h :=
begin
  sorry
end

end constant_width_distance_leq_l268_268702


namespace problem_l268_268939

/-
Given a circle passing through the points (1, 2), (-3, 2), and (-1, 2*sqrt(2)).
1. Prove the equation of the circle is (x+1)^2 + y^2 = 8.
2. Given a chord AB passing through P (-1, 2) with length 2*sqrt(7), prove the equation of line AB is x+y-1=0 or x-y+3=0.
-/

def point (x y : ℝ) := (x, y)

def passes_through (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) := Real.sqrt ((p.1 - c.1)^2 + (p.2 - c.2)^2) = r

def equation_of_circle (x y : ℝ) (a b : ℝ) (r : ℝ) : Prop :=
  ((x - a) ^ 2 + (y - b) ^ 2 = r^2)

def distance_from_center_to_line (c : ℝ × ℝ) (eq_line : ℝ → ℝ → ℝ) := 
  abs (eq_line c.1 c.2) / Real.sqrt ((eq_line 2 0)^2 + (eq_line 0 2)^2)

theorem problem (h1 : passes_through (-1, 0) (Real.sqrt 8) (1, 2)) 
                (h2 : passes_through (-1, 0) (Real.sqrt 8) (-3, 2))
                (h3 : passes_through (-1, 0) (Real.sqrt 8) (-1, 2*Real.sqrt 2))
                (h4 : distance_from_center_to_line (-1, 0) (λ x y, y - 2 - 1 * (x + 1)) = 1)
                (chord_length : Real.abs ((-4) - (6)) / Real.sqrt 2 = 2 * Real.sqrt 7):
                equation_of_circle x y (-1) 0 (Real.sqrt 8) ∧ 
                    ((∀ k, ((k = 1) ∨ (k = -1)) → 
                    (passes_through (-1, 0) (Real.sqrt 8) ((-1 + 2*Real.sqrt 7 * ((1 / k) - 1)/(1 + (1/k)^2)), 2 + 2*Real.sqrt 7 / (1 + (1/k)^2))) ∧ 
                    (passes_through (-1, 0) (Real.sqrt 8) ((-1 + 2*Real.sqrt 7 * (k - 1)/(1 + k^2)), 2 + 2*Real.sqrt 7 / (1 + k^2))) 
                    ∨ chord_length = 2 * Real.sqrt 7 → 
                        ((x + y - 1 = 0) ∨ (x - y + 3 = 0)))) := sorry

end problem_l268_268939


namespace number_of_small_triangles_l268_268850

noncomputable def area_of_large_triangle (hypotenuse_large : ℝ) : ℝ :=
  let leg := hypotenuse_large / Real.sqrt 2
  (1 / 2) * (leg * leg)

noncomputable def area_of_small_triangle (hypotenuse_small : ℝ) : ℝ :=
  let leg := hypotenuse_small / Real.sqrt 2
  (1 / 2) * (leg * leg)

theorem number_of_small_triangles (hypotenuse_large : ℝ) (hypotenuse_small : ℝ) :
  hypotenuse_large = 14 → hypotenuse_small = 2 →
  let number_of_triangles := (area_of_large_triangle hypotenuse_large) / (area_of_small_triangle hypotenuse_small)
  number_of_triangles = 49 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end number_of_small_triangles_l268_268850


namespace ZYX_syndrome_diagnosis_l268_268846

theorem ZYX_syndrome_diagnosis (p : ℕ) (h1 : p = 26) (h2 : ∀ c, c = 2 * p) : ∃ n, n = c / 4 ∧ n = 13 :=
by
  sorry

end ZYX_syndrome_diagnosis_l268_268846


namespace reciprocal_of_neg_2023_l268_268765

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268765


namespace area_of_inner_triangles_l268_268944

theorem area_of_inner_triangles (ABC : Triangle ℝ) (M : Point ℝ) (K : Point ℝ) (L : Point ℝ)
  (hM : M ∈ ABC.segment AB) (hK : K ∈ ABC.segment BC) (hL : L ∈ ABC.segment CA)
  (hM_not_vert : M ≠ A ∧ M ≠ B) (hK_not_vert : K ≠ B ∧ K ≠ C) (hL_not_vert : L ≠ C ∧ L ≠ A) :
  ∃ (∆ : Triangle ℝ), (∆ = Triangle.MAL ∨ ∆ = Triangle.KBM ∨ ∆ = Triangle.LCK) ∧
  Triangle.area ∆ ≤ (1 / 4) * Triangle.area ABC :=
by
  sorry

end area_of_inner_triangles_l268_268944


namespace shape_of_constant_phi_l268_268538

-- Define the spherical coordinates structure
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition that φ is a constant c
def constant_phi (c : ℝ) (coords : SphericalCoordinates) : Prop :=
  coords.φ = c

-- Define the type for shapes
inductive Shape
  | Line : Shape
  | Circle : Shape
  | Plane : Shape
  | Sphere : Shape
  | Cylinder : Shape
  | Cone : Shape

-- The theorem statement
theorem shape_of_constant_phi (c : ℝ) (coords : SphericalCoordinates) 
  (h : constant_phi c coords) : Shape :=
  Shape.Cone

end shape_of_constant_phi_l268_268538


namespace problem_statement_l268_268089

noncomputable def f : ℕ → ℝ
| 0       := 1
| 1       := 0
| (n + 1) := (n + 1) * f n + n * f (n - 1) - f n

theorem problem_statement (n : ℕ) : 
  \frac{f n}{n!} = \sum_{k=0}^n \frac{(-1)^k}{k!} :=
sorry

end problem_statement_l268_268089


namespace temperature_on_friday_is_72_l268_268352

-- Define the temperatures on specific days
def temp_sunday := 40
def temp_monday := 50
def temp_tuesday := 65
def temp_wednesday := 36
def temp_thursday := 82
def temp_saturday := 26

-- Average temperature over the week
def average_temp := 53

-- Total number of days in a week
def days_in_week := 7

-- Calculate the total sum of temperatures given the average temperature
def total_sum_temp : ℤ := average_temp * days_in_week

-- Sum of known temperatures from specific days
def known_sum_temp : ℤ := temp_sunday + temp_monday + temp_tuesday + temp_wednesday + temp_thursday + temp_saturday

-- Define the temperature on Friday
def temp_friday : ℤ := total_sum_temp - known_sum_temp

theorem temperature_on_friday_is_72 : temp_friday = 72 :=
by
  -- Placeholder for the proof
  sorry

end temperature_on_friday_is_72_l268_268352


namespace math_problem_l268_268604

theorem math_problem (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y + x * y = 1) :
  x * y + 1 / (x * y) - y / x - x / y = 4 :=
sorry

end math_problem_l268_268604


namespace samantha_kim_probability_correct_l268_268322

def probability_samantha_and_kim_next_to_each_other : ℚ :=
  2 / 3

theorem samantha_kim_probability_correct :
  (∃ (arrangements : list (list string)), 
    arrangements.length = 4 ∧
    (∀ a b c d : string, 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
      ∃ arrangement ∈ arrangements, 
        (arrangement = [a, b, c, d] ∨ arrangement = [d, a, b, c] ∨ arrangement = [c, d, a, b] ∨ arrangement = [b, c, d, a]) ∧
        list.mem "Samantha" arrangement ∧ list.mem "Kim" arrangement ∧
        (arrangement.index_of "Samantha" + 1 = arrangement.index_of "Kim" ∨ arrangement.index_of "Samantha" - 1 = arrangement.index_of "Kim"))) →
  probability_samantha_and_kim_next_to_each_other = 2 / 3 := 
sorry

end samantha_kim_probability_correct_l268_268322


namespace cylinder_height_relationship_l268_268793

variables (π r₁ r₂ h₁ h₂ : ℝ)

theorem cylinder_height_relationship
  (h_volume_eq : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius_rel : r₂ = 1.2 * r₁) :
  h₁ = 1.44 * h₂ :=
by {
  sorry -- proof not required as per instructions
}

end cylinder_height_relationship_l268_268793


namespace train_crossing_lamp_post_l268_268054

theorem train_crossing_lamp_post (bridge_length train_length time_cross_bridge : ℝ) 
  (h_bridge_length : bridge_length = 200) 
  (h_train_length : train_length = 200)
  (h_time_cross_bridge : time_cross_bridge = 10) : 
  let v := (bridge_length + train_length) / time_cross_bridge in
  let t_prime := train_length / v in
  t_prime = 5 := 
by
  -- We'll insert the proof here
  sorry

end train_crossing_lamp_post_l268_268054


namespace roots_of_polynomial_l268_268507

noncomputable def polynomial (m z : ℝ) : ℝ :=
  z^3 - (m^2 - m + 7) * z - (3 * m^2 - 3 * m - 6)

theorem roots_of_polynomial (m z : ℝ) (h : polynomial m (-1) = 0) :
  (m = 3 ∧ z = 4 ∨ z = -3) ∨ (m = -2 ∧ sorry) :=
sorry

end roots_of_polynomial_l268_268507


namespace solve_frac_eq_l268_268013

theorem solve_frac_eq (x : ℝ) (h : 1 / x^2 + 2 * (1 / x) = 1.25) :
  x = 2 ∨ x = -(2 / 5) :=
begin
  sorry
end

end solve_frac_eq_l268_268013


namespace last_digit_zero_in_modified_fibonacci_l268_268730

def modified_fibonacci (n : Nat) : Nat :=
  Nat.recOn n
    2
    (λ _ ih, Nat.recOn ih 1 (λ _ ih1 ih2, (ih1 + ih2) % 10))

theorem last_digit_zero_in_modified_fibonacci :
  ∃ n, (∀ m < n, modified_fibonacci m < 10) ∧ (modified_fibonacci n = 0) :=
  sorry

end last_digit_zero_in_modified_fibonacci_l268_268730


namespace determinant_geometric_sequence_zero_l268_268600

variables (a1 a2 a3 a4 : ℝ)
-- Condition: a_1, a_2, a_3, a_4 are in geometric sequence.
def geometric_sequence (a1 a2 a3 a4 : ℝ) : Prop := 
  ∃ r : ℝ, a2 = r * a1 ∧ a3 = r * a2 ∧ a4 = r * a3

-- Theorem: The determinant of the given 2x2 matrix is 0.
theorem determinant_geometric_sequence_zero :
  geometric_sequence a1 a2 a3 a4 →
  (a1 * a4 - a2 * a3 = 0) :=
by
  intros h
  sorry

end determinant_geometric_sequence_zero_l268_268600


namespace range_of_a_l268_268130

theorem range_of_a (a : ℝ) : (∃ x > 0, (2 * x - a) / (x + 1) = 1) ↔ a > -1 :=
by {
    sorry
}

end range_of_a_l268_268130


namespace lateral_surface_area_of_prism_l268_268658

theorem lateral_surface_area_of_prism : 
  let x : ℝ
  let r : ℝ := sqrt (4 / 3)
  let lateral_edge_length := 2 * x
  let base_height := x * (sqrt 3 / 2)
  let base_area := (sqrt 3 / 4) * x^2
  let lateral_surface_area := 3 * x * lateral_edge_length
  in 
  r^2 = 4 / 3 → base_height > 0 → base_area > 0 → lateral_surface_area = 6 :=
by 
  intros x r lateral_edge_length base_height base_area lateral_surface_area
  -- derive required conditions from the given problem
  assume hr2 : r^2 = 4 / 3,
  assume h_base_height : base_height > 0,
  assume h_base_area : base_area > 0,
  -- state the final result
  show lateral_surface_area = 6, from sorry

end lateral_surface_area_of_prism_l268_268658


namespace p_investment_amount_l268_268814

-- Define the conditions
variables (P : ℝ) (q_investment : ℝ) (p_profit_ratio q_profit_ratio : ℝ)
variables (ratio_condition : p_profit_ratio / q_profit_ratio = 3 / 5)
variables (q_investment_condition : q_investment = 30000)

-- The proof statement
theorem p_investment_amount :
  (P / q_investment) = (3 / 5) → P = 18000 :=
begin
  intro h,
  rw q_investment_condition at h,
  rw ← mul_eq_mul_right_iff at h,
  { cases h,
    { have h2 : 5 * P = 3 * 30000 := by linarith,
      rw mul_comm at h2,
      have h3 : P = 18000 := by linarith,
      exact h3,
    },
    { have h2 : P = 0 := by linarith,
      exact h2,
    },
  },
  sorry
end

end p_investment_amount_l268_268814


namespace greatest_third_side_l268_268998

theorem greatest_third_side (a b c : ℝ) (h₀: a = 5) (h₁: b = 11) (h₂ : 6 < c ∧ c < 16) : c ≤ 15 :=
by
  -- assumption applying that c needs to be within 6 and 16
  have h₃ : 6 < c := h₂.1
  have h₄: c < 16 := h₂.2
  -- need to show greatest integer c is 15
  sorry

end greatest_third_side_l268_268998


namespace exists_positive_integer_m_such_that_sqrt_8m_is_integer_l268_268306

theorem exists_positive_integer_m_such_that_sqrt_8m_is_integer :
  ∃ (m : ℕ), m > 0 ∧ ∃ (k : ℕ), 8 * m = k^2 :=
by
  sorry

end exists_positive_integer_m_such_that_sqrt_8m_is_integer_l268_268306


namespace problem_solution_l268_268061

def irrational_numbers_count : ℕ :=
  let s := {3.14, Real.sqrt 7, Real.pi, 1/3, 0.1010010001} in
    s.count (λ x, ¬ Rational x)

theorem problem_solution : irrational_numbers_count = 3 := 
by
  sorry

end problem_solution_l268_268061


namespace certain_number_sum_421_l268_268437

theorem certain_number_sum_421 :
  ∃ n, (∃ k, n = 423 * k) ∧ k = 2 →
  n + 421 = 1267 :=
by
  sorry

end certain_number_sum_421_l268_268437


namespace probability_of_distance_l268_268943

noncomputable def square_side_length : ℝ := 1
def center_coord : (ℝ × ℝ) := (0, 0)
def vertex_coords : (List (ℝ × ℝ)) := [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)]

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def prob_correct_dist : ℝ :=
  let all_points := [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)] in
  let total_ways := Finset.card (Finset.powersetLen 2 (Finset.fromList all_points)) in
  let favorable_ways := Finset.card (Finset.filter (λ (pair : Finset (ℝ × ℝ)), distance (List.head pair) (List.head (List.tail pair)) = Real.sqrt 2 / 2)
                      (Finset.powersetLen 2 (Finset.fromList all_points))) in
  favorable_ways / total_ways

theorem probability_of_distance : prob_correct_dist = 2 / 5 := sorry

end probability_of_distance_l268_268943


namespace expected_number_of_matches_variance_of_number_of_matches_l268_268425

-- Definitions of conditions
def num_pairs (N : ℕ) : Type := Fin N -> bool -- Type representing pairs of cards matching or not for an N-pair scenario

def indicator_variable (N : ℕ) (k : Fin N) : num_pairs N -> Prop :=
  λ (pairs : num_pairs N), pairs k

def matching_probability (N : ℕ) : ℝ :=
  1 / (N : ℝ)

-- Statement of the first proof problem
theorem expected_number_of_matches (N : ℕ) (pairs : num_pairs N) : 
  (∑ k, (if indicator_variable N k pairs then 1 else 0)) / N = 1 :=
sorry

-- Statement of the second proof problem
theorem variance_of_number_of_matches (N : ℕ) (pairs : num_pairs N) :
  (∑ k, (if indicator_variable N k pairs then 1 else 0) * (if indicator_variable N k pairs then 1 else 0) + 
  2 * ∑ i j, if i ≠ j then 
  (if indicator_variable N i pairs then 1 else 0) * (if indicator_variable N j pairs then 1 else 0) else 0) - 
  ((∑ k, (if indicator_variable N k pairs then 1 else 0)) / N) ^ 2 = 1 :=
sorry

end expected_number_of_matches_variance_of_number_of_matches_l268_268425


namespace reciprocal_of_negative_2023_l268_268755

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l268_268755


namespace find_number_satisfying_condition_l268_268903

-- Define the condition where fifteen percent of x equals 150
def fifteen_percent_eq (x : ℝ) : Prop :=
  (15 / 100) * x = 150

-- Statement to prove the existence of a number x that satisfies the condition, and this x equals 1000
theorem find_number_satisfying_condition : ∃ x : ℝ, fifteen_percent_eq x ∧ x = 1000 :=
by
  -- Proof will be added here
  sorry

end find_number_satisfying_condition_l268_268903


namespace sqrt_eq_4_implies_x_eq_169_l268_268987

-- Statement of the problem
theorem sqrt_eq_4_implies_x_eq_169 (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
begin
  sorry  -- proof not required
end

end sqrt_eq_4_implies_x_eq_169_l268_268987


namespace largest_angle_90_degrees_l268_268055

def triangle_altitudes (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  (9 * a = 12 * b) ∧ (9 * a = 18 * c)

theorem largest_angle_90_degrees (a b c : ℝ) 
  (h : triangle_altitudes a b c) : 
  exists (A B C : ℝ) (hApos : A > 0) (hBpos : B > 0) (hCpos : C > 0),
    (A^2 = B^2 + C^2) ∧ (B * C / 2 = 9 * a / 2 ∨ 
                         B * A / 2 = 12 * b / 2 ∨ 
                         C * A / 2 = 18 * c / 2) :=
sorry

end largest_angle_90_degrees_l268_268055


namespace solve_exponential_eq_l268_268709

theorem solve_exponential_eq (x : ℝ) (h : 4^x * 4^x * 16^(x+1) = 1024^2) : x = 2 :=
sorry

end solve_exponential_eq_l268_268709


namespace floor_sqrt_26_squared_l268_268512

theorem floor_sqrt_26_squared :
  (⟨5 < Real.sqrt 26, Real.sqrt 26 < 6⟩) ∧ (Real.sqrt 25 = 5) ∧ (Real.sqrt 36 = 6) → ⌊Real.sqrt 26⌋^2 = 25 := 
by
  introduction
  sorry

end floor_sqrt_26_squared_l268_268512


namespace Kostya_always_wins_l268_268795

-- Definitions representing the game and its conditions
def players_turn_order (turn : ℕ) : string :=
  [ "Vanya", "Kostya", "Lesha" ].get (turn % 3)

-- Conditions for each player picking matches
def Vanya_picks : ℕ → Prop :=
  λ n, n = 1 ∨ n = 2

def Lesha_picks : ℕ → Prop :=
  λ n, n = 1 ∨ n = 2

def Kostya_picks : ℕ → Prop :=
  λ n, n = 1 ∨ n = 2 ∨ n = 3

-- Initial state of the game
def initial_matches : ℕ := 2008

-- Winning condition
def wins (matches_left : ℕ) : Prop :=
  matches_left = 0

-- Kostya's winning strategy predicate
def kostya_wins : Prop :=
  ∀ matches_left, matches_left > 0 → 
    ∃ n, Kostya_picks n ∧ 
        ∀ v, Vanya_picks v → 
          ∀ l, Lesha_picks l → 
            wins (matches_left - n - v - l) ∨ 
            ∃ k_next, Kostya_picks k_next ∧ 
              wins (matches_left - n - v - l - k_next)

theorem Kostya_always_wins : kostya_wins :=
  sorry

end Kostya_always_wins_l268_268795


namespace ratio_of_width_perimeter_is_3_16_l268_268049

-- We define the conditions
def length_of_room : ℕ := 25
def width_of_room : ℕ := 15

-- We define the calculation and verification of the ratio
theorem ratio_of_width_perimeter_is_3_16 :
  let P := 2 * (length_of_room + width_of_room)
  let ratio := width_of_room / P
  let a := 15 / Nat.gcd 15 80
  let b := 80 / Nat.gcd 15 80
  (a, b) = (3, 16) :=
by 
  -- The proof is skipped with sorry
  sorry

end ratio_of_width_perimeter_is_3_16_l268_268049


namespace spring_length_increase_l268_268461

-- Define the weight (x) and length (y) data points
def weights : List ℝ := [0, 1, 2, 3, 4, 5]
def lengths : List ℝ := [20, 20.5, 21, 21.5, 22, 22.5]

-- Prove that for each increase of 1 kg in weight, the length of the spring increases by 0.5 cm
theorem spring_length_increase (h : weights.length = lengths.length) :
  ∀ i, i < weights.length - 1 → (lengths.get! (i+1) - lengths.get! i) = 0.5 :=
by
  -- Proof goes here, omitted for now
  sorry

end spring_length_increase_l268_268461


namespace find_a_l268_268962

/-- Given function -/
def f (x: ℝ) : ℝ := (x + 1)^2 - 2 * (x + 1)

/-- Problem statement -/
theorem find_a (a : ℝ) (h : f a = 3) : a = 2 ∨ a = -2 := 
by
  sorry

end find_a_l268_268962


namespace reciprocal_of_neg_2023_l268_268764

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268764


namespace percentage_x_minus_y_l268_268994

variable (x y : ℝ)

theorem percentage_x_minus_y (P : ℝ) :
  P / 100 * (x - y) = 20 / 100 * (x + y) ∧ y = 20 / 100 * x → P = 30 :=
by
  intros h
  sorry

end percentage_x_minus_y_l268_268994


namespace find_abc_sum_l268_268329

variables {x : ℤ}

def poly1 (a b : ℤ) : Polynomial ℤ := Polynomial.C b + Polynomial.X * (Polynomial.C a + Polynomial.X)
def poly2 (b c : ℤ) : Polynomial ℤ := Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X)

-- Conditions
axiom gcd_condition (a b c : ℤ) : Polynomial.gcd (poly1 a b) (poly2 b c) = Polynomial.C 1 * (Polynomial.X + Polynomial.C 1)
axiom lcm_condition (a b c : ℤ) : Polynomial.lcm (poly1 a b) (poly2 b c) = Polynomial.X^3 - Polynomial.C 2 * Polynomial.X^2 - Polynomial.C 7 * Polynomial.X - Polynomial.C 6

-- Theorem statement
theorem find_abc_sum (a b c : ℤ) : a + b + c = 6 :=
by
  sorry

end find_abc_sum_l268_268329


namespace sine_fourier_transform_eq_l268_268533

noncomputable def f (x : ℝ) : ℝ :=
  if h₀ : 0 < x ∧ x < 1 then 0 else
  if h₁ : 1 < x ∧ x < 2 then 1 else
  0

def F_s (p : ℝ) : ℝ :=
  sqrt (2 / Real.pi) * (Real.cos p - Real.cos (2 * p)) / p

theorem sine_fourier_transform_eq (p : ℝ) : 
  F_s p = sqrt (2 / Real.pi) * (Real.cos p - Real.cos (2 * p)) / p :=
by
  sorry

end sine_fourier_transform_eq_l268_268533


namespace collinear_H_O_D_l268_268638

variable (A B C H M N O D : Type)
variable [AcuteTriangle A B C]
variable (angleGtr60 : angle B A C > 60)
variable [Orthocenter H A B C]
variable (lineMB : M ∈ lineSegment A B)
variable (lineNC : N ∈ lineSegment A C)
variable (angleHMB_eq_60 : angle H M B = 60)
variable (angleHNC_eq_60 : angle H N C = 60)
variable [Circumcenter O H M N]
variable [EquilateralDBC D B C]
variable (same_side_A : SameSideOfLine A D (line B C))

theorem collinear_H_O_D :
  collinear {H, O, D} :=
sorry

end collinear_H_O_D_l268_268638


namespace eric_has_more_than_500_paperclips_on_saturday_l268_268511

theorem eric_has_more_than_500_paperclips_on_saturday :
  ∃ k : ℕ, (4 * 3 ^ k > 500) ∧ (∀ m : ℕ, m < k → 4 * 3 ^ m ≤ 500) ∧ ((k + 1) % 7 = 6) :=
by
  sorry

end eric_has_more_than_500_paperclips_on_saturday_l268_268511


namespace max_value_S_n_minus_S_m_l268_268587

noncomputable def a : ℕ → ℤ :=
  λ n, -n ^ 2 + 12 * n - 32

noncomputable def S : ℕ → ℤ :=
  λ n, (Finset.range n).sum a

theorem max_value_S_n_minus_S_m (n m : ℕ) (hm : 0 < m) (hnm : m < n) :
  S n - S m ≤ 10 :=
sorry

end max_value_S_n_minus_S_m_l268_268587


namespace count_odd_numbers_distinct_digits_l268_268028

theorem count_odd_numbers_distinct_digits : 
  ∃ n : ℕ, (∀ x : ℕ, 200 ≤ x ∧ x ≤ 999 ∧ x % 2 = 1 ∧ (∀ d ∈ [digit1, digit2, digit3], d ≤ 7) ∧ (digit1 ≠ digit2 ∧ digit2 ≠ digit3 ∧ digit1 ≠ digit3) → True) ∧
  n = 120 :=
sorry

end count_odd_numbers_distinct_digits_l268_268028


namespace digits_in_product_l268_268596

theorem digits_in_product : 
  ∀ (a b : ℕ), a = 3^4 → b = 6^8 → nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b ha hb,
  rw [ha, hb],
  calc
    (nat.log10 (3^4 * 6^8) + 1)
        = (nat.log10 ((3^4 * 6^8)) + 1) : by sorry
        ... = 9 : by sorry
end

end digits_in_product_l268_268596


namespace cos_angle_between_planes_l268_268670

noncomputable def cos_theta_between_planes : ℝ :=
  let n1 : ℝ × ℝ × ℝ := (1, -3, 2)
  let n2 : ℝ × ℝ × ℝ := (3, 2, -4)
  let dot_product n1 n2 := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3
  let magnitude n := Real.sqrt (n.1^2 + n.2^2 + n.3^2)
  let cos_theta := dot_product n1 n2 / (magnitude n1 * magnitude n2)
  cos_theta

theorem cos_angle_between_planes 
  (plane1_eq : ∀ x y z, x - 3 * y + 2 * z - 4 = 0)
  (plane2_eq : ∀ x y z, 3 * x + 2 * y - 4 * z + 7 = 0) :
  cos_theta_between_planes = -11 / Real.sqrt 406 := 
  sorry

end cos_angle_between_planes_l268_268670


namespace average_is_12_or_15_l268_268092

variable {N : ℝ} (h : 12 < N ∧ N < 22)

theorem average_is_12_or_15 : (∃ x ∈ ({12, 15} : Set ℝ), x = (9 + 15 + N) / 3) :=
by
  have h1 : 12 < (24 + N) / 3 := by sorry
  have h2 : (24 + N) / 3 < 15.3333 := by sorry
  sorry

end average_is_12_or_15_l268_268092


namespace largest_unattainable_sum_l268_268246

noncomputable def largestUnattainableSum (n : ℕ) : ℕ :=
  12 * n^2 + 8 * n - 1

theorem largest_unattainable_sum (n : ℕ) :
  ∀ s, (¬∃ a b c d, s = (a * (6 * n + 1) + b * (6 * n + 3) + c * (6 * n + 5) + d * (6 * n + 7)))
  ↔ s > largestUnattainableSum n := by
  sorry

end largest_unattainable_sum_l268_268246


namespace line_intersects_circle_l268_268996

theorem line_intersects_circle 
  (k : ℝ)
  (h_tangent : ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 2 → y = k * x - 1) :
  ∀ x y : ℝ, ((x - 2)^2 + y^2 = 3) → (y = k * x - 1) ∨ (y ≠ k * x - 1) :=
by
  sorry

end line_intersects_circle_l268_268996


namespace unique_sundaes_l268_268066

theorem unique_sundaes (flavors : ℕ) (h : flavors = 8) : 
  (finset.card (finset.powerset_len 1 (finset.range flavors)) +
   finset.card (finset.powerset_len 2 (finset.range flavors)) = 36) :=
by
  have h1 : finset.card (finset.powerset_len 1 (finset.range flavors)) = 8,
  {
    rw [h, finset.card_powerset_len],
    norm_num,
  },
  have h2 : finset.card (finset.powerset_len 2 (finset.range flavors)) = 28,
  {
    rw [h, finset.card_powerset_len],
    norm_num,
  },
  rw [h1, h2],
  norm_num,
  sorry

end unique_sundaes_l268_268066


namespace equivalence_of_daps_and_dips_l268_268211

variable (daps dops dips : Type)
variable (to_dops : daps → dops)
variable (to_dips : dops → dips)

-- Conditions given
axiom cond1 : ∀ x : daps, to_dops (5 * x) = 4 * to_dops x
axiom cond2 : ∀ y : dops, to_dips (3 * y) = 10 * to_dips y

-- The statement to prove
theorem equivalence_of_daps_and_dips : 
  ∃ x : daps, to_dips (to_dops x) * 60 = 22.5 * x :=
  sorry

end equivalence_of_daps_and_dips_l268_268211


namespace transport_cost_l268_268713

-- Define the conditions
def cost_per_kg : ℕ := 15000
def grams_per_kg : ℕ := 1000
def weight_in_grams : ℕ := 500

-- Define the main theorem stating the proof problem
theorem transport_cost
  (c : ℕ := cost_per_kg)
  (gpk : ℕ := grams_per_kg)
  (w : ℕ := weight_in_grams)
  : c * w / gpk = 7500 :=
by
  -- Since we are not required to provide the proof, adding sorry here
  sorry

end transport_cost_l268_268713


namespace product_of_integers_is_eight_l268_268351

-- Define three different positive integers a, b, c such that they sum to 7
def sum_to_seven (a b c : ℕ) : Prop := a + b + c = 7 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Prove that the product of these integers is 8
theorem product_of_integers_is_eight (a b c : ℕ) (h : sum_to_seven a b c) : a * b * c = 8 := by sorry

end product_of_integers_is_eight_l268_268351


namespace odd_function_behavior_l268_268224

theorem odd_function_behavior (f : ℝ → ℝ)
  (h_odd: ∀ x, f (-x) = -f x)
  (h_increasing: ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_max: ∀ x, 3 ≤ x → x ≤ 7 → f x ≤ 5) :
  (∀ x, -7 ≤ x → x ≤ -3 → f x ≥ -5) ∧ (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) :=
sorry

end odd_function_behavior_l268_268224


namespace number_of_ways_to_form_divisible_number_l268_268252

def valid_digits : List ℕ := [0, 2, 4, 7, 8, 9]

def is_divisible_by_4 (d1 d2 : ℕ) : Prop :=
  (d1 * 10 + d2) % 4 = 0

def is_divisible_by_3 (sum_of_digits : ℕ) : Prop :=
  sum_of_digits % 3 = 0

def replace_asterisks_to_form_divisible_number : Prop :=
  ∃ (a1 a2 a3 a4 a5 l : ℕ), a1 ∈ valid_digits ∧ a2 ∈ valid_digits ∧ a3 ∈ valid_digits ∧ a4 ∈ valid_digits ∧ a5 ∈ valid_digits ∧
  l ∈ [0, 2, 4, 8] ∧
  is_divisible_by_4 0 l ∧
  is_divisible_by_3 (11 + a1 + a2 + a3 + a4 + a5) ∧
  (4 * 324 = 1296)

theorem number_of_ways_to_form_divisible_number :
  replace_asterisks_to_form_divisible_number :=
  sorry

end number_of_ways_to_form_divisible_number_l268_268252


namespace math_problem_statement_l268_268148

noncomputable def ellipse_standard_equation {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a > b) (focus : ℝ × ℝ) (point_on_ellipse : ℝ × ℝ) : Prop :=
  ∃ (ell_eq : ℝ → ℝ → Prop), 
    ell_eq = (λ x y, x^2 / a^2 + y^2 / b^2 = 1) ∧
      let F := focus in
      let M := point_on_ellipse in
      F.1 = 1 ∧ F.2 = 0 ∧ 
      M.1 = sqrt 6 / 2 ∧ M.2 = 1 / 2 ∧ 
      a = sqrt 2 ∧ b = 1 ∧ 
      ell_eq M.1 M.2

noncomputable def minimum_value_OQ (P : ℝ × ℝ) (λ : ℝ) (hλ : 0 < λ) : ℝ :=
  let O := (0, 0) in
  let Q := (1/2, 1/2) in
  dist O Q

/-- Main theorem combining both parts of the problem -/
theorem math_problem_statement {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a > b) 
  (focus : ℝ × ℝ) (point_on_ellipse : ℝ × ℝ) (P : ℝ × ℝ) (λ : ℝ) (hλ : 0 < λ) :
  ellipse_standard_equation ha hb hab focus point_on_ellipse ∧ minimum_value_OQ P λ hλ = sqrt 2 / 2 :=
  sorry

end math_problem_statement_l268_268148


namespace expected_number_of_matches_variance_of_number_of_matches_l268_268422

-- Definitions of conditions
def num_pairs (N : ℕ) : Type := Fin N -> bool -- Type representing pairs of cards matching or not for an N-pair scenario

def indicator_variable (N : ℕ) (k : Fin N) : num_pairs N -> Prop :=
  λ (pairs : num_pairs N), pairs k

def matching_probability (N : ℕ) : ℝ :=
  1 / (N : ℝ)

-- Statement of the first proof problem
theorem expected_number_of_matches (N : ℕ) (pairs : num_pairs N) : 
  (∑ k, (if indicator_variable N k pairs then 1 else 0)) / N = 1 :=
sorry

-- Statement of the second proof problem
theorem variance_of_number_of_matches (N : ℕ) (pairs : num_pairs N) :
  (∑ k, (if indicator_variable N k pairs then 1 else 0) * (if indicator_variable N k pairs then 1 else 0) + 
  2 * ∑ i j, if i ≠ j then 
  (if indicator_variable N i pairs then 1 else 0) * (if indicator_variable N j pairs then 1 else 0) else 0) - 
  ((∑ k, (if indicator_variable N k pairs then 1 else 0)) / N) ^ 2 = 1 :=
sorry

end expected_number_of_matches_variance_of_number_of_matches_l268_268422


namespace max_distance_theorem_l268_268157

-- Define the moving point P on the line y = -2
def P (t : ℝ) : ℝ × ℝ := (t, -2)

-- Define the circle with the equation x^2 + y^2 = 1
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define points A, B, and C
def A : ℝ × ℝ := (2, 1)

-- Assume B and C are points of tangency, define a line through B and C
def lineBC (t x y : ℝ) : Prop := t * x - 2 * y = 1

-- Define the distance from point A to the line BC
def distance_from_A_to_BC (t : ℝ) : ℝ :=
  abs (2 * t - 3) / real.sqrt (t^2 + 4)

-- Define the maximum value of the distance
def max_distance : ℝ := 5 / 2

-- The theorem statement to prove the maximum distance is 5/2
theorem max_distance_theorem : 
  ∀ t : ℝ, circle (P t).1 (P t).2 → 
  ∃ B C : ℝ × ℝ, 
    lineBC t B.1 B.2 → 
    lineBC t C.1 C.2 → 
    distance_from_A_to_BC t ≤ max_distance :=
sorry

end max_distance_theorem_l268_268157


namespace prove_tan2alpha_and_cosbeta_l268_268554

theorem prove_tan2alpha_and_cosbeta (α β : ℝ) (h1 : cos α = 5 / 13)
                                   (h2 : cos (α - β) = 4 / 5)
                                   (h3 : 0 < β ∧ β < α ∧ α < π / 2) :
  tan (2 * α) = -120 / 119 ∧ cos β = 56 / 65 :=
by
  sorry

end prove_tan2alpha_and_cosbeta_l268_268554


namespace ZYX_syndrome_diagnosis_l268_268845

theorem ZYX_syndrome_diagnosis (p : ℕ) (h1 : p = 26) (h2 : ∀ c, c = 2 * p) : ∃ n, n = c / 4 ∧ n = 13 :=
by
  sorry

end ZYX_syndrome_diagnosis_l268_268845


namespace minimum_A_in_interval_1_3_l268_268816

def A (x y : ℝ) := 
  (3 * x * y + x ^ 2) * (Real.sqrt (3 * x * y + x - 3 * y)) + 
  (3 * x * y + y ^ 2) * (Real.sqrt (3 * x * y + y - 3 * x)) / 
  (x ^ 2 * y + y ^ 2 * x)

theorem minimum_A_in_interval_1_3 : 
  ∀ (x y : ℝ), (1 ≤ x) → (x ≤ 3) → (1 ≤ y) → (y ≤ 3) → A x y = 4 := 
by 
  sorry

end minimum_A_in_interval_1_3_l268_268816


namespace necessary_not_sufficient_l268_268337

theorem necessary_not_sufficient (a b c : ℝ) : (a < b) → (ac^2 < b * c^2) ∧ ∀a b c : ℝ, (ac^2 < b * c^2) → (a < b) :=
sorry

end necessary_not_sufficient_l268_268337


namespace sum_of_ages_l268_268332

theorem sum_of_ages (a b c d : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b = 24 ∨ a * c = 24 ∨ a * d = 24 ∨ b * c = 24 ∨ b * d = 24 ∨ c * d = 24)
  (h8 : a * b = 35 ∨ a * c = 35 ∨ a * d = 35 ∨ b * c = 35 ∨ b * d = 35 ∨ c * d = 35)
  (h9 : a < 10) (h10 : b < 10) (h11 : c < 10) (h12 : d < 10)
  (h13 : 0 < a) (h14 : 0 < b) (h15 : 0 < c) (h16 : 0 < d) :
  a + b + c + d = 23 := sorry

end sum_of_ages_l268_268332


namespace prime_pair_condition_l268_268109

open Nat

theorem prime_pair_condition (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h : (3 * p^(q-1) + 1) ∣ (11^p + 17^p)) : 
  p = 3 ∧ q = 3 := 
sorry

end prime_pair_condition_l268_268109


namespace f_neg_log3_2_f_f_t_range_l268_268684

def f (x : Real) : Real :=
  if x ≤ 1 then 3 ^ x
  else if 1 < x ∧ x < 3 then (9 / 2) - (3 * x / 2)
  else arbitrary

theorem f_neg_log3_2 : f (-Real.log (2) / Real.log (3)) = 1 / 2 := sorry

theorem f_f_t_range (t : Real) (h : f (f t) ∈ Icc 0 1) : t ∈ Icc (Real.log 7 / 3 / Real.log 3) 1 ∪ Ioo 1 ((9 * 3 / 13) / Real.log 3) := sorry

end f_neg_log3_2_f_f_t_range_l268_268684


namespace leading_coefficient_l268_268097

theorem leading_coefficient (x : ℝ) :
  let p := 5 * (x^5 - 2 * x^3 + x^2) - 8 * (x^5 + x^4 + 3) + 6 * (3 * x^5 - x^3 - 2)
  in p.leading_coeff = 15 :=
by
  sorry

end leading_coefficient_l268_268097


namespace expected_value_matches_variance_matches_l268_268429

variables {N : ℕ} (I : Fin N → Bool)

-- Define the probability that a randomly chosen pair of cards matches
def p_match : ℝ := 1 / N

-- Define the indicator variable I_k
def I_k (k : Fin N) : ℝ :=
if I k then 1 else 0

-- Define the sum S of all the indicator variables
def S : ℝ := (Finset.univ.sum I_k)

-- Expected value E[I_k] is 1/N
def E_I_k : ℝ := 1 / N

-- Expected value E[S] is the sum of E[I_k] over all k, which is 1
theorem expected_value_matches : ∑ k, E_I_k = 1 := sorry

-- Variance calculation: Var[S] = E[S^2] - (E[S])^2
def E_S_sq : ℝ := (Finset.univ.sum (λ k, I_k k * I_k k)) + 
                  2 * (Finset.univ.sum (λ (jk : Fin N × Fin N), if jk.1 < jk.2 then I_k jk.1 * I_k jk.2 else 0))

theorem variance_matches : (E_S_sq - 1) = 1 := sorry

end expected_value_matches_variance_matches_l268_268429


namespace largest_even_number_with_unique_digits_sum_20_l268_268005

theorem largest_even_number_with_unique_digits_sum_20 :
  ∃ n : ℕ, even n ∧ (∀ i j : ℕ, i ≠ j → ¬ (i ∈ digits 10 n) ∨ ¬ (j ∈ digits 10 n)) ∧ (digits 10 n).sum = 20 ∧ n = 86420 := 
by 
  sorry

end largest_even_number_with_unique_digits_sum_20_l268_268005


namespace lena_optimal_income_l268_268023

def monthly_salary : ℝ := 50000
def monthly_expense : ℝ := 45000
def bank_deposit_rate : ℝ := 0.01
def debit_card_annual_rate : ℝ := 0.08
def credit_card_limit : ℝ := 100000
def credit_card_fee : ℝ := 0.03
def savings_per_month : ℝ := monthly_salary - monthly_expense

-- Lena's strategy will involve investing in her deposit account and using her debit card efficiently.
def total_deposit_income : ℝ :=
  let monthly_savings := 5000
  let rate := 1 + bank_deposit_rate
  5000 * ((rate^12 - 1) / (rate - 1))

-- The total interest income from the deposit account in a year
def interest_from_deposit : ℝ := total_deposit_income - (monthly_savings * 12)

-- The total income from the debit card in a year
def interest_from_debit_card : ℝ := 45000 * debit_card_annual_rate

-- Total annual income
def total_annual_income : ℝ := interest_from_deposit + interest_from_debit_card

-- Proof statement
theorem lena_optimal_income : total_annual_income = 7000 := 
  by 
  -- Proof would go here
  sorry

end lena_optimal_income_l268_268023


namespace sum_of_two_digit_divisors_l268_268858

theorem sum_of_two_digit_divisors :
  let S := {x : ℕ | 10 ≤ x ∧ x ≤ 99 ∧ 109 % x = 4} in
  (∑ x in S, x) = 71 :=
by
  sorry

end sum_of_two_digit_divisors_l268_268858


namespace perp_vectors_k_eq_6_l268_268164

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

variables (a b : ℝ) (k : ℝ)

-- Conditions
def norm_a : ∥a∥ = 1 := sorry 
def norm_b : ∥b∥ = 1 := sorry 
def a_perp_b : inner_product_space.inner a b = 0 := sorry

-- Goal
theorem perp_vectors_k_eq_6 (a b : ℝ) (norm_a : ∥a∥ = 1) (norm_b : ∥b∥ = 1) (a_perp_b : inner_product_space.inner a b = 0) : 
    (inner_product_space.inner (2 * a + 3 * b) (k * a - 4 * b) = 0 ↔ k = 6) := 
sorry

end perp_vectors_k_eq_6_l268_268164


namespace interval_of_a_l268_268578

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then Real.exp x + x^2 else Real.exp (-x) + x^2

theorem interval_of_a (a : ℝ) :
  f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 :=
sorry

end interval_of_a_l268_268578


namespace mean_minus_median_l268_268916

theorem mean_minus_median (students : List ℕ)
  (h_size : students.length = 20)
  (h_count_0 : students.count 0 = 4)
  (h_count_1 : students.count 1 = 3)
  (h_count_2 : students.count 2 = 7)
  (h_count_3 : students.count 3 = 2)
  (h_count_4 : students.count 4 = 2)
  (h_count_5 : students.count 5 = 1)
  (h_count_6 : students.count 6 = 1)
: (1:ℚ) / 10 :=
by
  sorry

end mean_minus_median_l268_268916


namespace more_grandsons_or_granddaughters_probability_l268_268297

theorem more_grandsons_or_granddaughters_probability :
  let total_outcomes := 2 ^ 12
  let equal_count_outcomes := Nat.choose 12 6
  let equal_probability := equal_count_outcomes / total_outcomes
  let result_probability := 1 - equal_probability
  result_probability = 793 / 1024 :=
by
  let total_outcomes := (2 : ℚ) ^ 12
  let equal_count_outcomes := Nat.choose 12 6
  let equal_probability := equal_count_outcomes / total_outcomes
  let result_probability := 1 - equal_probability
  have h1 : total_outcomes = 4096 := by norm_num
  have h2 : equal_count_outcomes = 924 := by norm_num
  have h3 : equal_probability = 231 / 1024 := by {
    rw [h2, h1],
    norm_num,
  }
  have h4 : result_probability = 793 / 1024 := by {
    rw [h3],
    norm_num,
  }
  exact h4

end more_grandsons_or_granddaughters_probability_l268_268297


namespace expected_number_of_matches_variance_of_number_of_matches_l268_268424

-- Definitions of conditions
def num_pairs (N : ℕ) : Type := Fin N -> bool -- Type representing pairs of cards matching or not for an N-pair scenario

def indicator_variable (N : ℕ) (k : Fin N) : num_pairs N -> Prop :=
  λ (pairs : num_pairs N), pairs k

def matching_probability (N : ℕ) : ℝ :=
  1 / (N : ℝ)

-- Statement of the first proof problem
theorem expected_number_of_matches (N : ℕ) (pairs : num_pairs N) : 
  (∑ k, (if indicator_variable N k pairs then 1 else 0)) / N = 1 :=
sorry

-- Statement of the second proof problem
theorem variance_of_number_of_matches (N : ℕ) (pairs : num_pairs N) :
  (∑ k, (if indicator_variable N k pairs then 1 else 0) * (if indicator_variable N k pairs then 1 else 0) + 
  2 * ∑ i j, if i ≠ j then 
  (if indicator_variable N i pairs then 1 else 0) * (if indicator_variable N j pairs then 1 else 0) else 0) - 
  ((∑ k, (if indicator_variable N k pairs then 1 else 0)) / N) ^ 2 = 1 :=
sorry

end expected_number_of_matches_variance_of_number_of_matches_l268_268424


namespace infinite_representable_and_nonrepresentable_terms_l268_268589

def seq_a (n : ℕ) : ℕ := 2^n + 2^(n / 2)

theorem infinite_representable_and_nonrepresentable_terms :
  (∃^∞ n, ∃ i j, i ≠ j ∧ seq_a n = seq_a i + seq_a j) ∧
  (∃^∞ n, ¬∃ i j, i ≠ j ∧ seq_a n = seq_a i + seq_a j) :=
sorry

end infinite_representable_and_nonrepresentable_terms_l268_268589


namespace digit_erasure_l268_268704

def N (n : Nat) := 10^n - 1

theorem digit_erasure (n : Nat) (m : Nat) (h : n < m) : 
  ∀ k : Nat, N n ^ k = (10^n - 1) ^ k → ∃ l : List Nat, N n ^ k = l.erase_some_list (N m ^ k) := 
by 
  sorry

end digit_erasure_l268_268704


namespace range_of_a_l268_268532

open Real

theorem range_of_a (a : ℝ) :
  (∀ θ ∈ set.Icc (0 : ℝ) (π / 2),
      sin (2 * θ) - 2 * sqrt 2 * a * cos (θ - π / 4) - sqrt 2 * a / sin (θ + π / 4) > -3 - a^2) ↔
      a ∈ set.Iio 1 ∪ set.Ioi 3 := 
  sorry

end range_of_a_l268_268532


namespace perp_planes_l268_268640

variable {α β : Type}
variable (m n : Type) [HasPerp α β] [HasPerp m n] [HasPerp n β] [HasPerp m α]

theorem perp_planes (hmn : m ⟂ n) (hnβ : n ⟂ β) (hmα : m ⟂ α) : 
  α ⟂ β := 
sorry

end perp_planes_l268_268640


namespace fibonacci_even_count_100_l268_268852

-- Fibonacci sequence definition
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- The problem statement: Prove that there are 33 even Fibonacci numbers in the first 100 numbers of the sequence.
theorem fibonacci_even_count_100 :
  (∑ i in finset.range 100, if (fib i) % 2 == 0 then 1 else 0) = 33 :=
sorry

end fibonacci_even_count_100_l268_268852


namespace find_C_coordinates_l268_268172

open Real

noncomputable def coordC (A B : ℝ × ℝ) : ℝ × ℝ :=
  let n := A.1
  let m := B.1
  let coord_n_y : ℝ := n
  let coord_m_y : ℝ := m
  let y_value (x : ℝ) : ℝ := sqrt 3 / x
  (sqrt 3 / 2, 2)

theorem find_C_coordinates :
  ∃ C : ℝ × ℝ, 
  (∃ A B : ℝ × ℝ, 
   A.2 = sqrt 3 / A.1 ∧
   B.2 = sqrt 3 / B.1 + 6 ∧
   A.2 + 6 = B.2 ∧
   B.2 > A.2 ∧ 
   (sqrt 3 / 2, 2) = coordC A B) ∧
   (sqrt 3 / 2, 2) = (C.1, C.2) :=
by
  sorry

end find_C_coordinates_l268_268172


namespace pipe_A_fills_tank_in_16_hours_l268_268305

theorem pipe_A_fills_tank_in_16_hours
  (A : ℝ)
  (h1 : ∀ t : ℝ, t = 12.000000000000002 → (1/A + 1/24) * t = 5/4) :
  A = 16 :=
by sorry

end pipe_A_fills_tank_in_16_hours_l268_268305


namespace book_distribution_l268_268836

theorem book_distribution : ∃ (n : ℕ), 
  n = (cardinality (setOf (λ (x : ℕ), 2 ≤ x ∧ x ≤ 6)) ∧ n = 5 :=
by 
  sorry

end book_distribution_l268_268836


namespace fraction_of_crop_brought_to_BC_l268_268887

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

end fraction_of_crop_brought_to_BC_l268_268887


namespace parabola_and_hyperbola_equation_l268_268171

theorem parabola_and_hyperbola_equation (a b c : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (hp_eq : c = 2)
    (intersect : (3 / 2, Real.sqrt 6) ∈ {p : ℝ × ℝ | p.2^2 = 4 * c * p.1}
                ∧ (3 / 2, Real.sqrt 6) ∈ {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1}) :
    (∀ x y : ℝ, y^2 = 4*x ↔ c = 1)
    ∧ (∃ a', a' = 1 / 2 ∧ ∀ x y : ℝ, 4 * x^2 - (4 * y^2) / 3 = 1 ↔ a = a') := 
by 
  -- Proof will be here
  sorry

end parabola_and_hyperbola_equation_l268_268171


namespace distance_focus_directrix_l268_268561

variable (P F : ℝ × ℝ)

noncomputable def parabola := { p : ℝ // p > 0 }

noncomputable def focus (p : parabola) : ℝ × ℝ :=
  (p.val / 2, 0)

def directrix (p : parabola) : ℝ :=
  -p.val / 2

noncomputable def on_parabola (P : ℝ × ℝ) (p : parabola) : Prop :=
  P.2 ^ 2 = 2 * p.val * P.1

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

theorem distance_focus_directrix (p : parabola) (hP : on_parabola (6, P.2) p) (hDist : distance (6, P.2) (focus p) = 8) :
  abs ((focus p).1 - directrix p) = 4 :=
begin
  sorry
end

end distance_focus_directrix_l268_268561


namespace no_solution_sin_eq_l268_268108

theorem no_solution_sin_eq {x y : ℝ} (hx : 0 < x) (hx' : x < π / 2) (hy : 0 < y) (hy' : y < π / 2) :
  ¬ (sin x + sin y = sin (x * y)) :=
sorry

end no_solution_sin_eq_l268_268108


namespace reciprocal_of_neg_2023_l268_268763

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268763


namespace sum_density_l268_268324

noncomputable def f_sum_density (λ μ z : ℝ) : ℝ :=
  if z > 0 then (λ * μ / (μ - λ)) * (Real.exp (-λ * z) - Real.exp (-μ * z)) else 0

variable (ξ : ℝ → ℝ) (η : ℝ → ℝ)

axiom xi_exp_dist (λ : ℝ) :
  ∀ x, x ≥ 0 → ξ x = λ * Real.exp (-λ * x)

axiom eta_exp_dist (μ : ℝ) :
  ∀ y, y ≥ 0 → η y = μ * Real.exp (-μ * y)

axiom xi_eta_independent (x : ℝ) (y : ℝ) :
  f_sum_density x y = xi_exp_dist x * eta_exp_dist y

axiom λ_ne_μ (λ μ : ℝ) : λ ≠ μ

theorem sum_density (λ μ : ℝ) (z : ℝ) :
  ξ ∈ exponential_distribution λ →
  η ∈ exponential_distribution μ →
  f_sum_density λ μ z =
    if z > 0 then (λ * μ / (μ - λ)) * (Real.exp (-λ * z) - Real.exp (-μ * z)) else 0 :=
by sorry

end sum_density_l268_268324


namespace new_ratio_after_adding_water_l268_268844

-- Define the initial conditions
variables (M W M_new W_new : ℕ)
def initial_conditions : Prop := 
  (M / (W : ℚ) = 3 / 2) ∧ 
  (M + W = 20) ∧ 
  (W_new = W + 10) ∧ 
  (M_new = M)

-- State the theorem to prove the new ratio
theorem new_ratio_after_adding_water :
  initial_conditions M W M_new W_new →
  M_new / (W_new : ℚ) = 2 / 3 :=
by
  sorry

end new_ratio_after_adding_water_l268_268844


namespace brianna_initial_marbles_l268_268076

-- Defining the variables and constants
def initial_marbles : Nat := 24
def marbles_lost : Nat := 4
def marbles_given : Nat := 2 * marbles_lost
def marbles_ate : Nat := marbles_lost / 2
def marbles_remaining : Nat := 10

-- The main statement to prove
theorem brianna_initial_marbles :
  marbles_remaining + marbles_ate + marbles_given + marbles_lost = initial_marbles :=
by
  sorry

end brianna_initial_marbles_l268_268076


namespace carson_giant_slide_rides_l268_268471

def minutes_in_hour : Nat := 60

def carnival_hours : Nat := 4
def total_time := carnival_hours * minutes_in_hour

def roller_coaster_time_per_ride : Nat := 30
def tilt_a_whirl_time_per_ride : Nat := 60
def giant_slide_time_per_ride : Nat := 15

def roller_coaster_rides : Nat := 4
def tilt_a_whirl_rides : Nat := 1

def total_roller_coaster_time := roller_coaster_rides * roller_coaster_time_per_ride
def total_tilt_a_whirl_time := tilt_a_whirl_rides * tilt_a_whirl_time_per_ride
def remaining_time := total_time - total_roller_coaster_time - total_tilt_a_whirl_time

def giant_slide_rides := remaining_time / giant_slide_time_per_ride

theorem carson_giant_slide_rides : giant_slide_rides = 4 := by
  have h1 : total_time = 240 := rfl
  have h2 : total_roller_coaster_time = 120 := rfl
  have h3 : total_tilt_a_whirl_time = 60 := rfl
  have h4 : remaining_time = 60 := by rw [h1, h2, h3]; simp
  show giant_slide_rides = 4 from sorry

end carson_giant_slide_rides_l268_268471


namespace real_solutions_sum_correct_l268_268098

noncomputable def real_solutions_sum (a : ℝ) (h : a > 0.5) : ℝ :=
  let x := λ a => sqrt (3 * a + sqrt (2 * a)) + sqrt (3 * a - sqrt (2 * a))
  x a

theorem real_solutions_sum_correct (a : ℝ) (h : a > 0.5) :
  (∑ x in {x : ℝ | sqrt (3 * a - sqrt (2 * a + x)) = x}, x) = 
   sqrt (3 * a + sqrt (2 * a)) + sqrt (3 * a - sqrt (2 * a)) :=
by
  sorry

end real_solutions_sum_correct_l268_268098


namespace maximize_profit_at_14_yuan_and_720_l268_268456

def initial_cost : ℝ := 8
def initial_price : ℝ := 10
def initial_units_sold : ℝ := 200
def decrease_units_per_half_yuan_increase : ℝ := 10
def increase_price_per_step : ℝ := 0.5

noncomputable def profit (x : ℝ) : ℝ := 
  let selling_price := initial_price + increase_price_per_step * x
  let units_sold := initial_units_sold - decrease_units_per_half_yuan_increase * x
  (selling_price - initial_cost) * units_sold

theorem maximize_profit_at_14_yuan_and_720 :
  profit 8 = 720 ∧ (initial_price + increase_price_per_step * 8 = 14) :=
by
  sorry

end maximize_profit_at_14_yuan_and_720_l268_268456


namespace find_g_inv_8_and_g_inv_neg64_l268_268284

def g (x : ℝ) : ℝ :=
if x >= 0 then x^4 else -x^4

 noncomputable def g_inv (y : ℝ) : ℝ :=
if y >= 0 then y^(1/4) else -(abs y)^(1/4)

theorem find_g_inv_8_and_g_inv_neg64 : g_inv 8 + g_inv (-64) = 2^(3/4) - 4 :=
by
  -- Proof to be filled in
  sorry

end find_g_inv_8_and_g_inv_neg64_l268_268284


namespace final_amounts_total_l268_268654

variable {Ben_initial Tom_initial Max_initial: ℕ}
variable {Ben_final Tom_final Max_final: ℕ}

theorem final_amounts_total (h1: Ben_initial = 48) 
                           (h2: Max_initial = 48) 
                           (h3: Ben_final = ((Ben_initial - Tom_initial - Max_initial) * 3 / 2))
                           (h4: Max_final = ((Max_initial * 3 / 2))) 
                           (h5: Tom_final = (Tom_initial * 2 - ((Ben_initial - Tom_initial - Max_initial) / 2) - 48))
                           (h6: Max_final = 48) :
  Ben_final + Tom_final + Max_final = 144 := 
by 
  sorry

end final_amounts_total_l268_268654


namespace overtake_time_is_correct_l268_268857

noncomputable def time_to_overtake (V_train V_motorbike : ℕ) (L_train : ℝ) : ℝ :=
  let relative_speed := (V_train - V_motorbike) * (1 / 3.6)
  L_train / relative_speed

theorem overtake_time_is_correct :
  time_to_overtake 100 64 180.0144 = 18.00144 :=
by
  sorry

end overtake_time_is_correct_l268_268857


namespace expected_value_matches_variance_matches_l268_268426

variables {N : ℕ} (I : Fin N → Bool)

-- Define the probability that a randomly chosen pair of cards matches
def p_match : ℝ := 1 / N

-- Define the indicator variable I_k
def I_k (k : Fin N) : ℝ :=
if I k then 1 else 0

-- Define the sum S of all the indicator variables
def S : ℝ := (Finset.univ.sum I_k)

-- Expected value E[I_k] is 1/N
def E_I_k : ℝ := 1 / N

-- Expected value E[S] is the sum of E[I_k] over all k, which is 1
theorem expected_value_matches : ∑ k, E_I_k = 1 := sorry

-- Variance calculation: Var[S] = E[S^2] - (E[S])^2
def E_S_sq : ℝ := (Finset.univ.sum (λ k, I_k k * I_k k)) + 
                  2 * (Finset.univ.sum (λ (jk : Fin N × Fin N), if jk.1 < jk.2 then I_k jk.1 * I_k jk.2 else 0))

theorem variance_matches : (E_S_sq - 1) = 1 := sorry

end expected_value_matches_variance_matches_l268_268426


namespace reciprocal_of_neg_2023_l268_268767

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268767


namespace unique_x_inequality_l268_268562

theorem unique_x_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 → (a = 1 ∨ a = 2)) :=
by
  sorry

end unique_x_inequality_l268_268562


namespace find_d_l268_268074

theorem find_d
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd4 : 4 = a * Real.sin 0 + d)
  (hdm2 : -2 = a * Real.sin (π) + d) :
  d = 1 := by
  sorry

end find_d_l268_268074


namespace initial_roses_l268_268366

theorem initial_roses {x : ℕ} (h : x + 11 = 14) : x = 3 := by
  sorry

end initial_roses_l268_268366


namespace impossible_to_make_all_white_l268_268228

def initial_grid : matrix (fin 3) (fin 3) bool :=
  λ i j, (i = 0 ∧ j = 0)

def is_white (m : matrix (fin 3) (fin 3) bool) (i j : fin 3) : Prop :=
  m i j = false

theorem impossible_to_make_all_white
  (G : matrix (fin 3) (fin 3) bool)
  (H1 : G = initial_grid)
  (H2 : ∀ i, (i < 3) → ∀ j, (j < 3) → (G i j = b ↔ (G (t i) j = ¬b ∨ G i (t j) = ¬b))) :
  ¬(∀ i j, is_white G i j) :=
by
  sorry

end impossible_to_make_all_white_l268_268228


namespace volume_of_rotated_rectangle_is_1920_pi_l268_268896

-- Definitions based on the conditions
def length : ℝ := 30
def width : ℝ := 16
def radius : ℝ := width / 2
def height : ℝ := length

-- Volume formula for a cylinder
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem volume_of_rotated_rectangle_is_1920_pi :
  volume_cylinder radius height = 1920 * Real.pi := 
sorry

end volume_of_rotated_rectangle_is_1920_pi_l268_268896


namespace find_c_l268_268181

noncomputable def f (x a b c : ℤ) := x^3 + a * x^2 + b * x + c

theorem find_c (a b c : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : f a a b c = a^3) (h₄ : f b a b c = b^3) : c = 16 :=
sorry

end find_c_l268_268181


namespace evaluate_custom_operation_l268_268977

def custom_operation (A B : ℕ) : ℕ :=
  (A + 2 * B) * (A - B)

theorem evaluate_custom_operation : custom_operation 7 5 = 34 :=
by
  sorry

end evaluate_custom_operation_l268_268977


namespace graph_is_C_l268_268727

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

def g (x : ℝ) : ℝ := 1 / 2 * f x + 3

theorem graph_is_C : ∀ (x : ℝ), (g x) = (λ x, 1/2 * f x + 3) := 
by sorry

end graph_is_C_l268_268727


namespace range_of_k_is_l268_268722

noncomputable def range_of_k (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : Set ℝ :=
{k : ℝ | ∀ x : ℝ, a^x + 4 * a^(-x) - k > 0}

theorem range_of_k_is (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  range_of_k a h₁ h₂ = { k : ℝ | k < 4 ∧ k ≠ 3 } :=
sorry

end range_of_k_is_l268_268722


namespace reciprocal_of_negative_2023_l268_268760

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l268_268760


namespace sin_225_eq_neg_sqrt2_over_2_l268_268490

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end sin_225_eq_neg_sqrt2_over_2_l268_268490


namespace quadrilaterals_equal_area_l268_268791

-- Definitions based on problem conditions
def quadrilateralI_area : ℕ := 3 * 1
def quadrilateralII_area : ℕ := 2 * (1 * 1.5)

-- Theorem we aim to prove
theorem quadrilaterals_equal_area : 
  quadrilateralI_area = quadrilateralII_area := 
by
  -- Proof will go here, but we use sorry to skip the actual proof
  sorry

end quadrilaterals_equal_area_l268_268791


namespace number_of_permutations_ninety_sixth_permutation_l268_268027

noncomputable theory -- Use noncomputable theory as we are dealing with permutations and large sequences.

open Finset

-- Define the main conditions and the target results as hypotheses.
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Define the first statement: Proving the number of permutations.
theorem number_of_permutations : (univ : Finset (List ℕ)).card = 120 := sorry

-- Define the second statement: Proving the 96th permutation in lexicographic order.
theorem ninety_sixth_permutation : 
  (sort (list.permutations digits)).nth 95 = some [4, 5, 3, 2, 1] := sorry


end number_of_permutations_ninety_sixth_permutation_l268_268027


namespace reciprocal_of_neg_2023_l268_268753

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l268_268753


namespace servant_leaves_after_nine_months_l268_268592

-- Definitions based on conditions
def yearly_salary : ℕ := 90 + 90
def monthly_salary : ℕ := yearly_salary / 12
def amount_received : ℕ := 45 + 90

-- The theorem to prove
theorem servant_leaves_after_nine_months :
    amount_received / monthly_salary = 9 :=
by
  -- Using the provided conditions, we establish the equality we need.
  sorry

end servant_leaves_after_nine_months_l268_268592


namespace shape_is_cone_l268_268543

-- Definition: spherical coordinates and constant phi
def spherical_coords (ρ θ φ : ℝ) : Type := ℝ × ℝ × ℝ
def phi_constant (c : ℝ) (φ : ℝ) : Prop := φ = c

-- Theorem: shape described by φ = c in spherical coordinates is a cone
theorem shape_is_cone (ρ θ c : ℝ) (h₁ : c ∈ set.Icc 0 real.pi) : 
  (∃ (ρ θ : ℝ), spherical_coords ρ θ c = (ρ, θ, c)) → 
  (∀ φ, phi_constant c φ) → 
  shape_is_cone := sorry

end shape_is_cone_l268_268543


namespace reciprocal_of_neg_2023_l268_268745

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l268_268745


namespace max_regions_of_n_lines_min_regions_of_n_lines_l268_268936

theorem max_regions_of_n_lines (n : ℕ) : 
  (n * (n + 1)) / 2 + 1 = ∑ i in range (n + 1), i + 1 :=
by
  sorry

theorem min_regions_of_n_lines (n : ℕ) : 
  n + 1 = n + 1 :=
by
  sorry

end max_regions_of_n_lines_min_regions_of_n_lines_l268_268936


namespace equation_of_line_line_BD_fixed_point_l268_268445

noncomputable section

-- Conditions: a line l passes through the focus of the parabola y^2 = 4x with slope k,
-- and intersects the parabola at points A and B such that the distance AB is 8.
variables {k : ℝ} {x y : ℝ}

-- Given conditions
def parabola : Prop := y^2 = 4 * x
def focus_line (k : ℝ) : Prop := y = k * (x - 1)
def intersection_A_B : Prop := ∃ (x1 y1 x2 y2 : ℝ), parabola ∧ focus_line k ∧
  y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1) ∧ (x2 - x1)^2 + (y2 - y1)^2 = 8^2

-- Question 1: Proving the equation of line l is y = x - 1 or y = -x + 1
theorem equation_of_line (k : ℝ) (H : focus_line k ∧ intersection_A_B) :
  focus_line 1 ∨ focus_line (-1) :=
sorry

-- Question 2: Proving line BD intersects the fixed point (-1, 0)
theorem line_BD_fixed_point (A D : ℝ × ℝ) :
  (A.fst * D.fst = 1) ∧ (2 * (A.snd - D.snd) = 8) →
  ∃ (B D : ℝ × ℝ), ∀ (x y : ℝ), x = -1 ∧ y = 0 :=
sorry

end equation_of_line_line_BD_fixed_point_l268_268445


namespace parabola_focus_directrix_l268_268997

-- Define the parabola and its conditions
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

-- Define the focus condition
def focus_at (p : ℝ) : Prop := (1 : ℝ) = p / 2

-- Define the directrix condition
def directrix (p : ℝ) : Prop := x = - (p / 2)

-- Main theorem combining these conditions
theorem parabola_focus_directrix :
  ∃ (p : ℝ), (focus_at p) ∧ (parabola y x p) ∧ (directrix p) ∧ (p = 2) ∧ (x = -1) :=
by
  exists (2 : ℝ)
  exact ⟨rfl, sorry⟩ -- Proof is omitted using sorry

end parabola_focus_directrix_l268_268997


namespace arithmetic_geometric_mean_inequality_l268_268137

variable {a b : ℝ}

noncomputable def A (a b : ℝ) := (a + b) / 2
noncomputable def B (a b : ℝ) := Real.sqrt (a * b)

theorem arithmetic_geometric_mean_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : a ≠ b) : A a b > B a b := 
by
  sorry

end arithmetic_geometric_mean_inequality_l268_268137


namespace find_selling_price_of_each_lamp_l268_268832

-- Define the given conditions
variable (cost_of_lamp selling_price_base units_sold_base stock target_profit : ℝ)
variable (increase_per_1 decrease_sales_per_1 increase_sales_per_1 : ℝ)

def lamps_sold (x : ℝ) : ℝ := units_sold_base + increase_sales_per_1 * x
def profit (x : ℝ) : ℝ := (selling_price_base - x - cost_of_lamp) * lamps_sold x

-- Given conditions
def conditions := cost_of_lamp = 30 ∧
                  selling_price_base = 40 ∧
                  units_sold_base = 600 ∧
                  stock = 1210 ∧
                  target_profit = 8400 ∧
                  increase_per_1 = 1 ∧
                  decrease_sales_per_1 = 20 ∧
                  increase_sales_per_1 = 200

-- Main theorem to prove the correct selling price for the given conditions
theorem find_selling_price_of_each_lamp 
  (h : conditions) : ∃ x : ℝ, 
  selling_price_base - x = 37 ∧
  profit x = target_profit := 
by
  sorry

end find_selling_price_of_each_lamp_l268_268832


namespace inequalities_sufficient_but_not_necessary_l268_268780

theorem inequalities_sufficient_but_not_necessary (a b c d : ℝ) :
  (a > b ∧ c > d) → (a + c > b + d) ∧ ¬((a + c > b + d) → (a > b ∧ c > d)) :=
by
  sorry

end inequalities_sufficient_but_not_necessary_l268_268780


namespace axis_of_symmetry_of_g_l268_268582

def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x - Real.pi / 5)

def g (x : ℝ) : ℝ := 3 * Real.cos (2 * x - 13 * Real.pi / 15)

theorem axis_of_symmetry_of_g :
  ∃ (k : ℤ), ∀ (x : ℝ), g x = g (-x) → x = k * Real.pi / 2 + 13 * Real.pi / 30 :=
sorry

end axis_of_symmetry_of_g_l268_268582


namespace negation_of_existential_l268_268342

theorem negation_of_existential (x : ℝ) : ¬(∃ x : ℝ, x^2 - 2 * x + 3 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 3 ≤ 0 := 
by
  sorry

end negation_of_existential_l268_268342


namespace findP_l268_268576

variables {θ t : ℝ}

-- Define the ellipse C
def isOnEllipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the parametric equation of the ellipse
def parametricEllipse (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the point A
def A : ℝ × ℝ := (1, 0)

-- Define the distance function between two points
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the condition of the point P on the ellipse
def pointP (θ : ℝ) : ℝ × ℝ :=
  parametricEllipse θ

-- Define the statement of the problem
theorem findP (θ : ℝ) :
  isOnEllipse (pointP θ).1 (pointP θ).2 →
  distance (pointP θ) A = 3/2 →
  (pointP θ = (1, Real.sqrt(3)/2) ∨ pointP θ = (1, -Real.sqrt(3)/2)) :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end findP_l268_268576


namespace area_of_triangle_OAB_l268_268044

def ellipse : Set (ℝ × ℝ) := {p | (p.1^2 / 5) + (p.2^2 / 4) = 1}

def right_focus : ℝ × ℝ := (1, 0)

def line (p : ℝ × ℝ) (slope : ℝ) : Set (ℝ × ℝ) := 
  {q | q.2 = slope * (q.1 - p.1)}

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

theorem area_of_triangle_OAB :
  let F2 := right_focus;
  let AB := line F2 2;
  let A := (0, -2);
  let B := (5 / 3, 4 / 3);
  triangle_area (0, 0) A B = 5 / 3 :=
by
  sorry

end area_of_triangle_OAB_l268_268044


namespace units_digit_of_modifiedLucas_L20_eq_d_l268_268090

def modifiedLucas : ℕ → ℕ
| 0 => 3
| 1 => 2
| n + 2 => 2 * modifiedLucas (n + 1) + modifiedLucas n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_modifiedLucas_L20_eq_d :
  ∃ d, units_digit (modifiedLucas (modifiedLucas 20)) = d :=
by
  sorry

end units_digit_of_modifiedLucas_L20_eq_d_l268_268090


namespace line_perpendicular_to_plane_l268_268221

variable (a : ℝ^3 := ⟨1, -1, 2⟩)
variable (u : ℝ^3 := ⟨-2, 2, -4⟩)

theorem line_perpendicular_to_plane (a u : ℝ^3) (h1 : a = ⟨1, -1, 2⟩) (h2 : u = ⟨-2, 2, -4⟩) : 
  is_collinear a u → line_perpendicular_to_plane a u :=
by
  sorry

end line_perpendicular_to_plane_l268_268221


namespace find_m_value_l268_268151

-- Define the points P and Q and the condition of perpendicularity
def points_PQ (m : ℝ) : Prop := 
  let P := (-2, m)
  let Q := (m, 4)
  let slope_PQ := (m - 4) / (-2 - m)
  slope_PQ * (-1) = -1

-- Problem statement: Find the value of m such that the above condition holds
theorem find_m_value : ∃ (m : ℝ), points_PQ m ∧ m = 1 :=
by sorry

end find_m_value_l268_268151


namespace min_cubes_needed_is_200_l268_268019

noncomputable def total_cubes := 4 * 10 * 7
noncomputable def hollow_core_cubes := (4 - 2) * (10 - 2) * (7 - 2)
noncomputable def min_cubes_needed := total_cubes - hollow_core_cubes

theorem min_cubes_needed_is_200 :
  min_cubes_needed = 200 :=
by
  have total_cubes_calc : total_cubes = 280 := by
    unfold total_cubes
    norm_num
  have hollow_core_cubes_calc : hollow_core_cubes = 80 := by
    unfold hollow_core_cubes
    norm_num
  rw [total_cubes_calc, hollow_core_cubes_calc]
  norm_num
  sorry

end min_cubes_needed_is_200_l268_268019


namespace perpendicular_OE_CD_l268_268069

-- Define the structures and conditions
variables (A B C O D E : Point)
variables (circumcenter_O : is_circumcenter O A B C)
variables (midpoint_D : is_midpoint D A B)
variables (centroid_E : is_centroid E A C D)
variables (equal_sides : dist A B = dist A C)

-- Define the theorem to be proved
theorem perpendicular_OE_CD :
  OE ⟂ CD :=
sorry

end perpendicular_OE_CD_l268_268069


namespace largest_unattainable_sum_l268_268249

theorem largest_unattainable_sum (n : ℕ) : ∃ s, s = 12 * n^2 + 8 * n - 1 ∧ 
  ∀ (k : ℕ), k ≤ s → ¬ ∃ a b c d, 
    k = (6 * n + 1) * a + (6 * n + 3) * b + (6 * n + 5) * c + (6 * n + 7) * d := 
sorry

end largest_unattainable_sum_l268_268249


namespace relation_between_Rachel_and_Patty_l268_268693

-- Definitions from the conditions
def MattTime : ℝ := 12
def PattyTime : ℝ := MattTime / 3
def RachelTime : ℝ := 13

-- Theorem to prove the relation between RachelTime and PattyTime
theorem relation_between_Rachel_and_Patty : RachelTime / PattyTime = 3.25 :=
by
  -- proof steps are omitted and replaced with sorry
  sorry

end relation_between_Rachel_and_Patty_l268_268693


namespace angle_bisector_between_median_and_altitude_l268_268314

variable {α : Type*}

structure Triangle (α : Type*) :=
(A B C : α)

structure Point (α : Type*) :=
(x y : α)

def is_scalene (t : Triangle ℝ) : Prop := 
  t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.A ≠ t.C

def altitude_foot (t : Triangle ℝ) (B : Point ℝ) : Point ℝ := sorry
def angle_bisector_point (t : Triangle ℝ) (B : Point ℝ) : Point ℝ := sorry
def midpoint (A C : Point ℝ) : Point ℝ := sorry

theorem angle_bisector_between_median_and_altitude 
  (t : Triangle ℝ) (h_scl : is_scalene t) (H : Point ℝ) (D : Point ℝ) (M : Point ℝ)
  (H_alt : H = altitude_foot t t.B) 
  (D_bis : D = angle_bisector_point t t.B) 
  (M_mid : M = midpoint t.A t.C) : 
  between (D : affine_combination H M) :=
begin
  sorry
end

end angle_bisector_between_median_and_altitude_l268_268314


namespace no_real_roots_l268_268703

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  (finset.range (2 * n + 1)).sum (λ k, (-1 : ℝ)^k * (2 * n - k + 1) * x^(2 * n - k))

theorem no_real_roots (n : ℕ) : ¬ ∃ x : ℝ, f n x = 0 :=
by
  sorry

end no_real_roots_l268_268703


namespace lena_optimal_income_l268_268024

def monthly_salary : ℝ := 50000
def monthly_expense : ℝ := 45000
def bank_deposit_rate : ℝ := 0.01
def debit_card_annual_rate : ℝ := 0.08
def credit_card_limit : ℝ := 100000
def credit_card_fee : ℝ := 0.03
def savings_per_month : ℝ := monthly_salary - monthly_expense

-- Lena's strategy will involve investing in her deposit account and using her debit card efficiently.
def total_deposit_income : ℝ :=
  let monthly_savings := 5000
  let rate := 1 + bank_deposit_rate
  5000 * ((rate^12 - 1) / (rate - 1))

-- The total interest income from the deposit account in a year
def interest_from_deposit : ℝ := total_deposit_income - (monthly_savings * 12)

-- The total income from the debit card in a year
def interest_from_debit_card : ℝ := 45000 * debit_card_annual_rate

-- Total annual income
def total_annual_income : ℝ := interest_from_deposit + interest_from_debit_card

-- Proof statement
theorem lena_optimal_income : total_annual_income = 7000 := 
  by 
  -- Proof would go here
  sorry

end lena_optimal_income_l268_268024


namespace income_to_expenditure_ratio_l268_268339

variable (I E S : ℕ)
variable (I_val : I = 40000)
variable (S_val : S = 5000)
variable (savings_eq : S = I - E)

theorem income_to_expenditure_ratio :
  I = 40000 → S = 5000 → S = I - E → I:E = 8:7 :=
by
  intros hI hS hSavings
  rw [hI, hS, hSavings]
  simp
  sorry

end income_to_expenditure_ratio_l268_268339


namespace isosceles_right_triangle_l268_268947

namespace Proof

variables {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem isosceles_right_triangle
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ⟪a + b, a⟫ = 0) (h2 : ⟪a, b⟫ = ∥a∥^2) :
  (∥a∥ = ∥b∥) ∧ (∠ a b = real.pi / 2) :=
sorry

end Proof

end isosceles_right_triangle_l268_268947


namespace circle_equation_l268_268908

theorem circle_equation
  (center : ℝ × ℝ)
  (line1 line2 : ℝ → ℝ → Prop)
  (intersect : ℝ × ℝ)
  (equation : ℝ → ℝ → ℝ)
  (hC : center = (4, 3))
  (hline1 : ∀ x y, line1 x y ↔ x + 2 * y + 1 = 0)
  (hline2 : ∀ x y, line2 x y ↔ 2 * x + y - 1 = 0)
  (hintersect : intersect = (1, -1))
  (hequation : ∀ x y, equation x y ↔ (x - 4)^2 + (y - 3)^2 = 25) :
  equation (1 : ℝ) (-1 : ℝ) :=
by
  sorry

end circle_equation_l268_268908


namespace removed_cubes_total_l268_268904

-- Define the large cube composed of 125 smaller cubes (5x5x5 cube)
def large_cube := 5 * 5 * 5

-- Number of smaller cubes removed from each face to opposite face
def removed_faces := (5 * 5 + 5 * 5 + 5 * 3)

-- Overlapping cubes deducted
def overlapping_cubes := (3 + 1)

-- Final number of removed smaller cubes
def removed_total := removed_faces - overlapping_cubes

-- Lean theorem statement
theorem removed_cubes_total : removed_total = 49 :=
by
  -- Definitions provided above imply the theorem
  sorry

end removed_cubes_total_l268_268904


namespace find_a_minus_b_l268_268925

theorem find_a_minus_b (a b : ℝ) (h : (a + 1 * complex.I) / complex.I = b + 2 * complex.I) : a - b = -3 :=
  sorry

end find_a_minus_b_l268_268925


namespace two_primes_between_29_div_4_and_17_l268_268783

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def prime_count_in_range (a b : ℤ) : ℕ :=
  (finset.filter (λ x, is_prime x) (finset.Icc a b)).card

theorem two_primes_between_29_div_4_and_17 :
  prime_count_in_range (int.of_real (29 / 4)) 17 = 2 :=
by
  sorry

end two_primes_between_29_div_4_and_17_l268_268783


namespace number_of_valid_n_l268_268530

def floor (x : ℝ) : ℤ := Int.floor x

def valid_n (n : ℕ) : Prop :=
  ∃ (m : ℕ), n = 7 * m ∨ n = 7 * m + 1 ∨ n = 7 * m + 3 ∨ n = 7 * m + 4

theorem number_of_valid_n (bound : ℕ) (h : bound = 1500) :
  (∑ n in Finset.range (bound + 1), if valid_n n then 1 else 0) = 854 :=
by
  sorry

end number_of_valid_n_l268_268530


namespace students_in_classroom_l268_268035

theorem students_in_classroom (n : ℕ) :
  n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 → n = 21 ∨ n = 45 :=
by
  sorry

end students_in_classroom_l268_268035


namespace exists_irrational_c_in_interval_l268_268657

theorem exists_irrational_c_in_interval (c : ℝ) : 
  (0 < c ∧ c < 1) ∧ (irrational c) ∧ 
  (∀ n : ℕ, decimal_digit (decimal_expansion c n) ≠ 0) ∧ 
  (∀ n : ℕ, decimal_digit (decimal_expansion (sqrt c) n) ≠ 0) :=
sorry

end exists_irrational_c_in_interval_l268_268657


namespace volume_larger_of_cube_cut_plane_l268_268095

/-- Define the vertices and the midpoints -/
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def R : Point := ⟨0, 0, 0⟩
def X : Point := ⟨1, 2, 0⟩
def Y : Point := ⟨2, 0, 1⟩

/-- Equation of the plane passing through R, X and Y -/
def plane_eq (p : Point) : Prop :=
p.x - 2 * p.y - 2 * p.z = 0

/-- The volume of the larger of the two solids formed by cutting the cube with the plane -/
noncomputable def volume_larger_solid : ℝ :=
8 - (4/3 - (1/6))

/-- The statement for the given math problem -/
theorem volume_larger_of_cube_cut_plane :
  volume_larger_solid = 41/6 :=
by
  sorry

end volume_larger_of_cube_cut_plane_l268_268095


namespace cost_of_each_math_book_l268_268002

theorem cost_of_each_math_book
    (total_books : ℕ)
    (math_books : ℕ)
    (history_books_cost : ℕ)
    (total_cost : ℕ)
    (total_books_eq : total_books = 80)
    (math_books_eq : math_books = 10)
    (history_books_cost_eq : history_books_cost = 5)
    (total_cost_eq : total_cost = 390) : 
    (total_cost - (total_books - math_books) * history_books_cost) / math_books = 4 :=
by
    rw [total_books_eq, math_books_eq, history_books_cost_eq, total_cost_eq]
    norm_num
    sorry

end cost_of_each_math_book_l268_268002


namespace sum_of_ratios_is_one_l268_268274

variables {A B C P D E F P' D' E' F' : Type}

-- Definitions associating variables with the geometric properties provided
def Point (α : Type) := α

variables [inhabited (Point ℝ)] -- assuming points are represented within a real coordinate system

-- Given conditions
variables (ABC : Point ℝ) (P : Point ℝ)
variables (A B C : Point ℝ) (D E F : Point ℝ)
variables (DEF : Triangle ℝ) (P' : Point ℝ)
variables (D' E' F' : Point ℝ)

-- Hypotheses based on the given problem
variable (h1 : Line_through A P intersects BC at D)
variable (h2 : Line_through B P intersects CA at E)
variable (h3 : Line_through C P intersects AB at F)
variable (h4 : P' ∈ perimeter (Triangle.mk D E F))
variable (h5 : Line_parallel_to P' PD intersects BC at D')
variable (h6 : Line_parallel_to P' PE intersects CA at E')
variable (h7 : Line_parallel_to P' PF intersects AB at F')

-- Formal statement of the problem to prove
theorem sum_of_ratios_is_one :
  ∃j : Finset ℝ, j.card = 3 ∧
    j.1 = (P' D' / PD) ∧
    j.2 = (P' E' / PE) ∧
    j.3 = (P' F' / PF) ∧
    (∃k : Finset ℝ, (k ∈ j → j.1 = (j.2 + j.3))) :=
sorry

end sum_of_ratios_is_one_l268_268274


namespace function_increasing_probability_l268_268557

noncomputable def is_increasing_on_interval (a b : ℤ) : Prop :=
∀ x : ℝ, x > 1 → 2 * a * x - 2 * b > 0

noncomputable def valid_pairs : List (ℤ × ℤ) :=
[(0, -1), (1, -1), (1, 1), (2, -1), (2, 1)]

noncomputable def total_pairs : ℕ :=
3 * 4

noncomputable def probability_of_increasing_function : ℚ :=
(valid_pairs.length : ℚ) / total_pairs

theorem function_increasing_probability :
  probability_of_increasing_function = 5 / 12 :=
by
  sorry

end function_increasing_probability_l268_268557


namespace remainder_when_divided_by_29_l268_268395

theorem remainder_when_divided_by_29 (N : ℤ) (h : N % 899 = 63) : N % 29 = 10 :=
sorry

end remainder_when_divided_by_29_l268_268395


namespace maximum_marked_points_1976_gon_l268_268635

theorem maximum_marked_points_1976_gon : 
  ∀ (n : ℕ), n = 1976 → 
  let num_sides_midpoints := n,
      num_diagonals_midpoints := (n * (n - 3)) / 2 
  in 
  num_sides_midpoints + num_diagonals_midpoints = 1976 → 
  ∃ (max_points_on_a_circle : ℕ), max_points_on_a_circle = 1976 :=
begin
  sorry
end

end maximum_marked_points_1976_gon_l268_268635


namespace probability_no_shaded_square_l268_268033

theorem probability_no_shaded_square :
  let rectangles := (1:ℕ) in
  let shaded := (1:ℕ) in
  let shaded_probability := (8/55 : ℚ) in
  (∃ total_shaded_rectangles total_rectangles : ℚ,
    total_rectangles = 165 ∧
    total_shaded_rectangles = 141 ∧
    shaded_probability = (total_rectangles - total_shaded_rectangles) / total_rectangles) :=
begin
  sorry -- The proof will be filled in here.
end

end probability_no_shaded_square_l268_268033


namespace point_in_second_quadrant_l268_268621

theorem point_in_second_quadrant (m : ℝ) (h1 : 3 - m < 0) (h2 : m - 1 > 0) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l268_268621


namespace probability_closer_to_origin_l268_268450

def point := (ℝ × ℝ)
def rectangle (a b c d : point) := 
  a = (0, 0) ∧ b = (6, 0) ∧ c = (6, 2) ∧ d = (0, 2)

def closer_to_origin (P : point) : Prop :=
  let d_origin := real.sqrt (P.1^2 + P.2^2)
  let d_2_1 := real.sqrt ((P.1 - 2)^2 + (P.2 - 1)^2)
  d_origin < d_2_1

noncomputable def area_closer_to_origin := 
  1 / 2 * (1.25 * 2) + 1 / 2 * (0.25 * 2) + (1.25 - 0.25) * 2

theorem probability_closer_to_origin :
  ∀ (P : point), rectangle (0,0) (6,0) (6,2) (0,2) →
  P ∈ set_of (closer_to_origin) → 
  5 / 24 :=
sorry

end probability_closer_to_origin_l268_268450


namespace problem_proof_l268_268062

noncomputable def largest_product_pair (S : Finset ℕ) : ℕ :=
  let pairs := S.product S
  let filtered_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  filtered_pairs.sup (λ p, p.1 * p.2)

noncomputable def smallest_sum_pair (S : Finset ℕ) : ℕ :=
  let pairs := S.product S
  let filtered_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  filtered_pairs.inf (λ p, p.1 + p.2)

def count_pairs_with_sum (S : Finset ℕ) (target_sum : ℕ) : ℕ :=
  let pairs := S.product S
  let filtered_pairs := pairs.filter (λ p, p.1 ≠ p.2 ∧ p.1 + p.2 = target_sum)
  filtered_pairs.card

theorem problem_proof :
  let S := Finset.range 10 in
  largest_product_pair S = 72 ∧
  smallest_sum_pair S = 1 ∧
  count_pairs_with_sum S 10 = 4 :=
by
  sorry

end problem_proof_l268_268062


namespace num_paths_from_E_to_G_through_F_and_H_l268_268595

-- Define points as tuples (x, y)
def E := (0, 5)
def F := (4, 4)
def H := (5, 2)
def G := (6, 0)

-- Number of paths using combinatorics
noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k+1 => 0
| n+1, k+1 => binom n k + binom n (k+1)

def paths (p1 p2 : ℕ × ℕ) : ℕ :=
  binom ((p2.1 - p1.1) + (p2.2 - p1.2)) (p2.2 - p1.2)

theorem num_paths_from_E_to_G_through_F_and_H :
  paths E F * paths F H * paths H G = 135 := by
  sorry

end num_paths_from_E_to_G_through_F_and_H_l268_268595


namespace max_intersection_points_l268_268379

theorem max_intersection_points (circle : Type) (rectangle : Type) (triangle : Type)
  (distinct : rectangle ≠ triangle) (non_overlap : ¬ ∃ pt, pt ∈ rectangle ∧ pt ∈ triangle) :
  (∀ (intersect1 intersect2 : circle → rectangle → ℕ) (intersect3 intersect4 : circle → triangle → ℕ),
    (intersect1 circle rectangle = 2 ∧ intersect2 circle rectangle = 8) ∧
    (intersect3 circle triangle = 2 ∧ intersect4 circle triangle = 6) → 
    intersect2 circle rectangle + intersect4 circle triangle = 14) :=
begin
  sorry
end

end max_intersection_points_l268_268379


namespace PDFT_is_cyclic_l268_268685

-- Define the basic geometric entities involved
variables {A B C Q P T D F : Point}
variables (InscribedCircle_Touch_AQAB : touches (incircle (triangle A B C)) (line A B) Q)
variables (Incircle_QACTouch : touches (incircle (triangle Q A C)) (line A Q) P ∧ touches (incircle (triangle Q A C)) (line A C) T)
variables (Incircle_QBCTouch : touches (incircle (triangle Q B C)) (line B Q) D ∧ touches (incircle (triangle Q B C)) (line B C) F)

-- State the lemma and main theorem
lemma IncirclesTouchSamePointOnCQ : 
  ∀ (E : Point), touches (incircle (triangle Q A C)) (line C Q) E ∧ touches (incircle (triangle Q B C)) (line C Q) E := 
sorry

theorem PDFT_is_cyclic :
  cyclic quadrilateral P D F T :=
sorry

end PDFT_is_cyclic_l268_268685


namespace probability_cosine_interval_l268_268572

theorem probability_cosine_interval :
  let I := Ioc (- (Real.pi / 2)) (Real.pi / 2) in
  ∫ x in I, ∫ y in I, ite (y ≤ Real.cos x) 1 0 =
    ((2 / Real.pi ^ 2) + (1 / 2)) * (Real.pi * Real.pi / 4) :=
by
  sorry

end probability_cosine_interval_l268_268572


namespace arithmetic_sequence_fifth_term_l268_268585

theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℕ), (∀ n, a n.succ = a n + 2) → a 1 = 2 → a 5 = 10 :=
by
  intros a h1 h2
  sorry

end arithmetic_sequence_fifth_term_l268_268585


namespace correct_angle_calculation_l268_268568

theorem correct_angle_calculation (α β : ℝ) (hα : 0 < α ∧ α < 90) (hβ : 90 < β ∧ β < 180) :
    22.5 < 0.25 * (α + β) ∧ 0.25 * (α + β) < 67.5 → 0.25 * (α + β) = 45.3 :=
by
  sorry

end correct_angle_calculation_l268_268568


namespace lovely_book_sale_ratio_l268_268859

theorem lovely_book_sale_ratio:
  ∃ (x y : ℕ), x + y = 10 ∧ 2.5 * x + 2 * y = 22 ∧ x / 10 = 2 / 5 :=
by
  sorry

end lovely_book_sale_ratio_l268_268859


namespace probability_sum_18_l268_268616

theorem probability_sum_18:
  (∑ k in {1,2,3,4,5,6}, k = 6)^4 * (probability {d₁ d₂ d₃ d₄ : ℕ | d₁ + d₂ + d₃ + d₄ = 18} 6 6) = 5 / 216 := 
sorry

end probability_sum_18_l268_268616


namespace girls_percentage_l268_268630

theorem girls_percentage (total_students girls boys : ℕ) 
    (total_eq : total_students = 42)
    (ratio : 3 * girls = 4 * boys)
    (total_students_eq : total_students = girls + boys) : 
    (girls * 100 / total_students : ℚ) = 57.14 := 
by 
  sorry

end girls_percentage_l268_268630


namespace sin_225_eq_neg_sqrt2_over_2_l268_268487

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end sin_225_eq_neg_sqrt2_over_2_l268_268487


namespace valid_pair_tiger_leopard_l268_268403

constant Lion : Prop
constant Tiger : Prop
constant Leopard : Prop
constant Elephant : Prop

axiom lion_tiger_implication : Lion → Tiger
axiom tiger_leopard_converse_implication : ¬Leopard → ¬Tiger
axiom leopard_elephant_implication : Leopard → ¬Elephant

theorem valid_pair_tiger_leopard (hT : Tiger) (hP : Leopard) : (Tiger ∧ Leopard) :=
by {
  -- Ensuring all conditions are satisfied given Tiger and Leopard are participants
  have h1: Leaping implies (Lion is true only if Tiger is true), from lion_tiger_implication,
  have h2: (Leopard is false implies Tiger is false), from tiger_leopard_converse_implication,
  have h3: Leopard is true implies Elephant is false, from leopard_elephant_implication,
  
  -- Given Tiger and Leopard participate
  exact And.intro hT hP
}

end valid_pair_tiger_leopard_l268_268403


namespace book_distribution_l268_268839

theorem book_distribution (n : ℕ) (h₀ : n = 8) : 
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ 6 ∧ k = n - i ∧ 2 ≤ i ∧ i ≤ 6) → card {k : ℕ | 2 ≤ k ∧ k ≤ 6} = 5 := 
by 
  sorry

end book_distribution_l268_268839


namespace correct_operation_l268_268386
variable (a x y: ℝ)

theorem correct_operation : 
  ¬ (5 * a - 2 * a = 3) ∧
  ¬ ((x + 2 * y)^2 = x^2 + 4 * y^2) ∧
  ¬ (x^8 / x^4 = x^2) ∧
  ((2 * a)^3 = 8 * a^3) :=
by
  sorry

end correct_operation_l268_268386


namespace apples_given_to_restaurant_is_correct_l268_268374

-- Definitions for the conditions
def total_harvest := 405
def apples_for_juice := 90
def total_sales_in_dollars := 408
def price_per_bag_in_dollars := 8
def weight_per_bag := 5

-- Mathematical statement to prove
theorem apples_given_to_restaurant_is_correct :
  let number_of_bags_sold := total_sales_in_dollars / price_per_bag_in_dollars in
  let apples_sold_in_bags := number_of_bags_sold * weight_per_bag in
  let apples_given_to_restaurant := total_harvest - (apples_for_juice + apples_sold_in_bags) in
  apples_given_to_restaurant = 60 :=
by
  sorry

end apples_given_to_restaurant_is_correct_l268_268374


namespace positive_integer_is_48_l268_268218

theorem positive_integer_is_48 (n p : ℕ) (h_prime : Prime p) (h_eq : n = 24 * p) (h_min : n ≥ 48) : n = 48 :=
by
  sorry

end positive_integer_is_48_l268_268218


namespace condition_d_not_determine_similarity_l268_268385

theorem condition_d_not_determine_similarity (A B C D : Prop) :
  (A = "Two corresponding angles are equal" →
   B = "Two sets of sides are proportional and the included angles are equal" →
   C = "Three sides are proportional" →
   D = "Two sets of sides are proportional" →
   (∀ A B C, determines_similarity A →
             determines_similarity B →
             determines_similarity C →
             ¬ determines_similarity D)) :=
by
  intros hA hB hC hD
  sorry

end condition_d_not_determine_similarity_l268_268385


namespace find_fraction_squares_l268_268203

theorem find_fraction_squares (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := 
by
  sorry

end find_fraction_squares_l268_268203


namespace congruent_triangles_exists_in_21_gon_l268_268236

theorem congruent_triangles_exists_in_21_gon :
  ∃ (triangle_r triangle_b : set ℕ), 
    -- Define the sets of red vertices and blue vertices
    (triangle_r ⊆ {0, 1, 2, 3, 4, 5}) ∧ (triangle_b ⊆ {0, 1, 2, 3, 4, 5, 6}) ∧
    -- Each triangle must have 3 vertices
    (triangle_r.card = 3) ∧ (triangle_b.card = 3) ∧
    -- Congruence means there's a rotation mapping one triangle to the other
    ∃ (k : ℕ) (hk : k ∈ {1, 2, ..., 20}),
      triangle_r.map (λ x, (x + k) % 21) = triangle_b :=
by
  sorry

end congruent_triangles_exists_in_21_gon_l268_268236


namespace gcd_xyz_square_l268_268677

theorem gcd_xyz_square (x y z : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : ∃ n : ℕ, \gcd x y z * x * y * z = n^2 := 
sorry

end gcd_xyz_square_l268_268677


namespace reciprocal_of_neg_2023_l268_268752

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l268_268752


namespace ratio_removing_middle_digit_l268_268053

theorem ratio_removing_middle_digit 
  (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9)
  (h1 : 10 * b + c = 8 * a) 
  (h2 : 10 * a + b = 8 * c) : 
  (10 * a + c) / b = 17 :=
by sorry

end ratio_removing_middle_digit_l268_268053


namespace general_terms_and_min_diff_l268_268942

noncomputable def seq_a (n : ℕ) : ℕ := n
noncomputable def seq_b (n : ℕ) : ℕ := n + 2

theorem general_terms_and_min_diff (T_n : ℕ → ℝ) :
  (∀ n : ℕ, seq_a n = n) ∧ (∀ n : ℕ, seq_b n = n + 2) ∧
  (∀ n : ℕ, T_n n - 2 * n ∈ set.Icc (4/3 : ℝ) 3) → 
  ∀ n : ℕ, ∀ a b : ℝ, (a ≤ 4 / 3) ∧ (3 ≤ b) → (b - a = 5 / 3) :=
by
  sorry

end general_terms_and_min_diff_l268_268942


namespace graph_intersects_itself_48_times_l268_268584

theorem graph_intersects_itself_48_times :
  let x := λ t: ℝ, (Real.cos t) + t / 3
  let y := λ t: ℝ, Real.sin (2 * t)
  let count_intersections := λ (a b: ℝ) (x: ℝ → ℝ)
    (intersections: ℝ → ℝ → ℝ → ℝ → Prop) → ℕ,
      (sorry : ℕ) := 48 in
  count_intersections 1 50 x
    (λ t1 t2 x1 x2, (x1 = x2 ∧ x (t1) = x1 ∧ x (t2) = x2 ∧ t1 ≠ t2) ) = 48 :=
  sorry

end graph_intersects_itself_48_times_l268_268584


namespace reciprocal_of_negative_2023_l268_268756

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l268_268756


namespace unique_solution_to_absolute_value_equation_l268_268973

theorem unique_solution_to_absolute_value_equation :
  {x : ℝ | abs (x - 2) = abs (x - 4) + abs (x - 6)}.finite ∧ {x : ℝ | abs (x - 2) = abs (x - 4) + abs (x - 6)}.card = 1 :=
by
  sorry

end unique_solution_to_absolute_value_equation_l268_268973


namespace minimum_m_l268_268286

-- Define the arithmetic sequence {a_n} and sum S_n
def arithmetic_seq (a d n : ℕ) : ℕ := a + (n - 1) * d
noncomputable def sum (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

-- Conditions
def cond1 : Prop := (arithmetic_seq 1 2 1 + arithmetic_seq 1 2 13 = 26)
def cond2 : Prop := (sum 1 2 9 = 81)

-- General term formula for {a_n}
def general_term_formula (n : ℕ) : ℕ := 2 * n - 1

-- Define the sequence {b_n} and the sum T_n
def b (n : ℕ) : ℝ := 1 / (general_term_formula (n+1) * general_term_formula (n+2))
def T (n : ℕ) : ℝ := ∑ i in range n, b i

-- Proof of the minimum value of m
theorem minimum_m (m : ℝ) : (∀ n : ℕ, 30 * T n - m ≤ 0) → m ≥ 5 :=
sorry

end minimum_m_l268_268286


namespace tan_8100_is_zero_l268_268491

noncomputable def tan_8100_deg : ℝ := Real.tan (8100 * Real.pi / 180)

theorem tan_8100_is_zero : tan_8100_deg = 0 :=
by
  -- Step through the mathematical reduction
  have : 8100 % 360 = 180 := by norm_num
  have : Real.tan (8100 * Real.pi / 180) = Real.tan (180 * Real.pi / 180) := by rw [← Real.tan_add_periodic (8100 * Real.pi / 180) this]
  rw [Real.tan_pi]
  exact this

end tan_8100_is_zero_l268_268491


namespace germs_per_dish_l268_268641

theorem germs_per_dish (n_germs n_dishes : ℝ) (h1 : n_germs = 0.036 * 10^5) (h2 : n_dishes = 45000 * 10^(-3)) :
  n_germs / n_dishes = 80 :=
by
  sorry

end germs_per_dish_l268_268641


namespace sin_odd_monotonically_increasing_l268_268807

theorem sin_odd_monotonically_increasing :
  (∀ x, sin (-x) = -sin x) ∧ (∀ x ∈ Ioo (0 : ℝ) 1, (deriv sin x > 0)) := by
  sorry

end sin_odd_monotonically_increasing_l268_268807


namespace total_amount_owed_l268_268294

-- Definition of conditions
def payment_per_lawn := (13 : ℝ) / 3
def mowed_lawns := (8 : ℝ) / 5
def base_fee := 5

-- The statement we aim to prove
theorem total_amount_owed:
  payment_per_lawn * mowed_lawns + base_fee = 179 / 15 := 
by
  sorry

end total_amount_owed_l268_268294


namespace find_c_l268_268091

theorem find_c (c : ℝ) (h1 : 0 < c ∧ c < 5) 
               (h2 : ∃ P : ℝ × ℝ, P = (0, c)) 
               (h3 : ∃ S : ℝ × ℝ, S = (5, c - 5))
               (h4 : ratio_of_areas : (4 : ℝ) / 16) : 
  c = 10 / 3 := 
sorry

end find_c_l268_268091


namespace perimeter_increase_correct_l268_268865

noncomputable def first_triangle_side_length : ℝ := 3
noncomputable def growth_factor : ℝ := 1.25
noncomputable def fourth_triangle_side_length : ℝ := (growth_factor^3) * first_triangle_side_length
noncomputable def first_perimeter : ℝ := 3 * first_triangle_side_length
noncomputable def fourth_perimeter : ℝ := 3 * fourth_triangle_side_length

noncomputable def percent_increase : ℝ := ((fourth_perimeter - first_perimeter) / first_perimeter) * 100

theorem perimeter_increase_correct : percent_increase ≈ 95.3 := by
  sorry

end perimeter_increase_correct_l268_268865


namespace log_inequality_l268_268880

open Real

theorem log_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  log (1 + sqrt (a * b)) ≤ (1 / 2) * (log (1 + a) + log (1 + b)) :=
sorry

end log_inequality_l268_268880


namespace reciprocal_of_neg_2023_l268_268741

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l268_268741


namespace number_of_unique_products_l268_268958

theorem number_of_unique_products (x y : ℕ) (hx : x ∈ {1, 2, 3, 4}) (hy : y ∈ {5, 6, 7, 8}) : 
  (finset.image (λ xy, xy.1 * xy.2) ((finset.product {1, 2, 3, 4} {5, 6, 7, 8}))).card = 15 :=
sorry

end number_of_unique_products_l268_268958


namespace no_finite_maximum_for_determinant_l268_268913

theorem no_finite_maximum_for_determinant :
  ∀ θ : Real, ¬ ∃ M : Real, ∀ θ : Real, (1 + tan θ) * 1 - 1 * 1 - ((1 + cos θ) * 1 - 1 * 1) + ((1 + cos θ) * (1 + tan θ) - 1 * 1) ≤ M :=
by
  sorry

end no_finite_maximum_for_determinant_l268_268913


namespace speed_ratio_l268_268455

theorem speed_ratio (v1 v2 : ℝ) (t1 t2 : ℝ) (dist_before dist_after : ℝ) (total_dist : ℝ)
  (h1 : dist_before + dist_after = total_dist)
  (h2 : dist_before = 20)
  (h3 : dist_after = 20)
  (h4 : t2 = t1 + 11)
  (h5 : t2 = 22)
  (h6 : t1 = dist_before / v1)
  (h7 : t2 = dist_after / v2) :
  v1 / v2 = 2 := 
sorry

end speed_ratio_l268_268455


namespace lena_optimal_strategy_yields_7000_rubles_l268_268021

noncomputable def lena_annual_income : ℕ := 7000

theorem lena_optimal_strategy_yields_7000_rubles
  (monthly_salary : ℕ)
  (monthly_expenses : ℕ)
  (deposit_interest_monthly : ℚ)
  (debit_card_interest_annual : ℚ)
  (credit_card_limit : ℕ)
  (credit_card_fee_percent : ℚ)
  (monthly_savings : ℕ)
  (P_total : ℚ)
  (deposit_interest : ℚ)
  (debit_card_interest : ℚ) :
  monthly_salary = 50000 ∧
  monthly_expenses = 45000 ∧
  deposit_interest_monthly = 0.01 ∧
  debit_card_interest_annual = 0.08 ∧
  credit_card_limit = 100000 ∧
  credit_card_fee_percent = 0.03 ∧
  monthly_savings = 5000 ∧
  P_total = 5000 * (sum (λ k, (1 : ℚ) * 1.01 ^ k) (finset.range 12)) ∧
  deposit_interest = P_total - 5000 * 12 ∧
  debit_card_interest = 45000 * 0.08 ->
  deposit_interest + debit_card_interest = lena_annual_income :=
by
  sorry

end lena_optimal_strategy_yields_7000_rubles_l268_268021


namespace pyramid_volume_eq_l268_268235

-- Define the lengths of the edges
def AB : ℝ := 5
def BC : ℝ := 2
def CG : ℝ := 4
def BE : ℝ := Real.sqrt (AB^2 + BC^2) -- diagonal of the base rectangle
def base_area_BCHE : ℝ := BC * BE -- area of the base rectangle

-- Define the ratios and height of pyramid members
def FM : ℝ := 3
def MG : ℝ := 1
def height_M : ℝ := 3 -- height from M to the base resulting from FM being 3 times MG

-- Proof that the volume of the pyramid is 2√29
theorem pyramid_volume_eq :  (1 / 3) * base_area_BCHE * height_M = 2 * Real.sqrt 29 :=
by
  sorry

end pyramid_volume_eq_l268_268235


namespace find_f_5_l268_268559

def f : ℕ → ℕ
| x => if x > 10 then x + 3
       else f (f (x + 5))

theorem find_f_5 : f 5 = 24 := sorry

end find_f_5_l268_268559


namespace radius_of_third_circle_l268_268790

open Real

theorem radius_of_third_circle (r : ℝ) :
  let r_large := 40
  let r_small := 25
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  let region_area := area_large - area_small
  let half_region_area := region_area / 2
  let third_circle_area := π * r^2
  (third_circle_area = half_region_area) -> r = 15 * sqrt 13 :=
by
  sorry

end radius_of_third_circle_l268_268790


namespace Lois_books_total_l268_268689

-- Definitions based on the conditions
def initial_books : ℕ := 150
def books_given_to_nephew : ℕ := initial_books / 4
def remaining_books : ℕ := initial_books - books_given_to_nephew
def non_fiction_books : ℕ := remaining_books * 60 / 100
def kept_non_fiction_books : ℕ := non_fiction_books / 2
def fiction_books : ℕ := remaining_books - non_fiction_books
def lent_fiction_books : ℕ := fiction_books / 3
def remaining_fiction_books : ℕ := fiction_books - lent_fiction_books
def newly_purchased_books : ℕ := 12

-- The total number of books Lois has now
def total_books_now : ℕ := kept_non_fiction_books + remaining_fiction_books + newly_purchased_books

-- Theorem statement
theorem Lois_books_total : total_books_now = 76 := by
  sorry

end Lois_books_total_l268_268689


namespace diagonals_intersect_at_single_point_l268_268636

theorem diagonals_intersect_at_single_point (α : ℝ) (hα : α = 6 * (Real.pi / 180))
  (h1 : sin(2 * α) * sin(2 * α) * sin(8 * α) = sin(α) * sin(3 * α) * sin(14 * α)) :
  ∃ (P : Point), intersects P (diagonal 1) (diagonal 2) ∧ intersects P (diagonal 3) (diagonal 4) ∧ intersects P (diagonal 5) (diagonal 6) := 
sorry

end diagonals_intersect_at_single_point_l268_268636


namespace probability_sum_18_is_1_over_54_l268_268612

open Finset

-- Definitions for a 6-faced die, four rolls, and a probability space.
def faces := {1, 2, 3, 4, 5, 6}
def dice_rolls : Finset (Finset ℕ) := product faces (product faces (product faces faces))

def valid_sum : ℕ := 18

noncomputable def probability_of_sum_18 : ℚ :=
  (dice_rolls.filter (λ r, r.sum = valid_sum)).card / dice_rolls.card

theorem probability_sum_18_is_1_over_54 :
  probability_of_sum_18 = 1 / 54 := 
  sorry

end probability_sum_18_is_1_over_54_l268_268612


namespace reciprocal_of_negative_2023_l268_268759

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l268_268759


namespace range_of_function_l268_268734

noncomputable def f (x : ℝ) : ℝ := (Real.sin x - 1) / Real.sqrt (3 - 2 * Real.cos x - 2 * Real.sin x)

theorem range_of_function : 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → f x ∈ set.Icc (-1 : ℝ) 0 :=
by
  sorry

end range_of_function_l268_268734


namespace point_in_second_quadrant_l268_268618

variable (m : ℝ)

-- Defining the conditions
def x_negative (m : ℝ) := 3 - m < 0
def y_positive (m : ℝ) := m - 1 > 0

theorem point_in_second_quadrant (h1 : x_negative m) (h2 : y_positive m) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l268_268618


namespace probability_of_2_successes_in_4_trials_l268_268380

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1-p)^(n-k))

theorem probability_of_2_successes_in_4_trials :
  binomial_probability 4 2 0.3 = 0.2646 :=
by sorry

end probability_of_2_successes_in_4_trials_l268_268380


namespace a5_gt_b5_l268_268649

variables {a_n b_n : ℕ → ℝ}
variables {a1 b1 a3 b3 : ℝ}
variables {q : ℝ} {d : ℝ}

/-- Given conditions -/
axiom h1 : a1 = b1
axiom h2 : a1 > 0
axiom h3 : a3 = b3
axiom h4 : a3 = a1 * q^2
axiom h5 : b3 = a1 + 2 * d
axiom h6 : a1 ≠ a3

/-- Prove that a_5 is greater than b_5 -/
theorem a5_gt_b5 : a1 * q^4 > a1 + 4 * d :=
by sorry

end a5_gt_b5_l268_268649


namespace expected_matches_is_one_variance_matches_is_one_l268_268421

noncomputable def indicator (k : ℕ) (matches : Finset ℕ) : ℕ :=
  if k ∈ matches then 1 else 0

def expected_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  (Finset.range N).sum (λ k, indicator k matches / N)

def variance_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  let E_S := expected_matches N matches in
  let E_S2 := (Finset.range N).sum (λ k, (indicator k matches) ^ 2 / N) in
  E_S2 - E_S ^ 2

theorem expected_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  expected_matches N matches = 1 := sorry

theorem variance_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  variance_matches N matches = 1 := sorry

end expected_matches_is_one_variance_matches_is_one_l268_268421


namespace green_people_count_l268_268239

-- Define the total number of people
def total_people := 150

-- Define the number of people who think teal is "kinda blue"
def kinda_blue_people := 90

-- Define the number of people who think teal is both "kinda blue" and "kinda green"
def both_blue_and_green_people := 35

-- Define the number of people who think teal is neither "kinda blue" nor "kinda green"
def neither_blue_nor_green_people := 25

theorem green_people_count : 
  let only_blue_people := kinda_blue_people - both_blue_and_green_people in
  let accounted_people := both_blue_and_green_people + only_blue_people + neither_blue_nor_green_people in
  let only_green_people := total_people - accounted_people in
  both_blue_and_green_people + only_green_people = 70 :=
by
  sorry

end green_people_count_l268_268239


namespace valid_seating_count_l268_268637

noncomputable def num_valid_seatings : ℕ :=
  -- Define the valid seating arrangements based on the given conditions
  let seats := ["Alice", "Bob", "Carla", "Derek", "Eric"] in
  let valid_arrangements := {arrangement | 
    arrangement ∈ seats.permutations ∧
    (∀ i, (arrangement[i] = "Alice" → i < 4 ∧ arrangement[i + 1] ≠ "Bob")) ∧ -- Alice not next to Bob
    (∀ i, (arrangement[i] = "Derek" → 
           i ≠ 0 ∧ arrangement[i - 1] ≠ "Eric" ∧ arrangement[i - 1] ≠ "Carla" ∧ 
           i < 4 ∧ arrangement[i + 1] ≠ "Eric" ∧ arrangement[i + 1] ≠ "Carla")) -- Derek not next to Eric or Carla
  } in
  valid_arrangements.card

theorem valid_seating_count : num_valid_seatings = 44 := 
  by
  sorry -- Proof to be completed

end valid_seating_count_l268_268637


namespace vector_dot_product_l268_268627

theorem vector_dot_product (a b c : ℝ) (B : ℝ)
  (h1 : b^2 = a * c)
  (h2 : a + c = 3)
  (h3 : real.cos B = 3 / 4) :
  (-(a * c * real.cos B)) = -3 / 2 :=
sorry

end vector_dot_product_l268_268627


namespace problem_statement_l268_268927

variable (a x : ℝ)

theorem problem_statement (ha : a > 2) :
  let m := a + 1 / (a - 2)
      n := 4 - x^2
  in m >= n :=
begin
  sorry
end

end problem_statement_l268_268927


namespace jennifer_fish_tank_problem_l268_268659

theorem jennifer_fish_tank_problem :
  let built_tanks := 3
  let fish_per_built_tank := 15
  let planned_tanks := 3
  let fish_per_planned_tank := 10
  let total_built_fish := built_tanks * fish_per_built_tank
  let total_planned_fish := planned_tanks * fish_per_planned_tank
  let total_fish := total_built_fish + total_planned_fish
  total_fish = 75 := by
    let built_tanks := 3
    let fish_per_built_tank := 15
    let planned_tanks := 3
    let fish_per_planned_tank := 10
    let total_built_fish := built_tanks * fish_per_built_tank
    let total_planned_fish := planned_tanks * fish_per_planned_tank
    let total_fish := total_built_fish + total_planned_fish
    have h₁ : total_built_fish = 45 := by sorry
    have h₂ : total_planned_fish = 30 := by sorry
    have h₃ : total_fish = 75 := by sorry
    exact h₃

end jennifer_fish_tank_problem_l268_268659


namespace book_distribution_l268_268835

theorem book_distribution : ∃ (n : ℕ), 
  n = (cardinality (setOf (λ (x : ℕ), 2 ≤ x ∧ x ≤ 6)) ∧ n = 5 :=
by 
  sorry

end book_distribution_l268_268835


namespace total_bill_is_756_l268_268040

-- Define the variables
def number_of_investment_bankers : ℕ := 4
def number_of_clients : ℕ := 5
def cost_per_person : ℝ := 70
def gratuity_percentage : ℝ := 0.20

-- Define the total cost before gratuity
def total_cost_before_gratuity : ℝ := (number_of_investment_bankers + number_of_clients) * cost_per_person

-- Define the gratuity amount
def gratuity_amount : ℝ := total_cost_before_gratuity * gratuity_percentage

-- Define the total bill amount including gratuity
def total_bill_including_gratuity : ℝ := total_cost_before_gratuity + gratuity_amount

theorem total_bill_is_756 : total_bill_including_gratuity = 756 := 
by 
  have h : total_cost_before_gratuity = (number_of_investment_bankers + number_of_clients) * cost_per_person := rfl
  have h1 : total_cost_before_gratuity = 9 * 70 := by rw [h]; norm_num
  have k : gratuity_amount = total_cost_before_gratuity * gratuity_percentage := rfl
  have k1 : gratuity_amount = 630 * 0.2 := by rw [h1]; norm_num
  have k2 : gratuity_amount = 126 := by rw k1; norm_num
  have total : total_bill_including_gratuity = total_cost_before_gratuity + gratuity_amount := rfl
  have total1 : total_bill_including_gratuity = 630 + 126 := by rw [h1, k2]; norm_num
  show total_bill_including_gratuity = 756 from total1

end total_bill_is_756_l268_268040


namespace hyperbola_center_l268_268041

/-- 
Given that the foci of the hyperbola are at coordinates (2, 0) and (8, 6),
prove that the coordinates of the center of the hyperbola are (5, 3).
-/
theorem hyperbola_center : 
  let f1 := (2, 0) in 
  let f2 := (8, 6) in 
  let center := (5, 3) in 
  (center.1 = (f1.1 + f2.1) / 2) ∧ (center.2 = (f1.2 + f2.2) / 2) :=
by {
  let f1 := (2, 0),
  let f2 := (8, 6),
  let center := (5, 3),
  sorry
}

end hyperbola_center_l268_268041


namespace ordered_notebooks_amount_l268_268059

def initial_notebooks : ℕ := 10
def ordered_notebooks (x : ℕ) : ℕ := x
def lost_notebooks : ℕ := 2
def current_notebooks : ℕ := 14

theorem ordered_notebooks_amount (x : ℕ) (h : initial_notebooks + ordered_notebooks x - lost_notebooks = current_notebooks) : x = 6 :=
by
  sorry

end ordered_notebooks_amount_l268_268059


namespace notebook_cost_l268_268267

-- Definitions and conditions
def initial_money : ℕ := 56
def notebooks_bought : ℕ := 7
def cost_per_book : ℕ := 7
def books_bought : ℕ := 2
def money_left : ℕ := 14
def total_spent := initial_money - money_left
def total_books_cost := books_bought * cost_per_book

-- Target value
def x := 4

-- Proof statement
theorem notebook_cost : (notebooks_bought * x + total_books_cost = total_spent) → x = 4 := by
  simp
  sorry

end notebook_cost_l268_268267


namespace reciprocal_of_neg_2023_l268_268768

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268768


namespace minimum_omega_l268_268179

theorem minimum_omega (ω : ℝ) (ϕ : ℝ) (T : ℝ) :
  (ω > 0) →
  (0 < ϕ ∧ ϕ < π) →
  (T = 2 * π / ω) →
  (cos (ω * T + ϕ) = 1/2) →
  (∃ x : ℝ, x = 7 * π / 3 ∧ deriv (λ x, cos (ω * x + ϕ)) x = 0) →
  ω = 2 / 7 :=
by
  sorry

end minimum_omega_l268_268179


namespace most_stable_athlete_l268_268920

theorem most_stable_athlete (s2_A s2_B s2_C s2_D : ℝ) 
  (hA : s2_A = 0.5) 
  (hB : s2_B = 0.5) 
  (hC : s2_C = 0.6) 
  (hD : s2_D = 0.4) :
  s2_D < s2_A ∧ s2_D < s2_B ∧ s2_D < s2_C :=
by
  sorry

end most_stable_athlete_l268_268920


namespace P_at_7_eq_5760_l268_268682

noncomputable def P (x : ℝ) : ℝ :=
  12 * (x - 1) * (x - 2) * (x - 3)^2 * (x - 6)^4

theorem P_at_7_eq_5760 : P 7 = 5760 :=
by
  -- Proof goes here
  sorry

end P_at_7_eq_5760_l268_268682


namespace find_s_base_10_l268_268101

-- Defining the conditions of the problem
def s_in_base_b_equals_42 (b : ℕ) : Prop :=
  let factor_1 := b + 3
  let factor_2 := b + 4
  let factor_3 := b + 5
  let produced_number := factor_1 * factor_2 * factor_3
  produced_number = 2 * b^3 + 3 * b^2 + 2 * b + 5

-- The proof problem as a Lean 4 statement
theorem find_s_base_10 :
  (∃ b : ℕ, s_in_base_b_equals_42 b) →
  13 + 14 + 15 = 42 :=
sorry

end find_s_base_10_l268_268101


namespace largest_k_l268_268666

noncomputable def sequence_behavior (k : ℕ) : ℤ → ℤ :=
  if k = 0 then 0 else
    let S_prev := sequence_behavior (k - 1) in
    let a_k := if S_prev < k then 1 else -1 in
    S_prev + k * a_k

theorem largest_k (h : ∀ k, sequence_behavior k k <= 2010 ∧ sequence_behavior k = 0) :
  ∃ k <= 2010, sequence_behavior k = 0 := by
  sorry

end largest_k_l268_268666


namespace probability_longer_piece_x_times_shorter_piece_l268_268050

variable (L : ℝ) (x : ℝ)

theorem probability_longer_piece_x_times_shorter_piece (L_pos : 0 < L) (x_pos : 0 < x) :
  (∃ C : ℝ, C > 0 ∧ C < L ∧ L - C = x * C) → (∀ C, C = L / (x + 1) → 0) :=
by sorry

end probability_longer_piece_x_times_shorter_piece_l268_268050


namespace inscribed_square_length_l268_268321

-- Define the right triangle PQR with given sides
variables (PQ QR PR : ℕ)
variables (h s : ℚ)

-- Given conditions
def right_triangle_PQR : Prop := PQ = 5 ∧ QR = 12 ∧ PR = 13
def altitude_Q_to_PR : Prop := h = (PQ * QR) / PR
def side_length_of_square : Prop := s = h * (1 - h / PR)

theorem inscribed_square_length (PQ QR PR h s : ℚ) 
    (right_triangle_PQR : PQ = 5 ∧ QR = 12 ∧ PR = 13)
    (altitude_Q_to_PR : h = (PQ * QR) / PR) 
    (side_length_of_square : s = h * (1 - h / PR)) 
    : s = 6540 / 2207 := by
  -- we skip the proof here as requested
  sorry

end inscribed_square_length_l268_268321


namespace binomial_constant_term_l268_268458

theorem binomial_constant_term (n : ℕ) (h : n > 0) :
  (∃ r : ℕ, n = 2 * r) ↔ (n = 6) :=
by
  sorry

end binomial_constant_term_l268_268458


namespace possible_locations_of_R_l268_268681

theorem possible_locations_of_R 
  (ABC : Triangle)
  (ω : Circle)
  (ell : Line)
  (P Q R : Point)
  (hω : is_incircle ABC ω)
  (h_tangent : is_tangent ell ω)
  (hPQ : intersects_segment ell (segment BC) P ∧ intersects_segment ell (segment CA) Q)
  (hR : (dist P R = dist P A) ∧ (dist Q R = dist Q B))
  :
  ∃ (C : Circle), 
    (C = Circle_incenter_radius_mul 2 ω ∨ C = Circle_incenter_radius_mul 4 ω) ∧ 
    lies_on_arc R C :=
sorry

end possible_locations_of_R_l268_268681


namespace range_of_a_l268_268220

theorem range_of_a (h : ∀ x ∈ set.Ioo 1 2, real.log x / real.log a > (x - 1)^2) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l268_268220


namespace trader_sells_cloth_l268_268856

variable (x : ℝ) (SP_total : ℝ := 6900) (profit_per_meter : ℝ := 20) (CP_per_meter : ℝ := 66.25)

theorem trader_sells_cloth : SP_total = x * (CP_per_meter + profit_per_meter) → x = 80 :=
by
  intro h
  -- Placeholder for actual proof
  sorry

end trader_sells_cloth_l268_268856


namespace pencils_evenly_distributed_l268_268899

-- Define the initial number of pencils Eric had
def initialPencils : Nat := 150

-- Define the additional pencils brought by another teacher
def additionalPencils : Nat := 30

-- Define the total number of containers
def numberOfContainers : Nat := 5

-- Define the total number of pencils after receiving additional pencils
def totalPencils := initialPencils + additionalPencils

-- Define the number of pencils per container after even distribution
def pencilsPerContainer := totalPencils / numberOfContainers

-- Statement of the proof problem
theorem pencils_evenly_distributed :
  pencilsPerContainer = 36 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end pencils_evenly_distributed_l268_268899


namespace phi_range_l268_268178

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ) + 1

theorem phi_range (φ : ℝ) : 
  (|φ| ≤ Real.pi / 2) ∧ 
  (∀ x ∈ Set.Ioo (Real.pi / 24) (Real.pi / 3), f x φ > 2) →
  (Real.pi / 12 ≤ φ ∧ φ ≤ Real.pi / 6) :=
by
  sorry

end phi_range_l268_268178


namespace f_3_2_plus_f_5_1_l268_268500

def f (a b : ℤ) : ℚ :=
  if a - b ≤ 2 then (a * b - a - 1) / (3 * a)
  else (a * b + b - 1) / (-3 * b)

theorem f_3_2_plus_f_5_1 :
  f 3 2 + f 5 1 = -13 / 9 :=
by
  sorry

end f_3_2_plus_f_5_1_l268_268500


namespace largest_square_area_correct_l268_268261

noncomputable def area_of_largest_square (x y z : ℝ) : Prop := 
  ∃ (area : ℝ), (z^2 = area) ∧ 
                 (x^2 + y^2 = z^2) ∧ 
                 (x^2 + y^2 + 2*z^2 = 722) ∧ 
                 (area = 722 / 3)

theorem largest_square_area_correct (x y z : ℝ) :
  area_of_largest_square x y z :=
  sorry

end largest_square_area_correct_l268_268261


namespace sin_225_correct_l268_268479

-- Define the condition of point being on the unit circle at 225 degrees.
noncomputable def P_225 := Complex.polar 1 (Real.pi + Real.pi / 4)

-- Define the goal statement that translates the question and correct answer.
theorem sin_225_correct : Complex.sin (Real.pi + Real.pi / 4) = -Real.sqrt 2 / 2 := 
by sorry

end sin_225_correct_l268_268479


namespace fraction_of_shaded_area_is_11_by_12_l268_268309

noncomputable def shaded_fraction_of_square : ℚ :=
  let s : ℚ := 1 -- Assume the side length of the square is 1 for simplicity.
  let P := (0, s / 2)
  let Q := (s / 3, s)
  let V := (0, s)
  let base := s / 2
  let height := s / 3
  let triangle_area := (1 / 2) * base * height
  let square_area := s * s
  let shaded_area := square_area - triangle_area
  shaded_area / square_area

theorem fraction_of_shaded_area_is_11_by_12 : shaded_fraction_of_square = 11 / 12 :=
  sorry

end fraction_of_shaded_area_is_11_by_12_l268_268309


namespace gcd_mult_product_is_perfect_square_l268_268679

-- The statement of the problem in Lean 4
theorem gcd_mult_product_is_perfect_square
  (x y z : ℕ)
  (h : 1/x - 1/y = 1/z) : 
  ∃ k : ℕ, k^2 = Nat.gcd x (Nat.gcd y z) * x * y * z :=
by 
  sorry

end gcd_mult_product_is_perfect_square_l268_268679


namespace expected_value_matches_variance_matches_l268_268427

variables {N : ℕ} (I : Fin N → Bool)

-- Define the probability that a randomly chosen pair of cards matches
def p_match : ℝ := 1 / N

-- Define the indicator variable I_k
def I_k (k : Fin N) : ℝ :=
if I k then 1 else 0

-- Define the sum S of all the indicator variables
def S : ℝ := (Finset.univ.sum I_k)

-- Expected value E[I_k] is 1/N
def E_I_k : ℝ := 1 / N

-- Expected value E[S] is the sum of E[I_k] over all k, which is 1
theorem expected_value_matches : ∑ k, E_I_k = 1 := sorry

-- Variance calculation: Var[S] = E[S^2] - (E[S])^2
def E_S_sq : ℝ := (Finset.univ.sum (λ k, I_k k * I_k k)) + 
                  2 * (Finset.univ.sum (λ (jk : Fin N × Fin N), if jk.1 < jk.2 then I_k jk.1 * I_k jk.2 else 0))

theorem variance_matches : (E_S_sq - 1) = 1 := sorry

end expected_value_matches_variance_matches_l268_268427


namespace OM_perp_ON_l268_268155

-- Define the conditions
def point_on_parabola (E : ℝ × ℝ) (p : ℝ) : Prop :=
  E.2 ^ 2 = 2 * p * E.1

def line_through_point (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  l P.1 = P.2

def intersects_parabola (l : ℝ → ℝ) (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ y1 y2, A = (y1^2 / (2 * p), y1) ∧ B = (y2^2 / (2 * p), y2) ∧ y1 ≠ 2 ∧ y2 ≠ 2

def line_intersection (line1 line2 : ℝ → ℝ) (x : ℝ) : ℝ :=
  line1 x

-- State the theorem
theorem OM_perp_ON (E : ℝ × ℝ) (A B : ℝ × ℝ) (p k : ℝ) :
  point_on_parabola E p →
  ∃ l, line_through_point l (2, 0) ∧ intersects_parabola l p A B →
  ∃ M N : ℝ × ℝ, M.1 = -2 ∧ N.1 = -2 ∧
  -- Define coordinates of M and N based on line intersection
  let yM := line_intersection (λ x, 2 / (A.2 + 2) * (x - 2) + 2) (-2),
      yN := line_intersection (λ x, 2 / (B.2 + 2) * (x - 2) + 2) (-2) in
  M = (-2, yM) ∧ N = (-2, yN) →
  -- Prove the vectors OM and ON are perpendicular
  ((-2, yM) - (0, 0)) • ((-2, yN) - (0, 0)) = 0 :=
by sorry

end OM_perp_ON_l268_268155


namespace claire_earnings_l268_268085

theorem claire_earnings
  (total_flowers : ℕ)
  (tulips : ℕ)
  (white_roses : ℕ)
  (price_per_red_rose : ℚ)
  (sell_fraction : ℚ)
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : price_per_red_rose = 0.75)
  (h5 : sell_fraction = 1/2) : 
  (total_flowers - tulips - white_roses) * sell_fraction * price_per_red_rose = 75 :=
by
  sorry

end claire_earnings_l268_268085


namespace triangle_angles_and_sides_l268_268431

theorem triangle_angles_and_sides (A B C a b c : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hAngleSum : A + B + C = 180) (hle_ABC : A ≤ B ∧ B ≤ C) 
  (hle_abc : a ≤ b ∧ b ≤ c) (hArea : (a * b * (Real.sin C))/2 = 1) :
  (0 < A ∧ A ≤ 60) ∧ (0 < B ∧ B < 90) ∧ (60 ≤ C ∧ C < 180) ∧ 
  (0 < a) ∧ (sqrt 2 ≤ b) ∧ (frac 2 (Real/*c.root ⟨4, rfl⟩) > c) := 
by
  sorry

end triangle_angles_and_sides_l268_268431


namespace area_of_AFCG_quadrilateral_l268_268012

-- Define the conditions directly appearing in the problem
def right_triangle (A B C : Type) := sorry
def angle_deg_eq (α β : Type) (deg : ℕ) := sorry
def side_length (A B : Type) (length : ℕ) := sorry

theorem area_of_AFCG_quadrilateral
  (A F H C G : Type)
  (AF : side_length A F 32)
  (angle_AFH_eq_90 : angle_deg_eq A F H 90)
  (angle_FHC_eq_90 : angle_deg_eq F H C 90)
  (angle_HFG_eq_45 : angle_deg_eq H F G 45)
  (triangle_AFH_is_right : right_triangle A F H)
  (triangle_FCH_is_right : right_triangle F C H)
  (triangle_CGH_is_right : right_triangle C G H) :
  -- Proving the area of quadrilateral AFCG is 1024 square units
  area_of_quadrilateral A F C G = 1024 := by
  sorry

end area_of_AFCG_quadrilateral_l268_268012


namespace find_angle_B_l268_268227

-- Define the properties and values
noncomputable def triangle_ABC : Type := (BC AC A B : ℝ) (BC_pos : BC = sqrt 3) (AC_pos : AC = sqrt 2) (A_angle : A = π / 3)

-- Create the theorem statement that we need to prove
theorem find_angle_B (t : triangle_ABC) : t.B = π / 4 :=
sorry

end find_angle_B_l268_268227


namespace permutations_1225_l268_268188

def digit_list_1225 : List Nat := [1, 2, 2, 5]
def number_of_permutations (l : List Nat) : Nat :=
  let n := l.length
  let freq := l.foldl (λ m d => m.insert d (m.find d + 1).getOrElse 1) Std.RBMap.empty
  let factorial (x : Nat) : Nat := if x = 0 then 1 else x * factorial (x - 1)
  factorial n / (freq.toList.foldl (λ p q => p * factorial q.snd) 1)

theorem permutations_1225 : number_of_permutations digit_list_1225 = 12 := by
  sorry

end permutations_1225_l268_268188


namespace magnitude_sum_is_4_l268_268150

variables (a b : EuclideanSpace ℝ (Fin 2))

def magnitude (v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  Real.sqrt (dot_product v v)

-- Given conditions
axiom ha : magnitude a = Real.sqrt 7 + 1
axiom hb : magnitude b = Real.sqrt 7 - 1
axiom hab : magnitude (a - b) = 4

-- Required proof
theorem magnitude_sum_is_4 : magnitude (a + b) = 4 :=
sorry

end magnitude_sum_is_4_l268_268150


namespace largest_k_divisibility_l268_268909

theorem largest_k_divisibility :
  ∃ k : ℕ, (∀ x y : ℤ, k ∣ (x * y + 1) → k ∣ (x + y)) ∧
  (∀ m : ℕ, (∀ x y : ℤ, m ∣ (x * y + 1) → m ∣ (x + y)) → m ≤ 24) ∧ k = 24 :=
begin
  sorry
end

end largest_k_divisibility_l268_268909


namespace find_expression_l268_268031

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_about_x2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem find_expression (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : symmetric_about_x2 f)
  (h3 : ∀ x, -2 < x ∧ x ≤ 2 → f x = -x^2 + 1) :
  ∀ x, -6 < x ∧ x < -2 → f x = -(x + 4)^2 + 1 :=
by
  sorry

end find_expression_l268_268031


namespace discarded_number_approx_l268_268336

theorem discarded_number_approx (sum_65_numbers : ℝ) 
                                (one_number_is_30 : ℝ)
                                (new_avg : ℝ) 
                                (sum_65_numbers = 2600)
                                (one_number_is_30 = 30)
                                (new_avg ≈ 39.476190476190474) :
                                (∃ D : ℝ, D ≈ 83) :=
sorry

end discarded_number_approx_l268_268336


namespace num_days_c_worked_l268_268017

theorem num_days_c_worked (d : ℕ) :
  let daily_wage_c := 100
  let daily_wage_b := (4 * 20)
  let daily_wage_a := (3 * 20)
  let total_earning := 1480
  let earning_a := 6 * daily_wage_a
  let earning_b := 9 * daily_wage_b
  let earning_c := d * daily_wage_c
  total_earning = earning_a + earning_b + earning_c →
  d = 4 :=
by {
  sorry
}

end num_days_c_worked_l268_268017


namespace reciprocal_of_neg_2023_l268_268744

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l268_268744


namespace theater_price_balcony_l268_268855

theorem theater_price_balcony 
  (price_orchestra : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (extra_balcony_tickets : ℕ) (price_balcony : ℕ) 
  (h1 : price_orchestra = 12) 
  (h2 : total_tickets = 380) 
  (h3 : total_revenue = 3320) 
  (h4 : extra_balcony_tickets = 240) 
  (h5 : ∃ (O : ℕ), O + (O + extra_balcony_tickets) = total_tickets ∧ (price_orchestra * O) + (price_balcony * (O + extra_balcony_tickets)) = total_revenue) : 
  price_balcony = 8 := 
by
  sorry

end theater_price_balcony_l268_268855


namespace bob_friends_l268_268469

-- Define the total price and the amount paid by each person
def total_price : ℕ := 40
def amount_per_person : ℕ := 8

-- Define the total number of people who paid
def total_people : ℕ := total_price / amount_per_person

-- Define Bob's presence and require proving the number of his friends
theorem bob_friends (total_price amount_per_person total_people : ℕ) (h1 : total_price = 40)
  (h2 : amount_per_person = 8) (h3 : total_people = total_price / amount_per_person) : 
  total_people - 1 = 4 :=
by
  sorry

end bob_friends_l268_268469


namespace expected_number_of_matches_variance_of_number_of_matches_l268_268414

-- Defining the conditions first, and then posing the proof statements
namespace MatchingPairs

open ProbabilityTheory

-- Probabilistic setup for indicator variables
variable (N : ℕ) (prob : ℝ := 1 / N)

-- Indicator variable Ik representing matches
@[simp] def I (k : ℕ) : ℝ := if k < N then prob else 0

-- Define the sum of expected matches S
@[simp] def S : ℝ := ∑ k in finset.range N, I N k

-- Statement: The expectation of the number of matching pairs is 1
theorem expected_number_of_matches : E[S] = 1 := sorry

-- Statement: The variance of the number of matching pairs is 1
theorem variance_of_number_of_matches : Var S = 1 := sorry

end MatchingPairs

end expected_number_of_matches_variance_of_number_of_matches_l268_268414


namespace initial_three_points_collinear_l268_268655

noncomputable theory

open_locale classical

def collinear (A B C : Point) : Prop :=
∃ l : Line, A ∈ l ∧ B ∈ l ∧ C ∈ l

theorem initial_three_points_collinear
  (A B C : Point)
  (h1 : distinct A B C)
  (∀ t : ℕ, ∃ A B C : Point, ∃ D : Point, symmetric D A (perpendicular_bisector B C))
  (∃ P Q R : Point, collinear P Q R) :
  collinear A B C :=
sorry

end initial_three_points_collinear_l268_268655


namespace geometric_sequence_sum_sequence_b_n_sum_sequence_b_n_odd_inequality_p_n_l268_268135

-- Definitions and conditions
def sequence_sn (n : ℕ) : ℕ → ℤ := λ n, 2 * a_n n - n^2 + 3 * n + 2

noncomputable def sequence_a_n (n : ℕ) : ℤ := 
if n = 0 then 0 else (sequence_sn n - sequence_sn (n - 1)) / 2

noncomputable def sequence_b_n (n : ℕ) : ℤ := 
sequence_a_n n * sin ((2 * n + 1) / 2 * real.pi)

noncomputable def sequence_cn (n : ℕ) : ℤ := 
-1 / (sequence_a_n n + n)

-- Proofs without solutions (we use sorry to skip proofs)
theorem geometric_sequence (n : ℕ) (h : 0 < n) : 
∀ n : ℕ+, sequence_a_n n + 2 * n = -2 * (2^n + 2n) := by sorry

theorem sum_sequence_b_n (n : ℕ) (h : n % 2 = 0) : 
∑ k in range n, sequence_b_n k = (2 * (1 - 2^n)) / 3 - n := by sorry

theorem sum_sequence_b_n_odd (n : ℕ) (h : n % 2 = 1) : 
∑ k in range n, sequence_b_n k = (2 + 2^(n + 1)) / 3 + n + 1 := by sorry

theorem inequality_p_n (n : ℕ) (h : 0 < n) : 
∑ k in range n, sequence_cn k < 5 / 6 := by sorry

end geometric_sequence_sum_sequence_b_n_sum_sequence_b_n_odd_inequality_p_n_l268_268135


namespace triangle_trisection_distance_l268_268262

theorem triangle_trisection_distance 
  (A B C : Point) 
  (H₁ H₂ : Point) 
  (AB AC : ℝ) 
  (p q : ℝ) 
  (x : ℝ) 
  (isosceles : AB = AC) 
  (H₁C_dist : dist C H₁ = 17) 
  (H₂C_dist : dist C H₂ = 20) :
  (dist A H₁ = x ∨ dist A H₂ = x) ↔ (x = 24.2 ∨ x = 10.2) := 
by
  sorry

end triangle_trisection_distance_l268_268262


namespace problem1_problem2_l268_268470

-- Problem 1: y(x + y) + (x + y)(x - y) = x^2
theorem problem1 (x y : ℝ) : y * (x + y) + (x + y) * (x - y) = x^2 := 
by sorry

-- Problem 2: ( (2m + 1) / (m + 1) + m - 1 ) ÷ ( (m + 2) / (m^2 + 2m + 1) ) = m^2 + m
theorem problem2 (m : ℝ) (h1 : m ≠ -1) : 
  ( (2 * m + 1) / (m + 1) + m - 1 ) / ( (m + 2) / ((m + 1)^2) ) = m^2 + m := 
by sorry

end problem1_problem2_l268_268470


namespace find_a_value_l268_268591

noncomputable def a := (2 * Real.sqrt 3) / 3

theorem find_a_value (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) :
  (∃ m, 2 * Real.sin x = a * Real.cos x) →
  (∃ m, (2 * Real.cos x) * (-a * Real.sin x) = -1) →
  a = (2 * Real.sqrt 3) / 3 :=
by
  intro h_intersect h_perpendicular
  -- assumption and use of the conditions h_intersect, h_perpendicular to prove the theorem
  sorry

end find_a_value_l268_268591


namespace chord_length_y_eq_x_plus_one_meets_circle_l268_268912

noncomputable def chord_length (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem chord_length_y_eq_x_plus_one_meets_circle 
  (A B : ℝ × ℝ) 
  (hA : A.2 = A.1 + 1) 
  (hB : B.2 = B.1 + 1) 
  (hA_on_circle : A.1^2 + A.2^2 + 2 * A.2 - 3 = 0)
  (hB_on_circle : B.1^2 + B.2^2 + 2 * B.2 - 3 = 0) :
  chord_length A B = 2 * Real.sqrt 2 := 
sorry

end chord_length_y_eq_x_plus_one_meets_circle_l268_268912


namespace ap_perpendicular_pd_l268_268871

/-- 
  Let ABC be a right triangle with ∠A = 90°. 
  E and F are points on AC and AB, respectively. 
  BE and CF intersect at D. 
  Let M be the circumcircle of ΔAEF and O be the circumcircle of ΔABC.
  P is the intersection of circles M and O.

  Prove that AP is perpendicular to PD. 
--/
theorem ap_perpendicular_pd 
  (A B C D E F P : Type) 
  [right_triangle A B C]
  (hab : B = 90°) 
  (hE : E ∈ line_segment A C) 
  (hF : F ∈ line_segment A B) 
  (hintersect : ∃ D, line BE intersects line CF at D) 
  (circumAEF : circle_circum A E F) 
  (circumABC : circle_circum A B C) 
  (hintersect_circ: P ∈ circle_intersect circumAEF circumABC) :
  perpendicular (line_through A P) (line_through P D) :=
begin
  sorry
end

end ap_perpendicular_pd_l268_268871


namespace evaluate_expression_l268_268121

theorem evaluate_expression : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200 :=
by
  sorry

end evaluate_expression_l268_268121


namespace probability_A_wins_within_5_tosses_probability_B_wins_within_5_tosses_probability_A_wins_until_win_probability_B_wins_until_win_l268_268440

theorem probability_A_wins_within_5_tosses : 
  -- Given conditions
  (∃ (die : ℕ → ℕ), (∀ n, 1 ≤ die n ∧ die n ≤ 6)) → 
  -- Question and correct answer
  (Probability (PlayerA_wins_within_5_tosses die) = 55 / 243) :=
sorry

theorem probability_B_wins_within_5_tosses : 
  -- Given conditions
  (∃ (die : ℕ → ℕ), (∀ n, 1 ≤ die n ∧ die n ≤ 6)) → 
  -- Question and correct answer
  (Probability (PlayerB_wins_within_5_tosses die) = 176 / 243) :=
sorry

theorem probability_A_wins_until_win : 
  -- Given conditions
  (∃ (die : ℕ → ℕ), (∀ n, 1 ≤ die n ∧ die n ≤ 6)) → 
  -- Question and correct answer
  (Probability (PlayerA_wins_until_win die) = 5 / 21) :=
sorry

theorem probability_B_wins_until_win : 
  -- Given conditions
  (∃ (die : ℕ → ℕ), (∀ n, 1 ≤ die n ∧ die n ≤ 6)) → 
  -- Question and correct answer
  (Probability (PlayerB_wins_until_win die) = 16 / 21) :=
sorry

end probability_A_wins_within_5_tosses_probability_B_wins_within_5_tosses_probability_A_wins_until_win_probability_B_wins_until_win_l268_268440


namespace castle_knights_liars_max_knights_l268_268232

-- Define the problem conditions and the goal
theorem castle_knights_liars_max_knights :
  ∃ (K : ℕ) (L : ℕ) (occupants : Fin 16 → Bool), 
    (K + L = 16) ∧ 
    (∀ (i : Fin 16), occupants i = true → 
      ((i.1 % 4 > 0 ∧ occupants ⟨i.1 - 1, by linarith [i.2]⟩ = false) ∨ 
       (i.1 % 4 < 3 ∧ occupants ⟨i.1 + 1, by linarith [i.2]⟩ = false) ∨ 
       (i.1 ≥ 4 ∧ occupants ⟨i.1 - 4, by linarith [i.2]⟩ = false) ∨ 
       (i.1 < 12 ∧ occupants ⟨i.1 + 4, by linarith [i.2]⟩ = false))) ∧ 
    (∀ (i : Fin 16), occupants i = false → 
      ((i.1 % 4 > 0 → occupants ⟨i.1 - 1, by linarith [i.2]⟩ = true) ∧ 
       (i.1 % 4 < 3 → occupants ⟨i.1 + 1, by linarith [i.2]⟩ = true) ∧ 
       (i.1 ≥ 4 → occupants ⟨i.1 - 4, by linarith [i.2]⟩ = true) ∧ 
       (i.1 < 12 → occupants ⟨i.1 + 4, by linarith [i.2]⟩ = true))) ∧ 
    K ≤ 12 := 
sorry

end castle_knights_liars_max_knights_l268_268232


namespace fourth_selected_number_is_3_l268_268256

def is_valid_red_ball_number (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 33

def data_from_row_1 : list ℕ := [2976, 3413, 2814, 2641]

def selected_numbers (n : ℕ) : list ℕ :=
  [28, 14, 26, 3, 22, 24] -- Determined from 1st row data given and reading method described

theorem fourth_selected_number_is_3 : selected_numbers 4 = 03 :=
by {
  -- Proof here
  sorry
}

end fourth_selected_number_is_3_l268_268256


namespace probability_sum_18_l268_268615

theorem probability_sum_18:
  (∑ k in {1,2,3,4,5,6}, k = 6)^4 * (probability {d₁ d₂ d₃ d₄ : ℕ | d₁ + d₂ + d₃ + d₄ = 18} 6 6) = 5 / 216 := 
sorry

end probability_sum_18_l268_268615


namespace Edward_money_left_l268_268897

theorem Edward_money_left {initial_amount item_cost sales_tax_rate sales_tax total_cost money_left : ℝ} 
    (h_initial : initial_amount = 18) 
    (h_item : item_cost = 16.35) 
    (h_rate : sales_tax_rate = 0.075) 
    (h_sales_tax : sales_tax = item_cost * sales_tax_rate) 
    (h_sales_tax_rounded : sales_tax = 1.23) 
    (h_total : total_cost = item_cost + sales_tax) 
    (h_money_left : money_left = initial_amount - total_cost) :
    money_left = 0.42 :=
by sorry

end Edward_money_left_l268_268897


namespace range_of_m_l268_268152

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 1/3

def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem range_of_m (m : ℝ) :
  (¬ (proposition_p m) ∧ proposition_q m) ∨ (proposition_p m ∧ ¬ (proposition_q m)) →
  (1/3 <= m ∧ m < 15) :=
sorry

end range_of_m_l268_268152


namespace point_in_second_quadrant_l268_268619

variable (m : ℝ)

-- Defining the conditions
def x_negative (m : ℝ) := 3 - m < 0
def y_positive (m : ℝ) := m - 1 > 0

theorem point_in_second_quadrant (h1 : x_negative m) (h2 : y_positive m) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l268_268619


namespace max_product_of_blackboards_l268_268360

-- Define the sets of integers allowed on the blackboards
def validNumbers := {n | 2 ≤ n ∧ n ≤ 20}

-- Function to check if two numbers are coprime
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the problem
theorem max_product_of_blackboards :
  ∃ (A B : Finset ℕ), (A ⊆ validNumbers ∧ B ⊆ validNumbers)
  ∧ (∀ a ∈ A, ∀ b ∈ B, coprime a b)
  ∧ (A.card * B.card = 65) := sorry

end max_product_of_blackboards_l268_268360


namespace sqrt_7_irrational_l268_268388

theorem sqrt_7_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a: ℝ) / b = Real.sqrt 7 := by
  sorry

end sqrt_7_irrational_l268_268388


namespace sin_225_correct_l268_268482

-- Define the condition of point being on the unit circle at 225 degrees.
noncomputable def P_225 := Complex.polar 1 (Real.pi + Real.pi / 4)

-- Define the goal statement that translates the question and correct answer.
theorem sin_225_correct : Complex.sin (Real.pi + Real.pi / 4) = -Real.sqrt 2 / 2 := 
by sorry

end sin_225_correct_l268_268482


namespace max_min_product_of_abc_l268_268671

theorem max_min_product_of_abc (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 12) (h_sum_products : a * b + b * c + c * a = 30) : 
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 9 * real.sqrt 2 := 
sorry

end max_min_product_of_abc_l268_268671


namespace area_of_rhombus_formed_by_roots_eq_2_sqrt_10_l268_268338

noncomputable def quartic_roots_area : ℂ :=
  let roots := ( solveQuartic (X^4 + 4*i*X^3 + (-5 + 5*i)*X^2 + (-10 - i)*X + (1 - 6*i)) ) in
  let rhombus_area := 2 * (sqrt 10) in
  rhombus_area
    
theorem area_of_rhombus_formed_by_roots_eq_2_sqrt_10 :
  quartic_roots_area = 2 * (sqrt 10) :=
sorry

end area_of_rhombus_formed_by_roots_eq_2_sqrt_10_l268_268338


namespace wire_cut_problem_l268_268828

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ :=
  let x := total_length / (1 + ratio)
  x

theorem wire_cut_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) :
  total_length = 35 → ratio = 5/2 → shorter_length = 10 → shorter_piece_length total_length ratio = shorter_length := by
  intros h1 h2 h3
  unfold shorter_piece_length
  rw [h1, h2, h3]
  sorry

end wire_cut_problem_l268_268828


namespace book_distribution_l268_268834

theorem book_distribution : ∃ (n : ℕ), 
  n = (cardinality (setOf (λ (x : ℕ), 2 ≤ x ∧ x ≤ 6)) ∧ n = 5 :=
by 
  sorry

end book_distribution_l268_268834


namespace find_lambda_l268_268556

open Real

def vector := (ℝ × ℝ × ℝ)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def a : vector := (-2, 1, 3)

def b : vector := (-1, 2, 1)

def orthogonal_condition (λ : ℝ) : Prop :=
  dot_product a (a.1 - λ * b.1, a.2 - λ * b.2, a.3 - λ * b.3) = 0

theorem find_lambda : ∃ λ : ℝ, orthogonal_condition λ ∧ λ = 2 :=
by
  sorry

end find_lambda_l268_268556


namespace library_books_distribution_l268_268840

theorem library_books_distribution :
  let total_books := 8
  let min_books_in_library := 2
  let min_books_checked_out := 2
  ∃ (ways : ℕ), ways = 5 :=
begin
  sorry
end

end library_books_distribution_l268_268840


namespace geometry_problem_l268_268226

-- Define the geometry within a triangle ABC where angle ACB is 90 degrees
variable {A B C D E : Type}
variable [metric_space A] [metric_space B] [metric_space C]
variable [metric_space D] [metric_space E]
variable (AC BC AD BD CD CE AE : ℝ)

-- Given conditions
variable (h1 : ∠ACB = 90)
variable (h2 : D ⊂ line(AB) ∧ D = foot(altitude(C, AB)))
variable (h3 : ∃ x ∈ (interval (B, C)), E = x ∧ CE = BD / 2)

-- The statement to be proven
theorem geometry_problem :
  AD + CE = AE :=
sorry

end geometry_problem_l268_268226


namespace area_AMN_eq_56_25_l268_268628

open_locale real -- for π
open_locale euclidean_geometry -- for Euclidean geometry

noncomputable section

-- Define the necessary geometrical points and angles
variables (A B C H M D N : Point ℝ³)

-- Main theorem
theorem area_AMN_eq_56_25 :
  ∀ (A B C H M D N : Point ℝ³), 
    dist A B = 15 ∧ ∠ BAC = π/4 ∧ ∠ BCA = π/6 ∧ 
    foot A (line3 B C) H ∧ mid_point (line3 B C) M ∧
    mid_point (line3 H M) N -> 
  (area (triangle3 A M N) = 56.25) :=
by
  intros
  {
    sorry
  }

end area_AMN_eq_56_25_l268_268628


namespace smallest_number_of_cards_theorem_l268_268668

def smallest_number_of_cards (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

theorem smallest_number_of_cards_theorem (n : ℕ) (hn : 0 < n) :
  (∀ t, t ∈ Icc 1 n! → ∃ (cards : Multiset ℕ), 
          (∀ x ∈ cards, x ∣ n!) ∧
          cards.sum = t ∧
          cards.card = smallest_number_of_cards n) :=
sorry

end smallest_number_of_cards_theorem_l268_268668


namespace total_chestnuts_weight_l268_268900

def eunsoo_kg := 2
def eunsoo_g := 600
def mingi_g := 3700

theorem total_chestnuts_weight :
  (eunsoo_kg * 1000 + eunsoo_g + mingi_g) = 6300 :=
by
  sorry

end total_chestnuts_weight_l268_268900


namespace garden_theorem_l268_268849

noncomputable def garden_problem : Prop :=
  let AB := 500
  let BC := 350
  let CD := 400
  let angle_ABC := 90
  let angle_BCD := 105
  let fourth_side_AD := 467.6
  let area_garden := 181000
  ∃ (AD : Real) (Area : Real), 
    (AD ≈ fourth_side_AD) ∧ 
    (Area ≈ area_garden)

theorem garden_theorem : garden_problem := sorry

end garden_theorem_l268_268849


namespace ellipse_other_x_intercept_l268_268065

theorem ellipse_other_x_intercept :
  ∃ x : ℚ, let f1 := (0, 3), f2 := (4, 0), p1 := (1, 0) in
  (∃ (sum_dist_to_foci : ℚ), 
     sum_dist_to_foci = real.sqrt(1^2 + 3^2) + real.sqrt((1-4)^2 + 0^2) ∧
     sum_dist_to_foci = real.sqrt(x^2 + 9) + |x - 4| ∧
     (x = 20 / 7)) := 
sorry

end ellipse_other_x_intercept_l268_268065


namespace vector_problem_l268_268161

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B C : V)
variables (λ : ℝ)

theorem vector_problem (h1 : A - B = 2 • (B - C)) (h2 : A - C = λ • (C - B)) : λ = -3 :=
sorry

end vector_problem_l268_268161


namespace behavior_at_infinity_l268_268496

noncomputable def g (x : ℝ) : ℝ := -3 * x ^ 4 + 5 * x ^ 3 - 6

theorem behavior_at_infinity :
  (filter.tendsto g filter.at_top filter.at_bot) ∧
  (filter.tendsto g filter.at_bot filter.at_bot) :=
by
  sorry

end behavior_at_infinity_l268_268496


namespace consecutive_odd_product_l268_268299

theorem consecutive_odd_product (n : ℤ) :
  (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by sorry

end consecutive_odd_product_l268_268299


namespace shape_of_fixed_phi_l268_268541

open EuclideanGeometry

def spherical_coordinates (ρ θ φ : ℝ) : Point ℝ :=
  ⟨ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ⟩

theorem shape_of_fixed_phi (c : ℝ) :
    {p : Point ℝ | ∃ ρ θ, p = spherical_coordinates ρ θ c} = cone :=
by sorry

end shape_of_fixed_phi_l268_268541


namespace product_of_center_coordinates_l268_268733

theorem product_of_center_coordinates :
  let p1 := (7, -8)
  let p2 := (-2, 3)
  let center := ( (p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2 )
  (center.1 * center.2 = -25 / 4) :=
by 
  let p1 := (7, -8)
  let p2 := (-2, 3)
  let center := ( (p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2 )
  show (center.1 * center.2 = -25 / 4)
  from sorry

end product_of_center_coordinates_l268_268733


namespace quaternary_to_decimal_l268_268094

theorem quaternary_to_decimal (q : ℕ := 10231) (base : ℕ := 4) : 
  let d := 1 + 3 * base + 2 * base^2 + 1 * base^4 
  in d = 301 :=
by {
  sorry, -- The proof is skipped
}

end quaternary_to_decimal_l268_268094


namespace probability_two_rolls_sum_9_l268_268815

/-- Given that each die has 8 sides numbered from 1 to 8, the probability of rolling 
    a sum of 9 with two dice and then rolling a sum of 9 again on a subsequent roll 
    is 1/64.
-/ 
theorem probability_two_rolls_sum_9 (die1 die2 : Fin 8) : 
  (Pr {s | die1 + die2 = 9}) ^ 2 = 1 / 64 :=
sorry

end probability_two_rolls_sum_9_l268_268815


namespace range_of_x_l268_268625

theorem range_of_x (x y : ℝ) (h : x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y)) : x ∈ set.Icc 4 20 ∪ {0} := 
sorry

end range_of_x_l268_268625


namespace triangle_area_l268_268350

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 180 :=
by
  sorry

end triangle_area_l268_268350


namespace area_of_quadrilateral_l268_268674

variables {A B C D M N : Type}

-- Assume that we have a convex quadrilateral ABCD
variable (ABCD : ConvexQuadrilateral A B C D)

-- Assume M and N are midpoints of BC and CD
variable (M_midpoint: Midpoint A B C M) (N_midpoint: Midpoint C D N)

-- Areas of quadrilaterals
variables (S_ABCD S_AMN : ℝ)

theorem area_of_quadrilateral (h : ConvexQuadrilateral A B C D) :
  S_ABCD < 4 * S_AMN :=
sorry

end area_of_quadrilateral_l268_268674


namespace primes_less_than_10000_l268_268301

noncomputable def number_of_primes (x : ℝ) : ℝ := x / Real.log x

theorem primes_less_than_10000 :
  let lge := 0.43429
  let ln10 := 1 / lge
  let ln10000 := 4 * ln10
  number_of_primes 10000 ≈ 5759 :=
by
  sorry

end primes_less_than_10000_l268_268301


namespace altitudes_iff_perimeter_relation_l268_268945

-- Given an acute-angled triangle ABC with circumradius R and points D, E, F on sides BC, CA, and AB respectively,
-- we want to prove that AD, BE, CF are the altitudes of triangle ABC if and only if the sum of the triangle's area (S) 
-- is equal to R/2 times the perimeter formed by segments EF, FD, and DE.

variables {A B C D E F : Point}
variables {R : Real}
variables (h_acute : ∀ {θ : Angle}, θ ∈ [angle A B C, angle B C A, angle C A B] → θ < π / 2)
variables (h_Rcirc : circumradius (mk_triangle A B C) = R)
variables (h_points : Collinear B C D ∧ Collinear C A E ∧ Collinear A B F)
variables (h_condition : area (mk_triangle A B C) = R / 2 * ((distance E F) + (distance F D) + (distance D E)))

theorem altitudes_iff_perimeter_relation :
  (is_altitude (mk_triangle A B C) A D ∧ is_altitude (mk_triangle A B C) B E ∧ is_altitude (mk_triangle A B C) C F) ↔
  area (mk_triangle A B C) = R / 2 * ((distance E F) + (distance F D) + (distance D E)) :=
sorry

end altitudes_iff_perimeter_relation_l268_268945


namespace inv_square_matrix_eq_l268_268926

open Matrix

variable (A : Matrix (Fin 2) (Fin 2) ℤ)

def A_inv : Matrix (Fin 2) (Fin 2) ℤ := ![![ -2, 3], ![1, -5]]

theorem inv_square_matrix_eq :
  A⁻¹ = A_inv → (A * A)⁻¹ = ![![7, -21], ![-7, 28]] :=
by
  intro h
  sorry

end inv_square_matrix_eq_l268_268926


namespace inequality_solution_reciprocal_inequality_l268_268409

-- Proof Problem (1)
theorem inequality_solution (x : ℝ) : |x-1| + (1/2)*|x-3| < 2 ↔ (1 < x ∧ x < 3) :=
sorry

-- Proof Problem (2)
theorem reciprocal_inequality (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 2) : 
  (1/a) + (1/b) + (1/c) ≥ 9/2 :=
sorry

end inequality_solution_reciprocal_inequality_l268_268409


namespace ants_collision_inevitable_l268_268000

theorem ants_collision_inevitable
    (faces : Fin 20) 
    (edges : Fin 3)
    (vertices : Fin 5)
    (length_of_side : ℝ)
    (counterclockwise : ℝ)
    (speed : ℝ)
    (meet_vertex : faces → vertices)
    (collision : vertices → Prop) :
    (length_of_side = 1) →
    (∀ ant : faces, counterclockwise = 1) →
    (speed ≥ 1) →
    (∀ {x y : faces}, x ≠ y → meet_vertex x ≠ meet_vertex y) →
    (∀ v : vertices, collision v = (∃ five_ants : Fin 5, True)) →
    ¬ ∀ t, ∀ ants : Fin 20, ¬ (collision (meet_vertex ants)) :=
by sorry

end ants_collision_inevitable_l268_268000


namespace ratio_of_areas_l268_268729

variables (s : ℝ)

def side_length_square := s
def longer_side_rect := 1.2 * s
def shorter_side_rect := 0.8 * s

noncomputable def area_rectangle := longer_side_rect s * shorter_side_rect s
noncomputable def area_triangle := (1 / 2) * (longer_side_rect s * shorter_side_rect s)

theorem ratio_of_areas :
  (area_triangle s) / (area_rectangle s) = 1 / 2 :=
by
  sorry

end ratio_of_areas_l268_268729


namespace riya_speed_l268_268706

theorem riya_speed 
  (R : ℝ)
  (priya_speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ)
  (h_priya_speed : priya_speed = 22)
  (h_time : time = 1)
  (h_distance : distance = 43)
  : R + priya_speed * time = distance → R = 21 :=
by 
  sorry

end riya_speed_l268_268706


namespace region_probability_l268_268642

theorem region_probability (k : ℝ) (k_pos : k > 0)
  (h_prob : (∫ x in 0..(k-1), k * x - (x + x^2)) / 
            (∫ x in -1..0, x + x^2 + (∫ x in 0..(k-1), k * x - (x + x^2))) = 8 / 27) :
  k = (3 + Real.sqrt 5) / 11 :=
sorry

end region_probability_l268_268642


namespace unfair_die_probability_even_sum_l268_268068

noncomputable def probability_even_sum (p_even p_odd : ℚ) : ℚ :=
  let p_all_even := p_even^3 in
  let p_two_odd_one_even := (3 : ℚ) * (p_odd^2) * p_even in
  p_all_even + p_two_odd_one_even

theorem unfair_die_probability_even_sum :
  let p_odd := (1 : ℚ) / 5 in
  let p_even := (4 : ℚ) / 5 in
  probability_even_sum p_even p_odd = (76 : ℚ) / 125 :=
by
  sorry

end unfair_die_probability_even_sum_l268_268068


namespace problem_l268_268497

theorem problem (a b : ℚ) (x : ℚ) (hx : 0 < x) :
  (a / (10^(x+1) - 1) + b / (10^(x+1) + 3) = (3 * 10^x + 4) / ((10^(x+1) - 1) * (10^(x+1) + 3))) →
  a - b = 37 / 20 :=
sorry

end problem_l268_268497


namespace functions_have_same_shape_l268_268060

def f (x : ℝ) : ℝ := 2 / x
def g (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem functions_have_same_shape (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) : f x = g x → ∀ y, y = f x ↔ y = g x :=
by 
  sorry

end functions_have_same_shape_l268_268060


namespace john_yasmin_child_ratio_l268_268268

theorem john_yasmin_child_ratio
  (gabriel_grandkids : ℕ)
  (yasmin_children : ℕ)
  (john_children : ℕ)
  (h1 : gabriel_grandkids = 6)
  (h2 : yasmin_children = 2)
  (h3 : john_children + yasmin_children = gabriel_grandkids) :
  john_children / yasmin_children = 2 :=
by 
  sorry

end john_yasmin_child_ratio_l268_268268


namespace a5_gt_b5_l268_268648

variables {a_n b_n : ℕ → ℝ}
variables {a1 b1 a3 b3 : ℝ}
variables {q : ℝ} {d : ℝ}

/-- Given conditions -/
axiom h1 : a1 = b1
axiom h2 : a1 > 0
axiom h3 : a3 = b3
axiom h4 : a3 = a1 * q^2
axiom h5 : b3 = a1 + 2 * d
axiom h6 : a1 ≠ a3

/-- Prove that a_5 is greater than b_5 -/
theorem a5_gt_b5 : a1 * q^4 > a1 + 4 * d :=
by sorry

end a5_gt_b5_l268_268648


namespace sum_of_log_difference_l268_268474

noncomputable def ceil (x : ℝ) : ℤ := (⌈x⌉ : ℤ)
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem sum_of_log_difference :
  (∑ k in Finset.range (2000 + 1) \ Finset.singleton 0, k * (ceil (Real.log k / Real.sqrt 3) - floor (Real.log k / Real.sqrt 3))) = 1999907 :=
by 
  /- Declare necessary facts about ceil and floor -/
  have ceil_floor_diff : ∀ x : ℝ, ceil x - floor x = 
    if x ∈ { x : ℝ | x ≠ (x.to_nat : ℝ) } then 1 else 0 := 
    sorry
  /- Proof continues -/

  sorry

end sum_of_log_difference_l268_268474


namespace profit_growth_satisfies_conditions_l268_268037

-- Define the conditions and the required logarithmic function.
variable {a : ℝ} (h_a : a > 1)
def profit_function (x : ℝ) := log x / log a

-- State the theorem that the logarithmic function satisfies the conditions.
theorem profit_growth_satisfies_conditions (x : ℝ) (hx : x > 0):
  (profit_function h_a x).derivative > 0 ∧ (profit_function h_a x).second_derivative < 0 :=
sorry

end profit_growth_satisfies_conditions_l268_268037


namespace number_of_triangles_l268_268976

-- Define the structure and conditions of the figure as outlined
def figure_has_bisecting_lines : Prop :=
  -- Conditions of the figure
  (∃ (rect : Type) (v_bisec h_bisec d1 d2 l1 l2 l3 l4 center_x : Type),
    bisecting_line v_bisec rect ∧
    bisecting_line h_bisec rect ∧
    diagonal_line d1 rect ∧
    diagonal_line d2 rect ∧
    diagonal_line l1 rect ∧
    diagonal_line l2 rect ∧
    diagonal_line l3 rect ∧
    diagonal_line l4 rect ∧
    intersecting_lines center_x rect)

-- Define the main theorem statement.
theorem number_of_triangles (h : figure_has_bisecting_lines) : 
  ∃ n, n = 38 :=
sorry

end number_of_triangles_l268_268976


namespace probability_of_at_least_two_white_balls_l268_268229

theorem probability_of_at_least_two_white_balls :
  let red_balls := 2 in
  let white_balls := 4 in
  let total_balls := 6 in
  let p_white := (white_balls : ℚ) / (total_balls : ℚ) in
  let p_red := (red_balls : ℚ) / (total_balls : ℚ) in
  let p_draw_at_least_two_white : ℚ :=
    ((C(3, 2) * (p_white ^ 2) * p_red) + (p_white ^ 3)) in
  p_draw_at_least_two_white = 20 / 27 :=
by
  sorry

end probability_of_at_least_two_white_balls_l268_268229


namespace find_value_of_x_l268_268398

theorem find_value_of_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := 
sorry

end find_value_of_x_l268_268398


namespace product_formation_ways_l268_268966

theorem product_formation_ways (n : ℕ) (hn : n > 0) : 
  ∃ a_n : ℕ, a_n = (2 * n - 2)! :=
by
  exists (2 * n - 2)!
  sorry

end product_formation_ways_l268_268966


namespace ellipse_eccentricity_l268_268955

theorem ellipse_eccentricity {m : ℝ} 
  (h1 : ∀ x y : ℝ, (x^2 / 16 + y^2 / m = 1)) 
  (h2 : ∀ {a b c e : ℝ}, a = 4 ∧ b = sqrt m ∧ c = sqrt (16 - m) ∧ e = c / a ∧ e = 1 / 2) 
  : m = 12 := sorry

end ellipse_eccentricity_l268_268955


namespace slope_of_line_l268_268223

theorem slope_of_line {m : ℝ} (h1: 2 * 0 - m * (1/4) + 1 = 0) :
  m = 4 ∧ ((2 : ℝ) / m = (1 / 2)) :=
by
  sorry

end slope_of_line_l268_268223


namespace total_cost_of_bicycles_is_2000_l268_268318

noncomputable def calculate_total_cost_of_bicycles (SP1 SP2 : ℝ) (profit1 profit2 : ℝ) : ℝ :=
  let C1 := SP1 / (1 + profit1)
  let C2 := SP2 / (1 - profit2)
  C1 + C2

theorem total_cost_of_bicycles_is_2000 :
  calculate_total_cost_of_bicycles 990 990 0.10 0.10 = 2000 :=
by
  -- Proof will be provided here
  sorry

end total_cost_of_bicycles_is_2000_l268_268318


namespace probability_x_gt_9y_correct_l268_268308

def rectangular_region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2023 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2024}

noncomputable def area_triangle : ℝ :=
  1 / 2 * 2023 * (2023 / 9)

noncomputable def area_rectangle : ℝ :=
  2023 * 2024

noncomputable def probability_x_gt_9y : ℝ :=
  area_triangle / area_rectangle

theorem probability_x_gt_9y_correct :
  probability_x_gt_9y = 2023 / 36432 :=
by
  unfold probability_x_gt_9y area_triangle area_rectangle
  norm_num
  sorry

end probability_x_gt_9y_correct_l268_268308


namespace locus_of_point_T_l268_268937

theorem locus_of_point_T (r : ℝ) (a b : ℝ) (x y x1 y1 x2 y2 : ℝ)
  (hM_inside : a^2 + b^2 < r^2)
  (hK_on_circle : x1^2 + y1^2 = r^2)
  (hP_on_circle : x2^2 + y2^2 = r^2)
  (h_midpoints_eq : (x + a) / 2 = (x1 + x2) / 2 ∧ (y + b) / 2 = (y1 + y2) / 2)
  (h_diagonal_eq : (x - a)^2 + (y - b)^2 = (x1 - x2)^2 + (y1 - y2)^2) :
  x^2 + y^2 = 2 * r^2 - (a^2 + b^2) :=
  sorry

end locus_of_point_T_l268_268937


namespace student_council_votes_l268_268255

theorem student_council_votes (total_students : ℕ) (percentage_voted : ℝ)
  (percentage_A percentage_B percentage_C percentage_D percentage_E : ℝ) :
  total_students = 5000 →
  percentage_voted = 0.60 →
  percentage_A = 0.40 →
  percentage_B = 0.25 →
  percentage_C = 0.20 →
  percentage_D = 0.10 →
  percentage_E = 0.05 →
  let students_voted := percentage_voted * total_students in
  let votes_A := percentage_A * students_voted in
  let votes_B := percentage_B * students_voted in
  let votes_C := percentage_C * students_voted in
  let votes_D := percentage_D * students_voted in
  let votes_E := percentage_E * students_voted in
  (votes_A - votes_B = 450) ∧ (votes_C + votes_D + votes_E = 1050) :=
by
  intros
  let students_voted := percentage_voted * total_students
  let votes_A := percentage_A * students_voted
  let votes_B := percentage_B * students_voted
  let votes_C := percentage_C * students_voted
  let votes_D := percentage_D * students_voted
  let votes_E := percentage_E * students_voted
  have h1 : votes_A - votes_B = 450 := sorry
  have h2 : votes_C + votes_D + votes_E = 1050 := sorry
  exact ⟨h1, h2⟩

end student_council_votes_l268_268255


namespace rope_length_total_l268_268369

theorem rope_length_total :
  let length1 := 24
  let length2 := 20
  let length3 := 14
  let length4 := 12
  length1 + length2 + length3 + length4 = 70 :=
by
  sorry

end rope_length_total_l268_268369


namespace P_is_centroid_of_triangle_ABC_l268_268675

variables {A B C P D E F : Point}
variables {triangle_A_triangle_B_triangle_P : Triangle}
variables {triangle_C_triangle_D_triangle_E : Triangle}
variables {triangle_F_triangle_P_triangle_E : Triangle}

-- Given conditions
axiom interior_P_in_triangle_ABC : P ∈ interior (△ A B C)
axiom AP_meet_BC_at_D : is_extension P A D ∧ lies_on_line A B C D ∧ D ∈ BC
axiom BP_meet_AC_at_E : is_extension P B E ∧ lies_on_line B A C E ∧ E ∈ AC
axiom CP_meet_AB_at_F : is_extension P C F ∧ lies_on_line C A B F ∧ F ∈ AB
axiom equal_areas : area (△ A P F) = area (△ B P D) = area (△ C P E)

-- The main statement to prove
theorem P_is_centroid_of_triangle_ABC : is_centroid P (△ A B C) :=
sorry

end P_is_centroid_of_triangle_ABC_l268_268675


namespace min_x_minus_y_l268_268139

theorem min_x_minus_y {x y : ℝ} (hx : 0 ≤ x) (hx2 : x ≤ 2 * Real.pi) (hy : 0 ≤ y) (hy2 : y ≤ 2 * Real.pi)
    (h : 2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1 / 2) : 
    x - y = -Real.pi / 2 := 
sorry

end min_x_minus_y_l268_268139


namespace convert_725_base9_to_base3_l268_268499

theorem convert_725_base9_to_base3 :
  ∃ (base9_number : ℕ), (to_base3 base9_number) = 210212 :=
begin
  -- Given base 9 number
  let base9_number := 7 * 9^2 + 2 * 9 + 5,
  
  -- Using the to_base3 function which we assume is defined,
  -- to convert base9_number to base 3 and compare it to 210212
  use base9_number,
  convert_to_base 3 base9_number = 210212,
  sorry, -- skipping the actual proof steps
end

end convert_725_base9_to_base3_l268_268499


namespace Seokjin_paper_count_l268_268661

theorem Seokjin_paper_count (Jimin_paper : ℕ) (h1 : Jimin_paper = 41) (h2 : ∀ x : ℕ, Seokjin_paper = Jimin_paper - 1) : Seokjin_paper = 40 :=
by {
  sorry
}

end Seokjin_paper_count_l268_268661


namespace sum_of_values_satisfying_l268_268009

theorem sum_of_values_satisfying (x : ℝ) (h : Real.sqrt ((x - 2) ^ 2) = 8) :
  ∃ x1 x2 : ℝ, (Real.sqrt ((x1 - 2) ^ 2) = 8) ∧ (Real.sqrt ((x2 - 2) ^ 2) = 8) ∧ x1 + x2 = 4 := 
by
  sorry

end sum_of_values_satisfying_l268_268009


namespace prob_sum_is_18_l268_268609

theorem prob_sum_is_18 : 
  let num_faces := 6
  let num_dice := 4
  let total_outcomes := num_faces ^ num_dice
  ∑ (d1 d2 d3 d4 : ℕ) in finset.Icc 1 num_faces, 
  if d1 + d2 + d3 + d4 = 18 then 1 else 0 = 35 → 
  (35 : ℚ) / total_outcomes = 35 / 648 :=
by
  sorry

end prob_sum_is_18_l268_268609


namespace total_number_of_fleas_l268_268551

theorem total_number_of_fleas :
  let G_fleas := 10
  let O_fleas := G_fleas / 2
  let M_fleas := 5 * O_fleas
  G_fleas + O_fleas + M_fleas = 40 := rfl

end total_number_of_fleas_l268_268551


namespace sqrt_eq_four_implies_x_eq_169_l268_268982

theorem sqrt_eq_four_implies_x_eq_169 (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 := by
  sorry

end sqrt_eq_four_implies_x_eq_169_l268_268982


namespace pyramid_top_number_is_9_l268_268373

theorem pyramid_top_number_is_9 (f : Fin 9 → ℕ) (s : ℕ) :
  (∀ i, f i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (Injective f) ∧
  (∀ line, ∑ i in line, f i = s) →
  f 0 = 9 :=
by
  sorry

end pyramid_top_number_is_9_l268_268373


namespace correct_calculation_l268_268805

theorem correct_calculation (a b m : ℤ) : 
  (¬((a^3)^2 = a^5)) ∧ ((-2 * m^3)^2 = 4 * m^6) ∧ (¬(a^6 / a^2 = a^3)) ∧ (¬((a + b)^2 = a^2 + b^2)) := 
by
  sorry

end correct_calculation_l268_268805


namespace kyle_jogged_during_track_practice_l268_268270

theorem kyle_jogged_during_track_practice :
  let laps_PE := 1.12
  let total_laps := 3.25
  let laps_track := total_laps - laps_PE
  in laps_track = 2.13 :=
by
  let laps_PE := 1.12
  let total_laps := 3.25
  let laps_track := total_laps - laps_PE
  have h1 : laps_track = 3.25 - 1.12 := rfl
  have h2 : laps_track = 2.13 := sorry
  show laps_track = 2.13 from h2


end kyle_jogged_during_track_practice_l268_268270


namespace sqrt_eq_four_implies_x_eq_169_l268_268984

theorem sqrt_eq_four_implies_x_eq_169 (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 := by
  sorry

end sqrt_eq_four_implies_x_eq_169_l268_268984


namespace implication_a_lt_b_implies_a_lt_b_plus_1_l268_268558

theorem implication_a_lt_b_implies_a_lt_b_plus_1 (a b : ℝ) (h : a < b) : a < b + 1 := by
  sorry

end implication_a_lt_b_implies_a_lt_b_plus_1_l268_268558


namespace train_crossing_time_l268_268459

/-!
## Problem Statement
A train 400 m in length crosses a telegraph post. The speed of the train is 90 km/h. Prove that it takes 16 seconds for the train to cross the telegraph post.
-/

-- Defining the given definitions based on the conditions in a)
def train_length : ℕ := 400
def train_speed_kmh : ℕ := 90
def train_speed_ms : ℚ := 25 -- Converting 90 km/h to 25 m/s

-- Proving the problem statement
theorem train_crossing_time : train_length / train_speed_ms = 16 := 
by
  -- convert conditions and show expected result
  sorry

end train_crossing_time_l268_268459


namespace TreyHasSevenTimesAsManyTurtles_l268_268789

variable (Kristen_turtles : ℕ)
variable (Kris_turtles : ℕ)
variable (Trey_turtles : ℕ)

-- Conditions
def KristenHas12 : Kristen_turtles = 12 := sorry
def KrisHasQuarterOfKristen : Kris_turtles = Kristen_turtles / 4 := sorry
def TreyHas9MoreThanKristen : Trey_turtles = Kristen_turtles + 9 := sorry

-- Question: Prove that Trey has 7 times as many turtles as Kris
theorem TreyHasSevenTimesAsManyTurtles :
  Kristen_turtles = 12 → 
  Kris_turtles = Kristen_turtles / 4 → 
  Trey_turtles = Kristen_turtles + 9 → 
  Trey_turtles = 7 * Kris_turtles := sorry

end TreyHasSevenTimesAsManyTurtles_l268_268789


namespace median_of_combined_list_l268_268800

def list1 := List.range 3030  -- generates the list [1, 2, ..., 3030]
def list2 := (List.range 3030).map (fun n => (n + 1)^2)  -- generates the list [1^2, 2^2, ..., 3030^2]
def combined_list := list1 ++ list2  -- concatenate both lists

theorem median_of_combined_list : 
  let combined_sorted := combined_list.qsort (<=)  -- sort the combined list
  let n := combined_sorted.length
  n = 6060 →
  (combined_sorted 3029 + combined_sorted 3030) / 2 = 2975.5 :=
by
  sorry

end median_of_combined_list_l268_268800


namespace solve_for_x_l268_268980

theorem solve_for_x (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
sorry

end solve_for_x_l268_268980


namespace bus_speed_excluding_stoppages_l268_268517

theorem bus_speed_excluding_stoppages : 
  let V := λ (minutes_active hours_per_hour total_distance total_minutes_stopped) => total_distance * hours_per_hour / (minutes_active / (hours_per_hour - total_minutes_stopped)) in
  V 56 60 84 4 = 90 :=
by
  let V := λ (minutes_active hours_per_hour total_distance total_minutes_stopped) => total_distance * hours_per_hour / (minutes_active / (hours_per_hour - total_minutes_stopped))
  show V 56 60 84 4 = 90
  sorry

end bus_speed_excluding_stoppages_l268_268517


namespace winning_candidate_percentage_l268_268030

theorem winning_candidate_percentage :
  let votes_candidate1 := 3136 in
  let votes_candidate2 := 7636 in
  let votes_candidate3 := 11628 in
  let total_votes := votes_candidate1 + votes_candidate2 + votes_candidate3 in
  let winning_votes := max votes_candidate1 (max votes_candidate2 votes_candidate3) in
  (winning_votes / total_votes : ℚ) * 100 ≈ 51.93 := by
sorry

end winning_candidate_percentage_l268_268030


namespace total_cost_l268_268295

-- Define the cost variables
variables (t p r c : ℕ)

-- Define the conditions
def condition1 : Prop := t + c + r = 47
def condition2 : Prop := t + r + p = 58
def condition3 : Prop := p + c = 15

-- Prove the total cost of the entire set
theorem total_cost (h1 : condition1 t p r c) (h2 : condition2 t p r c) (h3 : condition3 t p r c) : 
  t + r + p + c = 60 :=
by
  sorry

end total_cost_l268_268295


namespace division_result_l268_268384

theorem division_result (x : ℕ) (h : x + 8 = 88) : x / 10 = 8 := by
  sorry

end division_result_l268_268384


namespace team_a_scored_points_l268_268810

theorem team_a_scored_points : ∃ (points_per_person : ℕ) (num_people_playing : ℕ), points_per_person = 2 ∧ num_people_playing = 9 ∧ points_per_person * num_people_playing = 18 :=
by {
  use 2,
  use 9,
  repeat { split },
  exact rfl,
  exact rfl,
  norm_num,
}

end team_a_scored_points_l268_268810


namespace gain_percent_is_sixty_l268_268039

-- Definitions based on the conditions
def costPrice : ℝ := 675
def sellingPrice : ℝ := 1080
def gain : ℝ := sellingPrice - costPrice
def gainPercent : ℝ := (gain / costPrice) * 100

-- Proof statement
theorem gain_percent_is_sixty (h1 : costPrice = 675) (h2 : sellingPrice = 1080) :
  gainPercent = 60 :=
by
  rw [h1, h2]
  -- Additional steps to prove the equality can be abstracted here
  sorry

end gain_percent_is_sixty_l268_268039


namespace false_proposition_is_B_l268_268863

theorem false_proposition_is_B :
  (¬ (∃ x0 ∈ ℝ, sin x0 + cos x0 = 2)) :=
begin
  -- Since √2 * sin(x + π/4) has range [-√2, √2], it cannot be 2.
  sorry
end

end false_proposition_is_B_l268_268863


namespace price_jemma_sells_each_frame_is_5_l268_268099

noncomputable def jemma_price_per_frame : ℝ :=
  let num_frames_jemma := 400
  let num_frames_dorothy := num_frames_jemma / 2
  let total_income := 2500
  let P_jemma := total_income / (num_frames_jemma + num_frames_dorothy / 2)
  P_jemma

theorem price_jemma_sells_each_frame_is_5 :
  jemma_price_per_frame = 5 := by
  sorry

end price_jemma_sells_each_frame_is_5_l268_268099


namespace y_intercept_of_line_l268_268526

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 0) : y = 4 :=
by
  -- The proof goes here
  sorry

end y_intercept_of_line_l268_268526


namespace reciprocal_of_neg_2023_l268_268747

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l268_268747


namespace expected_value_matches_variance_matches_l268_268428

variables {N : ℕ} (I : Fin N → Bool)

-- Define the probability that a randomly chosen pair of cards matches
def p_match : ℝ := 1 / N

-- Define the indicator variable I_k
def I_k (k : Fin N) : ℝ :=
if I k then 1 else 0

-- Define the sum S of all the indicator variables
def S : ℝ := (Finset.univ.sum I_k)

-- Expected value E[I_k] is 1/N
def E_I_k : ℝ := 1 / N

-- Expected value E[S] is the sum of E[I_k] over all k, which is 1
theorem expected_value_matches : ∑ k, E_I_k = 1 := sorry

-- Variance calculation: Var[S] = E[S^2] - (E[S])^2
def E_S_sq : ℝ := (Finset.univ.sum (λ k, I_k k * I_k k)) + 
                  2 * (Finset.univ.sum (λ (jk : Fin N × Fin N), if jk.1 < jk.2 then I_k jk.1 * I_k jk.2 else 0))

theorem variance_matches : (E_S_sq - 1) = 1 := sorry

end expected_value_matches_variance_matches_l268_268428


namespace sqrt_seven_irrational_l268_268389

theorem sqrt_seven_irrational : irrational (Real.sqrt 7) :=
sorry

end sqrt_seven_irrational_l268_268389


namespace tangent_slope_probability_l268_268652

theorem tangent_slope_probability :
  let I := set.Icc (-6 : ℝ) 6,
      I_valid_slope := {x : ℝ | (x >= 1/2) ∨ (x <= -1/2)},
      intersection := set.union (set.Icc (-6) (-1/2)) (set.Icc (1/2) 6),
      interval_length := 12,
      valid_length := 11 in
  ∀ {x0 : ℝ}, x0 ∈ I →
    (dthetan (2 * x0) ∈ set.Icc (real.pi / 4) (3 * real.pi / 4)) →
    (finset.card (finset.filter (λ x0, x0 ∈ intersection) (finset.Icc (-6) 6))) / interval_length = 11/12 :=
sorry

end tangent_slope_probability_l268_268652


namespace large_spoons_count_l268_268687

-- Define the conditions

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def teaspoons := 15
def total_spoons := 39

-- Define the problem statement
theorem large_spoons_count :
  let children_spoons := num_children * spoons_per_child in
  let known_spoons := children_spoons + decorative_spoons + teaspoons in
  total_spoons - known_spoons = 10 :=
by
  sorry

end large_spoons_count_l268_268687


namespace probability_AB_together_l268_268007

theorem probability_AB_together :
  let total_permutations := Nat.factorial 4,
      ab_unit_permutations := Nat.factorial 3,
      internal_ab_permutations := Nat.factorial 2,
      favorable_outcomes := internal_ab_permutations * ab_unit_permutations,
      probability := favorable_outcomes / total_permutations
  in
  probability = 1 / 2 :=
by
  let total_permutations := Nat.factorial 4
  let ab_unit_permutations := Nat.factorial 3
  let internal_ab_permutations := Nat.factorial 2
  let favorable_outcomes := internal_ab_permutations * ab_unit_permutations
  let probability := favorable_outcomes / total_permutations
  sorry

end probability_AB_together_l268_268007


namespace point_A_equidistant_l268_268527

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2)

theorem point_A_equidistant (A B C : ℝ × ℝ × ℝ) (hB : B = (1, 2, 3)) (hC : C = (2, 6, 10)) :
  (∃ x : ℝ, A = (x, 0, 0) ∧ distance A B = distance A C) → A = (63, 0, 0) :=
by 
  sorry

end point_A_equidistant_l268_268527


namespace sum_of_exponents_of_1025_in_base_3_l268_268361

theorem sum_of_exponents_of_1025_in_base_3 :
  ∃ (r : ℕ) (n : Fin r → ℕ) (a : Fin r → ℤ),
    (∀ i j, i < j → n i > n j) ∧
    (∀ k, a k = 1 ∨ a k = -1) ∧
    (∑ i, a i * 3 ^ (n i) = 1025) →
    (∑ i, n i = 17) := by
  sorry

end sum_of_exponents_of_1025_in_base_3_l268_268361


namespace MN_coordinates_l268_268922

-- Define the given vectors OM and ON.
def OM : ℝ × ℝ := (3, -2)
def ON : ℝ × ℝ := (-5, -1)

-- Define the vector subtraction function for 2D vectors.
def vector_sub (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 - v₂.1, v₁.2 - v₂.2)

-- State the problem as a theorem.
theorem MN_coordinates : vector_sub ON OM = (-8, 1) :=
by
  simp only [OM, ON, vector_sub]
  exact rfl

end MN_coordinates_l268_268922


namespace trig_expr_value_l268_268122

theorem trig_expr_value :
  (Real.cos (7 * Real.pi / 24)) ^ 4 +
  (Real.sin (11 * Real.pi / 24)) ^ 4 +
  (Real.sin (17 * Real.pi / 24)) ^ 4 +
  (Real.cos (13 * Real.pi / 24)) ^ 4 = 3 / 2 :=
by
  sorry

end trig_expr_value_l268_268122


namespace num_vertical_asymptotes_one_l268_268531

def vertical_asymptotes (f : ℝ → ℝ) : set ℝ := {x | ∃ ε > 0, ∀ δ > 0, ∃ y ∈ set.Ioo (x - δ) (x + δ), abs (f y) > ε}

theorem num_vertical_asymptotes_one (f : ℝ → ℝ) (h : ∀ x, f x = (x + 2) / (x ^ 2 - 4)) :
  ∃! x, x ∈ vertical_asymptotes f := sorry

end num_vertical_asymptotes_one_l268_268531


namespace least_number_conditioned_l268_268117

theorem least_number_conditioned (n : ℕ) :
  n % 56 = 3 ∧ n % 78 = 3 ∧ n % 9 = 0 ↔ n = 2187 := 
sorry

end least_number_conditioned_l268_268117


namespace find_f_f_5_div_2_l268_268960

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 1 else -x + 3

theorem find_f_f_5_div_2 : f (f (5 / 2)) = 3 / 2 :=
by
  -- Proof skipped
  sorry

end find_f_f_5_div_2_l268_268960


namespace achievable_with_at_most_17_presses_l268_268251

open Finset

-- Define a type to represent positions in the grid
inductive Pos : Type
| mk : Fin 5 → Fin 5 → Pos

-- Next, define an instance for decidable equality on our Pos type.
instance : DecidableEq Pos := by
  intros x y
  cases x; cases y
  apply dite (a = a_1 ∧ b = b_1) (λ h => isTrue h) (λ h => isFalse h.1)

-- Define the toggling function and its neighbors
def toggle (pos : Pos) : Pos → Bool := sorry

-- Main theorem statement
theorem achievable_with_at_most_17_presses (target_pattern : Pos → Bool) :
  (∀ pos, target_pattern pos = false) →
  ∃ press_sequence : Fin 17 → Pos,
    (Π i, toggle (press_sequence i)) = target_pattern := sorry

end achievable_with_at_most_17_presses_l268_268251


namespace characteristics_transformed_curve_eq_l268_268149

-- Define the matrix A
def A (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 0], ![1, k]]

-- Define the characteristic vector alpha
def alpha : Fin 2 → ℝ := ![-1, 1]

-- Define the transformation of vector alpha by matrix A
def transform_alpha (k : ℝ) : Fin 2 → ℝ := λ i, (A k).mulVec alpha i

-- Define the transformed point equations based on transformation of curve C by matrix A
def transformed_point (x y : ℝ) (k : ℝ) : Fin 2 → ℝ :=
  match (A k).mulVec ![x, y] with
  | M => λ i, M i

-- Lean 4 statement to show k = 3 and λ = 2
theorem characteristics (k λ : ℝ) (hλ : ∀ i, transform_alpha k i = λ * alpha i) :
  λ = 2 ∧ k = 3 :=
begin
  sorry
end

-- Lean 4 statement to prove the equation of curve C
theorem transformed_curve_eq (x y : ℝ) (k : ℝ) (h : k = 3) :
  (let x' := 2 * x in
   let y' := x + 3 * y in
   x'^2 + y'^2 = 2 → 5 * x^2 + 6 * x * y + 9 * y^2 = 2) :=
begin
  sorry
end

end characteristics_transformed_curve_eq_l268_268149


namespace pie_price_l268_268509

theorem pie_price (cakes_sold : ℕ) (cake_price : ℕ) (cakes_total_earnings : ℕ)
                  (pies_sold : ℕ) (total_earnings : ℕ) (price_per_pie : ℕ)
                  (H1 : cakes_sold = 453)
                  (H2 : cake_price = 12)
                  (H3 : pies_sold = 126)
                  (H4 : total_earnings = 6318)
                  (H5 : cakes_total_earnings = cakes_sold * cake_price)
                  (H6 : price_per_pie * pies_sold = total_earnings - cakes_total_earnings) :
    price_per_pie = 7 := by
    sorry

end pie_price_l268_268509


namespace soccer_camp_ratio_l268_268784

theorem soccer_camp_ratio :
  let total_kids := 2000
  let half_total := total_kids / 2
  let afternoon_camp := 750
  let morning_camp := half_total - afternoon_camp
  half_total ≠ 0 → 
  (morning_camp / half_total) = 1 / 4 := by
  sorry

end soccer_camp_ratio_l268_268784


namespace gcd_97_power_l268_268884

theorem gcd_97_power (h : Nat.Prime 97) : 
  Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := 
by 
  sorry

end gcd_97_power_l268_268884


namespace absolute_value_one_minus_sqrt_three_l268_268353

theorem absolute_value_one_minus_sqrt_three : |1 - real.sqrt 3| = real.sqrt 3 - 1 := by
  have h : real.sqrt 3 > 1 := sorry
  sorry

end absolute_value_one_minus_sqrt_three_l268_268353


namespace find_number_l268_268520

theorem find_number (x : ℝ) (h : x^2 + 50 = (x - 10)^2) : x = 2.5 :=
sorry

end find_number_l268_268520


namespace min_value_of_function_l268_268529

noncomputable def y (x : ℝ) : ℝ := (Real.cos x) * (Real.sin (2 * x))

theorem min_value_of_function :
  ∃ x ∈ Set.Icc (-Real.pi) Real.pi, y x = -4 * Real.sqrt 3 / 9 :=
sorry

end min_value_of_function_l268_268529


namespace simplify_expression_l268_268376

theorem simplify_expression :
  (3 ^ 2 * 3 ^ (-5)) / (3 ^ 4 * 3 ^ (-3)) = (1 / 81) := by
  sorry

end simplify_expression_l268_268376


namespace distance_traveled_is_correct_l268_268831

-- Given conditions
def still_water_speed : ℝ := 15   -- Speed of boat in still water (km/hr)
def current_speed : ℝ := 3        -- Rate of the current (km/hr)
def wind_increase : ℝ := 0.05     -- Wind increase in speed (5%)
def travel_time_minutes : ℝ := 24 -- Travel time downstream (minutes)
def km_to_miles : ℝ := 0.621371   -- Conversion factor (1 km = 0.621371 miles)

-- Derived distances
def travel_time_hours : ℝ := travel_time_minutes / 60
def effective_speed_downstream : ℝ := (still_water_speed + current_speed) * (1 + wind_increase)
def distance_km : ℝ := effective_speed_downstream * travel_time_hours
def distance_miles : ℝ := distance_km * km_to_miles

-- Statement to be proved
theorem distance_traveled_is_correct :
  distance_miles = 4.699 :=
by
  sorry

end distance_traveled_is_correct_l268_268831


namespace inequality_abc_l268_268280

variables {a b c : ℝ}

theorem inequality_abc 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2) ∧ 
    (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := 
by
  sorry

end inequality_abc_l268_268280


namespace max_two_digit_times_max_one_digit_is_three_digit_l268_268341

def max_two_digit : ℕ := 99
def max_one_digit : ℕ := 9
def product := max_two_digit * max_one_digit

theorem max_two_digit_times_max_one_digit_is_three_digit :
  100 ≤ product ∧ product < 1000 :=
by
  -- Prove that the product is a three-digit number
  sorry

end max_two_digit_times_max_one_digit_is_three_digit_l268_268341


namespace min_value_of_A_l268_268819

noncomputable def A (x y : ℝ) : ℝ := 
  (3 * x * y + x^2) * sqrt (3 * x * y + x - 3 * y) + (3 * x * y + y^2) * sqrt (3 * x * y + y - 3 * x) / (x^2 * y + y^2 * x)

theorem min_value_of_A (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 3) : 
  4 ≤ A x y :=
sorry

end min_value_of_A_l268_268819


namespace range_of_a_l268_268291

variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Define the conditions given:
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def monotonic_increasing_on_nonnegative_reals (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2) → (f x1 < f x2)

def inequality_in_interval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, (1 / 2 ≤ x) → (x ≤ 1) → (f (a * x + 1) ≤ f (x - 2))

-- The theorem we want to prove
theorem range_of_a (h1 : even_function f)
                   (h2 : monotonic_increasing_on_nonnegative_reals f)
                   (h3 : inequality_in_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l268_268291


namespace arithmetic_sequence_proof_l268_268575

theorem arithmetic_sequence_proof (a : ℕ → ℕ) (d : ℕ) (λ : ℝ) :
  (∀ n : ℕ, n > 0 → 2 * a (n + 1) = λ * a n + 4) ∧ a 1 = 1 ∧ d ≠ 0 →
  λ = 2 ∧ (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∑ i in finset.range n, a (2 ^ i - i)) = 2 ^ (n + 2) - n^2 - 2*n - 4 :=
by
  sorry

end arithmetic_sequence_proof_l268_268575


namespace find_trapezoid_angles_l268_268067

-- Define the problem conditions
variables (α h R : ℝ)

-- The condition: ratio of height to the radius of the circumscribed circle
def height_to_radius_ratio (h R : ℝ) : Prop := h = R * Real.sqrt (2 / 3)

-- Conclusion about the angle α (α is 45 degrees)
def angle_is_45_degree (α : ℝ) : Prop := α = π / 4

-- Conclusion about angles of the trapezoid
def trapezoid_angles (α : ℝ) : Prop := angle_is_45_degree α ∧ angle_is_45_degree (π - α)

-- Lean 4 theorem statement to prove the angles of the trapezoid
theorem find_trapezoid_angles :
  ∀ (h R : ℝ), height_to_radius_ratio h R → trapezoid_angles α :=
by
  sorry

end find_trapezoid_angles_l268_268067


namespace solve_for_x_l268_268978

theorem solve_for_x (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
sorry

end solve_for_x_l268_268978


namespace length_of_chord_EF_l268_268244

theorem length_of_chord_EF 
  (rO rN rP : ℝ)
  (AB BC CD : ℝ)
  (AG_EF_intersec_E AG_EF_intersec_F : ℝ)
  (EF : ℝ)
  (cond1 : rO = 10)
  (cond2 : rN = 20)
  (cond3 : rP = 30)
  (cond4 : AB = 2 * rO)
  (cond5 : BC = 2 * rN)
  (cond6 : CD = 2 * rP)
  (cond7 : EF = 6 * Real.sqrt (24 + 2/3)) :
  EF = 6 * Real.sqrt 24.6666 := sorry

end length_of_chord_EF_l268_268244


namespace intercept_of_perpendicular_line_l268_268116

theorem intercept_of_perpendicular_line (C : ℝ) (h_area : (1/2) * (abs (C/4)) * (abs (-C/3)) = 6) :
  let x_intercept := -C / 3 in
  x_intercept = 4 ∨ x_intercept = -4 :=
by
  sorry

end intercept_of_perpendicular_line_l268_268116


namespace prob_Bryce_score_at_most_2_l268_268077

noncomputable def prob_score_at_most_2 (n : ℕ) : ℚ :=
  let total_pairs := (n * 2).choose 2
  let same_color_pairs := n.choose 2 * 2
  let diff_color_pairs := n * n
  let prob_same_color := (same_color_pairs : ℚ) / total_pairs
  let prob_diff_color := (diff_color_pairs : ℚ) / total_pairs
  (prob_diff_color^n + n * prob_same_color * prob_diff_color^(n-1) + 
    (n.choose 2) * prob_same_color^2 * prob_diff_color^(n-2))

theorem prob_Bryce_score_at_most_2 : ∀ n = 7, 
  let p := 184
  let q := 429
  p + q = 613 → 
  prob_score_at_most_2 n = 184 / 429 :=
by
  intro n
  intro h
  sorry

end prob_Bryce_score_at_most_2_l268_268077


namespace operation_result_l268_268891

def operation (a b : Int) : Int :=
  (a + b) * (a - b)

theorem operation_result :
  operation 4 (operation 2 (-1)) = 7 :=
by
  sorry

end operation_result_l268_268891


namespace imaginary_part_zi_l268_268643

theorem imaginary_part_zi (x y : ℝ) (h : x = -2 ∧ y = 1) : 
  (λ (z : ℂ), (z * complex.I).im) (complex.mk x y) = -2 :=
by 
  sorry

end imaginary_part_zi_l268_268643


namespace partition_contains_all_distances_l268_268310

open Set
open Real

theorem partition_contains_all_distances (P1 P2 P3 : set ℝ^3) (hP1P2P3 : ∀ x, x ∈ P1 ∨ x ∈ P2 ∨ x ∈ P3)
  (hDisjoint : ∀ x, ¬(x ∈ P1 ∧ x ∈ P2 ∧ x ∈ P3)) :
  ∃ (i : {1, 2, 3}), ∀ a ∈ ℝ, ∃ (M N : ℝ^3), M ∈ [if i = 1 then P1 else if i = 2 then P2 else P3] ∧ N ∈ [if i = 1 then P1 else if i = 2 then P2 else P3] ∧ dist M N = a :=
sorry

end partition_contains_all_distances_l268_268310


namespace min_moves_to_find_treasure_l268_268700

theorem min_moves_to_find_treasure (cells : List ℕ) (h1 : cells = [5, 5, 5]) : 
  ∃ n, n = 2 ∧ (∀ moves, moves ≥ n → true) := sorry

end min_moves_to_find_treasure_l268_268700


namespace mindy_emails_l268_268695

theorem mindy_emails (P E : ℕ) 
    (h1 : E = 9 * P - 7)
    (h2 : E + P = 93) :
    E = 83 := 
    sorry

end mindy_emails_l268_268695


namespace area_of_gray_part_l268_268001

theorem area_of_gray_part (A1 A2 A_black A_white A_grey : ℕ) (hA1 : A1 = 8 * 10) (hA2 : A2 = 12 * 9) (hA_black : A_black = 37) (hA_white : A_white = A1 - A_black) (hA_grey : A_grey = A2 - A_white) : A_grey = 65 :=
by {
  -- Using the provided hypotheses to prove the theorem
  rw [hA1, hA2, hA_black] at *,
  have h1 : A_white = 80 - 37 := hA_white,
  have h2 : A_grey = 108 - A_white := hA_grey,
  rw h1 at h2,
  have h3 : A_grey = 108 - (80 - 37) := h2,
  have h4 : A_grey = 108 - 43 := h3,
  have h5 : A_grey = 65 := h4,
  exact h5
}

end area_of_gray_part_l268_268001


namespace a_eq_1_sufficient_not_necessary_l268_268821

theorem a_eq_1_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → |x - 1| ≤ |x - a|) ∧ ¬(∀ x : ℝ, x ≤ 1 → |x - 1| = |x - a|) :=
by
  sorry

end a_eq_1_sufficient_not_necessary_l268_268821


namespace find_58th_sum_l268_268358

noncomputable def set_of_numbers : Set ℕ := {1, 3, 9, 27, 81, 243}

def unique_sums (s : Set ℕ) : Set ℕ := 
  {sum | ∃ (t : Finset ℕ), (↑t ⊆ s) ∧ (sum = t.sum id)}

def ordered_sums (s : Set ℕ) : List ℕ :=
  (unique_sums s).to_list.sorted (<)

theorem find_58th_sum : (ordered_sums set_of_numbers).nth 57 = some 354 :=
by
  -- The proof follows that 354 is indeed the 58th element when the sums are ordered.
  sorry

end find_58th_sum_l268_268358


namespace time_fraction_l268_268266

variable (t₅ t₁₅ : ℝ)

def total_distance (t₅ t₁₅ : ℝ) : ℝ :=
  5 * t₅ + 15 * t₁₅

def total_time (t₅ t₁₅ : ℝ) : ℝ :=
  t₅ + t₁₅

def average_speed_eq (t₅ t₁₅ : ℝ) : Prop :=
  10 * (t₅ + t₁₅) = 5 * t₅ + 15 * t₁₅

theorem time_fraction (t₅ t₁₅ : ℝ) (h : average_speed_eq t₅ t₁₅) :
  (t₁₅ / (t₅ + t₁₅)) = 1 / 2 := by
  sorry

end time_fraction_l268_268266


namespace eccentricity_of_ellipse_l268_268573

-- Define the condition of the ellipse's equation
def ellipse_equation (x y : ℝ) (k : ℝ) : Prop := 3 * x^2 + k * y^2 = 1

-- Define eccentricity of an ellipse given a^2 and b^2
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (a^2 / b^2))

-- Define one focus is at (0, 1)
def focus_condition (b a : ℝ) : Prop := Real.sqrt (b^2 - a^2) = 1

-- Problem statement: Given the conditions, prove that the eccentricity equals sqrt(2)/2
theorem eccentricity_of_ellipse : 
  ∃ (k : ℝ), 
  (∀ x y, ellipse_equation x y k) ∧
  (∃ a b : ℝ, 3 * a^2 = 1 ∧ k * b^2 = 1 ∧ focus_condition b a ∧ eccentricity a b = Real.sqrt 2 / 2) :=
by
  sorry

end eccentricity_of_ellipse_l268_268573


namespace sarahs_total_problems_l268_268323

def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def science_pages : ℕ := 5
def math_problems_per_page : ℕ := 4
def reading_problems_per_page : ℕ := 4
def science_problems_per_page : ℕ := 6

def total_math_problems : ℕ := math_pages * math_problems_per_page
def total_reading_problems : ℕ := reading_pages * reading_problems_per_page
def total_science_problems : ℕ := science_pages * science_problems_per_page

def total_problems : ℕ := total_math_problems + total_reading_problems + total_science_problems

theorem sarahs_total_problems :
  total_problems = 70 :=
by
  -- proof will be inserted here
  sorry

end sarahs_total_problems_l268_268323


namespace arithmetic_series_sum_base6_l268_268905

-- Define the terms in the arithmetic series in base 6
def a₁ := 1
def a₄₅ := 45
def n := a₄₅

-- Sum of arithmetic series in base 6
def sum_arithmetic_series := (n * (a₁ + a₄₅)) / 2

-- Expected result for the arithmetic series sum
def expected_result := 2003

theorem arithmetic_series_sum_base6 :
  sum_arithmetic_series = expected_result := by
  sorry

end arithmetic_series_sum_base6_l268_268905


namespace sin_225_eq_neg_sqrt2_div_2_l268_268485

noncomputable def sin_225_deg := real.sin (225 * real.pi / 180)

theorem sin_225_eq_neg_sqrt2_div_2 : sin_225_deg = -real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt2_div_2_l268_268485


namespace find_x2017_l268_268574

noncomputable def f (x : ℝ) := sorry  -- Assume f is some odd, increasing function defined on ℝ

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

def sequence (n : ℕ) : ℝ := -1 + 2 * (n - 7)

theorem find_x2017 (f : ℝ → ℝ) (h_odd : is_odd_function f) 
    (h_inc : is_increasing_function f) (h_seq : ∀ n, sequence 7 + 2 * (8 - n) = sequence n) 
    (h_condition : f (sequence 7) + f (sequence 8) = 0) :
  sequence 2017 = 4019 :=
by
  -- Proof goes here
  sorry

end find_x2017_l268_268574


namespace find_a_l268_268177

theorem find_a (f : ℝ → ℝ) (a : ℝ) (h₀ : ∀ x, f(x + 3) = 3 * f(x))
  (h₁ : ∀ x, 0 < x → x < 3 → f(x) = Real.log x - a * x)
  (h₂ : ∀ x, -6 < x → x < -3 → f(x) ≤ -1 / 9) :
  a = 1 :=
sorry

end find_a_l268_268177


namespace dihedral_angle_correct_l268_268941

noncomputable def dihedral_angle_C_FG_E 
  (A B C D E F G: Point)
  (h_tetrahedron: regular_tetrahedron A B C D)
  (h_E: midpoint E A B)
  (h_F: midpoint F B C)
  (h_G: midpoint G C D) : Real :=
  pi - arccot (sqrt 2 / 2)

-- We need to declare a theorem to assert the required proof.
theorem dihedral_angle_correct:
  ∀ (A B C D E F G: Point),
  regular_tetrahedron A B C D →
  midpoint E A B →
  midpoint F B C →
  midpoint G C D →
  dihedral_angle_C_FG_E A B C D E F G 
  = pi - arccot (sqrt 2 / 2) :=
sorry

end dihedral_angle_correct_l268_268941


namespace parking_lot_cars_l268_268363

theorem parking_lot_cars :
  ∀ (initial_cars cars_left cars_entered remaining_cars final_cars : ℕ),
    initial_cars = 80 →
    cars_left = 13 →
    remaining_cars = initial_cars - cars_left →
    cars_entered = cars_left + 5 →
    final_cars = remaining_cars + cars_entered →
    final_cars = 85 := 
by
  intros initial_cars cars_left cars_entered remaining_cars final_cars h1 h2 h3 h4 h5
  sorry

end parking_lot_cars_l268_268363


namespace solution_set_of_inequality_l268_268779

theorem solution_set_of_inequality : 
  {x : ℝ | |x - 2| - |2 * x - 1| > 0} = set.Ioo (-1 : ℝ) 1 :=
by
  sorry

end solution_set_of_inequality_l268_268779


namespace cara_between_pairs_l268_268879

-- Definitions based on the conditions
def friends := 7 -- Cara has 7 friends
def fixed_neighbor : Prop := true -- Alex must always be one of the neighbors

-- Problem statement to be proven
theorem cara_between_pairs (h : fixed_neighbor): 
  ∃ n : ℕ, n = 6 ∧ (1 + (friends - 1)) = n := by
  sorry

end cara_between_pairs_l268_268879


namespace problem_one_problem_two_l268_268288

-- Define p and q
def p (a x : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := |x - 3| < 1

-- Problem (1)
theorem problem_one (a : ℝ) (h_a : a = 1) (h_pq : p a x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem_two (a : ℝ) (h_a_pos : a > 0) (suff : ¬ p a x → ¬ q x) (not_necess : ¬ (¬ q x → ¬ p a x)) : 
  (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

end problem_one_problem_two_l268_268288


namespace sqrt_eq_4_implies_x_eq_169_l268_268988

-- Statement of the problem
theorem sqrt_eq_4_implies_x_eq_169 (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
begin
  sorry  -- proof not required
end

end sqrt_eq_4_implies_x_eq_169_l268_268988


namespace sequence_recursion_x1001_minus_x401_l268_268349

theorem sequence_recursion_x1001_minus_x401 :
  let x : ℕ → ℝ := λ n, (recfun (λ (x : ℝ) n, (x + (2 - sqrt 3)) / (1 - x * (2 - sqrt 3))) 1) n
  x 1001 - x 401 = 0 :=
by
  -- Define the sequence
  let seq : ℕ → ℝ := λ n, recfun (λ (x : ℝ) n, (x + (2 - sqrt 3)) / (1 - x * (2 - sqrt 3))) 1 n
  -- Apply modulo to compute equivalent elements
  have h1 : seq 1001 = seq 401 := sorry
  -- Conclude the proof with the required assertion
  exact h1.symm ▸ sub_self (seq 401)

end sequence_recursion_x1001_minus_x401_l268_268349


namespace simple_annual_interest_rate_l268_268467

theorem simple_annual_interest_rate (interest_per_month : ℕ) (investment_amount : ℕ) (months_per_year : ℕ) (time_in_years : ℕ) :
  interest_per_month = 228 ->
  investment_amount = 30400 ->
  months_per_year = 12 ->
  time_in_years = 1 ->
  (interest_per_month * months_per_year : ℝ) / investment_amount = 0.09 :=
begin
  intros h1 h2 h3 h4,
  sorry,
end

end simple_annual_interest_rate_l268_268467


namespace A_subset_A_bar_l268_268918

namespace ProofExample

open Set

variables {α : Type*} (Aₙ : ℕ → Set α)

def A : Set α := ⋃ m, ⋂ n ≥ m, Aₙ n
def A_bar : Set α := ⋂ m, ⋃ n ≥ m, Aₙ n

theorem A_subset_A_bar : A Aₙ ⊆ A_bar Aₙ := by
  sorry

end ProofExample

end A_subset_A_bar_l268_268918


namespace how_many_pounds_of_raisins_l268_268472

theorem how_many_pounds_of_raisins
  (r : ℝ) (n : ℝ) (x : ℝ) (h1 : n = 3 * r) 
  (h2 : x * r = (1 / 4) * (x * r + 4 * n)) : 
  x = 4 :=
begin
  sorry,
end

end how_many_pounds_of_raisins_l268_268472


namespace largest_prime_divisor_of_sum_of_squares_l268_268910

def largest_prime_divisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_sum_of_squares :
  largest_prime_divisor (11^2 + 90^2) = 89 :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l268_268910


namespace problem_statement_l268_268176

noncomputable def f (x : ℝ) : ℝ := log x / log 3 + x - 5

theorem problem_statement
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_diff : b - a = 1)
  (h_root : ∃ x0 : ℝ, x0 ∈ set.Icc a b ∧ f x0 = 0) :
  a + b = 7 :=
by
  sorry

end problem_statement_l268_268176


namespace square_area_difference_l268_268798

theorem square_area_difference (side_length : ℕ) (increase : ℕ) (h₁ : side_length = 6) (h₂ : increase = 1) : 
  let original_area := side_length * side_length,
      new_side_length := side_length + increase,
      new_area := new_side_length * new_side_length,
      area_difference := new_area - original_area
  in area_difference = 13 := by
  sorry

end square_area_difference_l268_268798


namespace shape_is_cone_l268_268542

-- Definition: spherical coordinates and constant phi
def spherical_coords (ρ θ φ : ℝ) : Type := ℝ × ℝ × ℝ
def phi_constant (c : ℝ) (φ : ℝ) : Prop := φ = c

-- Theorem: shape described by φ = c in spherical coordinates is a cone
theorem shape_is_cone (ρ θ c : ℝ) (h₁ : c ∈ set.Icc 0 real.pi) : 
  (∃ (ρ θ : ℝ), spherical_coords ρ θ c = (ρ, θ, c)) → 
  (∀ φ, phi_constant c φ) → 
  shape_is_cone := sorry

end shape_is_cone_l268_268542


namespace min_value_proof_l268_268968

noncomputable def find_min_value 
  (a b c : ℝ × ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 2) 
  (hab : a.1 * b.1 + a.2 * b.2 = 2) 
  (hdot : (a.1 - c.1) * (b.1 - 2*c.1) + (a.2 - c.2) * (b.2 - 2*c.2) = 0) :
  ℝ :=
  let distance := (c.1 - b.1)^2 + (c.2 - b.2)^2 in
  (distance).sqrt

theorem min_value_proof 
  (a b c : ℝ × ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 2) 
  (hab : a.1 * b.1 + a.2 * b.2 = 2) 
  (hdot : (a.1 - c.1) * (b.1 - 2 * c.1) + (a.2 - c.2) * (b.2 - 2 * c.2) = 0) :
  find_min_value a b c ha hb hab hdot = (sqrt(7) - sqrt(3))/2 :=
sorry

end min_value_proof_l268_268968


namespace complex_div_l268_268026

open Complex

theorem complex_div (i : ℂ) (hi : i = Complex.I) : 
  (6 + 7 * i) / (1 + 2 * i) = 4 - i := 
by 
  sorry

end complex_div_l268_268026


namespace hyperbola_eccentricity_proof_l268_268156

noncomputable def hyperbola_eccentricity (a b c : ℝ) (F1 F2 P : ℝ × ℝ) 
(h_a_gt_b : a > b) (h_foci : c = real.sqrt (a^2 + b^2))
(h_on_hyperbola : P ∈ {p : ℝ × ℝ | (p.1^2 / a^2 - p.2^2 / b^2 = 1) })
(h_dot_product : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 0)
(h_product : real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) * real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 2 * a * c) : 
ℝ :=
  (real.sqrt 5 + 1) / 2

theorem hyperbola_eccentricity_proof (a b c : ℝ) (F1 F2 P : ℝ × ℝ)
(h_a_gt_b : a > b) (h_foci : c = real.sqrt (a^2 + b^2))
(h_on_hyperbola : P ∈ {p : ℝ × ℝ | (p.1^2 / a^2 - p.2^2 / b^2 = 1) })
(h_dot_product : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 0)
(h_product : real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) * real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 2 * a * c) : 
hyperbola_eccentricity a b c F1 F2 P h_a_gt_b h_foci h_on_hyperbola h_dot_product h_product
= (real.sqrt 5 + 1) / 2 := 
sorry

end hyperbola_eccentricity_proof_l268_268156


namespace solve_p_plus_q_l268_268279

noncomputable def a : ℚ := p / q

def rx_condition (x : ℝ) : Prop :=
  let w := Real.floor x in
  let f := x - w in
  w * f = a * (x^2 + x)

def valid_sum (xs : Set ℝ) : Prop :=
  (∃ S, S = (xs.filter rx_condition).sum ∧ S = 666)

theorem solve_p_plus_q (p q : ℕ) (h_coprime : Nat.gcd p q = 1) (h_posp : 0 < p) (h_posq : 0 < q) :
  valid_sum {x : ℝ | true} → p + q = 4 :=
by
  sorry

end solve_p_plus_q_l268_268279


namespace distinct_sets_count_l268_268125

theorem distinct_sets_count (n : ℕ) (p : ℕ) (hp : nat.prime p) (h_cond : p > 3) (hn_pos : n > 0) :
  ∃ (S : set (ℕ × ℕ × ℕ)) (h : ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → xyz = p^n * (x + y + z)), 
  S.card = 3 * (n + 1) :=
sorry

end distinct_sets_count_l268_268125


namespace selection_methods_count_l268_268355

-- Define the set of athletes and conditions
inductive Player
| Veteran : Player
| New : ℕ → Player   -- Using natural numbers to distinguish new players (e.g., 1 to 8)

-- Define exclusion of player A (let's assume A is New 1)
def excludedPlayer : Player := Player.New 1

-- Define the condition for team selection
def valid_team (team : Finset Player) : Prop :=
  team.card = 3 ∧
  (team.filter (λ p, match p with | Player.Veteran => true | _ => false end)).card ≤ 1 ∧
  excludedPlayer ∉ team

-- Define the number of ways to choose a valid team
def number_of_ways_to_choose_valid_team : ℤ :=
  (Fintype.card {team // valid_team team} : ℤ)

theorem selection_methods_count : number_of_ways_to_choose_valid_team = 77 :=
by
  sorry

end selection_methods_count_l268_268355


namespace response_needed_l268_268264

-- Condition: Response rate in percentage
def response_rate : ℝ := 0.62

-- Condition: Minimum number of questionnaires
def min_questionnaires : ℝ := 483.87

-- Intermediate step: Round up to the nearest whole number of questionnaires
def rounded_questionnaires : ℕ := ⌈min_questionnaires⌉.toNat -- round 483.87 to 484

-- Proof problem statement:
theorem response_needed : (rounded_questionnaires * response_rate).floor = 300 := by
  sorry

end response_needed_l268_268264


namespace find_analytical_expression_of_f_l268_268571

-- Given conditions: f(1/x) = 1/(x+1)
def f (x : ℝ) : ℝ := sorry

-- Domain statement (optional for additional clarity):
def domain (x : ℝ) := x ≠ 0 ∧ x ≠ -1

-- Proof obligation: Prove that f(x) = x / (x + 1)
theorem find_analytical_expression_of_f :
  ∀ x : ℝ, domain x → f x = x / (x + 1) := sorry

end find_analytical_expression_of_f_l268_268571


namespace ant_path_distance_l268_268063

noncomputable def distance_to_nearest_corner (x : ℝ) := x - 22

theorem ant_path_distance (x : ℝ) (hx1 : x + (18 - x) + (18 - x) = 75) :
  distance_to_nearest_corner (3 * x / 75) = 3 :=
by 
  have hx2 : 3 * x = 75 := by linarith 
  have hx3 : x = 25 := by linarith [hx2]
  show distance_to_nearest_corner x = 3, from sorry

end ant_path_distance_l268_268063


namespace photos_on_remaining_pages_l268_268862

theorem photos_on_remaining_pages (total_photos pages first_15_pages next_15_pages following_10_pages remaining_pages : ℕ)
  (h_total_photos : total_photos = 500)
  (h_pages : pages = 60)
  (h_first_15_pages : first_15_pages = 15)
  (h_next_15_pages : next_15_pages = 15)
  (h_following_10_pages : following_10_pages = 10)
  (h_remaining_pages : remaining_pages = 20)
  (h_photos_on_first_15 : 3 * first_15_pages + 4 * next_15_pages + 5 * following_10_pages = 155)
  (h_photos_left : total_photos - 155 = 345) :
  (∃ n : ℕ, n = 17) :=
by
  use 17
  sorry

end photos_on_remaining_pages_l268_268862


namespace degrees_to_radians_157_30_l268_268412

-- Define the conversion from degrees to radians
def deg_to_rad (d : ℚ) : ℚ := d * (Real.pi / 180)

-- The given angle 157 degrees 30 minutes in degrees
noncomputable def angle_in_degrees : ℚ := 157.5

-- The equivalent angle in radians as computed
noncomputable def angle_in_radians : ℚ := (7 / 8) * Real.pi

-- The theorem to prove the conversion is correct
theorem degrees_to_radians_157_30 : deg_to_rad angle_in_degrees = angle_in_radians :=
by
  -- Proof would go here
  sorry

end degrees_to_radians_157_30_l268_268412


namespace reciprocal_of_neg_2023_l268_268743

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l268_268743


namespace find_fraction_squares_l268_268202

theorem find_fraction_squares (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := 
by
  sorry

end find_fraction_squares_l268_268202


namespace intersecting_lines_l268_268806

theorem intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ (x = 0 ∨ y = 0) := by
  sorry

end intersecting_lines_l268_268806


namespace women_bathing_suits_count_l268_268434

theorem women_bathing_suits_count :
  ∀ (total_bathing_suits men_bathing_suits women_bathing_suits : ℕ),
    total_bathing_suits = 19766 →
    men_bathing_suits = 14797 →
    women_bathing_suits = total_bathing_suits - men_bathing_suits →
    women_bathing_suits = 4969 := by
sorry

end women_bathing_suits_count_l268_268434


namespace volume_prism_is_12_times_smallest_volume_l268_268404

variables (A B C A1 B1 C1 : Type) 

-- Assuming the necessary geometric and volumetric structures are in place.
axiom triangular_prism : Type
axiom plane : triangular_prism → triangular_prism → triangular_prism → Type
axiom smallest_volume : geometric_volume

def volume_of_smallest_part (A B C1 A1 B1 C : triangular_prism) : geometric_volume :=
smallest_volume

def volume_of_prism (A B C1 A1 B1 C : triangular_prism) : geometric_volume :=
12 * volume_of_smallest_part A B C1 A1 B1 C

theorem volume_prism_is_12_times_smallest_volume
  (A B C A1 B1 C1 : triangular_prism)
  (plane1 : plane A B C1)
  (plane2 : plane A1 B1 C)
  (V : geometric_volume)
  (smallest_vol_eq : volume_of_smallest_part A B C1 A1 B1 C = V) :
  volume_of_prism A B C1 A1 B1 C = 12 * V :=
sorry

end volume_prism_is_12_times_smallest_volume_l268_268404


namespace problem1_problem2_l268_268289

variables {z1 z2 : ℂ}

theorem problem1 (hz1 : ℜ z1 > 0) (hz2 : ℜ z2 > 0)
  (hz1_sq : ℜ (z1 ^ 2) = 2) (hz2_sq : ℜ (z2 ^ 2) = 2) :
  (ℜ (z1 * z2)) = 2 := sorry

theorem problem2 (hz1 : ℜ z1 > 0) (hz2 : ℜ z2 > 0)
  (hz1_sq : ℜ (z1 ^ 2) = 2) (hz2_sq : ℜ (z2 ^ 2) = 2) :
  abs (z1 + 2) + abs (conj z2 + 2) - abs (conj z1 - z2) = 4 * Real.sqrt 2 := sorry

end problem1_problem2_l268_268289


namespace sqrt_7_irrational_l268_268387

theorem sqrt_7_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a: ℝ) / b = Real.sqrt 7 := by
  sorry

end sqrt_7_irrational_l268_268387


namespace solid_volume_l268_268888

theorem solid_volume 
  (d p S : ℝ) : 
  volume (solid d p S) = 2 * d * S + π * p * d ^ 2 + 4 / 3 * π * d^3 :=
sorry

end solid_volume_l268_268888


namespace transform_cubic_eq_trig_solutions_l268_268405

section

variable {p q x t r : ℝ}

/-- Statement (a): Show that for p < 0, the equation x³ + px + q = 0 can be transformed by substituting x = 2 * sqrt(-p / 3) * t into the equation 4 * t³ - 3 * t - r = 0 in the variable t -/
theorem transform_cubic_eq (hp : p < 0) (ht : t = 2 * sqrt(-p / 3) * x) :
  x^3 + p * x + q = 0 ↔ 4 * t^3 - 3 * t - r = 0 :=
sorry

/-- Statement (b): Prove that for 4p³ + 27q² ≤ 0, the solutions to the equation 4t³ - 3t - r = 0 will be t₁ = cos(φ / 3), t₂ = cos((φ + 2 * π) / 3), t₃ = cos((φ + 4 * π) / 3) where φ = arccos(r) -/
theorem trig_solutions (hineq : 4 * p^3 + 27 * q^2 ≤ 0) (φ : ℝ) (hφ : φ = arccos r) :
  (4 * (cos(φ / 3))^3 - 3 * cos(φ / 3) - r = 0) ∧
  (4 * (cos((φ + 2 * real.pi) / 3))^3 - 3 * cos((φ + 2 * real.pi) / 3) - r = 0) ∧
  (4 * (cos((φ + 4 * real.pi) / 3))^3 - 3 * cos((φ + 4 * real.pi) / 3) - r = 0) :=
sorry

end

end transform_cubic_eq_trig_solutions_l268_268405


namespace positive_divisors_8_fact_l268_268193

-- Factorial function definition
def factorial : Nat → Nat
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Function to compute the number of divisors from prime factors
def numDivisors (factors : List (Nat × Nat)) : Nat :=
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

-- Known prime factorization of 8!
noncomputable def factors_8_fact : List (Nat × Nat) :=
  [(2, 7), (3, 2), (5, 1), (7, 1)]

-- Theorem statement
theorem positive_divisors_8_fact : numDivisors factors_8_fact = 96 :=
  sorry

end positive_divisors_8_fact_l268_268193


namespace set1_not_open_set3_not_open_l268_268501

def open_set (A : set (ℝ × ℝ)) : Prop :=
  ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ A → ∃ (r : ℝ) (hr : r > 0), ∀ (x y : ℝ), ((x - x₀)^2 + (y - y₀)^2 < r^2) → (x, y) ∈ A

def set1 : set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 ^ 2 = 1 }
def set2 : set (ℝ × ℝ) := { p | p.1 + p.2 + 2 > 0 }
def set3 : set (ℝ × ℝ) := { p | |p.1 + p.2| ≤ 6 }
def set4 : set (ℝ × ℝ) := { p | (0 < p.1 ^ 2 + (p.2 - real.sqrt 2) ^ 2) ∧ (p.1 ^ 2 + (p.2 - real.sqrt 2) ^ 2 < 1) }

theorem set1_not_open : ¬ open_set set1 :=
sorry

theorem set3_not_open : ¬ open_set set3 :=
sorry

end set1_not_open_set3_not_open_l268_268501


namespace smallest_whole_number_l268_268802

theorem smallest_whole_number (a : ℕ) : 
  (a % 4 = 1) ∧ (a % 3 = 1) ∧ (a % 5 = 2) → a = 37 :=
by
  intros
  sorry

end smallest_whole_number_l268_268802


namespace permutation_remainder_prime_l268_268892

theorem permutation_remainder_prime (n : ℕ) 
  (h : ∃ (a : Fin n → Fin n), Function.Bijective a ∧
        ∃ b : Fin n → Fin n,
          (∀ i : Fin n, b i = (∑ j in Finset.range (i+1), a j) % n) ∧ Function.Bijective b) : 
        n.Prime := 
sorry

end permutation_remainder_prime_l268_268892


namespace lice_checks_time_l268_268778

theorem lice_checks_time (kindergarteners : ℕ) (first_graders : ℕ) (second_graders : ℕ) (third_graders : ℕ) (check_time_per_student : ℕ) : 
  kindergarteners = 26 → first_graders = 19 → second_graders = 20 → third_graders = 25 → check_time_per_student = 2 →
  (kindergarteners + first_graders + second_graders + third_graders) * check_time_per_student / 60 = 3 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end lice_checks_time_l268_268778


namespace simplify_expression_l268_268468

variable (x : ℝ)

def y : ℝ := real.sqrt (x^2 - 4 * x + 4) * real.sqrt (x^2 + 4 * x + 4)

theorem simplify_expression : y x = |x-2| * |x+2| := 
by 
  sorry

end simplify_expression_l268_268468


namespace solve_for_k_l268_268919

theorem solve_for_k : 
  ∃ k : ℤ, (k + 2) / 4 - (2 * k - 1) / 6 = 1 ∧ k = -4 := 
by
  use -4
  sorry

end solve_for_k_l268_268919


namespace lena_optimal_strategy_yields_7000_rubles_l268_268022

noncomputable def lena_annual_income : ℕ := 7000

theorem lena_optimal_strategy_yields_7000_rubles
  (monthly_salary : ℕ)
  (monthly_expenses : ℕ)
  (deposit_interest_monthly : ℚ)
  (debit_card_interest_annual : ℚ)
  (credit_card_limit : ℕ)
  (credit_card_fee_percent : ℚ)
  (monthly_savings : ℕ)
  (P_total : ℚ)
  (deposit_interest : ℚ)
  (debit_card_interest : ℚ) :
  monthly_salary = 50000 ∧
  monthly_expenses = 45000 ∧
  deposit_interest_monthly = 0.01 ∧
  debit_card_interest_annual = 0.08 ∧
  credit_card_limit = 100000 ∧
  credit_card_fee_percent = 0.03 ∧
  monthly_savings = 5000 ∧
  P_total = 5000 * (sum (λ k, (1 : ℚ) * 1.01 ^ k) (finset.range 12)) ∧
  deposit_interest = P_total - 5000 * 12 ∧
  debit_card_interest = 45000 * 0.08 ->
  deposit_interest + debit_card_interest = lena_annual_income :=
by
  sorry

end lena_optimal_strategy_yields_7000_rubles_l268_268022


namespace lineup_combinations_l268_268302

open Finset

-- Define the given conditions
def soccer_team : Finset ℕ := range 16
def quadruplets : Finset ℕ := {0, 1, 2, 3}

-- Define the proof statement
theorem lineup_combinations : 
  (∑ (k : ℕ) in range 3,
    (if k = 2 then 6 * (soccer_team \ quadruplets).choose (5)
     else if k = 1 then 4 * (soccer_team \ quadruplets).choose (6)
     else (soccer_team \ quadruplets).choose (7))) = 9240 :=
by
  -- Sum the combinations for up to 2 quadruplets in the lineup
  sorry

end lineup_combinations_l268_268302


namespace find_sum_of_valid_a_values_l268_268906

theorem find_sum_of_valid_a_values :
  (∑ a in finset.range 10, (if ∃ x y : ℝ, (x - 2 * y = y^2 + 2 ∧ a * x - 2 * y = y^2 + x^2 + 0.25 * a^2) then a else 0)) = 10 :=
by sorry

end find_sum_of_valid_a_values_l268_268906


namespace reciprocal_of_neg_2023_l268_268772

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l268_268772


namespace range_makes_p_and_not_q_true_l268_268153

def p (m : ℝ) := ∀ a ∈ set.Icc 1 2, abs (m - 5) ≤ real.sqrt (a^2 + 8)

def q (m : ℝ) : Prop :=
  let f : ℝ → ℝ := λ x, x^3 + m * x^2 + (m + 6) * x + 1
  ∃ a b, a ≠ b ∧ f' a = 0 ∧ f' b = 0

theorem range_makes_p_and_not_q_true :
  ∀ m : ℝ, (p m ∧ ¬ q m) ↔ m ∈ set.Icc 2 6 :=
begin
  sorry
end

end range_makes_p_and_not_q_true_l268_268153


namespace b_n_geometric_sequence_greatest_integer_not_exceeding_T_2013_l268_268781

-- Sum of the first n terms in the sequence {a_n} is S_n
def S_n (n : ℕ) : ℝ := -(1 / 2) * n^2 - (3 / 2) * n + 1

-- Define the sequence {a_n} with the given condition
def a_n (n : ℕ) : ℝ := if n = 0 then 0 else (S_n n) - (n * (S_n (n - 1)) + (n / 2))

-- Define b_n in terms of a_n
def b_n (n : ℕ) : ℝ := a_n n + n

-- Task 1: Prove that {b_n} is a geometric sequence with first term 1/2 and ratio 1/2
theorem b_n_geometric_sequence (n : ℕ) (hn : n > 0) : b_n n = (1 / 2)^n := sorry

-- Define the sequence {c_n}
def c_n (n : ℕ) : ℝ := 1 + 1 / (((1 / 2)^n - a_n n) * ((1 / 2)^(n + 1) - a_n (n + 1)))

-- Define the sum T_n of the first n terms of {c_n}
def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, c_n i

-- Task 2: Prove that the greatest integer not exceeding T_2013 is 2013
theorem greatest_integer_not_exceeding_T_2013 : floor (T_n 2013) = 2013 := sorry

end b_n_geometric_sequence_greatest_integer_not_exceeding_T_2013_l268_268781


namespace sin_sq_inv_inequality_l268_268311

theorem sin_sq_inv_inequality (t : ℝ) (ht : 0 < t ∧ t ≤ real.pi / 2) :
  1 / real.sin t ^ 2 ≤ 1 / t ^ 2 + 1 - 4 / real.pi ^ 2 := by
  sorry

end sin_sq_inv_inequality_l268_268311


namespace length_of_AC_l268_268646

open Real

theorem length_of_AC (AB DC AD AC : ℝ) (hAB : AB = 15) (hDC : DC = 24) (hAD : AD = 7) :
  AC ≈ 31.8 :=
by
  assume h1 : AB = 15
  assume h2 : DC = 24
  assume h3 : AD = 7
  sorry

end length_of_AC_l268_268646


namespace solve_a_plus_b_l268_268129

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
if x < 2 then a * x + b else 8 - 3 * x

theorem solve_a_plus_b (a b : ℝ) (h : ∀ x, f (f x a b) a b = x) :
  a + b = 7 / 3 :=
sorry

end solve_a_plus_b_l268_268129


namespace monotonicity_on_interval_range_of_k_no_coinciding_tangents_exist_eq_of_coinciding_tangents_l268_268964

-- Part (1) Monotonicity of the function on [0, 2pi]
theorem monotonicity_on_interval (f : ℝ → ℝ) (D : set ℝ) (hD : D = set.Icc 0 (2 * Real.pi)) (h₁ : ∀ x, f x = x + Real.sin x) :
  ∀ x y ∈ D, x ≤ y → f x ≤ f y := by
  sorry

-- Part (2) Range of the real number k
theorem range_of_k (f : ℝ → ℝ) (D : set ℝ) (hD : D = set.Ioc 0 (Real.pi / 2)) (h₁ : ∀ x, f x = x + Real.sin x) (k : ℝ)
  (h₂ : ∀ x ∈ D, f x > k * x) :
  k < (2 / Real.pi) + 1 := by
  sorry

-- Part (3) Existence of coinciding tangents
theorem no_coinciding_tangents (f : ℝ → ℝ) (h₁ : ∀ x, f x = x + Real.sin x) :
  ¬ ∃ (A B C : ℝ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (∃ m b, ∀ x ∈ {A, B, C}, f' x = m ∧ (λ x, m * x + b) = f x) := by
  sorry

theorem exist_eq_of_coinciding_tangents (f : ℝ → ℝ) (h₁ : ∀ x, f x = x + Real.sin x) :
  ∀ (m : ℝ), (m = 1 ∨ m = -1) ↔ (∃ b, ∀ x, f' x = m ∧ (λ x, m * x + b) = f x) := by
  sorry

end monotonicity_on_interval_range_of_k_no_coinciding_tangents_exist_eq_of_coinciding_tangents_l268_268964


namespace abby_bridget_adjacent_probability_l268_268898

-- Definitions for the problem's setup
structure Student : Type where
  name : String

constant A B : Student  -- Specific students: Abby (A) and Bridget (B)
constant seats : Fin 12 -> Option Student  -- Assignment of students to 12 seats
constant is_adjacent : Fin 12 -> Fin 12 -> Prop
constant total_seatings : Nat
constant favorable_seatings : Nat
constant probability : ℚ

-- Conditions derived from the problem
axiom total_seating_arrangements : total_seatings = (12.fact / 4.fact)
axiom number_of_favorable_seatings :
  favorable_seatings = 17 * (2 * 10.fact)

-- The probability calculation
axiom probability_calculation :
  probability = favorable_seatings / total_seatings

-- The main theorem to prove the probability
theorem abby_bridget_adjacent_probability :
  probability = (17 / 66) := by
  sorry

end abby_bridget_adjacent_probability_l268_268898


namespace trigonometric_product_value_l268_268895

theorem trigonometric_product_value :
  sin (4 / 3 * Real.pi) * cos (5 / 6 * Real.pi) * tan (-4 / 3 * Real.pi) = - (3 * Real.sqrt 3) / 4 :=
by
  sorry

end trigonometric_product_value_l268_268895


namespace find_k_l268_268672

def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 60 < f a b c 9 ∧ f a b c 9 < 70)
  (h3 : 90 < f a b c 10 ∧ f a b c 10 < 100)
  (h4 : ∃ k : ℤ, 10000 * k < f a b c 100 ∧ f a b c 100 < 10000 * (k + 1))
  : k = 2 :=
sorry

end find_k_l268_268672


namespace exists_nat_with_digit_sum_l268_268508

-- Definitions of the necessary functions
def digit_sum (n : ℕ) : ℕ := sorry -- Assume this is the sum of the digits of n

theorem exists_nat_with_digit_sum :
  ∃ n : ℕ, digit_sum n = 1000 ∧ digit_sum (n^2) = 1000000 :=
by
  sorry

end exists_nat_with_digit_sum_l268_268508


namespace probability_sum_18_is_1_over_54_l268_268613

open Finset

-- Definitions for a 6-faced die, four rolls, and a probability space.
def faces := {1, 2, 3, 4, 5, 6}
def dice_rolls : Finset (Finset ℕ) := product faces (product faces (product faces faces))

def valid_sum : ℕ := 18

noncomputable def probability_of_sum_18 : ℚ :=
  (dice_rolls.filter (λ r, r.sum = valid_sum)).card / dice_rolls.card

theorem probability_sum_18_is_1_over_54 :
  probability_of_sum_18 = 1 / 54 := 
  sorry

end probability_sum_18_is_1_over_54_l268_268613


namespace loop_result_l268_268058

theorem loop_result (S I : ℕ) :
  (∃ (n : ℕ), 
    S = 1 ∧ I = 1 ∧
    (∀ k, k ≤ n → (I k) < 8) ∧ 
    (∀ k, k ≤ n → (S k) = S k + 2) ∧
    (∀ k, k ≤ n → (I k) = I k + 3) ∧
    (I n ≥ 8) ∧ S = 7) :=
sorry

end loop_result_l268_268058


namespace draw_balls_condition_l268_268357

-- Define the conditions in the problem
def colors := {black, white, red}
def numbers := {1, 2, 3, 4, 5}
def balls := colors × numbers

-- Define the specific problem statement
theorem draw_balls_condition (c1 c2 c3 c4 c5 : colors)
                              (n1 n2 n3 n4 n5 : numbers)
                              (distinct_numbers : n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ n1 ≠ n5 ∧ 
                                                  n2 ≠ n3 ∧ n2 ≠ n4 ∧ n2 ≠ n5 ∧ 
                                                  n3 ≠ n4 ∧ n3 ≠ n5 ∧ 
                                                  n4 ≠ n5)
                              (distinct_colors : c1 ≠ c2 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c1 ≠ c5 ∧ 
                                                c2 ≠ c3 ∧ c2 ≠ c4 ∧ c2 ≠ c5 ∧ 
                                                c3 ≠ c4 ∧ c3 ≠ c5 ∧ 
                                                c4 ≠ c5 ∧ 
                                                ∃ c1 c2 c3, c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3) :
                              -- We need to prove that there are 150 ways to draw such balls
                              ∃ (ways : ℕ), ways = 150 := 
by
  sorry

end draw_balls_condition_l268_268357


namespace division_result_l268_268383

-- Definitions for the values used in the problem
def numerator := 0.0048 * 3.5
def denominator := 0.05 * 0.1 * 0.004

-- Theorem statement
theorem division_result : numerator / denominator = 840 := by 
  sorry

end division_result_l268_268383


namespace large_rect_area_is_294_l268_268133

-- Define the dimensions of the smaller rectangles
def shorter_side : ℕ := 7
def longer_side : ℕ := 2 * shorter_side

-- Condition 1: Each smaller rectangle has a shorter side measuring 7 feet
axiom smaller_rect_shorter_side : ∀ (r : ℕ), r = shorter_side → r = 7

-- Condition 4: The longer side of each smaller rectangle is twice the shorter side
axiom smaller_rect_longer_side : ∀ (r : ℕ), r = longer_side → r = 2 * shorter_side

-- Condition 2: Three rectangles are aligned vertically
def vertical_height : ℕ := 3 * shorter_side

-- Condition 3: One rectangle is aligned horizontally adjoining them
def horizontal_length : ℕ := longer_side

-- The dimensions of the larger rectangle EFGH
def large_rect_width : ℕ := vertical_height
def large_rect_length : ℕ := horizontal_length

-- Calculate the area of the larger rectangle EFGH
def large_rect_area : ℕ := large_rect_width * large_rect_length

-- Prove that the area of the large rectangle is 294 square feet
theorem large_rect_area_is_294 : large_rect_area = 294 := by
  sorry

end large_rect_area_is_294_l268_268133


namespace total_votes_l268_268238

theorem total_votes (bob_votes total_votes : ℕ) (h1 : bob_votes = 48) (h2 : (2 : ℝ) / 5 * total_votes = bob_votes) :
  total_votes = 120 :=
by
  sorry

end total_votes_l268_268238


namespace sin_225_eq_neg_sqrt2_over_2_l268_268488

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end sin_225_eq_neg_sqrt2_over_2_l268_268488


namespace harmonic_mean_is_correct_l268_268079

-- Define the numbers
def a := 2
def b := 3
def c := 6.5

-- Define the reciprocals
def rec_a := 1 / a
def rec_b := 1 / b
def rec_c := 1 / c

-- Sum of reciprocals
def sum_reciprocals := rec_a + rec_b + rec_c

-- Average of reciprocals
def avg_reciprocals := sum_reciprocals / 3

-- Harmonic mean
def harmonic_mean := 1 / avg_reciprocals

-- Theorem: The harmonic mean of the numbers 2, 3, and 6.5 is 234/77
theorem harmonic_mean_is_correct : harmonic_mean = (234:ℚ) / 77 := 
by sorry

end harmonic_mean_is_correct_l268_268079


namespace find_m_l268_268186

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 2 * x
def symmetric_about_line (x1 y1 x2 y2 m : ℝ) : Prop := (y1 - y2) / (x1 - x2) = -1
def product_y (y1 y2 : ℝ) : Prop := y1 * y2 = -1 / 2

-- Theorem to be proven
theorem find_m 
  (x1 y1 x2 y2 m : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : symmetric_about_line x1 y1 x2 y2 m)
  (h4 : product_y y1 y2) :
  m = 9 / 4 :=
sorry

end find_m_l268_268186


namespace smallest_positive_c_unique_d_l268_268914

noncomputable def polynomial_roots (u v w : ℝ) := 
  u * v * w = 3 * Real.sqrt 3 ∧ (u + v + w) = 3 * Real.sqrt 3 ∧ 
  (u * v + v * w + w * u) = 9 ∧ u > 0 ∧ v > 0 ∧ w > 0

theorem smallest_positive_c_unique_d :
  ∃ (c : ℝ) (d : ℝ), c = 3 * Real.sqrt 3 ∧ d = 9 ∧ 
  (∃ (u v w : ℝ), polynomial_roots u v w) :=
begin
  sorry
end

end smallest_positive_c_unique_d_l268_268914


namespace find_value_of_x_l268_268430

theorem find_value_of_x (x : ℝ) : (45 * x = 0.4 * 900) -> x = 8 :=
by
  intro h
  sorry

end find_value_of_x_l268_268430


namespace geometric_sequence_general_term_l268_268277

theorem geometric_sequence_general_term (a : ℕ → ℝ) (a1 : ℝ) (S3 : ℝ) :
  (∀ n : ℕ, a n = a1 * (-2)^(n - 1) ∨ a n = a1) ∧ (a1 = 3/2) ∧ (S3 = 9/2) → 
  (a1 * (1 + (-2) + (-2)^2) = S3) :=
by
  assume h,
  sorry

end geometric_sequence_general_term_l268_268277


namespace barbecue_lcm_l268_268393

theorem barbecue_lcm : 
  let hot_dogs_and_buns_lcm := Nat.lcm 12 (Nat.lcm 10 (Nat.lcm 9 8)),
      toppings_lcm := Nat.lcm 18 (Nat.lcm 24 (Nat.lcm 20 30)) in
  hot_dogs_and_buns_lcm = 360 ∧ toppings_lcm = 360 :=
by
  let hot_dogs_and_buns_lcm := Nat.lcm 12 (Nat.lcm 10 (Nat.lcm 9 8))
  have h1 : hot_dogs_and_buns_lcm = 360 := by
    rw [Nat.lcm_assoc, Nat.lcm_comm 10 _, Nat.lcm_assoc, Nat.lcm_comm 9 _, Nat.lcm_assoc]
    norm_num
  let toppings_lcm := Nat.lcm 18 (Nat.lcm 24 (Nat.lcm 20 30))
  have h2 : toppings_lcm = 360 := by
    rw [Nat.lcm_assoc, Nat.lcm_comm 24 _, Nat.lcm_assoc, Nat.lcm_comm 20 _, Nat.lcm_assoc]
    norm_num
  exact ⟨h1, h2⟩

end barbecue_lcm_l268_268393


namespace number_added_to_x_is_2_l268_268545

/-- Prove that in a set of integers {x, x + y, x + 4, x + 7, x + 22}, 
    where the mean is 3 greater than the median, the number added to x 
    to get the second integer is 2. --/

theorem number_added_to_x_is_2 (x y : ℤ) (h_pos : 0 < x ∧ 0 < y) 
  (h_median : (x + 4) = ((x + y) + (x + (x + y) + (x + 4) + (x + 7) + (x + 22)) / 5 - 3)) : 
  y = 2 := by
  sorry

end number_added_to_x_is_2_l268_268545


namespace three_digit_number_ends_with_same_three_digits_l268_268524

theorem three_digit_number_ends_with_same_three_digits (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, k ≥ 1 → N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) := 
sorry

end three_digit_number_ends_with_same_three_digits_l268_268524


namespace max_area_orthogonal_projection_unit_cube_l268_268378

theorem max_area_orthogonal_projection_unit_cube: 
  let edge_length := 1
  let orthogonal_projection_area := 
    ∀ (P : ℝ → ℝ → ℝ), P ∈ (unit_cube_projection_set edge_length) → area(P) <= 2 * Real.sqrt 3
  edge_length = 1 → orthogonal_projection_area = 2 * Real.sqrt 3 :=
sorry

end max_area_orthogonal_projection_unit_cube_l268_268378


namespace extreme_points_exactly_one_zero_in_positive_interval_l268_268579

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1 / 3) * a * x^3

theorem extreme_points (a : ℝ) (h : a > Real.exp 1) :
  ∃ (x1 x2 x3 : ℝ), (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (deriv (f x) = 0) := sorry

theorem exactly_one_zero_in_positive_interval (a : ℝ) (h : a > Real.exp 1) :
  ∃! x : ℝ, (0 < x) ∧ (f x a = 0) := sorry

end extreme_points_exactly_one_zero_in_positive_interval_l268_268579


namespace range_of_a_l268_268961

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + Real.exp x - Real.exp (-x)

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l268_268961


namespace problem_equiv_none_of_these_l268_268208

variable {x y : ℝ}

theorem problem_equiv_none_of_these (hx : x ≠ 0) (hx3 : x ≠ 3) (hy : y ≠ 0) (hy5 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) →
  ¬(3 * x + 2 * y = x * y) ∧
  ¬(y = 3 * x / (5 - y)) ∧
  ¬(x / 3 + y / 2 = 3) ∧
  ¬(3 * y / (y - 5) = x) :=
sorry

end problem_equiv_none_of_these_l268_268208


namespace largest_y_coordinate_ellipse_l268_268495

theorem largest_y_coordinate_ellipse (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 := 
by
  -- proof to be filled in
  sorry

end largest_y_coordinate_ellipse_l268_268495


namespace sin_225_eq_neg_sqrt_two_div_two_l268_268478

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt_two_div_two_l268_268478


namespace walter_work_hours_per_day_l268_268372

open Set

-- Define the given conditions
def days_per_week := 5
def rate_per_hour := 5
def allocation_ratio := (3 : ℝ) / 4
def educational_expense := 75

-- Define the main theorem
theorem walter_work_hours_per_day : 
  (∃ h : ℝ, (allocation_ratio * (days_per_week * rate_per_hour * h) = educational_expense) ∧ h = 4) := 
by 
  sorry

end walter_work_hours_per_day_l268_268372


namespace smallest_square_area_example_l268_268827

noncomputable def smallest_square_area (l1 w1 l2 w2 : ℕ) : ℕ :=
  let min_side := max (l1 + l2) (max w1 w2)
  min_side * min_side

theorem smallest_square_area_example :
  smallest_square_area 1 4 2 5 = 81 :=
by {
  unfold smallest_square_area,
  simp,
}

end smallest_square_area_example_l268_268827


namespace range_of_t_log_inequality_l268_268823

theorem range_of_t_log_inequality :
  (∀ x > 0, log x + log t ≤ log (x^2 + t)) ↔ t ∈ Set.Ioc 0 4 := sorry

end range_of_t_log_inequality_l268_268823


namespace french_horn_trombone_difference_l268_268346

def flute_players : ℕ := 5
def trumpet_players : ℕ := 3 * flute_players
def trombone_players : ℕ := trumpet_players - 8
def drummers : ℕ := trombone_players + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players (x : ℕ) : ℕ := trombone_players + x
def total_members (x : ℕ) : ℕ := flute_players + trumpet_players + trombone_players + drummers + clarinet_players + french_horn_players x
def seats_required : ℕ := 65

theorem french_horn_trombone_difference : ∃ x : ℕ, total_members x = seats_required → x = 3 :=
begin
  sorry
end

end french_horn_trombone_difference_l268_268346


namespace avg_rate_of_change_l268_268959

def f (x : ℝ) := 1 + 1 / x

theorem avg_rate_of_change : (f 2 - f 1) / (2 - 1) = -1 / 2 := 
by
  sorry

end avg_rate_of_change_l268_268959


namespace cube_root_expression_l268_268875

theorem cube_root_expression : 
  ∑ x in {a b c : ℝ // a = 7 + 3 * ℝ.sqrt 21 &, b = 7 - 3 * ℝ.sqrt 21, c = 3*ℝ.sqrt (140) }, 
 (x.a ^ (1/3)) + (x.b ^ (1/3)) = 2 := by
  sorry

end cube_root_expression_l268_268875


namespace domain_of_f_l268_268502

def domain_of_function (f : ℝ → ℝ) (dom : Set ℝ) : Prop :=
  ∀ x, dom x → ∃ y, f x = y

noncomputable def f : ℝ → ℝ := λ x, sqrt x / (2^x - 1)

theorem domain_of_f : domain_of_function f (Set.Ioi 0) :=
by
  sorry

end domain_of_f_l268_268502


namespace expected_number_of_matches_variance_of_number_of_matches_l268_268417

-- Defining the conditions first, and then posing the proof statements
namespace MatchingPairs

open ProbabilityTheory

-- Probabilistic setup for indicator variables
variable (N : ℕ) (prob : ℝ := 1 / N)

-- Indicator variable Ik representing matches
@[simp] def I (k : ℕ) : ℝ := if k < N then prob else 0

-- Define the sum of expected matches S
@[simp] def S : ℝ := ∑ k in finset.range N, I N k

-- Statement: The expectation of the number of matching pairs is 1
theorem expected_number_of_matches : E[S] = 1 := sorry

-- Statement: The variance of the number of matching pairs is 1
theorem variance_of_number_of_matches : Var S = 1 := sorry

end MatchingPairs

end expected_number_of_matches_variance_of_number_of_matches_l268_268417


namespace sqrt_eq_4_implies_x_eq_169_l268_268986

-- Statement of the problem
theorem sqrt_eq_4_implies_x_eq_169 (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 :=
begin
  sorry  -- proof not required
end

end sqrt_eq_4_implies_x_eq_169_l268_268986


namespace ellipse_standard_eq_chord_length_major_axis_range_l268_268147

noncomputable def ellipse_eq := (a b : ℝ) (a > b > 0) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def hyperbola_vertices := ∃ (c : ℝ), c = 1

theorem ellipse_standard_eq (a b : ℝ) (h₀ : a = √3) (h₁ : b = √2) (h₂ : 2 * a + 2 * 1 = 2 * √3 + 2) :
  ellipse_eq a b :=
sorry

theorem chord_length (a b : ℝ) (h₀ : a = √3) (h₁ : b = √2) (h₂ : ellipse_eq a b) :
  ∃ (AB : ℝ), AB = 8 * √3 / 5 :=
sorry

theorem major_axis_range (a : ℝ) (h₀ : a = √3) (h₁ : 0 < √3) :
  (√5 ≤ 2 * a ∧ 2 * a ≤ √6) :=
sorry

end ellipse_standard_eq_chord_length_major_axis_range_l268_268147


namespace matrix_self_inverse_l268_268504

theorem matrix_self_inverse (a b : ℝ) :
  (matrix_mul (matrix [[4, -2], [a, b]]) (matrix [[4, -2], [a, b]]) = matrix_id 2) ↔
  (a = 7.5 ∧ b = -4) :=
by
  sorry

end matrix_self_inverse_l268_268504


namespace lattice_points_interval_l268_268275

def T : set (ℤ × ℤ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50}

theorem lattice_points_interval :
  ∃ (c d : ℕ), nat.coprime c d ∧ (500 = ((T : set (ℤ × ℤ)).filter (λ (p : ℤ × ℤ), p.2 ≤ (m * p.1))).card) ∧
  (∀ m, m ∈ {m | 1 ≤ (T : set (ℤ × ℤ)).filter (λ (p : ℤ × ℤ), p.2 ≤ (m * p.1)).card ≤ 500} → ∃ (c d : ℕ), nat.coprime c d ∧ 
  m ∈ Icc (1/300) (299/300)) →
  c + d = 301 := 
sorry

end lattice_points_interval_l268_268275


namespace number_of_ways_to_place_dishes_l268_268124

theorem number_of_ways_to_place_dishes (n : ℕ) (h : n = 8) : 
  let ways := 2 + 2 + 16 + 20 + 8 + 1 in
  ways = 49 :=
by
  sorry

end number_of_ways_to_place_dishes_l268_268124


namespace lim_sqrt_n_Gn_l268_268669

noncomputable def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

noncomputable def geometric_mean (n : ℕ) : ℝ := 
  real.exp ((list.sum (list.map (λ k, real.log (binom n k)) (list.range (n + 1))).to_real / (n + 1)))

theorem lim_sqrt_n_Gn : 
  tendsto (λ n : ℕ, real.sqrt (geometric_mean n)) at_top (nhds (real.sqrt real.exp 1)) :=
sorry

end lim_sqrt_n_Gn_l268_268669


namespace find_a_l268_268598

theorem find_a (a : ℝ) : 
  (2^n = 64) ∧ ((λ (x : ℝ), (x^2 + a / x)^6) = 729) ↔ (a = -4 ∨ a = 2) := 
by 
  sorry

end find_a_l268_268598


namespace h_range_M_max_k_range_l268_268583

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * Real.log x / Real.log 2
def g (x : ℝ) : ℝ := Real.log x / Real.log 2

def h (x : ℝ) : ℝ := (f x + 1) * g x
def M (x : ℝ) : ℝ := (f x + g x - abs (f x - g x)) / 2
def k_inequality (x : ℝ) (k : ℝ) : Prop := f (x^2) * f (Real.sqrt x) > k * g x

theorem h_range : ∀ x ∈ Icc 1 2, 0 ≤ h x ∧ h x ≤ 2 := 
sorry

theorem M_max : ∀ x ∈ Icc 1 2, M x ≤ 1 :=
sorry

theorem k_range : ∀ x ∈ Icc 1 2, (∀ k : ℝ, k_inequality x k) → k < -2 :=
sorry

end h_range_M_max_k_range_l268_268583


namespace bandi_Jan1_tie_green_l268_268535

-- Definitions of given conditions
variable (Aladar Bandi Feri Pista Geza : ℕ)
variable (Ties : List (ℕ -> String))
variable (Color : String -> Prop)

-- Condition 1: Each friend wears different tie each day and cycles through them.
def cycles_through (A : ℕ) (n : ℕ) : Prop :=
  ∃ k, n % k = A ∧ k = A

-- Condition 2: Each friend has at least 2 ties but no more than 11.
def proper_range (n : ℕ) : Prop := 2 ≤ n ∧ n ≤ 11

-- Condition 3: No two friends have the same number of ties.
def unique_ties (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- Condition 4 and specific dates ties
def dec_1_ties : Prop := Ties Aladar 1 = "blue" ∧ Ties Bandi 1 = "red" ∧ Ties Feri 1 = "red" ∧ Ties Pista 1 = "green" ∧ Ties Geza 1 = "yellow"
def dec_19_ties : Prop := Ties Pista 19 = "green" ∧ Ties Geza 19 = "yellow" ∧ Ties Feri 19 = "blue" ∧ Ties Aladar 19 = "red" ∧ Ties Bandi 19 = "red"
def dec_23_ties : Prop := Ties Pista 23 = "white"
def dec_26_ties : Prop := Ties Pista 26 = "yellow"
def dec_11_colors : Prop := ∃ a b c d e, a = "yellow" ∧ b = "red" ∧ c = "blue" ∧ d = "green" ∧ e = "white"
def dec_31_ties : Prop := Ties Aladar 31 = Ties Aladar 1 ∧ Ties Bandi 31 = Ties Bandi 1 ∧ Ties Feri 31 = Ties Feri 1 ∧ Ties Pista 31 = Ties Pista 1 ∧ Ties Geza 31 = Ties Geza 1

-- Prove Bandi's tie on January 1 is green
theorem bandi_Jan1_tie_green :
  proper_range Aladar ∧
  proper_range Bandi ∧
  proper_range Feri ∧
  proper_range Pista ∧
  proper_range Geza ∧
  unique_ties Aladar Bandi Feri Pista Geza ∧
  dec_1_ties ∧
  dec_19_ties ∧
  dec_23_ties ∧
  dec_26_ties ∧
  dec_11_colors ∧
  dec_31_ties →
  Ties Bandi 32 = "green"
:= sorry

end bandi_Jan1_tie_green_l268_268535


namespace valid_l_exists_l268_268145

noncomputable def valid_l (m l : ℕ) : Prop :=
  ml % 2 = 0 ∧ 1 ≤ l ∧ l < m

theorem valid_l_exists (m : ℕ) :
  ∀ l, valid_l m l → ∃ k, 1 ≤ l ∧ l < m ∧ (m * l) % 2 = 0 := by {
  sorry
}

end valid_l_exists_l268_268145


namespace radius_of_inscribed_circle_in_rhombus_l268_268381

theorem radius_of_inscribed_circle_in_rhombus
  (d1 d2 : ℕ) (h₁ : d1 = 8) (h₂ : d2 = 30) : 
  ∃ r : ℝ, r = 60 / Real.sqrt 241 :=
by
  use 60 / Real.sqrt 241
  sorry

end radius_of_inscribed_circle_in_rhombus_l268_268381


namespace dips_to_daps_conversion_l268_268213

variable (daps dops dips : ℝ)

-- Condition: 5 daps are equivalent to 4 dops
def condition1 := 5 * daps = 4 * dops

-- Condition: 3 dops are equivalent to 10 dips
def condition2 := 3 * dops = 10 * dips

-- Theorem: 60 dips are equivalent to 22.5 daps
theorem dips_to_daps_conversion (h1 : condition1) (h2 : condition2) : 60 * dips = 22.5 * daps :=
  sorry

end dips_to_daps_conversion_l268_268213


namespace cos_4_arccos_fraction_l268_268522

theorem cos_4_arccos_fraction :
  (Real.cos (4 * Real.arccos (2 / 5))) = (-47 / 625) :=
by
  sorry

end cos_4_arccos_fraction_l268_268522


namespace arithmetic_mean_is_one_l268_268113

variable a : ℝ := 1 + Real.sqrt 2
variable b : ℝ := 1 - Real.sqrt 2

theorem arithmetic_mean_is_one : (a + b) / 2 = 1 := by 
  sorry

end arithmetic_mean_is_one_l268_268113


namespace L_shape_area_correct_l268_268452

noncomputable def large_rectangle_area : ℕ := 12 * 7
noncomputable def small_rectangle_area : ℕ := 4 * 3
noncomputable def L_shape_area := large_rectangle_area - small_rectangle_area

theorem L_shape_area_correct : L_shape_area = 72 := by
  -- here goes your solution
  sorry

end L_shape_area_correct_l268_268452


namespace angle_bisector_divides_square_l268_268699

theorem angle_bisector_divides_square (a : ℝ) (A B C D E : ℝ → ℝ × ℝ)
  (h_square : (B = A + (a, 0)) ∧ (C = B + (0, a)) ∧ (D = A + (0, a)) ∧ (E = A + (a, 0))) 
  (h_triangle : ∡ E A B = 90) : 
  divides_square (bisector ∡ E A B) = true :=
sorry

end angle_bisector_divides_square_l268_268699


namespace max_knights_in_castle_l268_268230

/-- In a 4x4 grid of rooms, each room houses either a knight or a liar. Knights always tell the truth,
while liars always lie. Each person claims that at least one of the neighboring rooms houses a liar.
This theorem proves that the maximum number of knights in such a configuration is 12. -/
theorem max_knights_in_castle : 
  ∀ (room_condition : (ℕ × ℕ) → Prop),
  (∀ x y, room_condition (x, y) → (x < 4 ∧ y < 4)) →
  (∀ x y, room_condition (x, y) → (∃ (a b : ℕ), ((a = x + 1 ∨ a = x - 1 ∧ b = y) ∨ (b = y + 1 ∨ b = y - 1 ∧ a = x)) ∧ ¬ room_condition (a, b))) →
  (∃ k : ℕ, k ≤ 12 ∧ ∀ n, n > k → ¬(∃ (grid : (ℕ × ℕ) → Prop), (∀ x y, grid (x, y) → (x < 4 ∧ y < 4)) ∧ (∀ x y, grid (x, y) → (∃ (a b : ℕ), ((a = x + 1 ∨ a = x - 1 ∧ b = y) ∨ (b = y + 1 ∨ b = y - 1 ∧ a = x)) ∧ ¬ grid (a, b))) ∧ (∑ i j, if grid (i, j) then 1 else 0 = n))) := 
sorry

end max_knights_in_castle_l268_268230


namespace sum_of_digits_base8_product_l268_268873

theorem sum_of_digits_base8_product
  (a b : ℕ)
  (a_base8 : a = 3 * 8^1 + 4 * 8^0)
  (b_base8 : b = 2 * 8^1 + 2 * 8^0)
  (product : ℕ := a * b)
  (product_base8 : ℕ := (product / 64) * 8^2 + ((product / 8) % 8) * 8^1 + (product % 8)) :
  ((product_base8 / 8^2) + ((product_base8 / 8) % 8) + (product_base8 % 8)) = 1 * 8^1 + 6 * 8^0 :=
sorry

end sum_of_digits_base8_product_l268_268873


namespace expand_simplify_expression_l268_268518

def a (x : ℝ) := x + 3
def b (x : ℝ) := 4x - 8
def c (x : ℝ) := x^2

theorem expand_simplify_expression (x : ℝ) :
  (a x) * (b x) + (c x) = 5 * x^2 + 4 * x - 24 := 
by 
  sorry

end expand_simplify_expression_l268_268518


namespace sum_of_roots_l268_268566

noncomputable def f (x : ℝ) : ℝ :=
  if (0 < x ∧ x ≤ 1) then log (1/2) x
  else if (x > 1 ∧ x ≤ 2) then log (1/2) (2 - x)
  else if (x ≠ 0 ∧ x ≤ -1) then -f (-x)
  else f (x - 2)

theorem sum_of_roots (f_is_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_symmetric : ∀ x : ℝ, f (2 - x) = f x)
  (f_def : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = log (1/2) x) :
  let roots := {x | f x - 1 = 0 ∧ 0 < x ∧ x < 6} in
  ∑ x in roots, x = 12 :=
sorry

end sum_of_roots_l268_268566


namespace total_good_numbers_l268_268606

def is_good_number (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ a + b = c + d

theorem total_good_numbers : 
  (∃ n : ℕ, n = 615 ∧ ∀ a b c d : ℕ, is_good_number a b c d ↔ n) := 
sorry

end total_good_numbers_l268_268606


namespace exists_monochromatic_triangle_in_tripartite_graph_l268_268634

theorem exists_monochromatic_triangle_in_tripartite_graph :
  ∀ (V : Type) [Fintype V] [DecidableEq V] (e : V → V → Prop) [Symmetric (λ x y, e x y)]
  (color : V → V → fin 3) (hV : Fintype.card V = 17),
  ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (color a b = color b c ∧ color b c = color c a) :=
by
  -- Sorry to skip the proof.
  sorry

end exists_monochromatic_triangle_in_tripartite_graph_l268_268634


namespace shape_of_constant_phi_l268_268537

-- Define the spherical coordinates structure
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition that φ is a constant c
def constant_phi (c : ℝ) (coords : SphericalCoordinates) : Prop :=
  coords.φ = c

-- Define the type for shapes
inductive Shape
  | Line : Shape
  | Circle : Shape
  | Plane : Shape
  | Sphere : Shape
  | Cylinder : Shape
  | Cone : Shape

-- The theorem statement
theorem shape_of_constant_phi (c : ℝ) (coords : SphericalCoordinates) 
  (h : constant_phi c coords) : Shape :=
  Shape.Cone

end shape_of_constant_phi_l268_268537


namespace cars_in_parking_lot_l268_268364

theorem cars_in_parking_lot (initial_cars left_cars entered_cars : ℕ) (h1 : initial_cars = 80)
(h2 : left_cars = 13) (h3 : entered_cars = left_cars + 5) : 
initial_cars - left_cars + entered_cars = 85 :=
by
  rw [h1, h2, h3]
  sorry

end cars_in_parking_lot_l268_268364


namespace count_distinct_arrangements_l268_268032

/-- In a circular arrangement of 8 girls and 25 boys, 
such that there are at least two boys between any two girls,
the number of distinct seating arrangements is: 
16! * 25! / 9! -/
theorem count_distinct_arrangements :
  ∃ (girls boys : ℕ), 
    girls = 8 ∧ boys = 25 ∧
    (∃!(sequences : list ℕ), 
     sequences = 16! * 25! / 9!)  :=
begin
  use [8, 25],
  split, refl,
  split, refl,
  use 16! * 25! / 9!,
  split_ifs, 
  sorry
end

end count_distinct_arrangements_l268_268032


namespace math_proof_problem_l268_268142

variable (n : ℕ) (t : Fin n → ℝ)

theorem math_proof_problem 
  (h₀ : 0 < t 0)
  (h₁ : ∀ i j : Fin n, i ≤ j → t i ≤ t j)
  (h₂ : ∀ i : Fin n, t i < 1) :
  (1 - t ⟨n-1, by sorry⟩)^2 * (∑ i : Fin n, t i^(i + 1) / (1 - t i^(i + 2))^2) < 1 := 
by {
  sorry
}

end math_proof_problem_l268_268142


namespace min_value_expression_l268_268548

theorem min_value_expression :
  (∃ x, 1 < x ∧ x < 5 ∧ (x^2 - 4*x + 5) / (2*x - 6) = 1) :=
begin
  use 3,
  split,
  { linarith, },
  split,
  { linarith, },
  -- simplify the given expression at x = 3 to confirm it equals 1
  sorry
end

end min_value_expression_l268_268548


namespace solids_with_triangular_front_view_l268_268623

-- Definitions based on given conditions
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

def can_have_triangular_front_view : Solid → Prop
  | Solid.TriangularPyramid => true
  | Solid.SquarePyramid => true
  | Solid.TriangularPrism => true
  | Solid.SquarePrism => false
  | Solid.Cone => true
  | Solid.Cylinder => false

-- Theorem statement
theorem solids_with_triangular_front_view :
  {s : Solid | can_have_triangular_front_view s} = 
  {Solid.TriangularPyramid, Solid.SquarePyramid, Solid.TriangularPrism, Solid.Cone} :=
by
  sorry

end solids_with_triangular_front_view_l268_268623


namespace soda_cans_ratio_l268_268368

theorem soda_cans_ratio
  (initial_cans : ℕ := 22)
  (cans_taken : ℕ := 6)
  (final_cans : ℕ := 24)
  (x : ℚ := 1 / 2)
  (cans_left : ℕ := 16)
  (cans_bought : ℕ := 16 * 1 / 2) :
  (cans_bought / cans_left : ℚ) = 1 / 2 :=
sorry

end soda_cans_ratio_l268_268368


namespace game_ends_after_two_rallies_player_a_wins_game_l268_268073

theorem game_ends_after_two_rallies (prob_win_rally_serving prob_win_rally_not_serving : ℝ) :
  prob_win_rally_serving = 0.4 → prob_win_rally_not_serving = 0.5 →
  let prob_end_two_rallies :=
    prob_win_rally_serving * prob_win_rally_serving +
    (1 - prob_win_rally_serving) * (1 - prob_win_rally_not_serving)
  in prob_end_two_rallies = 0.46 :=
by
  intros h1 h2
  let prob_end_two_rallies :=
    prob_win_rally_serving * prob_win_rally_serving +
    (1 - prob_win_rally_serving) * (1 - prob_win_rally_not_serving)
  show prob_end_two_rallies = 0.46
  sorry

theorem player_a_wins_game (prob_win_rally_serving prob_win_rally_not_serving : ℝ) :
  prob_win_rally_serving = 0.4 → prob_win_rally_not_serving = 0.5 →
  let prob_a_wins :=
    prob_win_rally_serving * prob_win_rally_serving + 
    (prob_win_rally_serving * (1 - prob_win_rally_serving) * prob_win_rally_not_serving) +
    ((1 - prob_win_rally_serving) * prob_win_rally_not_serving * prob_win_rally_serving)
  in prob_a_wins = 0.4 :=
by
  intros h1 h2
  let prob_a_wins :=
    prob_win_rally_serving * prob_win_rally_serving + 
    (prob_win_rally_serving * (1 - prob_win_rally_serving) * prob_win_rally_not_serving) +
    ((1 - prob_win_rally_serving) * prob_win_rally_not_serving * prob_win_rally_serving)
  show prob_a_wins = 0.4
  sorry

end game_ends_after_two_rallies_player_a_wins_game_l268_268073


namespace number_of_sets_l268_268563

theorem number_of_sets :
  {M : Set ℕ // {1, 2, 3} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5}}.card = 4 :=
by
  sorry

end number_of_sets_l268_268563


namespace fib_100_mod_5_l268_268330

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib n + fib (n+1)

theorem fib_100_mod_5 : (fib 100) % 5 = 0 :=
by sorry

end fib_100_mod_5_l268_268330


namespace circle_equation_l268_268724

theorem circle_equation (x y : ℝ) (h k : ℝ) (r : ℝ) :
  (h = 1) →
  (k = 2) →
  (r = 5) →
  ((x - h) ^ 2 + (y - k) ^ 2 = r ^ 2) →
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25 :=
by
  intros h_eq k_eq r_eq circle_eq
  rw [h_eq, k_eq, r_eq] at circle_eq
  exact circle_eq

end circle_equation_l268_268724


namespace psychologist_diagnosis_l268_268847

theorem psychologist_diagnosis :
  let initial_patients := 26
  let doubling_factor := 2
  let probability := 1 / 4
  let total_patients := initial_patients * doubling_factor
  let expected_patients_with_ZYX := total_patients * probability
  expected_patients_with_ZYX = 13 := by
  sorry

end psychologist_diagnosis_l268_268847


namespace mod_21_solution_l268_268375

theorem mod_21_solution (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n < 21) (h₂ : 47635 ≡ n [MOD 21]) : n = 19 :=
by
  sorry

end mod_21_solution_l268_268375


namespace shape_of_constant_phi_l268_268536

-- Define the spherical coordinates structure
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition that φ is a constant c
def constant_phi (c : ℝ) (coords : SphericalCoordinates) : Prop :=
  coords.φ = c

-- Define the type for shapes
inductive Shape
  | Line : Shape
  | Circle : Shape
  | Plane : Shape
  | Sphere : Shape
  | Cylinder : Shape
  | Cone : Shape

-- The theorem statement
theorem shape_of_constant_phi (c : ℝ) (coords : SphericalCoordinates) 
  (h : constant_phi c coords) : Shape :=
  Shape.Cone

end shape_of_constant_phi_l268_268536


namespace exists_BC_with_given_ranks_A_satisfies_polynomial_eq_l268_268797

variables {n r : ℕ} (A : Matrix (Fin n) (Fin n) ℂ)
variables (B : Matrix (Fin n) (Fin r) ℂ) (C : Matrix (Fin r) (Fin n) ℂ)

-- Conditions
def isValidRank (A : Matrix (Fin n) (Fin n) ℂ) (r : ℕ) : Prop :=
  A.rank = r

def validDimensions : Prop := (n ≥ 2) ∧ (1 ≤ r) ∧ (r ≤ n - 1)

-- Prove that there exist B and C with required properties
theorem exists_BC_with_given_ranks :
  validDimensions → isValidRank A r →
  ∃ (B : Matrix (Fin n) (Fin r) ℂ) (C : Matrix (Fin r) (Fin n) ℂ), B.rank = r ∧ C.rank = r ∧ A = B ⬝ C :=
by
-- proof will be here
sorry

-- Prove that A satisfies a polynomial equation of degree r + 1
theorem A_satisfies_polynomial_eq :
  validDimensions → isValidRank A r →
  ∃ p : Polynomial ℂ, degree p = r + 1 ∧ ∀ x, p.eval x = 0 :=
by
-- proof will be here
sorry

end exists_BC_with_given_ranks_A_satisfies_polynomial_eq_l268_268797


namespace angle_bisector_between_median_and_altitude_l268_268313

variable {α : Type*}

structure Triangle (α : Type*) :=
(A B C : α)

structure Point (α : Type*) :=
(x y : α)

def is_scalene (t : Triangle ℝ) : Prop := 
  t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.A ≠ t.C

def altitude_foot (t : Triangle ℝ) (B : Point ℝ) : Point ℝ := sorry
def angle_bisector_point (t : Triangle ℝ) (B : Point ℝ) : Point ℝ := sorry
def midpoint (A C : Point ℝ) : Point ℝ := sorry

theorem angle_bisector_between_median_and_altitude 
  (t : Triangle ℝ) (h_scl : is_scalene t) (H : Point ℝ) (D : Point ℝ) (M : Point ℝ)
  (H_alt : H = altitude_foot t t.B) 
  (D_bis : D = angle_bisector_point t t.B) 
  (M_mid : M = midpoint t.A t.C) : 
  between (D : affine_combination H M) :=
begin
  sorry
end

end angle_bisector_between_median_and_altitude_l268_268313


namespace reciprocal_of_neg_2023_l268_268748

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end reciprocal_of_neg_2023_l268_268748


namespace trail_mix_total_weight_l268_268075

theorem trail_mix_total_weight :
  let peanuts := 0.16666666666666666
  let chocolate_chips := 0.16666666666666666
  let raisins := 0.08333333333333333
  let almonds := 0.14583333333333331
  let cashews := (1 / 8 : Real)
  let dried_cranberries := (3 / 32 : Real)
  (peanuts + chocolate_chips + raisins + almonds + cashews + dried_cranberries) = 0.78125 :=
by
  sorry

end trail_mix_total_weight_l268_268075


namespace diamond_4_3_l268_268096

def diamond (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem diamond_4_3 : diamond 4 3 = 1 :=
by
  -- The proof will go here.
  sorry

end diamond_4_3_l268_268096


namespace track_width_l268_268851

theorem track_width (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi) : r1 - r2 = 10 := by
  sorry

end track_width_l268_268851


namespace num_divisors_of_8_factorial_l268_268194

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the number of positive divisors function which uses prime factorization
noncomputable def num_divisors (n : ℕ) : ℕ :=
let factors := unique_factorization_monoid.factors n in
factors.to_finset.prod (λ p, factors.count p + 1)

-- Mathematical statement to prove
theorem num_divisors_of_8_factorial : num_divisors (factorial 8) = 96 := 
sorry

end num_divisors_of_8_factorial_l268_268194


namespace mutually_exclusive_not_necessarily_complementary_l268_268204

noncomputable theory
open_locale classical

variables {Ω : Type*} [probability_space Ω]

theorem mutually_exclusive_not_necessarily_complementary (A B : set Ω) 
  [measurable_set A] [measurable_set B] (h : P(A ∪ B) = P(A) + P(B) = 1) :
  (A ∩ B = ∅) ∧ ¬(A = Ω \ B ∧ B = Ω \ A) :=
by
  sorry

end mutually_exclusive_not_necessarily_complementary_l268_268204


namespace English_family_information_l268_268025

-- Define the statements given by the family members.
variables (father_statement : Prop)
          (mother_statement : Prop)
          (daughter_statement : Prop)

-- Conditions provided in the problem
variables (going_to_Spain : Prop)
          (coming_from_Newcastle : Prop)
          (stopped_in_Paris : Prop)

-- Define what each family member said
axiom Father : father_statement ↔ (going_to_Spain ∨ coming_from_Newcastle)
axiom Mother : mother_statement ↔ ((¬going_to_Spain ∧ coming_from_Newcastle) ∨ (stopped_in_Paris ∧ ¬going_to_Spain))
axiom Daughter : daughter_statement ↔ (¬coming_from_Newcastle ∨ stopped_in_Paris)

-- The final theorem to be proved:
theorem English_family_information : (¬going_to_Spain ∧ coming_from_Newcastle ∧ stopped_in_Paris) :=
by
  -- steps to prove the theorem should go here, but they are skipped with sorry
  sorry

end English_family_information_l268_268025


namespace num_mappings_from_A_to_B_l268_268154

set_option autoImplicit true

def A : Set ℕ := {1, 2}
def B : Set ℕ := {3, 4}

theorem num_mappings_from_A_to_B : (A → B).card = 2 ^ 2 :=
sorry

end num_mappings_from_A_to_B_l268_268154


namespace product_of_roots_l268_268940

theorem product_of_roots :
  (∏ n in Finset.range 2012, let eq := Quadratic (n+1) 1 (-n) in (eq.a * eq.b)) = -1/2012 :=
by sorry

end product_of_roots_l268_268940


namespace jellybean_total_count_l268_268432

theorem jellybean_total_count :
  let black := 8
  let green := 2 * black
  let orange := (2 * green) - 5
  let red := orange + 3
  let yellow := black / 2
  let purple := red + 4
  let brown := (green + purple) - 3
  black + green + orange + red + yellow + purple + brown = 166 := by
  -- skipping proof for brevity
  sorry

end jellybean_total_count_l268_268432


namespace grocery_cost_l268_268701

/-- Potatoes and celery costs problem. -/
theorem grocery_cost (a b : ℝ) (potato_cost_per_kg celery_cost_per_kg : ℝ) 
(h1 : potato_cost_per_kg = 1) (h2 : celery_cost_per_kg = 0.7) :
  potato_cost_per_kg * a + celery_cost_per_kg * b = a + 0.7 * b :=
by
  rw [h1, h2]
  sorry

end grocery_cost_l268_268701


namespace sin_225_eq_neg_sqrt2_div_2_l268_268484

noncomputable def sin_225_deg := real.sin (225 * real.pi / 180)

theorem sin_225_eq_neg_sqrt2_div_2 : sin_225_deg = -real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt2_div_2_l268_268484


namespace reciprocal_of_neg_2023_l268_268740

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l268_268740


namespace rationalize_sqrt_sum_l268_268737

noncomputable def rationalization_factor (a b : ℝ) : ℝ :=
if a ≥ b then sqrt a - sqrt b else sqrt b - sqrt a

theorem rationalize_sqrt_sum (a b : ℝ) :
  (rationalization_factor a b = sqrt a - sqrt b ∨ rationalization_factor a b = sqrt b - sqrt a) ∧
  ((sqrt a + sqrt b) * (sqrt a - sqrt b) = a - b ∨ (sqrt a + sqrt b) * (sqrt b - sqrt a) = b - a) :=
by
  split
  . unfold rationalization_factor
    split_ifs
    . left; refl
    . right; refl
  . unfold rationalization_factor
    split_ifs
    . left
      calc 
        (sqrt a + sqrt b) * (sqrt a - sqrt b)
          = a - b           : by ring
    . right
      calc
        (sqrt a + sqrt b) * (sqrt b - sqrt a)
          = -(a - b)       : by ring
          = b - a          : by ring

end rationalize_sqrt_sum_l268_268737


namespace difference_between_sum_and_difference_l268_268785

theorem difference_between_sum_and_difference :
  ∃ (x y : ℕ), 
    x * y = 2688 ∧ 
    x = 84 ∧ 
    ((x + y) - (x - y) = 64) :=
by
  use 84
  use 32
  sorry

end difference_between_sum_and_difference_l268_268785


namespace arithmetic_sequence_length_l268_268196

theorem arithmetic_sequence_length :
  ∃ n : ℕ, let a := 10 in
           let d := 5 in
           let l := 140 in
           a + (n - 1) * d = l ∧ n = 27 :=
by
  -- Proof is not required
  sorry

end arithmetic_sequence_length_l268_268196


namespace dan_worked_hours_l268_268890

theorem dan_worked_hours 
  (W : ℝ)
  (t : ℝ)
  (dan_work_rate : ℝ := 1 / 12)
  (annie_work_rate : ℝ := 1 / 9)
  (annie_time : ℝ := 3.0000000000000004)
  (total_work : ℝ := 1) : 
  t * dan_work_rate + annie_time * annie_work_rate = total_work → t = 8 :=
by
  intros h
  have eq1 : t * (1 / 12) = total_work - (annie_time * (1 / 9)), by sorry
  have eq2 : t * (1 / 12) = 1 - (3.0000000000000004 * (1 / 9)), by sorry
  have eq3 : t * (1 / 12) = 1 - (1 / 3), by sorry
  have eq4 : t * (1 / 12) = 2 / 3, by sorry
  have eq5 : t = (2 / 3) * 12, by sorry
  have eq6 : t = 8, by sorry
  exact eq6

end dan_worked_hours_l268_268890


namespace num_divisors_of_8_factorial_l268_268195

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the number of positive divisors function which uses prime factorization
noncomputable def num_divisors (n : ℕ) : ℕ :=
let factors := unique_factorization_monoid.factors n in
factors.to_finset.prod (λ p, factors.count p + 1)

-- Mathematical statement to prove
theorem num_divisors_of_8_factorial : num_divisors (factorial 8) = 96 := 
sorry

end num_divisors_of_8_factorial_l268_268195


namespace function_value_sum_l268_268265

namespace MathProof

variable {f : ℝ → ℝ}

theorem function_value_sum :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 5) = f x) →
  f (1 / 3) = 2022 →
  f (1 / 2) = 17 →
  f (-7) + f 12 + f (16 / 3) + f (9 / 2) = 2005 :=
by
  intros h_odd h_periodic h_f13 h_f12
  sorry

end MathProof

end function_value_sum_l268_268265


namespace root_condition_l268_268993

-- Let f(x) = x^2 + ax + a^2 - a - 2
noncomputable def f (a x : ℝ) : ℝ := x^2 + a * x + a^2 - a - 2

theorem root_condition (a : ℝ) (h1 : ∀ ζ : ℝ, (ζ > 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0) ∧ (ζ < 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0)) :
  -1 < a ∧ a < 1 :=
sorry

end root_condition_l268_268993


namespace quadratic_reciprocal_squares_l268_268889

theorem quadratic_reciprocal_squares :
  (∃ p q : ℝ, (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = p ∨ x = q)) ∧ (1 / p^2 + 1 / q^2 = 13 / 4)) :=
by
  have quadratic_eq : (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = 1 ∨ x = 2 / 3)) := sorry
  have identity_eq : 1 / (1:ℝ)^2 + 1 / (2 / 3)^2 = 13 / 4 := sorry
  exact ⟨1, 2 / 3, quadratic_eq, identity_eq⟩

end quadratic_reciprocal_squares_l268_268889


namespace shift_right_symmetric_l268_268182

open Real

/-- Given the function y = sin(2x + π/3), after shifting the graph of the function right
    by φ (0 < φ < π/2) units, the resulting graph is symmetric about the y-axis.
    Prove that the value of φ is 5π/12.
-/
theorem shift_right_symmetric (φ : ℝ) (hφ₁ : 0 < φ) (hφ₂ : φ < π / 2)
  (h_sym : ∃ k : ℤ, -2 * φ + π / 3 = k * π + π / 2) : φ = 5 * π / 12 :=
sorry

end shift_right_symmetric_l268_268182


namespace seconds_in_9_point_4_minutes_l268_268597

def seconds_in_minute : ℕ := 60
def minutes : ℝ := 9.4
def expected_seconds : ℝ := 564

theorem seconds_in_9_point_4_minutes : minutes * seconds_in_minute = expected_seconds :=
by 
  sorry

end seconds_in_9_point_4_minutes_l268_268597


namespace sin_225_correct_l268_268480

-- Define the condition of point being on the unit circle at 225 degrees.
noncomputable def P_225 := Complex.polar 1 (Real.pi + Real.pi / 4)

-- Define the goal statement that translates the question and correct answer.
theorem sin_225_correct : Complex.sin (Real.pi + Real.pi / 4) = -Real.sqrt 2 / 2 := 
by sorry

end sin_225_correct_l268_268480


namespace sqrt_eq_four_implies_x_eq_169_l268_268985

theorem sqrt_eq_four_implies_x_eq_169 (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 := by
  sorry

end sqrt_eq_four_implies_x_eq_169_l268_268985


namespace pythagorean_triple_exists_l268_268307

theorem pythagorean_triple_exists :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 = c^2 :=
begin
  use 12,
  use 16,
  use 20,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  norm_num,
end

end pythagorean_triple_exists_l268_268307


namespace expected_matches_is_one_variance_matches_is_one_l268_268420

noncomputable def indicator (k : ℕ) (matches : Finset ℕ) : ℕ :=
  if k ∈ matches then 1 else 0

def expected_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  (Finset.range N).sum (λ k, indicator k matches / N)

def variance_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  let E_S := expected_matches N matches in
  let E_S2 := (Finset.range N).sum (λ k, (indicator k matches) ^ 2 / N) in
  E_S2 - E_S ^ 2

theorem expected_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  expected_matches N matches = 1 := sorry

theorem variance_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  variance_matches N matches = 1 := sorry

end expected_matches_is_one_variance_matches_is_one_l268_268420


namespace simplify_expression_l268_268138

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
    (sqrt (a^3 * b)) / (cbrt (a * b)) = a^(7/6) * b^(1/6) := 
by 
  sorry

end simplify_expression_l268_268138


namespace max_largest_root_l268_268935

noncomputable def poly (a b c x : ℝ) := x^5 - 10 * x^3 + a * x^2 + b * x + c

theorem max_largest_root {a b c : ℝ} (h_real: ∀ x : ℝ, poly a b c x = 0 → real.root (poly a b c) x) :
  ∃ m : ℝ, ∀ r : ℝ, real.root (poly a b c) r → r ≤ m ∧ m = 4 :=
by
  sorry

end max_largest_root_l268_268935


namespace vector_at_t_zero_is_correct_l268_268042

open Matrix Real

-- The parameters for the problem
def v1 : Matrix (Fin 3) (Fin 1) ℝ := ![![2], ![-1], ![7]]
def v2 : Matrix (Fin 3) (Fin 1) ℝ := ![[-1], ![-6], ![-2]]

-- the vector at t = 0
def v0 : Matrix (Fin 3) (Fin 1) ℝ := ![![1/2], ![-7/2], ![5/2]]

theorem vector_at_t_zero_is_correct :
  v0 = 
  (let d := (v1 - v2) / 2 in
  v1 - d) := 
  sorry

end vector_at_t_zero_is_correct_l268_268042


namespace sphere_cap_cone_volume_eq_l268_268072

theorem sphere_cap_cone_volume_eq (R x : ℝ) (h1 : 0 < R) (h2 : 0 ≤ x) :
  (2 / 3) * real.pi * (R^2) * (R - x) = (1 / 3) * real.pi * ((R^2) - (x^2)) * x →
  x = R * (real.sqrt 5 - 1) / 2 :=
by
  sorry

end sphere_cap_cone_volume_eq_l268_268072


namespace find_min_n_for_integer_pq_l268_268250

-- Problem setup
def edge_length (n : ℕ) : ℝ := Real.sqrt n
def ps (n : ℕ) : ℝ := 4 * edge_length n
def sr (n : ℕ) : ℝ := 6 * edge_length n
def rq (n : ℕ) : ℝ := 4 * edge_length n
def pq_squared (n : ℕ) : ℝ := ps n ^ 2 + sr n ^ 2 + rq n ^ 2

-- The main theorem statement
theorem find_min_n_for_integer_pq : ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, pq_squared n = ↑(k^2) ∧ n = 17 :=
by
  -- Proof omitted
  sorry

end find_min_n_for_integer_pq_l268_268250


namespace shape_of_fixed_phi_l268_268539

open EuclideanGeometry

def spherical_coordinates (ρ θ φ : ℝ) : Point ℝ :=
  ⟨ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ⟩

theorem shape_of_fixed_phi (c : ℝ) :
    {p : Point ℝ | ∃ ρ θ, p = spherical_coordinates ρ θ c} = cone :=
by sorry

end shape_of_fixed_phi_l268_268539


namespace find_value_of_expression_l268_268930

theorem find_value_of_expression
  (x y : ℝ)
  (h : x^2 - 2*x + y^2 - 6*y + 10 = 0) :
  x^2 * y^2 + 2 * x * y + 1 = 16 :=
sorry

end find_value_of_expression_l268_268930


namespace painting_cost_is_88_l268_268454

noncomputable def total_painting_cost : Nat :=
  let east_addresses := List.range' 5 124 5
  let west_addresses := List.range' 2 96 4
  let num_digits (n: Nat) : Nat := n.digits.length
  let east_cost := east_addresses.sum (fun n => num_digits n)
  let west_cost := west_addresses.sum (fun n => num_digits n)
  east_cost + west_cost

theorem painting_cost_is_88 : total_painting_cost = 88 := 
  sorry

end painting_cost_is_88_l268_268454


namespace gcd_xyz_square_l268_268678

theorem gcd_xyz_square (x y z : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : ∃ n : ℕ, \gcd x y z * x * y * z = n^2 := 
sorry

end gcd_xyz_square_l268_268678


namespace rectangle_not_always_similar_l268_268809

theorem rectangle_not_always_similar :
  ∃ (r₁ r₂ : Type) [rectangle r₁] [rectangle r₂], 
  ¬ ((∀ (s₁ s₂ : ℝ), proportionally_corresponding_sides r₁ r₂ s₁ s₂ ∧ equal_corresponding_angles r₁ r₂) → similar r₁ r₂) :=
sorry

end rectangle_not_always_similar_l268_268809


namespace lcm_condition_l268_268163

theorem lcm_condition (m : ℕ) (h_m_pos : m > 0) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 36 :=
by
  sorry

end lcm_condition_l268_268163


namespace tenth_roll_is_last_l268_268991

def rollSequence (n : ℕ) : ℕ → Prop := fun r => r > 0 ∧ r ≤ 8

def isRepeatOnTenthRoll : Prop :=
  (∀ k, 1 ≤ k ∧ k ≤ 7 → (λ r, rollSequence 8 r ∧ r ≠ r.succ)) ∧ 
  (λ r, rollSequence 8 r ∧ r = r.succ)

noncomputable def probabilityOfTenthRollBeingLast : ℚ :=
  (7/8)^8 * (1/8)

theorem tenth_roll_is_last : 
  isRepeatOnTenthRoll → round (probabilityOfTenthRollBeingLast.toReal * 1000) / 1000 = 0.043 :=
by 
  sorry

end tenth_roll_is_last_l268_268991


namespace sum_first_n_arithmetic_sequence_l268_268158

theorem sum_first_n_arithmetic_sequence (a1 d : ℝ) (S : ℕ → ℝ) :
  (S 3 + S 6 = 18) → 
  S 3 = 3 * a1 + 3 * d → 
  S 6 = 6 * a1 + 15 * d → 
  S 5 = 10 :=
by
  sorry

end sum_first_n_arithmetic_sequence_l268_268158


namespace matrix_inverse_self_l268_268506

variable (a b : ℝ)

theorem matrix_inverse_self (h : matrix.of ![![4, -2], ![a, b]] ⬝ matrix.of ![![4, -2], ![a, b]] = 1) :
  a = 15 / 2 ∧ b = -4 :=
by
sorry

end matrix_inverse_self_l268_268506
