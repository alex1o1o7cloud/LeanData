import Mathlib

namespace value_of_f_m_plus_one_l149_14983

variable (a m : ℝ)

def f (x : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one 
  (h : f a (-m) < 0) : f a (m + 1) < 0 := by
  sorry

end value_of_f_m_plus_one_l149_14983


namespace no_real_roots_f_of_f_x_eq_x_l149_14947

theorem no_real_roots_f_of_f_x_eq_x (a b c : ℝ) (h: (b - 1)^2 - 4 * a * c < 0) : 
  ¬(∃ x : ℝ, (a * (a * x^2 + b * x + c)^2 + b * (a * x^2 + b * x + c) + c = x)) := 
by
  sorry

end no_real_roots_f_of_f_x_eq_x_l149_14947


namespace winning_strategy_ping_pong_l149_14978

theorem winning_strategy_ping_pong:
  ∀ {n : ℕ}, n = 18 → (∀ a : ℕ, 1 ≤ a ∧ a ≤ 4 → (∀ k : ℕ, k = 3 * a → (∃ b : ℕ, 1 ≤ b ∧ b ≤ 4 ∧ n - k - b = 18 - (k + b))) → (∃ c : ℕ, c = 3)) :=
by
sorry

end winning_strategy_ping_pong_l149_14978


namespace remuneration_difference_l149_14903

-- Define the conditions and question
def total_sales : ℝ := 12000
def commission_rate_old : ℝ := 0.05
def fixed_salary_new : ℝ := 1000
def commission_rate_new : ℝ := 0.025
def sales_threshold_new : ℝ := 4000

-- Define the remuneration for the old scheme
def remuneration_old : ℝ := total_sales * commission_rate_old

-- Define the remuneration for the new scheme
def sales_exceeding_threshold_new : ℝ := total_sales - sales_threshold_new
def commission_new : ℝ := sales_exceeding_threshold_new * commission_rate_new
def remuneration_new : ℝ := fixed_salary_new + commission_new

-- Statement of the theorem to be proved
theorem remuneration_difference : remuneration_new - remuneration_old = 600 :=
by
  -- The proof goes here but is omitted as per the instructions
  sorry

end remuneration_difference_l149_14903


namespace paired_divisors_prime_properties_l149_14908

theorem paired_divisors_prime_properties (n : ℕ) (h : n > 0) (h_pairing : ∃ (pairing : (ℕ × ℕ) → Prop), 
  (∀ d1 d2 : ℕ, 
    pairing (d1, d2) → d1 * d2 = n ∧ Prime (d1 + d2))) : 
  (∀ (d1 d2 : ℕ), d1 ≠ d2 → d1 + d2 ≠ d3 + d4) ∧ (∀ p : ℕ, Prime p → ¬ p ∣ n) :=
by
  sorry

end paired_divisors_prime_properties_l149_14908


namespace find_angle_NCB_l149_14913

def triangle_ABC_with_point_N (A B C N : Point) : Prop :=
  ∃ (angle_ABC angle_ACB angle_NAB angle_NBC : ℝ),
    angle_ABC = 50 ∧
    angle_ACB = 20 ∧
    angle_NAB = 40 ∧
    angle_NBC = 30 

theorem find_angle_NCB (A B C N : Point) 
  (h : triangle_ABC_with_point_N A B C N) :
  ∃ (angle_NCB : ℝ), 
  angle_NCB = 10 :=
sorry

end find_angle_NCB_l149_14913


namespace max_marks_l149_14958

theorem max_marks (M : ℝ) (h1 : 0.25 * M = 185 + 25) : M = 840 :=
by
  sorry

end max_marks_l149_14958


namespace initial_tabs_count_l149_14980

theorem initial_tabs_count (T : ℕ) (h1 : T > 0)
  (h2 : (3 / 4 : ℚ) * T - (2 / 5 : ℚ) * ((3 / 4 : ℚ) * T) > 0)
  (h3 : (9 / 20 : ℚ) * T - (1 / 2 : ℚ) * ((9 / 20 : ℚ) * T) = 90) :
  T = 400 :=
sorry

end initial_tabs_count_l149_14980


namespace fraction_subtraction_result_l149_14928

theorem fraction_subtraction_result :
  (3 * 5 + 5 * 7 + 7 * 9) / (2 * 4 + 4 * 6 + 6 * 8) - (2 * 4 + 4 * 6 + 6 * 8) / (3 * 5 + 5 * 7 + 7 * 9) = 74 / 119 :=
by sorry

end fraction_subtraction_result_l149_14928


namespace sum_a_b_is_95_l149_14911

-- Define the conditions
def product_condition (a b : ℕ) : Prop :=
  (a : ℤ) / 3 = 16 ∧ b = a - 1

-- Define the theorem to be proven
theorem sum_a_b_is_95 (a b : ℕ) (h : product_condition a b) : a + b = 95 :=
by
  sorry

end sum_a_b_is_95_l149_14911


namespace range_of_a_l149_14990

open Set

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 3 → x^2 - a * x - a + 1 ≥ 0) ↔ a ≤ 5 / 2 :=
sorry

end range_of_a_l149_14990


namespace maximum_triangle_area_within_circles_l149_14915

noncomputable def radius1 : ℕ := 71
noncomputable def radius2 : ℕ := 100
noncomputable def largest_triangle_area : ℕ := 24200

theorem maximum_triangle_area_within_circles : 
  ∃ (L : ℕ), L = largest_triangle_area ∧ 
             ∀ (r1 r2 : ℕ), r1 = radius1 → 
                             r2 = radius2 → 
                             L ≥ (r1 * r1 + 2 * r1 * r2) :=
by
  sorry

end maximum_triangle_area_within_circles_l149_14915


namespace woman_alone_days_l149_14973

theorem woman_alone_days (M W : ℝ) (h1 : (10 * M + 15 * W) * 5 = 1) (h2 : M * 100 = 1) : W * 150 = 1 :=
by
  sorry

end woman_alone_days_l149_14973


namespace victor_draw_order_count_l149_14995

-- Definitions based on the problem conditions
def num_piles : ℕ := 3
def num_cards_per_pile : ℕ := 3
def total_cards : ℕ := num_piles * num_cards_per_pile

-- The cardinality of the set of valid sequences where within each pile cards must be drawn in order
def valid_sequences_count : ℕ :=
  Nat.factorial total_cards / (Nat.factorial num_cards_per_pile ^ num_piles)

-- Now we state the problem: proving the valid sequences count is 1680
theorem victor_draw_order_count :
  valid_sequences_count = 1680 :=
by
  sorry

end victor_draw_order_count_l149_14995


namespace minimum_value_f_1_minimum_value_f_e_minimum_value_f_m_minus_1_range_of_m_l149_14942

noncomputable def f (x m : ℝ) : ℝ := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f_1 (m : ℝ) : (m ≤ 2) → f 1 m = 2 - m := sorry

theorem minimum_value_f_e (m : ℝ) : (m ≥ Real.exp 1 + 1) → f (Real.exp 1) m = Real.exp 1 - m - (m - 1) / Real.exp 1 := sorry

theorem minimum_value_f_m_minus_1 (m : ℝ) : (2 < m ∧ m < Real.exp 1 + 1) → 
  f (m - 1) m = m - 2 - m * Real.log (m - 1) := sorry

theorem range_of_m (m : ℝ) : 
  (m ≤ 2) → 
  (∃ x1 ∈ Set.Icc (Real.exp 1) (Real.exp 1 ^ 2), ∀ x2 ∈ Set.Icc (-2 : ℝ) 0, f x1 m ≤ g x2) → 
  Real.exp 1 - m - (m - 1) / Real.exp 1 ≤ 1 → 
  (m ≥ (Real.exp 1 ^ 2 - Real.exp 1 + 1) / (Real.exp 1 + 1) ∧ m ≤ 2) := sorry

end minimum_value_f_1_minimum_value_f_e_minimum_value_f_m_minus_1_range_of_m_l149_14942


namespace area_enclosed_by_circle_l149_14946

theorem area_enclosed_by_circle :
  let center := (3, -10)
  let radius := 3
  let equation := ∀ (x y : ℝ), (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2
  ∃ enclosed_area : ℝ, enclosed_area = 9 * Real.pi :=
by
  sorry

end area_enclosed_by_circle_l149_14946


namespace prime_square_sum_l149_14956

theorem prime_square_sum (p q m : ℕ) (hp : Prime p) (hq : Prime q) (hne : p ≠ q)
  (hp_eq : p^2 - 2001 * p + m = 0) (hq_eq : q^2 - 2001 * q + m = 0) :
  p^2 + q^2 = 3996005 :=
sorry

end prime_square_sum_l149_14956


namespace differential_system_solution_l149_14904

noncomputable def x (t : ℝ) := 1 - t - Real.exp (-6 * t) * Real.cos t
noncomputable def y (t : ℝ) := 1 - 7 * t + Real.exp (-6 * t) * Real.cos t + Real.exp (-6 * t) * Real.sin t

theorem differential_system_solution :
  (∀ t : ℝ, (deriv x t) = -7 * x t + y t + 5) ∧
  (∀ t : ℝ, (deriv y t) = -2 * x t - 5 * y t - 37 * t) ∧
  (x 0 = 0) ∧
  (y 0 = 0) :=
by 
  sorry

end differential_system_solution_l149_14904


namespace initial_speed_100_l149_14924

/-- Conditions of the problem:
1. The total distance from A to D is 100 km.
2. At point B, the navigator shows that 30 minutes are remaining.
3. At point B, the motorist reduces his speed by 10 km/h.
4. At point C, the navigator shows 20 km remaining, and the motorist again reduces his speed by 10 km/h.
5. The distance from C to D is 20 km.
6. The journey from B to C took 5 minutes longer than from C to D.
-/
theorem initial_speed_100 (x v : ℝ) (h1 : x = 100 - v / 2)
  (h2 : ∀ t, t = x / v)
  (h3 : ∀ t1 t2, t1 = (80 - x) / (v - 10) ∧ t2 = 20 / (v - 20))
  (h4 : (80 - x) / (v - 10) - 20 / (v - 20) = 1/12) :
  v = 100 := 
sorry

end initial_speed_100_l149_14924


namespace correct_system_of_equations_l149_14940

theorem correct_system_of_equations (x y : ℕ) :
  (8 * x - 3 = y ∧ 7 * x + 4 = y) ↔ 
  (8 * x - 3 = y ∧ 7 * x + 4 = y) := 
by 
  sorry

end correct_system_of_equations_l149_14940


namespace find_power_l149_14907

noncomputable def x : Real := 14.500000000000002
noncomputable def target : Real := 126.15

theorem find_power (n : Real) (h : (3/5) * x^n = target) : n = 2 :=
sorry

end find_power_l149_14907


namespace neg_3_14_gt_neg_pi_l149_14954

theorem neg_3_14_gt_neg_pi (π : ℝ) (h : 0 < π) : -3.14 > -π := 
sorry

end neg_3_14_gt_neg_pi_l149_14954


namespace ab_parallel_to_x_axis_and_ac_parallel_to_y_axis_l149_14932

theorem ab_parallel_to_x_axis_and_ac_parallel_to_y_axis
  (a b : ℝ)
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (a, -1))
  (hB : B = (2, 3 - b))
  (hC : C = (-5, 4))
  (hAB_parallel_x : A.2 = B.2)
  (hAC_parallel_y : A.1 = C.1) : a + b = -1 := by
  sorry


end ab_parallel_to_x_axis_and_ac_parallel_to_y_axis_l149_14932


namespace sum_of_interior_angles_of_decagon_l149_14931

def sum_of_interior_angles_of_polygon (n : ℕ) : ℕ := (n - 2) * 180

theorem sum_of_interior_angles_of_decagon : sum_of_interior_angles_of_polygon 10 = 1440 :=
by
  -- Proof goes here
  sorry

end sum_of_interior_angles_of_decagon_l149_14931


namespace roots_of_third_quadratic_l149_14953

/-- Given two quadratic equations with exactly one common root and a non-equal coefficient condition, 
prove that the other roots are roots of a third quadratic equation -/
theorem roots_of_third_quadratic 
  (a1 a2 a3 α β γ : ℝ)
  (h1 : α ≠ β)
  (h2 : α ≠ γ)
  (h3 : a1 ≠ a2)
  (h_eq1 : α^2 + a1*α + a2*a3 = 0)
  (h_eq2 : β^2 + a1*β + a2*a3 = 0)
  (h_eq3 : α^2 + a2*α + a1*a3 = 0)
  (h_eq4 : γ^2 + a2*γ + a1*a3 = 0) :
  β^2 + a3*β + a1*a2 = 0 ∧ γ^2 + a3*γ + a1*a2 = 0 :=
by
  sorry

end roots_of_third_quadratic_l149_14953


namespace number_of_people_adopting_cats_l149_14949

theorem number_of_people_adopting_cats 
    (initial_cats : ℕ)
    (monday_kittens : ℕ)
    (tuesday_injured_cat : ℕ)
    (final_cats : ℕ)
    (cats_per_person_adopting : ℕ)
    (h_initial : initial_cats = 20)
    (h_monday : monday_kittens = 2)
    (h_tuesday : tuesday_injured_cat = 1)
    (h_final: final_cats = 17)
    (h_cats_per_person: cats_per_person_adopting = 2) :
    ∃ (people_adopting : ℕ), people_adopting = 3 :=
by
  sorry

end number_of_people_adopting_cats_l149_14949


namespace min_value_of_a_l149_14930

noncomputable def f (x a : ℝ) : ℝ :=
  Real.exp x * (x + (3 / x) - 3) - (a / x)

noncomputable def g (x : ℝ) : ℝ :=
  (x^2 - 3 * x + 3) * Real.exp x

theorem min_value_of_a (a : ℝ) :
  (∃ x > 0, f x a ≤ 0) → a ≥ Real.exp 1 :=
by
  sorry

end min_value_of_a_l149_14930


namespace find_y_value_l149_14926

theorem find_y_value : (12^3 * 6^3 / 432) = 864 := by
  sorry

end find_y_value_l149_14926


namespace probability_is_correct_l149_14974

-- Define the ratios for the colors: red, yellow, blue, black
def red_ratio := 6
def yellow_ratio := 2
def blue_ratio := 1
def black_ratio := 4

-- Define the total ratio
def total_ratio := red_ratio + yellow_ratio + blue_ratio + black_ratio

-- Define the ratio of red or blue regions
def red_or_blue_ratio := red_ratio + blue_ratio

-- Define the probability of landing on a red or blue region
def probability_red_or_blue := red_or_blue_ratio / total_ratio

-- State the theorem to prove
theorem probability_is_correct : probability_red_or_blue = 7 / 13 := 
by 
  -- Proof will go here
  sorry

end probability_is_correct_l149_14974


namespace combined_population_lake_bright_and_sunshine_hills_l149_14910

theorem combined_population_lake_bright_and_sunshine_hills
  (p_toadon p_gordonia p_lake_bright p_riverbank p_sunshine_hills : ℕ)
  (h1 : p_toadon + p_gordonia + p_lake_bright + p_riverbank + p_sunshine_hills = 120000)
  (h2 : p_gordonia = 1 / 3 * 120000)
  (h3 : p_toadon = 3 / 4 * p_gordonia)
  (h4 : p_riverbank = p_toadon + 2 / 5 * p_toadon) :
  p_lake_bright + p_sunshine_hills = 8000 :=
by
  sorry

end combined_population_lake_bright_and_sunshine_hills_l149_14910


namespace proportion_equation_l149_14969

theorem proportion_equation (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
by 
  sorry

end proportion_equation_l149_14969


namespace evaluate_fraction_sum_l149_14921

theorem evaluate_fraction_sum : (5 / 50) + (4 / 40) + (6 / 60) = 0.3 :=
by
  sorry

end evaluate_fraction_sum_l149_14921


namespace union_complement_eq_l149_14917

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

theorem union_complement_eq :
  (complement U A ∪ B) = {2, 3, 4} :=
by
  sorry

end union_complement_eq_l149_14917


namespace part1_part2_l149_14976

variables (A B C : ℝ)
variables (a b c : ℝ) -- sides of the triangle opposite to angles A, B, and C respectively

-- Part (I): Prove that c / a = 2 given b(cos A - 2 * cos C) = (2 * c - a) * cos B
theorem part1 (h1 : b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B) : c / a = 2 :=
sorry

-- Part (II): Prove that b = 2 given the results from part (I) and additional conditions
theorem part2 (h1 : c / a = 2) (h2 : Real.cos B = 1 / 4) (h3 : a + b + c = 5) : b = 2 :=
sorry

end part1_part2_l149_14976


namespace square_pizza_area_larger_by_27_percent_l149_14964

theorem square_pizza_area_larger_by_27_percent :
  let r := 5
  let A_circle := Real.pi * r^2
  let s := 2 * r
  let A_square := s^2
  let delta_A := A_square - A_circle
  let percent_increase := (delta_A / A_circle) * 100
  Int.floor (percent_increase + 0.5) = 27 :=
by
  sorry

end square_pizza_area_larger_by_27_percent_l149_14964


namespace even_function_and_inverse_property_l149_14997

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem even_function_and_inverse_property (x : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  f (-x) = f x ∧ f (1 / x) = -f x := by
  sorry

end even_function_and_inverse_property_l149_14997


namespace minimum_prism_volume_l149_14988

theorem minimum_prism_volume (l m n : ℕ) (h1 : l > 0) (h2 : m > 0) (h3 : n > 0)
    (hidden_volume_condition : (l - 1) * (m - 1) * (n - 1) = 420) :
    ∃ N : ℕ, N = l * m * n ∧ N = 630 := by
  sorry

end minimum_prism_volume_l149_14988


namespace scientific_notation_suzhou_blood_donors_l149_14977

theorem scientific_notation_suzhou_blood_donors : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 124000 = a * 10^n ∧ a = 1.24 ∧ n = 5 :=
by
  sorry

end scientific_notation_suzhou_blood_donors_l149_14977


namespace percent_equivalence_l149_14994

theorem percent_equivalence (x : ℝ) : (0.6 * 0.3 * x - 0.1 * x) / x * 100 = 8 := by
  sorry

end percent_equivalence_l149_14994


namespace moon_speed_conversion_correct_l149_14993

-- Define the conversions
def kilometers_per_second_to_miles_per_hour (kmps : ℝ) : ℝ :=
  kmps * 0.621371 * 3600

-- Condition: The moon's speed
def moon_speed_kmps : ℝ := 1.02

-- Correct answer in miles per hour
def expected_moon_speed_mph : ℝ := 2281.34

-- Theorem stating the equivalence of converted speed to expected speed
theorem moon_speed_conversion_correct :
  kilometers_per_second_to_miles_per_hour moon_speed_kmps = expected_moon_speed_mph :=
by 
  sorry

end moon_speed_conversion_correct_l149_14993


namespace map_length_25_cm_represents_125_km_l149_14982

-- Define the conditions
def map_scale (cm: ℝ) : ℝ := 5 * cm

-- Define the main statement to be proved
theorem map_length_25_cm_represents_125_km : map_scale 25 = 125 := by
  sorry

end map_length_25_cm_represents_125_km_l149_14982


namespace molecular_weight_correct_l149_14955

-- Atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 15.999
def atomic_weight_H : ℝ := 1.008

-- Number of each type of atom in the compound
def num_Al : ℕ := 1
def num_O : ℕ := 3
def num_H : ℕ := 3

-- Molecular weight calculation
def molecular_weight : ℝ :=
  (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H)

theorem molecular_weight_correct : molecular_weight = 78.001 := by
  sorry

end molecular_weight_correct_l149_14955


namespace paint_intensity_l149_14901

theorem paint_intensity (I : ℝ) (F : ℝ) (I_initial I_new : ℝ) : 
  I_initial = 50 → I_new = 30 → F = 2 / 3 → I = 20 :=
by
  intros h1 h2 h3
  sorry

end paint_intensity_l149_14901


namespace place_integers_on_cube_l149_14920

theorem place_integers_on_cube:
  ∃ (A B C D A₁ B₁ C₁ D₁ : ℤ),
    A = B + D + A₁ ∧ 
    B = A + C + B₁ ∧ 
    C = B + D + C₁ ∧ 
    D = A + C + D₁ ∧ 
    A₁ = B₁ + D₁ + A ∧ 
    B₁ = A₁ + C₁ + B ∧ 
    C₁ = B₁ + D₁ + C ∧ 
    D₁ = A₁ + C₁ + D :=
sorry

end place_integers_on_cube_l149_14920


namespace m_perpendicular_beta_l149_14965

variables {Plane : Type*} {Line : Type*}

-- Definitions of the perpendicularity and parallelism
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Given variables
variables (α β : Plane) (m : Line)

-- Conditions
axiom M_perpendicular_Alpha : perpendicular m α
axiom Alpha_parallel_Beta : parallel α β

-- Proof goal
theorem m_perpendicular_beta 
  (h1 : perpendicular m α) 
  (h2 : parallel α β) : 
  perpendicular m β := 
  sorry

end m_perpendicular_beta_l149_14965


namespace smallest_number_l149_14992

theorem smallest_number (a b c d : ℤ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -1) (h4 : d = -3) :
  d = -3 ∧ d < c ∧ d < b ∧ d < a :=
by
  sorry

end smallest_number_l149_14992


namespace remaining_volume_after_pours_l149_14981

-- Definitions based on the problem conditions
def initial_volume_liters : ℝ := 2
def initial_volume_milliliters : ℝ := initial_volume_liters * 1000
def pour_amount (x : ℝ) : ℝ := x

-- Statement of the problem as a theorem in Lean 4
theorem remaining_volume_after_pours (x : ℝ) : 
  ∃ remaining_volume : ℝ, remaining_volume = initial_volume_milliliters - 4 * pour_amount x :=
by
  -- To be filled with the proof
  sorry

end remaining_volume_after_pours_l149_14981


namespace negation_of_cos_proposition_l149_14979

variable (x : ℝ)

theorem negation_of_cos_proposition (h : ∀ x : ℝ, Real.cos x ≤ 1) : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end negation_of_cos_proposition_l149_14979


namespace tan_neg_405_eq_one_l149_14939

theorem tan_neg_405_eq_one : Real.tan (-(405 * Real.pi / 180)) = 1 :=
by
-- Proof omitted
sorry

end tan_neg_405_eq_one_l149_14939


namespace find_ab_l149_14975

theorem find_ab 
(a b : ℝ) 
(h1 : a + b = 2) 
(h2 : a * b = 1 ∨ a * b = -1) :
(a = 1 ∧ b = 1) ∨
(a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
(a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
sorry

end find_ab_l149_14975


namespace m_divides_product_iff_composite_ne_4_l149_14960

theorem m_divides_product_iff_composite_ne_4 (m : ℕ) : 
  (m ∣ Nat.factorial (m - 1)) ↔ 
  (∃ a b : ℕ, a ≠ b ∧ 1 < a ∧ 1 < b ∧ m = a * b ∧ m ≠ 4) := 
sorry

end m_divides_product_iff_composite_ne_4_l149_14960


namespace intersection_A_B_l149_14929

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_A_B : (A ∩ B) = { x | 0 < x ∧ x ≤ 1 } := by
  sorry

end intersection_A_B_l149_14929


namespace shifted_function_is_correct_l149_14998

-- Given conditions
def original_function (x : ℝ) : ℝ := -(x + 2) ^ 2 + 1

def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Resulting function after shifting 1 unit to the right
def shifted_function : ℝ → ℝ := shift_right original_function 1

-- Correct answer
def correct_function (x : ℝ) : ℝ := -(x + 1) ^ 2 + 1

-- Proof Statement
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = correct_function x := by
  sorry

end shifted_function_is_correct_l149_14998


namespace radius_of_circumcircle_of_triangle_l149_14996

theorem radius_of_circumcircle_of_triangle (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (∃ (R : ℝ), R = 2.5) :=
by {
  sorry
}

end radius_of_circumcircle_of_triangle_l149_14996


namespace length_of_goods_train_l149_14966

theorem length_of_goods_train 
  (speed_kmh : ℕ) 
  (platform_length_m : ℕ) 
  (cross_time_s : ℕ) :
  speed_kmh = 72 → platform_length_m = 280 → cross_time_s = 26 → 
  ∃ train_length_m : ℕ, train_length_m = 240 :=
by
  intros h1 h2 h3
  sorry

end length_of_goods_train_l149_14966


namespace gen_formula_is_arith_seq_l149_14909

-- Given: The sum of the first n terms of the sequence {a_n} is S_n = n^2 + 2n
def sum_seq (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 2 * n

-- The general formula for {a_n} is a_n = 2n + 1
theorem gen_formula (S : ℕ → ℕ) (h : sum_seq S) : ∀ n : ℕ,  n > 0 → (∃ a : ℕ → ℕ, a n = 2 * n + 1 ∧ ∀ m : ℕ, m < n → a m = S (m + 1) - S m) :=
by sorry

-- The sequence {a_n} defined by a_n = 2n + 1 is an arithmetic sequence
theorem is_arith_seq : ∀ n : ℕ, n > 0 → (∀ a : ℕ → ℕ, (∀ k, k > 0 → a k = 2 * k + 1) → ∃ d : ℕ, d = 2 ∧ ∀ j > 0, a j - a (j - 1) = d) :=
by sorry

end gen_formula_is_arith_seq_l149_14909


namespace x_minus_y_values_l149_14952

theorem x_minus_y_values (x y : ℝ) (h₁ : |x + 1| = 4) (h₂ : (y + 2)^2 = 4) (h₃ : x + y ≥ -5) :
  x - y = -5 ∨ x - y = 3 ∨ x - y = 7 :=
by
  sorry

end x_minus_y_values_l149_14952


namespace find_p0_over_q0_l149_14936

-- Definitions

def p (x : ℝ) := 3 * (x - 4) * (x - 2)
def q (x : ℝ) := (x - 4) * (x + 3)

theorem find_p0_over_q0 : (p 0) / (q 0) = -2 :=
by
  -- Prove the equality given the conditions
  sorry

end find_p0_over_q0_l149_14936


namespace alcohol_water_ratio_l149_14948

theorem alcohol_water_ratio (a b : ℚ) (h₀ : a > 0) (h₁ : b > 0) :
  (3 * a / (a + 2) + 8 / (4 + b)) / (6 / (a + 2) + 2 * b / (4 + b)) = (3 * a + 8) / (6 + 2 * b) :=
by
  sorry

end alcohol_water_ratio_l149_14948


namespace cyclist_motorcyclist_intersection_l149_14914

theorem cyclist_motorcyclist_intersection : 
  ∃ t : ℝ, (4 * t^2 + (t - 1)^2 - 2 * |t| * |t - 1| = 49) ∧ (t = 4 ∨ t = -4) := 
by 
  sorry

end cyclist_motorcyclist_intersection_l149_14914


namespace roots_of_quadratic_eq_l149_14945

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l149_14945


namespace find_y_value_l149_14944

theorem find_y_value (y : ℝ) (h : 1 / (3 + 1 / (3 + 1 / (3 - y))) = 0.30337078651685395) : y = 0.3 :=
sorry

end find_y_value_l149_14944


namespace line_eq_489_l149_14951

theorem line_eq_489 (m b : ℤ) (h1 : m = 5) (h2 : 3 = m * 5 + b) : m + b^2 = 489 :=
by
  sorry

end line_eq_489_l149_14951


namespace factorization_correct_l149_14934

theorem factorization_correct :
  (¬ (x^2 - 2 * x - 1 = x * (x - 2) - 1)) ∧
  (¬ (2 * x + 1 = x * (2 + 1 / x))) ∧
  (¬ ((x + 2) * (x - 2) = x^2 - 4)) ∧
  (x^2 - 1 = (x + 1) * (x - 1)) :=
by
  sorry

end factorization_correct_l149_14934


namespace find_radius_of_tangent_circle_l149_14961

def tangent_circle_radius : Prop :=
  ∃ (r : ℝ), 
    (r > 0) ∧ 
    (∀ (θ : ℝ),
      (∃ (x y : ℝ),
        x = 1 + r * Real.cos θ ∧ 
        y = 1 + r * Real.sin θ ∧ 
        x + y - 1 = 0))
    → r = (Real.sqrt 2) / 2

theorem find_radius_of_tangent_circle : tangent_circle_radius :=
sorry

end find_radius_of_tangent_circle_l149_14961


namespace num_men_employed_l149_14916

noncomputable def original_number_of_men (M : ℕ) : Prop :=
  let total_work_original := M * 5
  let total_work_actual := (M - 8) * 15
  total_work_original = total_work_actual

theorem num_men_employed (M : ℕ) (h : original_number_of_men M) : M = 12 :=
by sorry

end num_men_employed_l149_14916


namespace value_of_f_f_f_2_l149_14972

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem value_of_f_f_f_2 : f (f (f 2)) = 2 :=
by {
  sorry
}

end value_of_f_f_f_2_l149_14972


namespace solve_quadratic_equation_l149_14963

theorem solve_quadratic_equation :
  ∀ x : ℝ, (10 - x) ^ 2 = 2 * x ^ 2 + 4 * x ↔ x = 3.62 ∨ x = -27.62 := by
  sorry

end solve_quadratic_equation_l149_14963


namespace nick_charges_l149_14912

theorem nick_charges (y : ℕ) :
  let travel_cost := 7
  let hourly_rate := 10
  10 * y + 7 = travel_cost + hourly_rate * y :=
by sorry

end nick_charges_l149_14912


namespace part1_max_area_part2_find_a_l149_14957

-- Part (1): Define the function and prove maximum area of the triangle
noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp x - 3 * a * x + 2 * Real.sin x - 1

theorem part1_max_area (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let f' := a^2 - 3 * a + 2
  ∃ h_a_max, h_a_max == 3 / 8 :=
  sorry

-- Part (2): Prove that the function reaches an extremum at x = 0 and determine the value of a.
theorem part2_find_a (a : ℝ) : (a^2 - 3 * a + 2 = 0) → (a = 1 ∨ a = 2) :=
  sorry

end part1_max_area_part2_find_a_l149_14957


namespace triangle_length_l149_14991

theorem triangle_length (DE DF : ℝ) (Median_to_EF : ℝ) (EF : ℝ) :
  DE = 2 ∧ DF = 3 ∧ Median_to_EF = EF → EF = (13:ℝ).sqrt / 5 := by
  sorry

end triangle_length_l149_14991


namespace obtuse_triangle_two_acute_angles_l149_14989

-- Define the angle type (could be Real between 0 and 180 in degrees).
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define an obtuse triangle using three angles α, β, γ
structure obtuse_triangle :=
(angle1 angle2 angle3 : ℝ)
(sum_angles_eq : angle1 + angle2 + angle3 = 180)
(obtuse_condition : is_obtuse angle1 ∨ is_obtuse angle2 ∨ is_obtuse angle3)

-- The theorem to prove the number of acute angles in an obtuse triangle is 2.
theorem obtuse_triangle_two_acute_angles (T : obtuse_triangle) : 
  (is_acute T.angle1 ∧ is_acute T.angle2 ∧ ¬ is_acute T.angle3) ∨ 
  (is_acute T.angle1 ∧ ¬ is_acute T.angle2 ∧ is_acute T.angle3) ∨ 
  (¬ is_acute T.angle1 ∧ is_acute T.angle2 ∧ is_acute T.angle3) :=
by sorry

end obtuse_triangle_two_acute_angles_l149_14989


namespace largest_number_in_L_shape_l149_14986

theorem largest_number_in_L_shape (x : ℤ) (sum : ℤ) (h : sum = 2015) : x = 676 :=
by
  sorry

end largest_number_in_L_shape_l149_14986


namespace find_eccentricity_l149_14959

-- Define the hyperbola structure
structure Hyperbola where
  a : ℝ
  b : ℝ
  (a_pos : 0 < a)
  (b_pos : 0 < b)

-- Define the point P and focus F₁ F₂ relationship
structure PointsRelation (C : Hyperbola) where
  P : ℝ × ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  (distance_condition : dist P F1 = 3 * dist P F2)
  (dot_product_condition : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = C.a^2)

noncomputable def eccentricity (C : Hyperbola) (rel : PointsRelation C) : ℝ :=
  Real.sqrt (1 + (C.b ^ 2) / (C.a ^ 2))

theorem find_eccentricity (C : Hyperbola) (rel : PointsRelation C) : eccentricity C rel = Real.sqrt 2 := by
  sorry

end find_eccentricity_l149_14959


namespace function_range_x2_minus_2x_l149_14900

theorem function_range_x2_minus_2x : 
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 3 → -1 ≤ x^2 - 2 * x ∧ x^2 - 2 * x ≤ 3 :=
by
  intro x hx
  sorry

end function_range_x2_minus_2x_l149_14900


namespace find_ethanol_percentage_l149_14987

noncomputable def ethanol_percentage_in_fuel_A (P_A : ℝ) (V_A : ℝ) : Prop :=
  (P_A / 100) * V_A + 0.16 * (200 - V_A) = 18

theorem find_ethanol_percentage (P_A : ℝ) (V_A : ℝ) (h₀ : V_A ≤ 200) (h₁ : 0 ≤ V_A) :
  ethanol_percentage_in_fuel_A P_A V_A :=
by
  sorry

end find_ethanol_percentage_l149_14987


namespace cube_sum_l149_14923

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l149_14923


namespace alice_number_l149_14968

theorem alice_number (n : ℕ) 
  (h1 : 243 ∣ n) 
  (h2 : 36 ∣ n) 
  (h3 : 1000 < n) 
  (h4 : n < 3000) : 
  n = 1944 ∨ n = 2916 := 
sorry

end alice_number_l149_14968


namespace area_calculation_l149_14999

variable (x : ℝ)

def area_large_rectangle : ℝ := (2 * x + 9) * (x + 6)
def area_rectangular_hole : ℝ := (x - 1) * (2 * x - 5)
def area_square : ℝ := (x + 3) ^ 2
def area_remaining : ℝ := area_large_rectangle x - area_rectangular_hole x - area_square x

theorem area_calculation : area_remaining x = -x^2 + 22 * x + 40 := by
  sorry

end area_calculation_l149_14999


namespace solve_for_z_l149_14937

theorem solve_for_z (a b s z : ℝ) (h1 : z ≠ 0) (h2 : 1 - 6 * s ≠ 0) (h3 : z = a^3 * b^2 + 6 * z * s - 9 * s^2) :
  z = (a^3 * b^2 - 9 * s^2) / (1 - 6 * s) := 
 by
  sorry

end solve_for_z_l149_14937


namespace luke_total_points_l149_14950

-- Definitions based on conditions
def points_per_round : ℕ := 3
def rounds_played : ℕ := 26

-- Theorem stating the question and correct answer
theorem luke_total_points : points_per_round * rounds_played = 78 := 
by 
  sorry

end luke_total_points_l149_14950


namespace ratio_initial_to_doubled_l149_14933

theorem ratio_initial_to_doubled (x : ℝ) (h : 3 * (2 * x + 8) = 84) : x / (2 * x) = 1 / 2 :=
by
  have h1 : 2 * x + 8 = 28 := by
    sorry
  have h2 : x = 10 := by
    sorry
  rw [h2]
  norm_num

end ratio_initial_to_doubled_l149_14933


namespace tan_sum_pi_over_12_l149_14970

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l149_14970


namespace simplify_expression_l149_14985

variables {x y : ℝ}
-- Ensure that x and y are not zero to avoid division by zero errors.
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
sorry

end simplify_expression_l149_14985


namespace investment_total_l149_14927

theorem investment_total (x y : ℝ) (h₁ : 0.08 * x + 0.05 * y = 490) (h₂ : x = 3000 ∨ y = 3000) : x + y = 8000 :=
by
  sorry

end investment_total_l149_14927


namespace exponent_equivalence_l149_14971

theorem exponent_equivalence (a b : ℕ) (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) (h1 : 9 ^ m = a) (h2 : 3 ^ n = b) : 
  3 ^ (2 * m + 4 * n) = a * b ^ 4 := 
by 
  sorry

end exponent_equivalence_l149_14971


namespace right_triangle_of_angle_condition_l149_14922

-- Defining the angles of the triangle
variables (α β γ : ℝ)

-- Defining the condition where the sum of angles in a triangle is 180 degrees
def sum_of_angles_in_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Defining the given condition 
def angle_condition (γ α β : ℝ) : Prop :=
  γ = α + β

-- Stating the theorem to be proved
theorem right_triangle_of_angle_condition (α β γ : ℝ) :
  sum_of_angles_in_triangle α β γ → angle_condition γ α β → γ = 90 :=
by
  intro hsum hcondition
  sorry

end right_triangle_of_angle_condition_l149_14922


namespace each_child_ate_3_jellybeans_l149_14905

-- Define the given conditions
def total_jellybeans : ℕ := 100
def total_kids : ℕ := 24
def sick_kids : ℕ := 2
def leftover_jellybeans : ℕ := 34

-- Calculate the number of kids who attended
def attending_kids : ℕ := total_kids - sick_kids

-- Calculate the total jellybeans eaten
def total_jellybeans_eaten : ℕ := total_jellybeans - leftover_jellybeans

-- Calculate the number of jellybeans each child ate
def jellybeans_per_child : ℕ := total_jellybeans_eaten / attending_kids

theorem each_child_ate_3_jellybeans : jellybeans_per_child = 3 :=
by sorry

end each_child_ate_3_jellybeans_l149_14905


namespace integer_solution_l149_14935

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n^2 > -27) : n = 2 :=
by {
  sorry
}

end integer_solution_l149_14935


namespace hair_ratio_l149_14918

theorem hair_ratio (washed : ℕ) (grow_back : ℕ) (brushed : ℕ) (n : ℕ)
  (hwashed : washed = 32)
  (hgrow_back : grow_back = 49)
  (heq : washed + brushed + 1 = grow_back) :
  (brushed : ℚ) / washed = 1 / 2 := 
by 
  sorry

end hair_ratio_l149_14918


namespace part1_part2_l149_14943

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin ((1 / 3) * x - (Real.pi / 6))

theorem part1 : f (5 * Real.pi / 4) = Real.sqrt 2 :=
by sorry

theorem part2 (α β : ℝ) (hαβ : 0 ≤ α ∧ α ≤ Real.pi / 2 ∧ 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h1: f (3 * α + Real.pi / 2) = 10 / 13) (h2: f (3 * β + 2 * Real.pi) = 6 / 5) :
  Real.cos (α + β) = 16 / 65 :=
by sorry

end part1_part2_l149_14943


namespace average_age_6_members_birth_correct_l149_14919

/-- The average age of 7 members of a family is 29 years. -/
def average_age_7_members := 29

/-- The present age of the youngest member is 5 years. -/
def age_youngest_member := 5

/-- Total age of 7 members of the family -/
def total_age_7_members := 7 * average_age_7_members

/-- Total age of 6 members at present -/
def total_age_6_members_present := total_age_7_members - age_youngest_member

/-- Total age of 6 members at time of birth of youngest member -/
def total_age_6_members_birth := total_age_6_members_present - (6 * age_youngest_member)

/-- Average age of 6 members at time of birth of youngest member -/
def average_age_6_members_birth := total_age_6_members_birth / 6

/-- Prove the average age of 6 members at the time of birth of the youngest member -/
theorem average_age_6_members_birth_correct :
  average_age_6_members_birth = 28 :=
by
  sorry

end average_age_6_members_birth_correct_l149_14919


namespace angle_A_is_30_degrees_l149_14962

theorem angle_A_is_30_degrees
    (a b : ℝ)
    (B A : ℝ)
    (a_eq_4 : a = 4)
    (b_eq_4_sqrt2 : b = 4 * Real.sqrt 2)
    (B_eq_45 : B = Real.pi / 4) : 
    A = Real.pi / 6 := 
by 
    sorry

end angle_A_is_30_degrees_l149_14962


namespace responses_needed_l149_14984

noncomputable def Q : ℝ := 461.54
noncomputable def percentage : ℝ := 0.65
noncomputable def required_responses : ℝ := percentage * Q

theorem responses_needed : required_responses = 300 := by
  sorry

end responses_needed_l149_14984


namespace sale_price_of_trouser_l149_14902

theorem sale_price_of_trouser (original_price : ℝ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) (h2 : discount_percentage = 0.5) : sale_price = 50 :=
by
  sorry

end sale_price_of_trouser_l149_14902


namespace min_a5_of_geom_seq_l149_14941

-- Definition of geometric sequence positivity and difference condition.
def geom_seq_pos_diff (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 3 - a 1 = 2)

-- The main theorem stating that the minimum value of a_5 is 8.
theorem min_a5_of_geom_seq {a : ℕ → ℝ} {q : ℝ} (h : geom_seq_pos_diff a q) :
  a 5 ≥ 8 :=
sorry

end min_a5_of_geom_seq_l149_14941


namespace max_value_of_x1_squared_plus_x2_squared_l149_14938

theorem max_value_of_x1_squared_plus_x2_squared :
  ∀ (k : ℝ), -4 ≤ k ∧ k ≤ -4 / 3 → (∃ x1 x2 : ℝ, x1^2 + x2^2 = 18) :=
by
  sorry

end max_value_of_x1_squared_plus_x2_squared_l149_14938


namespace angle_C_is_30_degrees_l149_14967

theorem angle_C_is_30_degrees
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (A_rad: 0 ≤ A ∧ A ≤ Real.pi)
  (B_rad: 0 ≤ B ∧ B ≤ Real.pi)
  (C_rad : 0 ≤ C ∧ C ≤ Real.pi)
  (triangle_condition: A + B + C = Real.pi) :
  C = Real.pi / 6 :=
sorry

end angle_C_is_30_degrees_l149_14967


namespace roots_reciprocal_l149_14925

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 4 * x1 - 2 = 0) (h2 : x2^2 - 4 * x2 - 2 = 0) (h3 : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = -2 := 
sorry

end roots_reciprocal_l149_14925


namespace sin_double_angle_l149_14906

theorem sin_double_angle (α : ℝ) (h_tan : Real.tan α < 0) (h_sin : Real.sin α = - (Real.sqrt 3) / 3) :
  Real.sin (2 * α) = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end sin_double_angle_l149_14906
