import Mathlib

namespace NUMINAMATH_GPT_tetrahedron_circumscribed_sphere_radius_l1439_143900

open Real

theorem tetrahedron_circumscribed_sphere_radius :
  ∀ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 5 →
    dist C D = 5 →
    dist A C = sqrt 34 →
    dist B D = sqrt 34 →
    dist A D = sqrt 41 →
    dist B C = sqrt 41 →
    ∃ (R : ℝ), R = 5 * sqrt 2 / 2 :=
by
  intros A B C D hAB hCD hAC hBD hAD hBC
  sorry

end NUMINAMATH_GPT_tetrahedron_circumscribed_sphere_radius_l1439_143900


namespace NUMINAMATH_GPT_quadratic_intersection_l1439_143932

theorem quadratic_intersection
  (a b c d h : ℝ)
  (h_a : a ≠ 0)
  (h_b : b ≠ 0)
  (h_h : h ≠ 0)
  (h_d : d ≠ c) :
  ∃ x y : ℝ, (y = a * x^2 + b * x + c) ∧ (y = a * (x - h)^2 + b * (x - h) + d)
    ∧ x = (d - c) / b
    ∧ y = a * (d - c)^2 / b^2 + d :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_intersection_l1439_143932


namespace NUMINAMATH_GPT_math_competition_probs_l1439_143970

-- Definitions related to the problem conditions
def boys : ℕ := 3
def girls : ℕ := 3
def total_students := boys + girls
def total_combinations := (total_students.choose 2)

-- Definition of the probabilities
noncomputable def prob_exactly_one_boy : ℚ := 0.6
noncomputable def prob_at_least_one_boy : ℚ := 0.8
noncomputable def prob_at_most_one_boy : ℚ := 0.8

-- Lean statement for the proof problem
theorem math_competition_probs :
  prob_exactly_one_boy = 0.6 ∧
  prob_at_least_one_boy = 0.8 ∧
  prob_at_most_one_boy = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_math_competition_probs_l1439_143970


namespace NUMINAMATH_GPT_find_f_2008_l1439_143975

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the problem statement with all given conditions
theorem find_f_2008 (h_odd : is_odd f) (h_f2 : f 2 = 0) (h_rec : ∀ x, f (x + 4) = f x + f 4) : f 2008 = 0 := 
sorry

end NUMINAMATH_GPT_find_f_2008_l1439_143975


namespace NUMINAMATH_GPT_percentage_increase_in_cellphone_pay_rate_l1439_143991

theorem percentage_increase_in_cellphone_pay_rate
    (regular_rate : ℝ)
    (total_surveys : ℕ)
    (cellphone_surveys : ℕ)
    (total_earnings : ℝ)
    (regular_surveys : ℕ := total_surveys - cellphone_surveys)
    (higher_rate : ℝ := (total_earnings - (regular_surveys * regular_rate)) / cellphone_surveys)
    : regular_rate = 30 ∧ total_surveys = 100 ∧ cellphone_surveys = 50 ∧ total_earnings = 3300
    → ((higher_rate - regular_rate) / regular_rate) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_cellphone_pay_rate_l1439_143991


namespace NUMINAMATH_GPT_average_speed_l1439_143978

-- Define the conditions
def distance1 := 350 -- miles
def time1 := 6 -- hours
def distance2 := 420 -- miles
def time2 := 7 -- hours

-- Define the total distance and total time (excluding break)
def total_distance := distance1 + distance2
def total_time := time1 + time2

-- Define the statement to prove
theorem average_speed : 
  (total_distance / total_time : ℚ) = 770 / 13 := by
  sorry

end NUMINAMATH_GPT_average_speed_l1439_143978


namespace NUMINAMATH_GPT_product_or_double_is_perfect_square_l1439_143902

variable {a b c : ℤ}

-- Conditions
def sides_of_triangle (a b c : ℤ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def no_common_divisor (a b c : ℤ) : Prop := gcd (gcd a b) c = 1

def all_fractions_are_integers (a b c : ℤ) : Prop :=
  (a + b - c) ≠ 0 ∧ (b + c - a) ≠ 0 ∧ (c + a - b) ≠ 0 ∧
  ((a^2 + b^2 - c^2) % (a + b - c) = 0) ∧ 
  ((b^2 + c^2 - a^2) % (b + c - a) = 0) ∧ 
  ((c^2 + a^2 - b^2) % (c + a - b) = 0)

-- Mathematical proof problem statement in Lean 4
theorem product_or_double_is_perfect_square (a b c : ℤ) 
  (h1 : sides_of_triangle a b c)
  (h2 : no_common_divisor a b c)
  (h3 : all_fractions_are_integers a b c) :
  ∃ k : ℤ, k^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
           k^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := sorry

end NUMINAMATH_GPT_product_or_double_is_perfect_square_l1439_143902


namespace NUMINAMATH_GPT_factorization_a_minus_b_l1439_143987

theorem factorization_a_minus_b (a b: ℤ) 
  (h : (4 * y + a) * (y + b) = 4 * y * y - 3 * y - 28) : a - b = -11 := by
  sorry

end NUMINAMATH_GPT_factorization_a_minus_b_l1439_143987


namespace NUMINAMATH_GPT_area_of_triangle_l1439_143909

theorem area_of_triangle (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 144) : 
  1/2 * a * b = 30 :=
by sorry

end NUMINAMATH_GPT_area_of_triangle_l1439_143909


namespace NUMINAMATH_GPT_g_at_3_l1439_143944

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2

theorem g_at_3 : g 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_g_at_3_l1439_143944


namespace NUMINAMATH_GPT_min_distance_from_P_to_origin_l1439_143939

noncomputable def distance_to_origin : ℝ := 8 / 5

theorem min_distance_from_P_to_origin
  (P : ℝ × ℝ)
  (hA : P.1^2 + P.2^2 = 1)
  (hB : (P.1 - 3)^2 + (P.2 + 4)^2 = 10)
  (h_tangent : PE = PD) :
  dist P (0, 0) = distance_to_origin := 
sorry

end NUMINAMATH_GPT_min_distance_from_P_to_origin_l1439_143939


namespace NUMINAMATH_GPT_integer_solutions_b_l1439_143941

theorem integer_solutions_b (b : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ ∀ x : ℤ, x1 ≤ x ∧ x ≤ x2 → x^2 + b * x + 3 ≤ 0) ↔ b = -4 ∨ b = 4 := 
sorry

end NUMINAMATH_GPT_integer_solutions_b_l1439_143941


namespace NUMINAMATH_GPT_number_of_chips_per_day_l1439_143992

def total_chips : ℕ := 100
def chips_first_day : ℕ := 10
def total_days : ℕ := 10
def days_remaining : ℕ := total_days - 1
def chips_remaining : ℕ := total_chips - chips_first_day

theorem number_of_chips_per_day : 
  chips_remaining / days_remaining = 10 :=
by 
  unfold chips_remaining days_remaining total_chips chips_first_day total_days
  sorry

end NUMINAMATH_GPT_number_of_chips_per_day_l1439_143992


namespace NUMINAMATH_GPT_greatest_whole_number_satisfying_inequality_l1439_143937

theorem greatest_whole_number_satisfying_inequality :
  ∃ x : ℕ, (∀ y : ℕ, y < 1 → y ≤ x) ∧ 4 * x - 3 < 2 - x :=
sorry

end NUMINAMATH_GPT_greatest_whole_number_satisfying_inequality_l1439_143937


namespace NUMINAMATH_GPT_roses_picked_later_l1439_143983

/-- Represents the initial number of roses the florist had. -/
def initial_roses : ℕ := 37

/-- Represents the number of roses the florist sold. -/
def sold_roses : ℕ := 16

/-- Represents the final number of roses the florist ended up with. -/
def final_roses : ℕ := 40

/-- Theorem which states the number of roses picked later is 19 given the conditions. -/
theorem roses_picked_later : (final_roses - (initial_roses - sold_roses)) = 19 :=
by
  -- proof steps are omitted, sorry as a placeholder
  sorry

end NUMINAMATH_GPT_roses_picked_later_l1439_143983


namespace NUMINAMATH_GPT_option_A_option_C_l1439_143928

variable {a : ℕ → ℝ} (q : ℝ)
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n + 1) = q * (a n)

def decreasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n > a (n + 1)

theorem option_A (h₁ : a 1 > 0) (hq : geometric_sequence a q) : 0 < q ∧ q < 1 → decreasing_sequence a := 
  sorry

theorem option_C (h₁ : a 1 < 0) (hq : geometric_sequence a q) : q > 1 → decreasing_sequence a := 
  sorry

end NUMINAMATH_GPT_option_A_option_C_l1439_143928


namespace NUMINAMATH_GPT_roof_area_l1439_143915

-- Definitions of the roof's dimensions based on the given conditions.
def length (w : ℝ) := 4 * w
def width (w : ℝ) := w
def difference (l w : ℝ) := l - w
def area (l w : ℝ) := l * w

-- The proof problem: Given the conditions, prove the area is 576 square feet.
theorem roof_area : ∀ w : ℝ, (length w) - (width w) = 36 → area (length w) (width w) = 576 := by
  intro w
  intro h_diff
  sorry

end NUMINAMATH_GPT_roof_area_l1439_143915


namespace NUMINAMATH_GPT_smallest_m_inequality_l1439_143911

theorem smallest_m_inequality (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end NUMINAMATH_GPT_smallest_m_inequality_l1439_143911


namespace NUMINAMATH_GPT_problem1_problem2_l1439_143918

variable (x : ℝ)

-- Statement for the first problem
theorem problem1 : (-1 + 3 * x) * (-3 * x - 1) = 1 - 9 * x^2 := 
by
  sorry

-- Statement for the second problem
theorem problem2 : (x + 1)^2 - (1 - 3 * x) * (1 + 3 * x) = 10 * x^2 + 2 * x := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1439_143918


namespace NUMINAMATH_GPT_taimour_paints_fence_alone_in_15_hours_l1439_143908

theorem taimour_paints_fence_alone_in_15_hours :
  ∀ (T : ℝ), (∀ (J : ℝ), J = T / 2 → (1 / J + 1 / T = 1 / 5)) → T = 15 :=
by
  intros T h
  have h1 := h (T / 2) rfl
  sorry

end NUMINAMATH_GPT_taimour_paints_fence_alone_in_15_hours_l1439_143908


namespace NUMINAMATH_GPT_infinite_non_prime_seq_l1439_143962

-- Let's state the theorem in Lean
theorem infinite_non_prime_seq (k : ℕ) : 
  ∃ᶠ n in at_top, ∀ i : ℕ, (1 ≤ i ∧ i ≤ k) → ¬ Nat.Prime (n + i) := 
sorry

end NUMINAMATH_GPT_infinite_non_prime_seq_l1439_143962


namespace NUMINAMATH_GPT_problem_equivalent_l1439_143952

theorem problem_equivalent : ∀ m : ℝ, 2 * m^2 + m = -1 → 4 * m^2 + 2 * m + 5 = 3 := 
by
  intros m h
  sorry

end NUMINAMATH_GPT_problem_equivalent_l1439_143952


namespace NUMINAMATH_GPT_Eliane_schedule_combinations_l1439_143935

def valid_schedule_combinations : ℕ :=
  let mornings := 6 * 3 -- 6 days (Monday to Saturday) each with 3 time slots
  let afternoons := 5 * 2 -- 5 days (Monday to Friday) each with 2 time slots
  let mon_or_fri_comb := 2 * 3 * 3 * 2 -- Morning on Monday or Friday
  let sat_comb := 1 * 3 * 4 * 2 -- Morning on Saturday
  let tue_wed_thu_comb := 3 * 3 * 2 * 2 -- Morning on Tuesday, Wednesday, or Thursday
  mon_or_fri_comb + sat_comb + tue_wed_thu_comb

theorem Eliane_schedule_combinations :
  valid_schedule_combinations = 96 := by
  sorry

end NUMINAMATH_GPT_Eliane_schedule_combinations_l1439_143935


namespace NUMINAMATH_GPT_correct_samples_for_senior_l1439_143951

-- Define the total number of students in each section
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_students : ℕ := junior_students + senior_students

-- Define the total number of samples to be drawn
def total_samples : ℕ := 60

-- Calculate the number of samples to be drawn from each section
def junior_samples : ℕ := total_samples * junior_students / total_students
def senior_samples : ℕ := total_samples - junior_samples

-- The theorem to prove
theorem correct_samples_for_senior :
  senior_samples = 20 :=
by
  sorry

end NUMINAMATH_GPT_correct_samples_for_senior_l1439_143951


namespace NUMINAMATH_GPT_find_length_of_first_dimension_of_tank_l1439_143956

theorem find_length_of_first_dimension_of_tank 
    (w : ℝ) (h : ℝ) (cost_per_sq_ft : ℝ) (total_cost : ℝ) (l : ℝ) :
    w = 5 → h = 3 → cost_per_sq_ft = 20 → total_cost = 1880 → 
    1880 = (2 * l * w + 2 * l * h + 2 * w * h) * cost_per_sq_ft →
    l = 4 := 
by
  intros hw hh hcost htotal heq
  sorry

end NUMINAMATH_GPT_find_length_of_first_dimension_of_tank_l1439_143956


namespace NUMINAMATH_GPT_color_pairings_correct_l1439_143938

noncomputable def num_color_pairings (bowls : ℕ) (glasses : ℕ) : ℕ :=
  bowls * glasses

theorem color_pairings_correct : 
  num_color_pairings 4 5 = 20 :=
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_color_pairings_correct_l1439_143938


namespace NUMINAMATH_GPT_tv_sale_increase_l1439_143923

theorem tv_sale_increase (P Q : ℝ) :
  let new_price := 0.9 * P
  let original_sale_value := P * Q
  let increased_percentage := 1.665
  ∃ x : ℝ, (new_price * (1 + x / 100) * Q = increased_percentage * original_sale_value) → x = 85 :=
by
  sorry

end NUMINAMATH_GPT_tv_sale_increase_l1439_143923


namespace NUMINAMATH_GPT_range_of_m_l1439_143930

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  ∀ m, m = 9 / 4 → (1 / x + 4 / y) ≥ m := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1439_143930


namespace NUMINAMATH_GPT_part1_sales_volume_part2_price_reduction_l1439_143943

noncomputable def daily_sales_volume (x : ℝ) : ℝ :=
  100 + 200 * x

noncomputable def profit_eq (x : ℝ) : Prop :=
  (4 - 2 - x) * (100 + 200 * x) = 300

theorem part1_sales_volume (x : ℝ) : daily_sales_volume x = 100 + 200 * x :=
sorry

theorem part2_price_reduction (hx : profit_eq (1 / 2)) : 1 / 2 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_part1_sales_volume_part2_price_reduction_l1439_143943


namespace NUMINAMATH_GPT_sum_of_sines_leq_3_sqrt3_over_2_l1439_143957

theorem sum_of_sines_leq_3_sqrt3_over_2 (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_sines_leq_3_sqrt3_over_2_l1439_143957


namespace NUMINAMATH_GPT_cone_lateral_area_l1439_143922

noncomputable def lateral_area_of_cone (θ : ℝ) (r_base : ℝ) : ℝ :=
  if θ = 120 ∧ r_base = 2 then 
    12 * Real.pi 
  else 
    0 -- default case for the sake of definition, not used in our proof

theorem cone_lateral_area :
  lateral_area_of_cone 120 2 = 12 * Real.pi :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_cone_lateral_area_l1439_143922


namespace NUMINAMATH_GPT_biquadratic_exactly_two_distinct_roots_l1439_143947

theorem biquadratic_exactly_two_distinct_roots {a : ℝ} :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^4 + a*x1^2 + a - 1 = 0) ∧ (x2^4 + a*x2^2 + a - 1 = 0) ∧
   ∀ x, x^4 + a*x^2 + a - 1 = 0 → (x = x1 ∨ x = x2)) ↔ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_biquadratic_exactly_two_distinct_roots_l1439_143947


namespace NUMINAMATH_GPT_fred_seashells_l1439_143974

def seashells_given : ℕ := 25
def seashells_left : ℕ := 22
def seashells_found : ℕ := 47

theorem fred_seashells :
  seashells_found = seashells_given + seashells_left :=
  by sorry

end NUMINAMATH_GPT_fred_seashells_l1439_143974


namespace NUMINAMATH_GPT_honey_nectar_relationship_l1439_143982

-- Definitions representing the conditions
def nectarA_water_content (x : ℝ) := 0.7 * x
def nectarB_water_content (y : ℝ) := 0.5 * y
def final_honey_water_content := 0.3
def evaporation_loss (initial_content : ℝ) := 0.15 * initial_content

-- The system of equations to prove
theorem honey_nectar_relationship (x y : ℝ) :
  (x + y = 1) ∧ (0.595 * x + 0.425 * y = 0.3) :=
sorry

end NUMINAMATH_GPT_honey_nectar_relationship_l1439_143982


namespace NUMINAMATH_GPT_grilled_cheese_sandwiches_l1439_143927

theorem grilled_cheese_sandwiches (h g : ℕ) (c_ham c_grilled total_cheese : ℕ)
  (h_count : h = 10)
  (ham_cheese : c_ham = 2)
  (grilled_cheese : c_grilled = 3)
  (cheese_used : total_cheese = 50)
  (sandwich_eq : total_cheese = h * c_ham + g * c_grilled) :
  g = 10 :=
by
  sorry

end NUMINAMATH_GPT_grilled_cheese_sandwiches_l1439_143927


namespace NUMINAMATH_GPT_certain_amount_added_l1439_143993

theorem certain_amount_added {x y : ℕ} 
    (h₁ : x = 15) 
    (h₂ : 3 * (2 * x + y) = 105) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_certain_amount_added_l1439_143993


namespace NUMINAMATH_GPT_parallelogram_area_l1439_143933

theorem parallelogram_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let y_top := a
  let y_bottom := -b
  let x_left := -c + 2*y
  let x_right := d - 2*y 
  (d + c) * (a + b) = ad + ac + bd + bc :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1439_143933


namespace NUMINAMATH_GPT_inequality_solution_l1439_143980

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1439_143980


namespace NUMINAMATH_GPT_fair_decision_l1439_143950

def fair_selection (b c : ℕ) : Prop :=
  (b - c)^2 = b + c

theorem fair_decision (b c : ℕ) : fair_selection b c := by
  sorry

end NUMINAMATH_GPT_fair_decision_l1439_143950


namespace NUMINAMATH_GPT_range_of_t_l1439_143955

variable (t : ℝ)

def point_below_line (x y a b c : ℝ) : Prop :=
  a * x - b * y + c < 0

theorem range_of_t (t : ℝ) : point_below_line 2 (3 * t) 2 (-1) 6 → t < 10 / 3 :=
  sorry

end NUMINAMATH_GPT_range_of_t_l1439_143955


namespace NUMINAMATH_GPT_inequality_not_always_hold_l1439_143948

variables {a b c d : ℝ}

theorem inequality_not_always_hold 
  (h1 : a > b) 
  (h2 : c > d) 
: ¬ (a + d > b + c) :=
  sorry

end NUMINAMATH_GPT_inequality_not_always_hold_l1439_143948


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1439_143914

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 2) : (1/x < 1/2 ∧ (∃ y : ℝ, 1/y < 1/2 ∧ y ≤ 2)) :=
by { sorry }

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1439_143914


namespace NUMINAMATH_GPT_perpendicular_planes_condition_l1439_143963

variables (α β : Plane) (m : Line) 

-- Assuming the basic definitions:
def perpendicular (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

-- Conditions
axiom α_diff_β : α ≠ β
axiom m_in_α : in_plane m α

-- Proving the necessary but not sufficient condition
theorem perpendicular_planes_condition : 
  (perpendicular α β → perpendicular_to_plane m β) ∧ 
  (¬ perpendicular_to_plane m β → ¬ perpendicular α β) ∧ 
  ¬ (perpendicular_to_plane m β → perpendicular α β) :=
sorry

end NUMINAMATH_GPT_perpendicular_planes_condition_l1439_143963


namespace NUMINAMATH_GPT_arithmetic_sequence_a_100_l1439_143996

theorem arithmetic_sequence_a_100 :
  ∀ (a : ℕ → ℕ), 
  (a 1 = 100) → 
  (∀ n : ℕ, a (n + 1) = a n + 2) → 
  a 100 = 298 :=
by
  intros a h1 hrec
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a_100_l1439_143996


namespace NUMINAMATH_GPT_three_distinct_divisors_l1439_143905

theorem three_distinct_divisors (M : ℕ) : (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ M ∧ b ∣ M ∧ c ∣ M ∧ (∀ d, d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬ d ∣ M)) ↔ (∃ p : ℕ, Prime p ∧ M = p^2) := 
by sorry

end NUMINAMATH_GPT_three_distinct_divisors_l1439_143905


namespace NUMINAMATH_GPT_smallest_m_plus_n_l1439_143912

theorem smallest_m_plus_n : ∃ (m n : ℕ), m > 1 ∧ 
  (∃ (a b : ℝ), a = (1 : ℝ) / (m * n : ℝ) ∧ b = (m : ℝ) / (n : ℝ) ∧ b - a = (1 : ℝ) / 1007) ∧
  (∀ (k l : ℕ), k > 1 ∧ 
    (∃ (c d : ℝ), c = (1 : ℝ) / (k * l : ℝ) ∧ d = (k : ℝ) / (l : ℝ) ∧ d - c = (1 : ℝ) / 1007) → m + n ≤ k + l) ∧ 
  m + n = 19099 :=
sorry

end NUMINAMATH_GPT_smallest_m_plus_n_l1439_143912


namespace NUMINAMATH_GPT_product_ab_zero_l1439_143968

variable {a b : ℝ}

theorem product_ab_zero (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
  sorry

end NUMINAMATH_GPT_product_ab_zero_l1439_143968


namespace NUMINAMATH_GPT_simplest_square_root_l1439_143958

theorem simplest_square_root (A B C D : Real) 
    (hA : A = Real.sqrt 0.1) 
    (hB : B = 1 / 2) 
    (hC : C = Real.sqrt 30) 
    (hD : D = Real.sqrt 18) : 
    C = Real.sqrt 30 := 
by 
    sorry

end NUMINAMATH_GPT_simplest_square_root_l1439_143958


namespace NUMINAMATH_GPT_sequence_value_proof_l1439_143954

theorem sequence_value_proof : 
  (∃ (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    (∀ n : ℕ, a (2 * n) = 2 * n * a n) ∧ 
    a (2^50) = 2^1276) :=
sorry

end NUMINAMATH_GPT_sequence_value_proof_l1439_143954


namespace NUMINAMATH_GPT_hexagon_area_l1439_143949

noncomputable def area_of_hexagon (P Q R P' Q' R' : Point) (radius : ℝ) : ℝ :=
  -- a placeholder for the actual area calculation
  sorry 

theorem hexagon_area (P Q R P' Q' R' : Point) 
  (radius : ℝ) (perimeter : ℝ) :
  radius = 9 → perimeter = 42 →
  area_of_hexagon P Q R P' Q' R' radius = 189 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_hexagon_area_l1439_143949


namespace NUMINAMATH_GPT_problem1_problem2_l1439_143920

-- Define Sn as given
def S (n : ℕ) : ℕ := (n ^ 2 + n) / 2

-- Define a sequence a_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define b_n using a_n = log_2 b_n
def b (n : ℕ) : ℕ := 2 ^ n

-- Define the sum of first n terms of sequence b_n
def T (n : ℕ) : ℕ := (2 ^ (n + 1)) - 2

-- Our main theorem statements
theorem problem1 (n : ℕ) : a n = n := by
  sorry

theorem problem2 (n : ℕ) : (Finset.range n).sum b = T n := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1439_143920


namespace NUMINAMATH_GPT_total_gift_money_l1439_143929

-- Definitions based on the conditions given in the problem
def initialAmount : ℕ := 159
def giftFromGrandmother : ℕ := 25
def giftFromAuntAndUncle : ℕ := 20
def giftFromParents : ℕ := 75

-- Lean statement to prove the total amount of money Chris has after receiving his birthday gifts
theorem total_gift_money : 
    initialAmount + giftFromGrandmother + giftFromAuntAndUncle + giftFromParents = 279 := by
sorry

end NUMINAMATH_GPT_total_gift_money_l1439_143929


namespace NUMINAMATH_GPT_scientific_notation_correct_l1439_143973

-- Defining the given number in terms of its scientific notation components.
def million : ℝ := 10^6
def num_million : ℝ := 15.276

-- Expressing the number 15.276 million using its definition.
def fifteen_point_two_seven_six_million : ℝ := num_million * million

-- Scientific notation representation to be proved.
def scientific_notation : ℝ := 1.5276 * 10^7

-- The theorem statement.
theorem scientific_notation_correct :
  fifteen_point_two_seven_six_million = scientific_notation :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1439_143973


namespace NUMINAMATH_GPT_distance_between_tangency_points_l1439_143977

theorem distance_between_tangency_points
  (circle_radius : ℝ) (M_distance : ℝ) (A_distance : ℝ) 
  (h1 : circle_radius = 7)
  (h2 : M_distance = 25)
  (h3 : A_distance = 7) :
  ∃ AB : ℝ, AB = 48 :=
by
  -- Definitions and proofs will go here.
  sorry

end NUMINAMATH_GPT_distance_between_tangency_points_l1439_143977


namespace NUMINAMATH_GPT_simplify_fraction_l1439_143942

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : 
  (x / (x - y) - y / (x + y)) = (x^2 + y^2) / (x^2 - y^2) :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l1439_143942


namespace NUMINAMATH_GPT_perimeter_of_square_field_l1439_143931

variable (s a p : ℕ)

-- Given conditions as definitions
def area_eq_side_squared (a s : ℕ) : Prop := a = s^2
def perimeter_eq_four_sides (p s : ℕ) : Prop := p = 4 * s
def given_equation (a p : ℕ) : Prop := 6 * a = 6 * (2 * p + 9)

-- The proof statement
theorem perimeter_of_square_field (s a p : ℕ) 
  (h1 : area_eq_side_squared a s)
  (h2 : perimeter_eq_four_sides p s)
  (h3 : given_equation a p) :
  p = 36 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_field_l1439_143931


namespace NUMINAMATH_GPT_DongfangElementary_total_students_l1439_143926

theorem DongfangElementary_total_students (x y : ℕ) 
  (h1 : x = y + 2)
  (h2 : 10 * (y + 2) = 22 * 11 * (y - 22))
  (h3 : x - x / 11 = 2 * (y - 22)) :
  x + y = 86 :=
by
  sorry

end NUMINAMATH_GPT_DongfangElementary_total_students_l1439_143926


namespace NUMINAMATH_GPT_fraction_equality_l1439_143906

theorem fraction_equality : (2 + 4) / (1 + 2) = 2 := by
  sorry

end NUMINAMATH_GPT_fraction_equality_l1439_143906


namespace NUMINAMATH_GPT_total_distance_covered_is_correct_fuel_cost_excess_is_correct_l1439_143988

-- Define the ratios and other conditions for Car A
def carA_ratio_gal_per_mile : ℚ := 4 / 7
def carA_gallons_used : ℚ := 44
def carA_cost_per_gallon : ℚ := 3.50

-- Define the ratios and other conditions for Car B
def carB_ratio_gal_per_mile : ℚ := 3 / 5
def carB_gallons_used : ℚ := 27
def carB_cost_per_gallon : ℚ := 3.25

-- Define the budget
def budget : ℚ := 200

-- Combined total distance covered by both cars
theorem total_distance_covered_is_correct :
  (carA_gallons_used * (7 / 4) + carB_gallons_used * (5 / 3)) = 122 :=
by
  sorry

-- Total fuel cost and whether it stays within budget
theorem fuel_cost_excess_is_correct :
  ((carA_gallons_used * carA_cost_per_gallon) + (carB_gallons_used * carB_cost_per_gallon)) - budget = 41.75 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_covered_is_correct_fuel_cost_excess_is_correct_l1439_143988


namespace NUMINAMATH_GPT_eval_expression_l1439_143999

theorem eval_expression :
  ((-2 : ℤ) ^ 3 : ℝ) ^ (1/3 : ℝ) - (-1 : ℤ) ^ 0 = -3 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1439_143999


namespace NUMINAMATH_GPT_number_of_integer_length_chords_through_point_l1439_143979

theorem number_of_integer_length_chords_through_point 
  (r : ℝ) (d : ℝ) (P_is_5_units_from_center : d = 5) (circle_has_radius_13 : r = 13) :
  ∃ n : ℕ, n = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_integer_length_chords_through_point_l1439_143979


namespace NUMINAMATH_GPT_car_speed_l1439_143953

def travel_time : ℝ := 5
def travel_distance : ℝ := 300

theorem car_speed :
  travel_distance / travel_time = 60 := sorry

end NUMINAMATH_GPT_car_speed_l1439_143953


namespace NUMINAMATH_GPT_inequality_xyz_l1439_143946

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz / (x^3 + y^3 + xyz) + xyz / (y^3 + z^3 + xyz) + xyz / (z^3 + x^3 + xyz) ≤ 1) := by
  sorry

end NUMINAMATH_GPT_inequality_xyz_l1439_143946


namespace NUMINAMATH_GPT_youngest_child_age_l1439_143995

theorem youngest_child_age (x y z : ℕ) 
  (h1 : 3 * x + 6 = 48) 
  (h2 : 3 * y + 9 = 60) 
  (h3 : 2 * z + 4 = 30) : 
  z = 13 := 
sorry

end NUMINAMATH_GPT_youngest_child_age_l1439_143995


namespace NUMINAMATH_GPT_find_x_l1439_143965

noncomputable def value_of_x (x : ℝ) := (5 * x) ^ 4 = (15 * x) ^ 3

theorem find_x : ∀ (x : ℝ), (value_of_x x) ∧ (x ≠ 0) → x = 27 / 5 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l1439_143965


namespace NUMINAMATH_GPT_ab_value_l1439_143976

theorem ab_value 
  (a b : ℕ) 
  (a_pos : a > 0)
  (b_pos : b > 0)
  (h1 : a + b = 30)
  (h2 : 3 * a * b + 4 * a = 5 * b + 318) : 
  (a * b = 56) :=
sorry

end NUMINAMATH_GPT_ab_value_l1439_143976


namespace NUMINAMATH_GPT_treasure_probability_l1439_143903

variable {Island : Type}

-- Define the probabilities.
def prob_treasure : ℚ := 1 / 3
def prob_trap : ℚ := 1 / 6
def prob_neither : ℚ := 1 / 2

-- Define the number of islands.
def num_islands : ℕ := 5

-- Define the probability of encountering exactly 4 islands with treasure and one with neither traps nor treasures.
theorem treasure_probability :
  (num_islands.choose 4) * (prob_ttreasure^4) * (prob_neither^1) = (5 : ℚ) * (1 / 81) * (1 / 2) :=
  by
  sorry

end NUMINAMATH_GPT_treasure_probability_l1439_143903


namespace NUMINAMATH_GPT_solve_equation_l1439_143981

theorem solve_equation (x y : ℝ) : 
    ((16 * x^2 + 1) * (y^2 + 1) = 16 * x * y) ↔ 
        ((x = 1/4 ∧ y = 1) ∨ (x = -1/4 ∧ y = -1)) := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1439_143981


namespace NUMINAMATH_GPT_total_books_l1439_143910

-- Define the given conditions
def books_per_shelf : ℕ := 8
def mystery_shelves : ℕ := 12
def picture_shelves : ℕ := 9

-- Define the number of books on each type of shelves
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := picture_shelves * books_per_shelf

-- Define the statement to prove
theorem total_books : total_mystery_books + total_picture_books = 168 := by
  sorry

end NUMINAMATH_GPT_total_books_l1439_143910


namespace NUMINAMATH_GPT_numPerfectSquareFactorsOf450_l1439_143972

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end NUMINAMATH_GPT_numPerfectSquareFactorsOf450_l1439_143972


namespace NUMINAMATH_GPT_seventh_number_fifth_row_l1439_143916

theorem seventh_number_fifth_row : 
  ∀ (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ → ℕ), 
  (∀ i, 1 <= i ∧ i <= n  → b 1 i = 2 * i - 1) →
  (∀ j i, 2 <= j ∧ 1 <= i ∧ i <= n - (j-1)  → b j i = b (j-1) i + b (j-1) (i+1)) →
  (b : ℕ → ℕ → ℕ) →
  b 5 7 = 272 :=
by {
  sorry
}

end NUMINAMATH_GPT_seventh_number_fifth_row_l1439_143916


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l1439_143959

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a = 1 ∧ b = 1) → (1 / a + 1 / b = 2) := by
  intros h
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l1439_143959


namespace NUMINAMATH_GPT_right_triangle_ratio_l1439_143994

theorem right_triangle_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∃ (x y : ℝ), 5 * (x * y) = x^2 + y^2 ∧ 5 * (a^2 + b^2) = (x + y)^2 ∧ 
    ((x - y)^2 < x^2 + y^2 ∧ x^2 + y^2 < (x + y)^2)):
  (1/2 < a / b) ∧ (a / b < 2) := by
  sorry

end NUMINAMATH_GPT_right_triangle_ratio_l1439_143994


namespace NUMINAMATH_GPT_total_salary_l1439_143997

-- Define the salaries and conditions.
def salaryN : ℝ := 280
def salaryM : ℝ := 1.2 * salaryN

-- State the theorem we want to prove
theorem total_salary : salaryM + salaryN = 616 :=
by
  sorry

end NUMINAMATH_GPT_total_salary_l1439_143997


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1439_143940

variable (x : ℝ) (h : x = Real.sqrt 2 - 1)

theorem simplify_and_evaluate_expression : 
  (1 - 1 / (x + 1)) / (x / (x^2 + 2 * x + 1)) = Real.sqrt 2 :=
by
  -- Using the given definition of x
  have hx : x = Real.sqrt 2 - 1 := h
  
  -- Required proof should go here 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1439_143940


namespace NUMINAMATH_GPT_value_of_S6_l1439_143945

theorem value_of_S6 (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 :=
by sorry

end NUMINAMATH_GPT_value_of_S6_l1439_143945


namespace NUMINAMATH_GPT_factor_expression_l1439_143907

theorem factor_expression (x : ℝ) : 9 * x^2 + 3 * x = 3 * x * (3 * x + 1) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1439_143907


namespace NUMINAMATH_GPT_b_range_l1439_143901

noncomputable def f (a b x : ℝ) := (x - 1) * Real.log x - a * x + a + b

theorem b_range (a b : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = 0 ∧ f a b x2 = 0) :
  b < 0 :=
sorry

end NUMINAMATH_GPT_b_range_l1439_143901


namespace NUMINAMATH_GPT_number_of_students_is_20_l1439_143989

-- Define the constants and conditions
def average_age_all_students (N : ℕ) : ℕ := 20
def average_age_9_students : ℕ := 11
def average_age_10_students : ℕ := 24
def age_20th_student : ℕ := 61

theorem number_of_students_is_20 (N : ℕ) 
  (h1 : N * average_age_all_students N = 99 + 240 + 61) 
  (h2 : 99 = 9 * average_age_9_students) 
  (h3 : 240 = 10 * average_age_10_students) 
  (h4 : N = 9 + 10 + 1) : N = 20 :=
sorry

end NUMINAMATH_GPT_number_of_students_is_20_l1439_143989


namespace NUMINAMATH_GPT_express_inequality_l1439_143913

theorem express_inequality (x : ℝ) : x + 4 ≥ -1 := sorry

end NUMINAMATH_GPT_express_inequality_l1439_143913


namespace NUMINAMATH_GPT_no_integer_y_such_that_abs_g_y_is_prime_l1439_143966

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 → m ≤ n → m ∣ n → m = 1 ∨ m = n

def g (y : ℤ) : ℤ := 8 * y^2 - 55 * y + 21

theorem no_integer_y_such_that_abs_g_y_is_prime : 
  ∀ y : ℤ, ¬ is_prime (|g y|) :=
by sorry

end NUMINAMATH_GPT_no_integer_y_such_that_abs_g_y_is_prime_l1439_143966


namespace NUMINAMATH_GPT_initial_salty_cookies_l1439_143964

theorem initial_salty_cookies (sweet_init sweet_eaten sweet_left salty_eaten : ℕ) 
  (h1 : sweet_init = 34)
  (h2 : sweet_eaten = 15)
  (h3 : sweet_left = 19)
  (h4 : salty_eaten = 56) :
  sweet_left + sweet_eaten = sweet_init → 
  sweet_init - sweet_eaten = sweet_left →
  ∃ salty_init, salty_init = salty_eaten :=
by
  sorry

end NUMINAMATH_GPT_initial_salty_cookies_l1439_143964


namespace NUMINAMATH_GPT_solve_for_x_l1439_143960

theorem solve_for_x
  (n m x : ℕ)
  (h1 : 7 / 8 = n / 96)
  (h2 : 7 / 8 = (m + n) / 112)
  (h3 : 7 / 8 = (x - m) / 144) :
  x = 140 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1439_143960


namespace NUMINAMATH_GPT_parabola_properties_l1439_143904

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties (a b c : ℝ) (h₀ : a ≠ 0)
    (h₁ : parabola a b c (-1) = -1)
    (h₂ : parabola a b c 0 = 1)
    (h₃ : parabola a b c (-2) > 1) :
    (a * b * c > 0) ∧
    (∃ Δ : ℝ, Δ > 0 ∧ (Δ = b^2 - 4*a*c)) ∧
    (a + b + c > 7) :=
sorry

end NUMINAMATH_GPT_parabola_properties_l1439_143904


namespace NUMINAMATH_GPT_boys_in_class_l1439_143917

theorem boys_in_class
  (g b : ℕ)
  (h_ratio : g = (3 * b) / 5)
  (h_total : g + b = 32) :
  b = 20 :=
sorry

end NUMINAMATH_GPT_boys_in_class_l1439_143917


namespace NUMINAMATH_GPT_proof_problem_l1439_143967

def x : ℝ := 0.80 * 1750
def y : ℝ := 0.35 * 3000
def z : ℝ := 0.60 * 4500
def w : ℝ := 0.40 * 2800
def a : ℝ := z * w
def b : ℝ := x + y

theorem proof_problem : a - b = 3021550 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1439_143967


namespace NUMINAMATH_GPT_least_y_l1439_143919

theorem least_y (y : ℝ) : (2 * y ^ 2 + 7 * y + 3 = 5) → y = -2 :=
sorry

end NUMINAMATH_GPT_least_y_l1439_143919


namespace NUMINAMATH_GPT_sum_of_angles_l1439_143925

theorem sum_of_angles (ABC ABD : ℝ) (n_octagon n_triangle : ℕ) 
(h1 : n_octagon = 8) 
(h2 : n_triangle = 3) 
(h3 : ABC = 180 * (n_octagon - 2) / n_octagon)
(h4 : ABD = 180 * (n_triangle - 2) / n_triangle) : 
ABC + ABD = 195 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_angles_l1439_143925


namespace NUMINAMATH_GPT_jim_miles_driven_l1439_143985

theorem jim_miles_driven (total_journey : ℕ) (miles_needed : ℕ) (h : total_journey = 1200 ∧ miles_needed = 985) : total_journey - miles_needed = 215 := 
by sorry

end NUMINAMATH_GPT_jim_miles_driven_l1439_143985


namespace NUMINAMATH_GPT_sequence_a4_value_l1439_143971

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = 2 * a n + 1) ∧ a 4 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a4_value_l1439_143971


namespace NUMINAMATH_GPT_find_point_A_l1439_143998

-- Definitions of the conditions
def point_A_left_translated_to_B (A B : ℝ × ℝ) : Prop :=
  ∃ l : ℝ, A.1 - l = B.1 ∧ A.2 = B.2

def point_A_upward_translated_to_C (A C : ℝ × ℝ) : Prop :=
  ∃ u : ℝ, A.1 = C.1 ∧ A.2 + u = C.2

-- Given points B and C
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (3, 4)

-- The statement to prove the coordinates of point A
theorem find_point_A (A : ℝ × ℝ) : 
  point_A_left_translated_to_B A B ∧ point_A_upward_translated_to_C A C → A = (3, 2) :=
by 
  sorry

end NUMINAMATH_GPT_find_point_A_l1439_143998


namespace NUMINAMATH_GPT_omega_value_l1439_143924

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem omega_value (ω x₁ x₂ : ℝ) (h_ω : ω > 0) (h_x1 : f ω x₁ = -2) (h_x2 : f ω x₂ = 0) (h_min : |x₁ - x₂| = Real.pi) :
  ω = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_omega_value_l1439_143924


namespace NUMINAMATH_GPT_ratio_of_DE_EC_l1439_143934

noncomputable def ratio_DE_EC (a x : ℝ) : ℝ :=
  let DE := a - x
  x / DE

theorem ratio_of_DE_EC (a : ℝ) (H1 : ∀ x, x = 5 * a / 7) :
  ratio_DE_EC a (5 * a / 7) = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_DE_EC_l1439_143934


namespace NUMINAMATH_GPT_geometric_progression_x_unique_l1439_143990

theorem geometric_progression_x_unique (x : ℝ) :
  (70+x)^2 = (30+x)*(150+x) ↔ x = 10 := by
  sorry

end NUMINAMATH_GPT_geometric_progression_x_unique_l1439_143990


namespace NUMINAMATH_GPT_number_of_baggies_l1439_143961

/-- Conditions -/
def cookies_per_bag : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41

/-- Question: Prove the total number of baggies Olivia can make is 6 --/
theorem number_of_baggies : (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := sorry

end NUMINAMATH_GPT_number_of_baggies_l1439_143961


namespace NUMINAMATH_GPT_part_I_part_II_l1439_143921

-- Part I
theorem part_I (x : ℝ) : (|x + 1| + |x - 4| ≤ 2 * |x - 4|) ↔ (x < 1.5) :=
sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x : ℝ, |x + a| + |x - 4| ≥ 3) → (a ≤ -7 ∨ a ≥ -1) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1439_143921


namespace NUMINAMATH_GPT_Kyle_monthly_income_l1439_143936

theorem Kyle_monthly_income :
  let rent := 1250
  let utilities := 150
  let retirement_savings := 400
  let groceries_eatingout := 300
  let insurance := 200
  let miscellaneous := 200
  let car_payment := 350
  let gas_maintenance := 350
  rent + utilities + retirement_savings + groceries_eatingout + insurance + miscellaneous + car_payment + gas_maintenance = 3200 :=
by
  -- Informal proof was provided in the solution.
  sorry

end NUMINAMATH_GPT_Kyle_monthly_income_l1439_143936


namespace NUMINAMATH_GPT_arun_age_in_6_years_l1439_143986

theorem arun_age_in_6_years
  (A D n : ℕ)
  (h1 : D = 42)
  (h2 : A = (5 * D) / 7)
  (h3 : A + n = 36) 
  : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_arun_age_in_6_years_l1439_143986


namespace NUMINAMATH_GPT_camera_pics_l1439_143969

-- Definitions of the given conditions
def phone_pictures := 22
def albums := 4
def pics_per_album := 6

-- The statement to prove the number of pictures uploaded from camera
theorem camera_pics : (albums * pics_per_album) - phone_pictures = 2 :=
by
  sorry

end NUMINAMATH_GPT_camera_pics_l1439_143969


namespace NUMINAMATH_GPT_lcm_36_100_l1439_143984

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end NUMINAMATH_GPT_lcm_36_100_l1439_143984
