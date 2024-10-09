import Mathlib

namespace radius_tangent_circle_l1699_169989

theorem radius_tangent_circle (r r1 r2 : ℝ) (h_r1 : r1 = 3) (h_r2 : r2 = 5)
    (h_concentric : true) : r = 1 := by
  -- Definitions are given as conditions
  have h1 := r1 -- radius of smaller concentric circle
  have h2 := r2 -- radius of larger concentric circle
  have h3 := h_concentric -- the circles are concentric
  have h4 := h_r1 -- r1 = 3
  have h5 := h_r2 -- r2 = 5
  sorry

end radius_tangent_circle_l1699_169989


namespace cubic_root_expression_l1699_169913

theorem cubic_root_expression (p q r : ℝ) (h1 : p + q + r = 0) (h2 : p * q + p * r + q * r = -2) (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -24 :=
sorry

end cubic_root_expression_l1699_169913


namespace vanya_correct_answers_l1699_169954

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l1699_169954


namespace wall_building_time_l1699_169965

variable (r : ℝ) -- rate at which one worker can build the wall
variable (W : ℝ) -- the wall in units, let’s denote one whole wall as 1 unit

theorem wall_building_time:
  (∀ (w t : ℝ), W = (60 * r) * t → W = (30 * r) * 6) :=
by
  sorry

end wall_building_time_l1699_169965


namespace value_of_a2_l1699_169919

theorem value_of_a2 (a0 a1 a2 a3 a4 : ℝ) (x : ℝ) 
  (h : x^4 = a0 + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3 + a4 * (x - 2)^4) :
  a2 = 24 :=
sorry

end value_of_a2_l1699_169919


namespace possible_values_for_p_t_l1699_169981

theorem possible_values_for_p_t (p q r s t : ℝ)
(h₁ : |p - q| = 3)
(h₂ : |q - r| = 4)
(h₃ : |r - s| = 5)
(h₄ : |s - t| = 6) :
  ∃ (v : Finset ℝ), v = {0, 2, 4, 6, 8, 10, 12, 18} ∧ |p - t| ∈ v :=
sorry

end possible_values_for_p_t_l1699_169981


namespace number_of_passed_boys_l1699_169904

theorem number_of_passed_boys 
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 36 * 120) :
  P = 105 := 
sorry

end number_of_passed_boys_l1699_169904


namespace part1_part2_l1699_169914

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Part (1): Prove range for m
theorem part1 (m : ℝ) : (∃ x : ℝ, quadratic 1 (-5) m x = 0) ↔ m ≤ 25 / 4 := sorry

-- Part (2): Prove value of m given conditions on roots
theorem part2 (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : 3 * x1 - 2 * x2 = 5) : 
  m = x1 * x2 → m = 6 := sorry

end part1_part2_l1699_169914


namespace ellipse_eq_line_eq_l1699_169963

-- Conditions for part (I)
def cond1 (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a > b
def pt_p_cond (PF1 PF2 : ℝ) : Prop := PF1 = 4 / 3 ∧ PF2 = 14 / 3 ∧ PF1^2 + PF2^2 = 1

-- Theorem for part (I)
theorem ellipse_eq (a b : ℝ) (PF1 PF2 : ℝ) (h₁ : cond1 a b) (h₂ : pt_p_cond PF1 PF2) : 
  (a = 3 ∧ b = 2 ∧ PF1 = 4 / 3 ∧ PF2 = 14 / 3) → 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

-- Conditions for part (II)
def center_circle (M : ℝ × ℝ) : Prop := M = (-2, 1)
def pts_symmetric (A B M : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * M.1 ∧ A.2 + B.2 = 2 * M.2

-- Theorem for part (II)
theorem line_eq (A B M : ℝ × ℝ) (k : ℝ) (h₁ : center_circle M) (h₂ : pts_symmetric A B M) :
  k = 8 / 9 → (∀ x y : ℝ, 8 * x - 9 * y + 25 = 0) :=
sorry

end ellipse_eq_line_eq_l1699_169963


namespace integer_solution_l1699_169996

theorem integer_solution (n : ℤ) (hneq : n ≠ -2) :
  ∃ (m : ℤ), (n^3 + 8) = m * (n^2 - 4) ↔ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end integer_solution_l1699_169996


namespace pear_sales_ratio_l1699_169909

theorem pear_sales_ratio : 
  ∀ (total_sold afternoon_sold morning_sold : ℕ), 
  total_sold = 420 ∧ afternoon_sold = 280 ∧ total_sold = afternoon_sold + morning_sold 
  → afternoon_sold / morning_sold = 2 :=
by 
  intros total_sold afternoon_sold morning_sold 
  intro h 
  have h_total : total_sold = 420 := h.1 
  have h_afternoon : afternoon_sold = 280 := h.2.1 
  have h_morning : total_sold = afternoon_sold + morning_sold := h.2.2
  sorry

end pear_sales_ratio_l1699_169909


namespace coordinates_of_a_l1699_169974

theorem coordinates_of_a
  (a : ℝ × ℝ)
  (b : ℝ × ℝ := (1, 2))
  (h1 : (a.1)^2 + (a.2)^2 = 5)
  (h2 : ∃ k : ℝ, a = (k, 2 * k))
  : a = (1, 2) ∨ a = (-1, -2) :=
  sorry

end coordinates_of_a_l1699_169974


namespace total_rabbits_and_chickens_l1699_169966

theorem total_rabbits_and_chickens (r c : ℕ) (h₁ : r = 64) (h₂ : r = c + 17) : r + c = 111 :=
by {
  sorry
}

end total_rabbits_and_chickens_l1699_169966


namespace intersection_eq_l1699_169927

def A : Set ℝ := {x : ℝ | (x - 2) / (x + 3) ≤ 0 }
def B : Set ℝ := {x : ℝ | x ≤ 1 }

theorem intersection_eq : A ∩ B = {x : ℝ | -3 < x ∧ x ≤ 1 } :=
sorry

end intersection_eq_l1699_169927


namespace remainder_13_pow_150_mod_11_l1699_169982

theorem remainder_13_pow_150_mod_11 : (13^150) % 11 = 1 := 
by 
  sorry

end remainder_13_pow_150_mod_11_l1699_169982


namespace range_of_a_l1699_169986

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f' x ≥ a) → (a ≤ 2) :=
by
  sorry

end range_of_a_l1699_169986


namespace newly_grown_uneaten_potatoes_l1699_169930

variable (u : ℕ)

def initially_planted : ℕ := 8
def total_now : ℕ := 11

theorem newly_grown_uneaten_potatoes : u = total_now - initially_planted := by
  sorry

end newly_grown_uneaten_potatoes_l1699_169930


namespace girls_joined_school_l1699_169956

theorem girls_joined_school
  (initial_girls : ℕ)
  (initial_boys : ℕ)
  (total_pupils_after : ℕ)
  (computed_new_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  total_pupils_after = 1346 →
  computed_new_girls = total_pupils_after - (initial_girls + initial_boys) →
  computed_new_girls = 418 :=
by
  intros h_initial_girls h_initial_boys h_total_pupils_after h_computed_new_girls
  sorry

end girls_joined_school_l1699_169956


namespace smallest_positive_multiple_of_45_l1699_169935

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l1699_169935


namespace a_horses_is_18_l1699_169931

-- Definitions of given conditions
def total_cost : ℕ := 435
def b_share : ℕ := 180
def horses_b : ℕ := 16
def months_b : ℕ := 9
def cost_b : ℕ := horses_b * months_b

def horses_c : ℕ := 18
def months_c : ℕ := 6
def cost_c : ℕ := horses_c * months_c

def total_cost_eq (x : ℕ) : Prop :=
  x * 8 + cost_b + cost_c = total_cost

-- Statement of the proof problem
theorem a_horses_is_18 (x : ℕ) : total_cost_eq x → x = 18 := 
sorry

end a_horses_is_18_l1699_169931


namespace range_of_m_l1699_169964

def P (m : ℝ) : Prop :=
  9 - m > 2 * m ∧ 2 * m > 0

def Q (m : ℝ) : Prop :=
  m > 0 ∧ (Real.sqrt (6) / 2 < Real.sqrt (5 + m) / Real.sqrt (5)) ∧ (Real.sqrt (5 + m) / Real.sqrt (5) < Real.sqrt (2))

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) → (0 < m ∧ m ≤ 5 / 2) ∨ (3 ≤ m ∧ m < 5) :=
sorry

end range_of_m_l1699_169964


namespace breadth_remains_the_same_l1699_169936

variable (L B : ℝ)

theorem breadth_remains_the_same 
  (A : ℝ) (hA : A = L * B) 
  (L_half : ℝ) (hL_half : L_half = L / 2) 
  (B' : ℝ)
  (A' : ℝ) (hA' : A' = L_half * B') 
  (hA_change : A' = 0.5 * A) : 
  B' = B :=
  sorry

end breadth_remains_the_same_l1699_169936


namespace dennis_rocks_left_l1699_169917

-- Definitions based on conditions:
def initial_rocks : ℕ := 10
def rocks_eaten_by_fish (initial : ℕ) : ℕ := initial / 2
def rocks_spat_out_by_fish : ℕ := 2

-- Total rocks left:
def total_rocks_left (initial : ℕ) (spat_out : ℕ) : ℕ :=
  (rocks_eaten_by_fish initial) + spat_out

-- Statement to be proved:
theorem dennis_rocks_left : total_rocks_left initial_rocks rocks_spat_out_by_fish = 7 :=
by
  -- Conclusion by calculation (Proved in steps)
  sorry

end dennis_rocks_left_l1699_169917


namespace largest_C_inequality_l1699_169942

theorem largest_C_inequality :
  ∃ C : ℝ, C = Real.sqrt (8 / 3) ∧ ∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z) :=
by
  sorry

end largest_C_inequality_l1699_169942


namespace ShielaDrawingsPerNeighbor_l1699_169990

-- Defining our problem using the given conditions:
def ShielaTotalDrawings : ℕ := 54
def ShielaNeighbors : ℕ := 6

-- Mathematically restating the problem:
theorem ShielaDrawingsPerNeighbor : (ShielaTotalDrawings / ShielaNeighbors) = 9 := by
  sorry

end ShielaDrawingsPerNeighbor_l1699_169990


namespace equal_costs_at_60_minutes_l1699_169968

-- Define the base rates and the per minute rates for each company
def base_rate_united : ℝ := 9.00
def rate_per_minute_united : ℝ := 0.25
def base_rate_atlantic : ℝ := 12.00
def rate_per_minute_atlantic : ℝ := 0.20

-- Define the total cost functions
def cost_united (m : ℝ) : ℝ := base_rate_united + rate_per_minute_united * m
def cost_atlantic (m : ℝ) : ℝ := base_rate_atlantic + rate_per_minute_atlantic * m

-- State the theorem to be proved
theorem equal_costs_at_60_minutes : 
  ∃ (m : ℝ), cost_united m = cost_atlantic m ∧ m = 60 :=
by
  -- Pending proof
  sorry

end equal_costs_at_60_minutes_l1699_169968


namespace problem1_problem2_l1699_169901

-- Define the first problem as a proof statement in Lean
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 25 → (x = 7 ∨ x = -3) := sorry

-- Define the second problem as a proof statement in Lean
theorem problem2 (x : ℝ) : (x - 5) ^ 2 = 2 * (5 - x) → (x = 5 ∨ x = 3) := sorry

end problem1_problem2_l1699_169901


namespace area_triangle_MDA_l1699_169977

noncomputable def area_of_triangle_MDA (r : ℝ) : ℝ := 
  let AM := r / 3
  let OM := (r ^ 2 - (AM ^ 2)).sqrt
  let AD := AM / 2
  let DM := AD / (1 / 2)
  1 / 2 * AD * DM

theorem area_triangle_MDA (r : ℝ) : area_of_triangle_MDA r = r ^ 2 / 36 := by
  sorry

end area_triangle_MDA_l1699_169977


namespace minimum_and_maximum_S_l1699_169911

theorem minimum_and_maximum_S (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 30) :
  3 * (a^3 + b^3 + c^3 + d^3) - 3 * a^2 - 3 * b^2 - 3 * c^2 - 3 * d^2 = 7.5 :=
sorry

end minimum_and_maximum_S_l1699_169911


namespace find_m_l1699_169937

theorem find_m (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (4 / x) + (9 / y) = m) (h4 : ∃ x y , x + y = 5/6) : m = 30 :=
sorry

end find_m_l1699_169937


namespace initial_boys_down_slide_l1699_169960

variable (B : Int)

theorem initial_boys_down_slide:
  B + 13 = 35 → B = 22 := by
  sorry

end initial_boys_down_slide_l1699_169960


namespace units_digit_of_square_l1699_169959

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 :=
by
  sorry

end units_digit_of_square_l1699_169959


namespace line_through_point_parallel_to_line_l1699_169987

theorem line_through_point_parallel_to_line {x y : ℝ} 
  (point : x = 1 ∧ y = 0) 
  (parallel_line : ∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0) :
  x - 2 * y - 1 = 0 := 
by
  sorry

end line_through_point_parallel_to_line_l1699_169987


namespace quadratic_with_roots_1_and_2_l1699_169972

theorem quadratic_with_roots_1_and_2 : ∃ (a b c : ℝ), (a = 1 ∧ b = 2) ∧ (∀ x : ℝ, x ≠ 1 → x ≠ 2 → a * x^2 + b * x + c = 0) ∧ (a * x^2 + b * x + c = x^2 - 3 * x + 2) :=
by
  sorry

end quadratic_with_roots_1_and_2_l1699_169972


namespace parallelogram_area_l1699_169991

variable (base height : ℝ) (tripled_area_factor original_area new_area : ℝ)

theorem parallelogram_area (h_base : base = 6) (h_height : height = 20)
    (h_tripled_area_factor : tripled_area_factor = 9)
    (h_original_area_calc : original_area = base * height)
    (h_new_area_calc : new_area = original_area * tripled_area_factor) :
    original_area = 120 ∧ tripled_area_factor = 9 ∧ new_area = 1080 := by
  sorry

end parallelogram_area_l1699_169991


namespace sum_2019_l1699_169988

noncomputable def a : ℕ → ℝ := sorry
def S (n : ℕ) : ℝ := sorry

axiom prop_1 : (a 2 - 1)^3 + (a 2 - 1) = 2019
axiom prop_2 : (a 2018 - 1)^3 + (a 2018 - 1) = -2019
axiom arithmetic_sequence : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom sum_formula : S 2019 = (2019 * (a 1 + a 2019)) / 2

theorem sum_2019 : S 2019 = 2019 :=
by sorry

end sum_2019_l1699_169988


namespace sandy_friday_hours_l1699_169980

-- Define the conditions
def hourly_rate := 15
def saturday_hours := 6
def sunday_hours := 14
def total_earnings := 450

-- Define the proof problem
theorem sandy_friday_hours (F : ℝ) (h1 : F * hourly_rate + saturday_hours * hourly_rate + sunday_hours * hourly_rate = total_earnings) : F = 10 :=
sorry

end sandy_friday_hours_l1699_169980


namespace simple_interest_rate_l1699_169944

theorem simple_interest_rate (P R: ℝ) (T: ℝ) (H: T = 5) (H1: P * (1/6) = P * (R * T / 100)) : R = 10/3 :=
by {
  sorry
}

end simple_interest_rate_l1699_169944


namespace union_sets_l1699_169952

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

-- State the theorem
theorem union_sets : A ∪ B = { x | -1 < x ∧ x ≤ 4 } := 
by
   sorry

end union_sets_l1699_169952


namespace max_n_minus_m_l1699_169998

/-- The function defined with given parameters. -/
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem max_n_minus_m (a b : ℝ) (h1 : -a / 2 = 1)
    (h2 : ∀ x, f x a b ≥ 2)
    (h3 : ∃ m n, (∀ x, f x a b ≤ 6 → m ≤ x ∧ x ≤ n) ∧ (n = 3 ∧ m = -1)) : 
    (∀ m n, (m ≤ n) → (n - m ≤ 4)) :=
by sorry

end max_n_minus_m_l1699_169998


namespace maximum_value_of_parabola_eq_24_l1699_169922

theorem maximum_value_of_parabola_eq_24 (x : ℝ) : 
  ∃ x, x = -2 ∧ (-2 * x^2 - 8 * x + 16) = 24 :=
by
  use -2
  sorry

end maximum_value_of_parabola_eq_24_l1699_169922


namespace curve_is_hyperbola_l1699_169962

theorem curve_is_hyperbola (m n x y : ℝ) (h_eq : m * x^2 - m * y^2 = n) (h_mn : m * n < 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b^2/a^2 - x^2/a^2 = 1 := 
sorry

end curve_is_hyperbola_l1699_169962


namespace trigonometric_identity_l1699_169907

theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (-1 + Real.sqrt 3) / 2 :=
sorry

end trigonometric_identity_l1699_169907


namespace Mary_more_than_Tim_l1699_169955

-- Define the incomes
variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.80 * J
def Mary_income : Prop := M = 1.28 * J

-- Theorem statement to prove
theorem Mary_more_than_Tim (J T M : ℝ) (h1 : Tim_income J T)
  (h2 : Mary_income J M) : ((M - T) / T) * 100 = 60 :=
by
  -- Including sorry to skip the proof
  sorry

end Mary_more_than_Tim_l1699_169955


namespace sum_of_fractions_and_decimal_l1699_169903

theorem sum_of_fractions_and_decimal :
  (6 / 5 : ℝ) + (1 / 10 : ℝ) + 1.56 = 2.86 :=
by
  sorry

end sum_of_fractions_and_decimal_l1699_169903


namespace half_of_number_l1699_169918

theorem half_of_number (N : ℝ)
  (h1 : (4 / 15) * (5 / 7) * N = (4 / 9) * (2 / 5) * N + 8) : 
  (N / 2) = 315 := 
sorry

end half_of_number_l1699_169918


namespace fishing_probability_correct_l1699_169908

-- Definitions for probabilities
def P_sunny : ℝ := 0.3
def P_rainy : ℝ := 0.5
def P_cloudy : ℝ := 0.2

def P_fishing_given_sunny : ℝ := 0.7
def P_fishing_given_rainy : ℝ := 0.3
def P_fishing_given_cloudy : ℝ := 0.5

-- The total probability function
def P_fishing : ℝ :=
  P_sunny * P_fishing_given_sunny +
  P_rainy * P_fishing_given_rainy +
  P_cloudy * P_fishing_given_cloudy

theorem fishing_probability_correct : P_fishing = 0.46 :=
by 
  sorry -- Proof goes here

end fishing_probability_correct_l1699_169908


namespace ratio_tough_to_good_sales_l1699_169971

-- Define the conditions
def tough_week_sales : ℤ := 800
def total_sales : ℤ := 10400
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the problem in Lean 4:
theorem ratio_tough_to_good_sales : ∃ G : ℤ, (good_weeks * G) + (tough_weeks * tough_week_sales) = total_sales ∧ 
  (tough_week_sales : ℚ) / (G : ℚ) = 1 / 2 :=
sorry

end ratio_tough_to_good_sales_l1699_169971


namespace covered_area_of_strips_l1699_169997

/-- Four rectangular strips of paper, each 16 cm long and 2 cm wide, overlap on a table. 
    We need to prove that the total area of the table surface covered by these strips is 112 cm². --/

theorem covered_area_of_strips (length width : ℝ) (number_of_strips : ℕ) (intersections : ℕ) 
    (area_of_strip : ℝ) (total_area_without_overlap : ℝ) (overlap_area : ℝ) 
    (actual_covered_area : ℝ) :
  length = 16 →
  width = 2 →
  number_of_strips = 4 →
  intersections = 4 →
  area_of_strip = length * width →
  total_area_without_overlap = number_of_strips * area_of_strip →
  overlap_area = intersections * (width * width) →
  actual_covered_area = total_area_without_overlap - overlap_area →
  actual_covered_area = 112 := 
by
  intros
  sorry

end covered_area_of_strips_l1699_169997


namespace find_range_of_a_l1699_169921

noncomputable def range_of_a (a : ℝ) : Prop :=
∀ (x : ℝ) (θ : ℝ), (0 ≤ θ ∧ θ ≤ (Real.pi / 2)) → 
  let α := (x + 3, x)
  let β := (2 * Real.sin θ * Real.cos θ, a * Real.sin θ + a * Real.cos θ)
  let sum := (α.1 + β.1, α.2 + β.2)
  (sum.1^2 + sum.2^2)^(1/2) ≥ Real.sqrt 2

theorem find_range_of_a : range_of_a a ↔ (a ≤ 1 ∨ a ≥ 5) :=
sorry

end find_range_of_a_l1699_169921


namespace j_mod_2_not_zero_l1699_169958

theorem j_mod_2_not_zero (x j : ℤ) (h : 2 * x - j = 11) : j % 2 ≠ 0 :=
sorry

end j_mod_2_not_zero_l1699_169958


namespace heaviest_lightest_difference_total_excess_weight_total_selling_price_l1699_169995

-- Define deviations from standard weight and their counts
def deviations : List (ℚ × ℕ) := [(-3.5, 2), (-2, 4), (-1.5, 2), (0, 1), (1, 3), (2.5, 8)]

-- Define standard weight and price per kg
def standard_weight : ℚ := 18
def price_per_kg : ℚ := 1.8

-- Prove the three statements:
theorem heaviest_lightest_difference :
  (2.5 - (-3.5)) = 6 := by
  sorry

theorem total_excess_weight :
  (2 * -3.5 + 4 * -2 + 2 * -1.5 + 1 * 0 + 3 * 1 + 8 * 2.5) = 5 := by
  sorry

theorem total_selling_price :
  (standard_weight * 20 + 5) * price_per_kg = 657 := by
  sorry

end heaviest_lightest_difference_total_excess_weight_total_selling_price_l1699_169995


namespace contractor_initial_hire_l1699_169929

theorem contractor_initial_hire :
  ∃ (P : ℕ), 
    (∀ (total_work : ℝ), 
      (P * 20 = (1/4) * total_work) ∧ 
      ((P - 2) * 75 = (3/4) * total_work)) → 
    P = 10 :=
by
  sorry

end contractor_initial_hire_l1699_169929


namespace find_d_squared_l1699_169970

noncomputable def g (z : ℂ) (c d : ℝ) : ℂ := (c + d * Complex.I) * z

theorem find_d_squared (c d : ℝ) (z : ℂ) (h1 : ∀ z : ℂ, Complex.abs (g z c d - z) = 2 * Complex.abs (g z c d)) (h2 : Complex.abs (c + d * Complex.I) = 6) : d^2 = 11305 / 4 := 
sorry

end find_d_squared_l1699_169970


namespace expression_is_correct_l1699_169950

theorem expression_is_correct (a : ℝ) : 2 * (a + 1) = 2 * a + 1 := 
sorry

end expression_is_correct_l1699_169950


namespace function_relation4_l1699_169933

open Set

section
  variable (M : Set ℤ) (N : Set ℤ)

  def relation1 (x : ℤ) := x ^ 2
  def relation2 (x : ℤ) := x + 1
  def relation3 (x : ℤ) := x - 1
  def relation4 (x : ℤ) := abs x

  theorem function_relation4 : 
    M = {-1, 1, 2, 4} →
    N = {1, 2, 4} →
    (∀ x ∈ M, relation4 x ∈ N) :=
  by
    intros hM hN
    simp [relation4]
    sorry
end

end function_relation4_l1699_169933


namespace probability_two_red_marbles_l1699_169985

theorem probability_two_red_marbles
  (red_marbles : ℕ)
  (white_marbles : ℕ)
  (total_marbles : ℕ)
  (prob_first_red : ℚ)
  (prob_second_red_after_first_red : ℚ)
  (combined_probability : ℚ) :
  red_marbles = 5 →
  white_marbles = 7 →
  total_marbles = 12 →
  prob_first_red = 5 / 12 →
  prob_second_red_after_first_red = 4 / 11 →
  combined_probability = 5 / 33 →
  combined_probability = prob_first_red * prob_second_red_after_first_red := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_two_red_marbles_l1699_169985


namespace Nancy_more_pearl_beads_l1699_169975

-- Define the problem conditions
def metal_beads_Nancy : ℕ := 40
def crystal_beads_Rose : ℕ := 20
def stone_beads_Rose : ℕ := crystal_beads_Rose * 2
def total_beads_needed : ℕ := 20 * 8
def total_Rose_beads : ℕ := crystal_beads_Rose + stone_beads_Rose
def pearl_beads_Nancy : ℕ := total_beads_needed - total_Rose_beads

-- State the theorem to prove
theorem Nancy_more_pearl_beads :
  pearl_beads_Nancy = metal_beads_Nancy + 60 :=
by
  -- We leave the proof as an exercise
  sorry

end Nancy_more_pearl_beads_l1699_169975


namespace probability_first_die_l1699_169953

theorem probability_first_die (n : ℕ) (n_pos : n = 4025) (m : ℕ) (m_pos : m = 2012) : 
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  (favorable_outcomes / total_outcomes : ℚ) = 1006 / 4025 :=
by
  have h_n : n = 4025 := n_pos
  have h_m : m = 2012 := m_pos
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  sorry

end probability_first_die_l1699_169953


namespace find_M_at_x_eq_3_l1699_169951

noncomputable def M (a b c d x : ℝ) := a * x^5 + b * x^3 + c * x + d

theorem find_M_at_x_eq_3
  (a b c d M : ℝ)
  (h₀ : d = -5)
  (h₁ : 243 * a + 27 * b + 3 * c = -12) :
  M = -17 :=
by
  sorry

end find_M_at_x_eq_3_l1699_169951


namespace eccentricity_of_ellipse_l1699_169932

noncomputable def calculate_eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a ^ 2 - b ^ 2)
  c / a

theorem eccentricity_of_ellipse : 
  (calculate_eccentricity 5 4) = 3 / 5 :=
by
  sorry

end eccentricity_of_ellipse_l1699_169932


namespace ratio_equivalence_l1699_169961

theorem ratio_equivalence (m n s u : ℚ) (h1 : m / n = 5 / 4) (h2 : s / u = 8 / 15) :
  (5 * m * s - 2 * n * u) / (7 * n * u - 10 * m * s) = 4 / 3 :=
by
  sorry

end ratio_equivalence_l1699_169961


namespace consecutive_sum_divisible_by_12_l1699_169969

theorem consecutive_sum_divisible_by_12 
  (b : ℤ) 
  (a : ℤ := b - 1) 
  (c : ℤ := b + 1) 
  (d : ℤ := b + 2) :
  ∃ k : ℤ, ab + ac + ad + bc + bd + cd + 1 = 12 * k := by
  sorry

end consecutive_sum_divisible_by_12_l1699_169969


namespace day_of_week_after_6_pow_2023_l1699_169992

def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem day_of_week_after_6_pow_2023 :
  day_of_week_after_days 4 (6^2023) = 3 :=
by
  sorry

end day_of_week_after_6_pow_2023_l1699_169992


namespace dogs_not_liking_any_food_l1699_169925

-- Declare variables
variable (n w s ws c cs : ℕ)

-- Define problem conditions
def total_dogs := n
def dogs_like_watermelon := w
def dogs_like_salmon := s
def dogs_like_watermelon_and_salmon := ws
def dogs_like_chicken := c
def dogs_like_chicken_and_salmon_but_not_watermelon := cs

-- Define the statement proving the number of dogs that do not like any of the three foods
theorem dogs_not_liking_any_food : 
  n = 75 → 
  w = 15 → 
  s = 54 → 
  ws = 12 → 
  c = 20 → 
  cs = 7 → 
  (75 - ((w - ws) + (s - ws - cs) + (c - cs) + ws + cs) = 5) :=
by
  intros _ _ _ _ _ _
  sorry

end dogs_not_liking_any_food_l1699_169925


namespace area_of_sandbox_is_correct_l1699_169967

-- Define the length and width of the sandbox
def length_sandbox : ℕ := 312
def width_sandbox : ℕ := 146

-- Define the area calculation
def area_sandbox (length width : ℕ) : ℕ := length * width

-- The theorem stating that the area of the sandbox is 45552 cm²
theorem area_of_sandbox_is_correct : area_sandbox length_sandbox width_sandbox = 45552 := sorry

end area_of_sandbox_is_correct_l1699_169967


namespace part_I_part_II_l1699_169910

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -4 ∨ x > -2}
def C (m : ℝ) : Set ℝ := {x | 3 - 2 * m ≤ x ∧ x ≤ 2 + m}
def D : Set ℝ := {y | y < -6 ∨ y > -5}

theorem part_I (m : ℝ) : (∀ x, x ∈ A ∧ x ∈ B → x ∈ C m) → m ≥ 5 / 2 :=
sorry

theorem part_II (m : ℝ) : 
  (B ∪ (C m) = Set.univ) ∧ 
  (C m ⊆ D) → 
  7 / 2 ≤ m ∧ m < 4 :=
sorry

end part_I_part_II_l1699_169910


namespace distance_between_lines_l1699_169924

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem distance_between_lines : 
  ∀ (x y : ℝ), line1 x y → line2 x y → (|1 - (-1)| / Real.sqrt (1^2 + (-1)^2)) = Real.sqrt 2 := 
by 
  sorry

end distance_between_lines_l1699_169924


namespace solve_fractional_eq_l1699_169900

-- Defining the fractional equation as a predicate
def fractional_eq (x : ℝ) : Prop :=
  (5 / x) = (7 / (x - 2))

-- The main theorem to be proven
theorem solve_fractional_eq : ∃ x : ℝ, fractional_eq x ∧ x = -5 := by
  sorry

end solve_fractional_eq_l1699_169900


namespace fuel_consumption_gallons_l1699_169957

theorem fuel_consumption_gallons
  (distance_per_liter : ℝ)
  (speed_mph : ℝ)
  (time_hours : ℝ)
  (mile_to_km : ℝ)
  (gallon_to_liters : ℝ)
  (fuel_consumption : ℝ) :
  distance_per_liter = 56 →
  speed_mph = 91 →
  time_hours = 5.7 →
  mile_to_km = 1.6 →
  gallon_to_liters = 3.8 →
  fuel_consumption = 3.9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end fuel_consumption_gallons_l1699_169957


namespace max_proj_area_l1699_169905

variable {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem max_proj_area : 
  ∃ max_area : ℝ, max_area = Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) :=
by
  sorry

end max_proj_area_l1699_169905


namespace max_expression_value_l1699_169999

theorem max_expression_value {x y : ℝ} (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) :
  x^2 + y^2 ≤ 10 :=
sorry

end max_expression_value_l1699_169999


namespace bird_families_difference_l1699_169978

theorem bird_families_difference {initial_families flown_away : ℕ} (h1 : initial_families = 87) (h2 : flown_away = 7) :
  (initial_families - flown_away) - flown_away = 73 := by
sorry

end bird_families_difference_l1699_169978


namespace find_r_l1699_169947

theorem find_r (b r : ℝ) (h1 : b / (1 - r) = 18) (h2 : b * r^2 / (1 - r^2) = 6) : r = 1/2 :=
by
  sorry

end find_r_l1699_169947


namespace hoses_fill_time_l1699_169912

noncomputable def time_to_fill_pool {P A B C : ℝ} (h₁ : A + B = P / 3) (h₂ : A + C = P / 4) (h₃ : B + C = P / 5) : ℝ :=
  (120 / 47 : ℝ)

theorem hoses_fill_time {P A B C : ℝ} 
  (h₁ : A + B = P / 3) 
  (h₂ : A + C = P / 4) 
  (h₃ : B + C = P / 5) 
  : time_to_fill_pool h₁ h₂ h₃ = (120 / 47 : ℝ) :=
sorry

end hoses_fill_time_l1699_169912


namespace final_height_of_tree_in_4_months_l1699_169902

-- Definitions based on the conditions
def growth_rate_cm_per_two_weeks : ℕ := 50
def current_height_meters : ℕ := 2
def weeks_per_month : ℕ := 4
def months : ℕ := 4
def cm_per_meter : ℕ := 100

-- The final height of the tree after 4 months in centimeters
theorem final_height_of_tree_in_4_months : 
  (current_height_meters * cm_per_meter) + 
  (((months * weeks_per_month) / 2) * growth_rate_cm_per_two_weeks) = 600 := 
by
  sorry

end final_height_of_tree_in_4_months_l1699_169902


namespace curve_is_segment_l1699_169941

noncomputable def parametric_curve := {t : ℝ // 0 ≤ t ∧ t ≤ 5}

def x (t : parametric_curve) : ℝ := 3 * t.val ^ 2 + 2
def y (t : parametric_curve) : ℝ := t.val ^ 2 - 1

def line_equation (x y : ℝ) := x - 3 * y - 5 = 0

theorem curve_is_segment :
  ∀ (t : parametric_curve), line_equation (x t) (y t) ∧ 
  2 ≤ x t ∧ x t ≤ 77 :=
by
  sorry

end curve_is_segment_l1699_169941


namespace largest_divisor_of_expression_l1699_169983

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 + n^3 - n - 1) :=
sorry

end largest_divisor_of_expression_l1699_169983


namespace cyclist_time_to_climb_and_descend_hill_l1699_169993

noncomputable def hill_length : ℝ := 400 -- hill length in meters
noncomputable def ascent_speed_kmh : ℝ := 7.2 -- ascent speed in km/h
noncomputable def ascent_speed_ms : ℝ := ascent_speed_kmh * 1000 / 3600 -- ascent speed converted in m/s
noncomputable def descent_speed_ms : ℝ := 2 * ascent_speed_ms -- descent speed in m/s

noncomputable def time_to_climb : ℝ := hill_length / ascent_speed_ms -- time to climb in seconds
noncomputable def time_to_descend : ℝ := hill_length / descent_speed_ms -- time to descend in seconds
noncomputable def total_time : ℝ := time_to_climb + time_to_descend -- total time in seconds

theorem cyclist_time_to_climb_and_descend_hill : total_time = 300 :=
by
  sorry

end cyclist_time_to_climb_and_descend_hill_l1699_169993


namespace find_x_l1699_169946

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) : 
  let l5 := log_base 5 x
  let l6 := log_base 6 x
  let l7 := log_base 7 x
  let surface_area := 2 * (l5 * l6 + l5 * l7 + l6 * l7)
  let volume := l5 * l6 * l7 
  (surface_area = 2 * volume) → x = 210 :=
by 
  sorry

end find_x_l1699_169946


namespace sequence_divisible_by_11_l1699_169994

theorem sequence_divisible_by_11 
  (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
  (∀ n, n = 4 ∨ n = 8 ∨ n ≥ 10 → 11 ∣ a n) := sorry

end sequence_divisible_by_11_l1699_169994


namespace shorter_side_of_rectangular_room_l1699_169979

theorem shorter_side_of_rectangular_room 
  (a b : ℕ) 
  (h1 : 2 * a + 2 * b = 52) 
  (h2 : a * b = 168) : 
  min a b = 12 := 
  sorry

end shorter_side_of_rectangular_room_l1699_169979


namespace average_of_other_half_l1699_169948

theorem average_of_other_half (avg : ℝ) (sum_half : ℝ) (n : ℕ) (n_half : ℕ)
    (h_avg : avg = 43.1)
    (h_sum_half : sum_half = 158.4)
    (h_n : n = 8)
    (h_n_half : n_half = n / 2) :
    ((n * avg - sum_half) / n_half) = 46.6 :=
by
  -- The proof steps would be given here. We're omitting them as the prompt instructs.
  sorry

end average_of_other_half_l1699_169948


namespace sin_sum_less_than_zero_l1699_169916

noncomputable def is_acute_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2 ∧ 0 < γ ∧ γ < Real.pi / 2

theorem sin_sum_less_than_zero (n : ℕ) :
  (∀ (α β γ : ℝ), is_acute_triangle α β γ → (Real.sin (n * α) + Real.sin (n * β) + Real.sin (n * γ) < 0)) ↔ n = 4 :=
by
  sorry

end sin_sum_less_than_zero_l1699_169916


namespace add_to_fraction_eq_l1699_169938

theorem add_to_fraction_eq (n : ℕ) : (4 + n) / (7 + n) = 6 / 7 → n = 14 :=
by sorry

end add_to_fraction_eq_l1699_169938


namespace smallest_w_l1699_169984

theorem smallest_w (x y w : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 ^ x) ∣ (3125 * w)) (h4 : (3 ^ y) ∣ (3125 * w)) 
  (h5 : (5 ^ (x + y)) ∣ (3125 * w)) (h6 : (7 ^ (x - y)) ∣ (3125 * w))
  (h7 : (13 ^ 4) ∣ (3125 * w))
  (h8 : x + y ≤ 10) (h9 : x - y ≥ 2) :
  w = 33592336 :=
by
  sorry

end smallest_w_l1699_169984


namespace product_of_xy_l1699_169940

theorem product_of_xy (x y : ℝ) : 
  (1 / 5 * (x + y + 4 + 5 + 6) = 5) ∧ 
  (1 / 5 * ((x - 5) ^ 2 + (y - 5) ^ 2 + (4 - 5) ^ 2 + (5 - 5) ^ 2 + (6 - 5) ^ 2) = 2) 
  → x * y = 21 :=
by sorry

end product_of_xy_l1699_169940


namespace find_S2019_l1699_169945

-- Conditions given in the problem
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Definitions and conditions extracted: conditions for sum of arithmetic sequence
axiom arithmetic_sum (n : ℕ) : S n = n * a (n / 2)
axiom OB_condition : a 3 + a 2017 = 1

-- Lean statement to prove S2019
theorem find_S2019 : S 2019 = 2019 / 2 := by
  sorry

end find_S2019_l1699_169945


namespace probability_of_event_l1699_169928

noncomputable def drawing_probability : ℚ := 
  let total_outcomes := 81
  let successful_outcomes :=
    (9 + 9 + 9 + 9 + 9 + 7 + 5 + 3 + 1)
  successful_outcomes / total_outcomes

theorem probability_of_event :
  drawing_probability = 61 / 81 := 
by
  sorry

end probability_of_event_l1699_169928


namespace systematic_sampling_correct_l1699_169915

-- Conditions as definitions
def total_bags : ℕ := 50
def num_samples : ℕ := 5
def interval (total num : ℕ) : ℕ := total / num
def correct_sequence : List ℕ := [5, 15, 25, 35, 45]

-- Statement
theorem systematic_sampling_correct :
  ∃ l : List ℕ, (l.length = num_samples) ∧ 
               (∀ i ∈ l, i ≤ total_bags) ∧
               (∀ i j, i < j → l.indexOf i < l.indexOf j → j - i = interval total_bags num_samples) ∧
               l = correct_sequence :=
by
  sorry

end systematic_sampling_correct_l1699_169915


namespace number_of_triangles_and_squares_l1699_169943

theorem number_of_triangles_and_squares (x y : ℕ) (h1 : x + y = 13) (h2 : 3 * x + 4 * y = 47) : 
  x = 5 ∧ y = 8 :=
by
  sorry

end number_of_triangles_and_squares_l1699_169943


namespace new_milk_water_ratio_l1699_169923

theorem new_milk_water_ratio
  (original_milk : ℚ)
  (original_water : ℚ)
  (added_water : ℚ)
  (h_ratio : original_milk / original_water = 2 / 1)
  (h_milk_qty : original_milk = 45)
  (h_added_water : added_water = 10) :
  original_milk / (original_water + added_water) = 18 / 13 :=
by
  sorry

end new_milk_water_ratio_l1699_169923


namespace tan_alpha_value_l1699_169973

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 := 
by
  sorry

end tan_alpha_value_l1699_169973


namespace zilla_savings_deposit_l1699_169906

-- Definitions based on problem conditions
def monthly_earnings (E : ℝ) : Prop :=
  0.07 * E = 133

def tax_deduction (E : ℝ) : ℝ :=
  E - 0.10 * E

def expenditure (earnings : ℝ) : ℝ :=
  133 +  0.30 * earnings + 0.20 * earnings + 0.12 * earnings

def savings_deposit (remaining_earnings : ℝ) : ℝ :=
  0.15 * remaining_earnings

-- The final proof statement
theorem zilla_savings_deposit (E : ℝ) (total_spent : ℝ) (earnings_after_tax : ℝ) (remaining_earnings : ℝ) : 
  monthly_earnings E →
  tax_deduction E = earnings_after_tax →
  expenditure earnings_after_tax = total_spent →
  remaining_earnings = earnings_after_tax - total_spent →
  savings_deposit remaining_earnings = 77.52 :=
by
  intros
  sorry

end zilla_savings_deposit_l1699_169906


namespace system1_solution_correct_system2_solution_correct_l1699_169939

theorem system1_solution_correct (x y : ℝ) (h1 : x + y = 5) (h2 : 4 * x - 2 * y = 2) :
    x = 2 ∧ y = 3 :=
  sorry

theorem system2_solution_correct (x y : ℝ) (h1 : 3 * x - 2 * y = 13) (h2 : 4 * x + 3 * y = 6) :
    x = 3 ∧ y = -2 :=
  sorry

end system1_solution_correct_system2_solution_correct_l1699_169939


namespace tickets_left_l1699_169949

-- Define the number of tickets won by Dave
def tickets_won : ℕ := 14

-- Define the number of tickets lost by Dave
def tickets_lost : ℕ := 2

-- Define the number of tickets used to buy toys
def tickets_used : ℕ := 10

-- The theorem to prove that the number of tickets left is 2
theorem tickets_left : tickets_won - tickets_lost - tickets_used = 2 := by
  -- Initial computation of tickets left after losing some
  let tickets_after_lost := tickets_won - tickets_lost
  -- Computation of tickets left after using some
  let tickets_after_used := tickets_after_lost - tickets_used
  show tickets_after_used = 2
  sorry

end tickets_left_l1699_169949


namespace intersection_of_A_and_B_l1699_169920

section intersection_proof

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x + 1 > 0}

-- Statement of the theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := 
by {
  sorry
}

end intersection_proof

end intersection_of_A_and_B_l1699_169920


namespace coin_probability_not_unique_l1699_169934

variables (p : ℝ) (w : ℝ)
def binomial_prob := 10 * p^3 * (1 - p)^2

theorem coin_probability_not_unique (h : binomial_prob p = 144 / 625) : 
  ∃ p1 p2, p1 ≠ p2 ∧ binomial_prob p1 = 144 / 625 ∧ binomial_prob p2 = 144 / 625 :=
by 
  sorry

end coin_probability_not_unique_l1699_169934


namespace twenty_twenty_third_term_l1699_169926

def sequence_denominator (n : ℕ) : ℕ :=
  2 * n - 1

def sequence_numerator_pos (n : ℕ) : ℕ :=
  (n + 1) / 2

def sequence_numerator_neg (n : ℕ) : ℤ :=
  -((n + 1) / 2 : ℤ)

def sequence_term (n : ℕ) : ℚ :=
  if n % 2 = 1 then 
    (sequence_numerator_pos n) / (sequence_denominator n) 
  else 
    (sequence_numerator_neg n : ℚ) / (sequence_denominator n)

theorem twenty_twenty_third_term :
  sequence_term 2023 = 1012 / 4045 := 
sorry

end twenty_twenty_third_term_l1699_169926


namespace zombie_count_today_l1699_169976

theorem zombie_count_today (Z : ℕ) (h : Z < 50) : 16 * Z = 48 :=
by
  -- Assume Z, h conditions from a)
  -- Proof will go here, for now replaced with sorry
  sorry

end zombie_count_today_l1699_169976
