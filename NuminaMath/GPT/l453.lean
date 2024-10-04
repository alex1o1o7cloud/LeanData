import Mathlib

namespace find_other_x_intercept_l453_453701

noncomputable def distance (p1 p2: ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_x_intercept (point: ℝ × ℝ) (focus1 focus2: ℝ × ℝ) (sum_dist: ℝ) : Prop :=
  distance point focus1 + distance point focus2 = sum_dist ∧ point.2 = 0

theorem find_other_x_intercept (focus1 focus2 intercept1: ℝ × ℝ) :
  focus1 = (1, 2) → focus2 = (4, 0) → intercept1 = (1, 0) →
  ∃ intercept2: ℝ × ℝ, is_x_intercept intercept2 focus1 focus2 5 ∧ intercept2 ≠ intercept1 :=
by
  sorry

end find_other_x_intercept_l453_453701


namespace hyperbola_condition_l453_453168

theorem hyperbola_condition (m : ℝ) : (∀ x y : ℝ, x^2 + m * y^2 = 1 → m < 0 ↔ x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end hyperbola_condition_l453_453168


namespace triangle_BK_KH_ratio_l453_453494

theorem triangle_BK_KH_ratio (ABC : Triangle) (K M H : Point)
  (h1 : ABC.angle B C = 120)
  (h2 : ABC.on_side K AC)
  (h3 : ABC.on_side M AC)
  (h4 : ABC.side_length K A = ABC.side_length B A)
  (h5 : ABC.side_length M C = ABC.side_length C B)
  (h6 : is_perpendicular K H B M) :
  ABC.side_length B K / ABC.side_length K H = 2 :=
sorry

end triangle_BK_KH_ratio_l453_453494


namespace remaining_cooking_time_l453_453318

-- Define the recommended cooking time in minutes and the time already cooked in seconds
def recommended_cooking_time_min := 5
def time_cooked_seconds := 45

-- Define the conversion from minutes to seconds
def minutes_to_seconds (min : Nat) : Nat := min * 60

-- Define the total recommended cooking time in seconds
def total_recommended_cooking_time_seconds := minutes_to_seconds recommended_cooking_time_min

-- State the theorem to prove the remaining cooking time
theorem remaining_cooking_time :
  (total_recommended_cooking_time_seconds - time_cooked_seconds) = 255 :=
by
  sorry

end remaining_cooking_time_l453_453318


namespace smallest_number_after_operations_l453_453133

theorem smallest_number_after_operations :
  ∃ n : ℕ, n = 1 ∧ 
  ∀ nums : list ℕ, 
  (nums = list.range' 1 102) →
  (nums.length = 101) →
  ∀ f : list ℕ → ℕ,
  (∀ ns : list ℕ, ns.length = 2 → f ns = abs (ns.head - (ns.tail.head))) →
  ∃ final_nums : list ℕ, 
  (list.length final_nums = 1) →
  (100 operations from nums with f final_nums.head = 1) :=
sorry

end smallest_number_after_operations_l453_453133


namespace class_with_at_least_35_students_l453_453992

theorem class_with_at_least_35_students (classes : ℕ) (students : ℕ) (h_classes : classes = 33) (h_students : students = 1150) : 
  ∃ k, k ≤ classes ∧ ∀ i < classes, k ≥ 35 → students ≥ (k - 1) * classes :=
by
  -- Conditions derived from problem statement
  have h1 : students / classes ≥ 35 - 1 := sorry,
  have h2 : students > (34) * classes := sorry,
  -- Proof using the assumptions and inherent contradiction
  have h3 : ∃ k, k ≤ classes ∧ ∀ i < classes, k ≥ 35 := sorry,
  exact h3

end class_with_at_least_35_students_l453_453992


namespace train_length_is_correct_l453_453693

def speed_in_meters_per_second (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

def train_length (time_seconds : ℝ) (speed_kmh : ℕ) : ℝ :=
  (speed_in_meters_per_second speed_kmh) * (time_seconds.toFloat)

theorem train_length_is_correct (time_seconds : ℝ) (speed_kmh : ℕ) (length_meters : ℝ) :
  time_seconds = 3.499720022398208 ∧ speed_kmh = 180 ∧ length_meters = 174.9860011199104 →
  train_length time_seconds speed_kmh = length_meters :=
by
  intros h
  cases h with h_time_seconds h_rest
  cases h_rest with h_speed_kmh h_length_meters
  rw [h_time_seconds, h_speed_kmh, h_length_meters]
  have speed_conversion : speed_in_meters_per_second 180 = 50 :=
    by decide -- converts 180 km/hr to 50 m/s
  rw speed_conversion
  norm_num
  exact h_length_meters

end train_length_is_correct_l453_453693


namespace valid_param_A_D_l453_453594

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = 3 * x - 4

-- Define the parameterizations
def param_A (t : ℝ) : ℝ × ℝ := (0, -4) + t • (1, 3)
def param_B (t : ℝ) : ℝ × ℝ := (4/3, 0) + t • (3, 1)
def param_C (t : ℝ) : ℝ × ℝ := (2, 2) + t • (9, 3)
def param_D (t : ℝ) : ℝ × ℝ := (-4, -16) + t • (3, 9)
def param_E (t : ℝ) : ℝ × ℝ := (1, -1) + t • (1/3, 1)

-- Define a function to check if a parameterization is valid
def valid_param (f : ℝ → ℝ × ℝ) : Prop :=
  ∀ t, line_eq (f t).fst (f t).snd

-- The theorem stating the valid parameterizations
theorem valid_param_A_D : valid_param param_A ∧ valid_param param_D :=
by sorry

end valid_param_A_D_l453_453594


namespace at_least_two_fail_l453_453983

theorem at_least_two_fail (p q : ℝ) (n : ℕ) (h_p : p = 0.2) (h_q : q = 1 - p) :
  n ≥ 18 → (1 - ((q^n) * (1 + n * p / 4))) ≥ 0.9 :=
by
  sorry

end at_least_two_fail_l453_453983


namespace find_C_D_E_sum_l453_453600

noncomputable def polynomial_divisibility (C D E : ℝ) : Prop :=
  ∃ p : Polynomial ℝ, Polynomial.divides (Polynomial.x ^ 103 + C * Polynomial.x ^ 2 + D * Polynomial.x + Polynomial.C E)
    (Polynomial.x ^ 2 + Polynomial.x + 1)

theorem find_C_D_E_sum (C D E : ℝ) (h : polynomial_divisibility C D E) : C + D + E = 2 := by
  sorry

end find_C_D_E_sum_l453_453600


namespace chickens_and_rabbits_l453_453485

-- Let x be the number of chickens and y be the number of rabbits
variables (x y : ℕ)

-- Conditions: There are 35 heads and 94 feet in total
def heads_eq : Prop := x + y = 35
def feet_eq : Prop := 2 * x + 4 * y = 94

-- Proof statement (no proof is required, so we use sorry)
theorem chickens_and_rabbits :
  (heads_eq x y) ∧ (feet_eq x y) ↔ (x + y = 35 ∧ 2 * x + 4 * y = 94) :=
by
  sorry

end chickens_and_rabbits_l453_453485


namespace rectangular_plot_width_l453_453676

theorem rectangular_plot_width :
  ∀ (length width : ℕ), 
    length = 60 → 
    ∀ (poles spacing : ℕ), 
      poles = 44 → 
      spacing = 5 → 
      2 * length + 2 * width = poles * spacing →
      width = 50 :=
by
  intros length width h_length poles spacing h_poles h_spacing h_perimeter
  rw [h_length, h_poles, h_spacing] at h_perimeter
  linarith

end rectangular_plot_width_l453_453676


namespace cheyenne_earnings_l453_453336

def total_pots := 80
def cracked_fraction := (2 : ℕ) / 5
def price_per_pot := 40

def cracked_pots (total_pots : ℕ) (fraction : ℚ) : ℕ :=
  (fraction * total_pots).toNat

def remaining_pots (total_pots : ℕ) (cracked_pots : ℕ) : ℕ :=
  total_pots - cracked_pots

def total_earnings (remaining_pots : ℕ) (price_per_pot : ℕ) : ℕ :=
  remaining_pots * price_per_pot

theorem cheyenne_earnings :
  total_earnings (remaining_pots total_pots (cracked_pots total_pots cracked_fraction)) price_per_pot = 1920 :=
by
  sorry

end cheyenne_earnings_l453_453336


namespace solution_l453_453520

-- Defining S as the set of positive real numbers
def S := { x : ℝ | x > 0 }

-- Defining the function f from S to ℝ with the given functional equation
def f (x : S) : ℝ := sorry

-- The condition on the function f
axiom f_condition (x y : S) : f x * f y = f (S.mk (x * y) sorry) + 2023 * (1 / x + 1 / y + 2022)

-- The question asks us to prove n * s = 4047 / 2
noncomputable def n : ℕ := sorry
noncomputable def s : ℝ := sorry

theorem solution : n * s = 4047 / 2 := sorry

end solution_l453_453520


namespace find_m_l453_453816

theorem find_m (m : ℝ) : (let a := (m, 4)
                          let b := (3, -2)
                          ¬ collinear ℝ ![a.1, a.2] ![b.1, b.2]) → m = -6 := by
    unfold collinear
    sorry

end find_m_l453_453816


namespace range_of_a_l453_453006

noncomputable def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * a * x + 16 > 0

noncomputable def q (a : ℝ) : Prop :=
  let disc := (2 * a - 2)^2 - 8 * (3 * a - 7) in 
  disc ≥ 0

theorem range_of_a (a : ℝ) : p a ∧ q a → -4 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l453_453006


namespace CMU_PuyoPuyo_tournament_expected_value_l453_453993

theorem CMU_PuyoPuyo_tournament_expected_value : 
  let contestant_count := 36
  let expected_matches : Real := 22085 / 36
  (∃ n : ℕ, n = contestant_count ∧ 
   (∀ (matches : ℕ), matches ≥ 35 ∧ matches ≤ (contestant_count * (contestant_count - 1) / 2) → 
    ∑ (i : ℕ) in (finset.range (matches + 1)).filter (λ i, i ≥ 35), 
      (i * (real.to_rat i!.val.to_real) * (real.to_rat (595.choosen (i.val - 35))) / 
      (∏ j in (finset.range i).filter (λ j, j + 1 ≤ 630 - i.val), 630 - j) = expected_matches))) :=
by
  sorry

end CMU_PuyoPuyo_tournament_expected_value_l453_453993


namespace maclaurin_newton_inequality_part_1_maclaurin_newton_inequality_part_2_l453_453645

variable {n : ℕ} (x : Fin n → ℝ)
variable [h : Fact (1 < n)] [strictpos : ∀ i : Fin n, 0 < x i]

noncomputable def d (p : ℕ) : ℝ :=
  (Finset.univ.powersetLen p).sum (λ s, (∏ i in s, x i) / (Nat.choose n p))

theorem maclaurin_newton_inequality_part_1 
  (p q : ℕ) (hpq : 1 ≤ p ∧ p < q ∧ q ≤ n) : (d x p)^(1/p) ≥ (d x q)^(1/q) :=
by sorry

theorem maclaurin_newton_inequality_part_2
  (p : ℕ) (hp : 1 ≤ p ∧ p < n) : (d x p)^2 ≥ (d x (p-1)) * (d x (p+1)) :=
by sorry

end maclaurin_newton_inequality_part_1_maclaurin_newton_inequality_part_2_l453_453645


namespace average_speed_l453_453669

theorem average_speed (speed1 speed2 time1 time2: ℝ) (h1 : speed1 = 60) (h2 : time1 = 3) (h3 : speed2 = 85) (h4 : time2 = 2) : 
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 70 :=
by
  -- Definitions
  have distance1 := speed1 * time1
  have distance2 := speed2 * time2
  have total_distance := distance1 + distance2
  have total_time := time1 + time2
  -- Proof skeleton
  sorry

end average_speed_l453_453669


namespace problem_l453_453501

variable (f : ℝ → ℝ)
variable (P : ∀ x y > 0, f (x * y) ≤ f x * f y)

theorem problem 
  (x : ℝ) (hx : x > 0)
  (n : ℕ) (hn : n > 0) :
  f (x ^ n) ≤ f x * (f (x ^ 2))^(1/2) * (f (x ^ 3))^(1/3) * ... * (f (x ^ n))^(1/n) := sorry

end problem_l453_453501


namespace kerosene_price_increase_l453_453876

theorem kerosene_price_increase (P C : ℝ) (x : ℝ)
  (h1 : 1 = (1 + x / 100) * 0.8) :
  x = 25 := by
  sorry

end kerosene_price_increase_l453_453876


namespace compute_u_dot_v_cross_w_l453_453902

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 3))]

-- Given conditions: u, v, z are unit vectors
variables (u v z : euclidean_space ℝ (fin 3))
-- w defined by the given equation
variable (w : euclidean_space ℝ (fin 3)) 
-- Assume conditions
hypothesis (hu : ∥u∥ = 1)
hypothesis (hv : ∥v∥ = 1)
hypothesis (hz : ∥z∥ = 1)
hypothesis (h_w : w = (u ⨯ (v + z)) + z)
hypothesis (h_wu : ⟪w, u⟫ = 0)

theorem compute_u_dot_v_cross_w 
  : ⟪u, v ⨯ w⟫ = 1 + ⟪v, z⟫ := 
sorry

end compute_u_dot_v_cross_w_l453_453902


namespace balloons_remaining_l453_453102

variable (bags_round : ℕ) (balloons_per_bag_round : ℕ)
          (bags_long : ℕ) (balloons_per_bag_long : ℕ)
          (bags_heart : ℕ) (balloons_per_bag_heart : ℕ)
          (bags_star : ℕ) (balloons_per_bag_star : ℕ)
          (defective_rate_round : ℕ → ℝ) (defective_rate_long : ℕ → ℝ)
          (defective_rate_heart : ℕ → ℝ) (defective_rate_star : ℕ → ℝ)
          (burst_round : ℕ) (burst_long : ℕ)
          (burst_heart : ℕ) (burst_star : ℕ)
          (tot_remaining : ℕ)

def total_balloons_round := bags_round * balloons_per_bag_round
def total_balloons_long := bags_long * balloons_per_bag_long
def total_balloons_heart := bags_heart * balloons_per_bag_heart
def total_balloons_star := bags_star * balloons_per_bag_star

def defective_balloons_round := (defective_rate_round total_balloons_round).toNat
def defective_balloons_long := (defective_rate_long total_balloons_long).toNat
def defective_balloons_heart := (defective_rate_heart total_balloons_heart).toNat
def defective_balloons_star := (defective_rate_star total_balloons_star).toNat

def non_defective_round := total_balloons_round - defective_balloons_round
def non_defective_long := total_balloons_long - defective_balloons_long
def non_defective_heart := total_balloons_heart - defective_balloons_heart
def non_defective_star := total_balloons_star - defective_balloons_star

def remaining_round := non_defective_round - burst_round
def remaining_long := non_defective_long - burst_long
def remaining_heart := non_defective_heart - burst_heart
def remaining_star := non_defective_star - burst_star

theorem balloons_remaining (h1 : bags_round = 5) 
                           (h2 : balloons_per_bag_round = 25)
                           (h3 : bags_long = 4)
                           (h4 : balloons_per_bag_long = 35)
                           (h5 : bags_heart = 3)
                           (h6 : balloons_per_bag_heart = 40)
                           (h7 : bags_star = 2)
                           (h8 : balloons_per_bag_star = 50)
                           (h9 : ∀ n, defective_rate_round n = 0.1)
                           (h10 : ∀ n, defective_rate_long n = 0.05)
                           (h11 : ∀ n, defective_rate_heart n = 0.15)
                           (h12 : ∀ n, defective_rate_star n = 0.08)
                           (h13 : burst_round = 5)
                           (h14 : burst_long = 7)
                           (h15 : burst_heart = 3)
                           (h16 : burst_star = 4)
                           : remaining_round + remaining_long + remaining_heart + remaining_star = 421 := sorry

end balloons_remaining_l453_453102


namespace geometry_problem_l453_453870

variables {A B C E F M N D : Type*} [Field A] [EuclideanGeometry B C]
  (h₁ : is_acute_triangle A B C)
  (h₂ : collinear_points E F → F ∈ segment B C → ∠ B A E = ∠ C A F)
  (h₃ : foot_perpendicular F A B M)
  (h₄ : foot_perpendicular F A C N)
  (h₅ : extend_to_circumcircle A E D)
  (h_area : area_quadrilateral A M D N = area_triangle A B C)

-- The theorem statement
theorem geometry_problem (h₁ : is_acute_triangle A B C)
    (h₂ : collinear_points E F → F ∈ segment B C → ∠ B A E = ∠ C A F)
    (h₃ : foot_perpendicular F A B M)
    (h₄ : foot_perpendicular F A C N)
    (h₅ : extend_to_circumcircle A E D)
    (h_area : area_quadrilateral A M D N = area_triangle A B C) :
    h_area = area_quadrilateral A M D N := sorry

end geometry_problem_l453_453870


namespace travel_time_l453_453159

def speed : ℝ := 60  -- Speed of the car in miles per hour
def distance : ℝ := 300  -- Distance to the campground in miles

theorem travel_time : distance / speed = 5 := by
  sorry

end travel_time_l453_453159


namespace sum_of_roots_l453_453907

theorem sum_of_roots (k p x1 x2 : ℝ) (h1 : (4 * x1 ^ 2 - k * x1 - p = 0))
  (h2 : (4 * x2 ^ 2 - k * x2 - p = 0))
  (h3 : x1 ≠ x2) : 
  x1 + x2 = k / 4 :=
sorry

end sum_of_roots_l453_453907


namespace q_is_correct_l453_453573

noncomputable def q (x : ℝ) : ℝ :=
  8 * x^4 + 35 * x^3 + 40 * x^2 + 2 - (2 * x^6 + 5 * x^4 + 10 * x)

theorem q_is_correct (x : ℝ) :
  q(x) = -2 * x^6 + 3 * x^4 + 35 * x^3 + 40 * x^2 - 10 * x + 2 :=
by
  sorry

end q_is_correct_l453_453573


namespace tangent_line_at_a_eq_0_decreasing_on_interval_minimum_value_of_g_l453_453042

-- Definitions
def f (x a : ℝ) : ℝ := x^2 + a * x - Real.log x
def g (x a : ℝ) : ℝ := f x a - x^2

-- (I) Tangent line problem
theorem tangent_line_at_a_eq_0 : 
  let a := 0
  in ∀ (x : ℝ), 
    let y := f x a 
    in x = 1 → (tangent_slope f a 1) * (x - 1) + y = 0 := 
by
  sorry

-- (II) Decreasing function on interval
theorem decreasing_on_interval :
  (∀ x ∈ Icc (1 : ℝ) 2, deriv (λ x, f x a) x ≤ 0) ↔ a ≤ -7 / 2 :=
by 
  sorry

-- (III) Minimum value of g(x)
theorem minimum_value_of_g :
  (∃ a : ℝ, ∀ x ∈ Ioo 0 Real.exp 1, g x a = 3) ↔ a = Real.exp 2 :=
by
  sorry

end tangent_line_at_a_eq_0_decreasing_on_interval_minimum_value_of_g_l453_453042


namespace fraction_of_shaded_circle_l453_453487

theorem fraction_of_shaded_circle (total_regions shaded_regions : ℕ) (h1 : total_regions = 4) (h2 : shaded_regions = 1) :
  shaded_regions / total_regions = 1 / 4 := by
  sorry

end fraction_of_shaded_circle_l453_453487


namespace shells_picked_in_morning_l453_453915

-- Definitions based on conditions
def total_shells : ℕ := 616
def afternoon_shells : ℕ := 324

-- The goal is to prove that morning_shells = 292
theorem shells_picked_in_morning (morning_shells : ℕ) (h : total_shells = morning_shells + afternoon_shells) : morning_shells = 292 := 
by
  sorry

end shells_picked_in_morning_l453_453915


namespace incenters_symmetric_l453_453108

variables {α : Type*} [Euclidean_geometry α]

open Euclidean_geometry

-- Definitions and conditions from the problem
variables {A B C N K L P I J Q : α}
variable CN : line α
variables (on_AB : N ∈ segment A B) (on_BC : K ∈ segment B C) (on_CA : L ∈ segment C A)
variables (bisector_C : CN ∈ angle_bisector A C B) 
variables (eq_AL_BK : segment_len A L = segment_len B K)
variable (P_def : P ∈ (line_through B L ∩ line_through A K))
variables (incenter_I : is_incenter I (triangle B P K)) (incenter_J : is_incenter J (triangle A L P))
variables (Q_def : Q ∈ line_through I J ∩ CN)

-- Statement to prove
theorem incenters_symmetric (h1 : bisector_C)
    (h2 : eq_AL_BK) (h3 : P_def) (h4 : incenter_I) (h5 : incenter_J)
    (h6 : Q_def) : segment_len I Q = segment_len J P := sorry

end incenters_symmetric_l453_453108


namespace eccentricity_of_ellipse_l453_453767

open Real

theorem eccentricity_of_ellipse {a b : ℝ} (h₁ : a > b) (h₂ : b > 0)
  (F₁ F₂ P : ℝ × ℝ) (h₃ : is_ellipse a b (F₁, F₂))
  (h₄ : angle F₁ P F₂ = π / 3) (h₅ : dist P F₁ = 5 * dist P F₂) :
  ellipse_eccentricity a b (F₁, F₂) = (sqrt (21 : ℝ)) / 6 :=
sorry

end eccentricity_of_ellipse_l453_453767


namespace condition_a_gt_1_iff_a_gt_0_l453_453578

theorem condition_a_gt_1_iff_a_gt_0 : ∀ (a : ℝ), (a > 1) ↔ (a > 0) :=
by 
  sorry

end condition_a_gt_1_iff_a_gt_0_l453_453578


namespace tenth_term_arithmetic_sequence_l453_453607

theorem tenth_term_arithmetic_sequence (a d : ℤ) 
  (h1 : a + 2 * d = 23) (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
  by
    sorry

end tenth_term_arithmetic_sequence_l453_453607


namespace chandra_should_read_482_l453_453603

noncomputable def chandraLastPage (x : ℕ) : Prop :=
  let total_pages := 820
  let bob_reads_time := 50
  let chandra_reads_time := 35
  x = (820 * 10 / 17)

theorem chandra_should_read_482 : chandraLastPage 482 :=
by
  let total_pages := 820
  let bob_reads_time := 50
  let chandra_reads_time := 35
  let ratio := 7 / 10
  let x := 482
  have h : ratio * x = total_pages - x :=
    calc 
      7 / 10 * 482 = 337.4 := by sorry
      820 - 482 = 338 := by sorry
  sorry

end chandra_should_read_482_l453_453603


namespace g_inverse_of_point_two_five_l453_453909

noncomputable def g (x : ℝ) := (x^3 - 2) / 4

theorem g_inverse_of_point_two_five :
  ∃ x, g x = 0.25 ∧ x = real.cbrt 3 := 
by
  sorry

end g_inverse_of_point_two_five_l453_453909


namespace find_ellipse_equation_and_t_constant_l453_453381

-- Conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Problem Setup
def A : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (1, 0)

-- Main Theorem
theorem find_ellipse_equation_and_t_constant (a b : ℝ) :
  ellipse a b (-2) 0 ∧ F = (1, 0) →
  (a^2 = 4 ∧ b^2 = 3 ∧
    (∃ t > 1, ∀ (x0 y0 : ℝ), ellipse 2 sqrt(3) x0 y0 → y0 ≠ 0 → let M : ℝ × ℝ := ((x0 - 2) / 2, y0 / 2),
    line_through_origin_parallel_to_AP : ℝ → ℝ := λ x, (y0 / (x0 + 2)) * x,
    Q : ℝ × ℝ := (t, t * y0 / (x0 + 2)),
    OM_slope : ℝ := y0 / (x0 - 2),
    FQ_slope : ℝ := t * y0 / ((t - 1) * (x0 + 2)),
    OM_slope * FQ_slope = -1 )) :=
sorry

end find_ellipse_equation_and_t_constant_l453_453381


namespace interval_of_monotonic_increase_of_power_function_through_point_l453_453178

noncomputable def power_function (x : ℝ) (n : ℕ) : ℝ := x^n

theorem interval_of_monotonic_increase_of_power_function_through_point {n : ℕ} (h : power_function 2 n = 8) : 
  ∃ I : set ℝ, I = set.univ ∧ ∀ x₁ x₂ ∈ I, x₁ < x₂ → power_function x₁ 3 ≤ power_function x₂ 3 := 
by
  sorry

end interval_of_monotonic_increase_of_power_function_through_point_l453_453178


namespace leaves_fall_l453_453691

theorem leaves_fall (planned_trees : ℕ) (tree_multiplier : ℕ) (leaves_per_tree : ℕ) (h1 : planned_trees = 7) (h2 : tree_multiplier = 2) (h3 : leaves_per_tree = 100) :
  (planned_trees * tree_multiplier) * leaves_per_tree = 1400 :=
by
  rw [h1, h2, h3]
  -- Additional step suggestions for interactive proof environments, e.g.,
  -- Have: 7 * 2 = 14
  -- Goal: 14 * 100 = 1400
  sorry

end leaves_fall_l453_453691


namespace complex_calculation_1_complex_calculation_2_l453_453708

-- Proof for the first expression
theorem complex_calculation_1 :
  (-1/2 + (Real.sqrt 3) / 2 * Complex.i) * (2 - Complex.i) * (3 + Complex.i) = 
  (-3 / 2) + (5 * (Real.sqrt 3)) / 2 + (((7 * (Real.sqrt 3)) + 1) / 2 * Complex.i) :=
by
  sorry

-- Proof for the second expression
theorem complex_calculation_2 :
  ((Real.sqrt 2) + (Real.sqrt 2) * Complex.i)^2 * (4 + 5 * Complex.i) / 
  ((5 - 4 * Complex.i) * (1 - Complex.i)) = 
  62 / 41 + (80 / 41) * Complex.i :=
by
  sorry

end complex_calculation_1_complex_calculation_2_l453_453708


namespace problem_1_problem_2_l453_453040

-- Define the function f and its derivative condition
def f (x : ℝ) : ℝ := 3 * x ^ 2 + 2 * x * (f' 1)
def f' (x : ℝ) : ℝ := deriv f x

-- Problem 1: Prove f'(1) = -6
theorem problem_1 : f'(1) = -6 := 
by sorry

-- Problem 2: Prove the equation of the tangent line at (1, f(1)) is y = -6x - 3
theorem problem_2 : ∀ x : ℝ, (f'(1) * (x - 1) + f 1 = -6 * x - 3) := 
by sorry

end problem_1_problem_2_l453_453040


namespace three_digit_numbers_form_3_pow_l453_453831

theorem three_digit_numbers_form_3_pow (n : ℤ) : 
  ∃! (n : ℤ), 100 ≤ 3^n ∧ 3^n ≤ 999 :=
by {
  use [5, 6],
  sorry
}

end three_digit_numbers_form_3_pow_l453_453831


namespace proposition_truth_value_l453_453014

-- Definitions of the propositions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (|x|) ≥ 1

-- The proof problem statement
theorem proposition_truth_value : (p ∧ q) ∧ ¬ (¬p ∧ q) ∧ ¬ (p ∧ ¬q) ∧ ¬ (¬ (p ∨ q)) :=
by
  sorry

end proposition_truth_value_l453_453014


namespace fibonacci_divisibility_l453_453144

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_divisibility (m n : ℕ) (h : 2 < m) :
  (fibonacci n % fibonacci m = 0) ↔ (n % m = 0) :=
by sorry

end fibonacci_divisibility_l453_453144


namespace simplify_expression_l453_453565

theorem simplify_expression :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
  sorry

end simplify_expression_l453_453565


namespace powers_of_3_but_not_9_l453_453426

theorem powers_of_3_but_not_9 : 
  {n : ℕ | 0 < n ∧ n < 500000 ∧
    ∃ k : ℕ, n = 3^k ∧ ¬ ∃ m : ℕ, n = 9^m }.to_finset.card = 6 :=
by sorry

end powers_of_3_but_not_9_l453_453426


namespace domain_lg_sin_sqrt_cos_l453_453966

def domain_of_function : Set ℝ := { x | ∃ k : ℤ, (π / 3 + 2 * k * π ≤ x ∧ x < 5 * π / 6 + 2 * k * π) }

theorem domain_lg_sin_sqrt_cos (x : ℝ) :
  (2 * sin x - 1 > 0) → (1 - 2 * cos x ≥ 0) ↔ (x ∈ domain_of_function) :=
by {
  -- The proof step is omitted as per instructions
  sorry
}

end domain_lg_sin_sqrt_cos_l453_453966


namespace product_of_consecutive_integers_is_perfect_square_l453_453146

theorem product_of_consecutive_integers_is_perfect_square (n : ℤ) :
    n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 :=
sorry

end product_of_consecutive_integers_is_perfect_square_l453_453146


namespace max_a_condition_l453_453024

variable (a x : ℝ)

theorem max_a_condition (H : ∀ x ∈ set.Icc (1 / Real.exp 1) 2, (a + Real.exp 1) * x - 1 - Real.log x ≤ 0) :
  a ≤ 1 / 2 + Real.log 2 / 2 - Real.exp 1 :=
sorry

end max_a_condition_l453_453024


namespace max_rect_area_with_given_perimeter_l453_453213

-- Define the variables used in the problem
def length_of_wire := 12
def max_area (x : ℝ) := -(x - 3)^2 + 9

-- Lean Statement for the problem
theorem max_rect_area_with_given_perimeter : ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 6 → (x * (6 - x) ≤ A)) ∧ A = 9 :=
by
  sorry

end max_rect_area_with_given_perimeter_l453_453213


namespace inequality_proof_l453_453893

theorem inequality_proof (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = real.sqrt 2): 
    1 / real.sqrt (1 + a^2) + 1 / real.sqrt (1 + b^2) + 1 / real.sqrt (1 + c^2) ≥ 2 + 1 / real.sqrt 3 :=
by sorry

end inequality_proof_l453_453893


namespace verify_minor_premise_l453_453878

-- Define the premises and conclusion
def major_premise : Prop := ∀ r : Type, (rectangle r) → (parallelogram r)
def minor_premise : Prop := ∀ t : Type, ¬ (parallelogram t)
def conclusion : Prop := ∀ t : Type, ¬ (rectangle t)

-- Theorem to prove the minor premise
theorem verify_minor_premise (h_major : major_premise) (h_minor : minor_premise) (h_conclusion: conclusion) :
  ¬ (parallelogram (triangle : Type)) := 
begin
  apply h_minor,
  sorry
end

end verify_minor_premise_l453_453878


namespace luke_fish_fillets_l453_453729

theorem luke_fish_fillets : 
  (∃ (catch_rate : ℕ) (days : ℕ) (fillets_per_fish : ℕ), catch_rate = 2 ∧ days = 30 ∧ fillets_per_fish = 2 → 
  (catch_rate * days * fillets_per_fish = 120)) :=
by
  sorry

end luke_fish_fillets_l453_453729


namespace three_digit_powers_of_three_l453_453822

theorem three_digit_powers_of_three : 
  {n : ℤ | 100 ≤ 3^n ∧ 3^n ≤ 999}.finset.card = 2 :=
by
  sorry

end three_digit_powers_of_three_l453_453822


namespace haley_money_difference_l453_453818

def initial_amount : ℕ := 2
def chores : ℕ := 5
def birthday : ℕ := 10
def neighbor : ℕ := 7
def candy : ℕ := 3
def lost : ℕ := 2

theorem haley_money_difference : (initial_amount + chores + birthday + neighbor - candy - lost) - initial_amount = 17 := by
  sorry

end haley_money_difference_l453_453818


namespace max_min_on_interval_l453_453751

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 + 5

theorem max_min_on_interval : 
  (∃ x ∈ set.Icc (1 : ℝ) 3, ∀ y ∈ set.Icc (1 : ℝ) 3, f y ≤ f x ∧ f x = 5) ∧
  (∃ x ∈ set.Icc (1 : ℝ) 3, ∀ y ∈ set.Icc (1 : ℝ) 3, f x ≤ f y ∧ f x = 1) := 
by 
  sorry

end max_min_on_interval_l453_453751


namespace distance_rational_l453_453558

theorem distance_rational (a b : ℤ) : 
  ∃ k : ℚ, k = (| 4 * b - 3 * a - 4 |) / 5 :=
sorry

end distance_rational_l453_453558


namespace product_log_eq_l453_453789

theorem product_log_eq (m n : ℝ) (h1 : log 2 m = 3.5) (h2 : log 2 n = 0.5) : m * n = 16 :=
sorry

end product_log_eq_l453_453789


namespace inequality_proof_l453_453120

variable {n : ℕ}
variable {a : Fin n → ℝ}

theorem inequality_proof (h_sum : (∑ i, a i) = n) (h_nonneg : ∀ i, 0 ≤ a i) :
    (∑ i, (a i)^2 / (1 + (a i)^4)) ≤ ∑ i, 1 / (1 + a i) :=
sorry

end inequality_proof_l453_453120


namespace negation_of_proposition_l453_453183

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, sin x ≥ -1) ↔ ∃ x : ℝ, sin x < -1 :=
by sorry

end negation_of_proposition_l453_453183


namespace no_perfect_squares_in_sequence_l453_453770
noncomputable theory

def sequence (n : ℕ) : ℕ :=
nat.rec_on n 1 (λ n prev, nat.succ_rec_on n 3 (λ n x_n x_n_minus_1, 6 * x_n - x_n_minus_1))

theorem no_perfect_squares_in_sequence (n : ℕ) : ¬ ∃ m : ℕ, m * m = sequence n :=
sorry

end no_perfect_squares_in_sequence_l453_453770


namespace total_profit_l453_453641

theorem total_profit (a b c c_profit ratio_a ratio_b ratio_c : ℕ) (h1 : a = 8000)
(h2 : b = 4000) (h3 : c = 2000) (h4 : c_profit = 36000)
(h5 : ratio_a = 4) (h6 : ratio_b = 2) (h7 : ratio_c = 1) :
let total_ratio := ratio_a + ratio_b + ratio_c,
    total_profit := (c_profit / ratio_c) * total_ratio
in total_profit = 252000 := 
sorry

end total_profit_l453_453641


namespace topless_cubical_box_l453_453951

theorem topless_cubical_box (L : Type) [fintype L] 
  (squares : set L) 
  (L_shape : set (L → L → Prop)) 
  (can_fold : ∀ (s : L), s ∈ squares → bool) 
  (number_of_valid_configurations : ℕ) :
  # {(s: L) | can_fold s true} = 5 :=
sorry

end topless_cubical_box_l453_453951


namespace quotient_of_polynomial_division_l453_453759
noncomputable def polynomial_division_quotient (dividend divisor : Polynomial ℝ) : Polynomial ℝ × Polynomial ℝ :=
  Polynomial.divMod dividend divisor

theorem quotient_of_polynomial_division :
  polynomial_division_quotient (9 * X^3 - 5 * X^2 + 8 * X + 15) (X - 3) = (9 * X^2 + 22 * X + 74, 237) :=
by
  sorry

end quotient_of_polynomial_division_l453_453759


namespace michael_truck_meetings_l453_453539

theorem michael_truck_meetings :
  let michael_speed := 6
  let truck_speed := 12
  let pail_distance := 200
  let truck_stop_time := 20
  let initial_distance := pail_distance
  ∃ (meetings : ℕ), 
  (michael_speed, truck_speed, pail_distance, truck_stop_time, initial_distance, meetings) = 
  (6, 12, 200, 20, 200, 10) :=
sorry

end michael_truck_meetings_l453_453539


namespace equilateral_triangle_count_l453_453813

-- Defining the set of vertices for the ten-sided regular polygon
def vertices := {B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_{10} : Type}

-- Defining the ten-sided regular polygon
def is_regular_decagon (P : set vertices) : Prop :=
  ∃ p : fin 10 → vertices, (bijective p ∧ ∀ i : fin 10, P (p i))

-- Statement to prove the number of equilateral triangles as 82
theorem equilateral_triangle_count (P : set vertices) (hP : is_regular_decagon P):
  ∃ n : ℕ, n = 82 ∧ ∀ T : set vertices, (set.card T = 3 ∧ is_equilateral_triangle T P) → n = 82 :=
  sorry

end equilateral_triangle_count_l453_453813


namespace problem_statement_l453_453589

noncomputable def g (x : ℝ) : ℝ :=
  sorry

theorem problem_statement : (∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 + x^2) → g 3 = -201 / 8 :=
by
  intro h
  sorry

end problem_statement_l453_453589


namespace infinite_elements_in_Z_union_C_union_A_l453_453413

open Set

noncomputable def universal_set := {x : ℝ | True}

noncomputable def set_of_integers := {x : ℤ | True}

noncomputable def set_A := {x : ℝ | x^2 - x - 6 ≥ 0}

theorem infinite_elements_in_Z_union_C_union_A :
  Infinite (set_of_integers ∩ (Complex ∪ set_A)) :=
begin
  sorry
end

end infinite_elements_in_Z_union_C_union_A_l453_453413


namespace inverse_function_inequality_solution_set_l453_453397

-- Let f(x) = 1 - 2^(-x) for x in ℝ
def f (x : ℝ) : ℝ := 1 - 2^(-x)

-- Definition of the inverse function
def f_inv (y : ℝ) : ℝ := -Real.logb 2 (1 - y)

-- The given conditions
def condition1 (x : ℝ) : Prop := x < 1
def condition2 (x : ℝ) : Prop := -1 < x ∧ x < 1

-- Define the inequality to find the solution set
def inequality (x : ℝ) : Prop :=
  2 * Real.logb 2 (x + 1) + f_inv x ≥ 0

-- Define the solution set for the inequality
def solution_set (x : ℝ) : Prop :=
  0 ≤ x ∧ x < 1

-- Define the proof for the inverse function
theorem inverse_function :
  ∀ x : ℝ, condition1 x → f_inv (f x) = x :=
by sorry

-- Prove the solution set of the inequality given the conditions
theorem inequality_solution_set :
  ∀ x : ℝ, condition2 x → (inequality x ↔ solution_set x) :=
by sorry

end inverse_function_inequality_solution_set_l453_453397


namespace find_a8_l453_453025

-- Define the sum of a geometric sequence.
def geom_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 1 * (1 - q^n) / (1 - q)

-- Define the conditions
def arithmetic_sequence (S : ℕ → ℝ) : Prop :=
  2 * S 9 = S 3 + S 6

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 2 + a 5 = 4) ∧
  arithmetic_sequence (λ n, geom_sum a q n)

-- Theorem with given condition that S_3, S_9, S_6 form an arithmetic sequence
-- and a_2 + a_5 = 4, prove a_8 = 2
theorem find_a8 (a : ℕ → ℝ) (q : ℝ) (h : geometric_sequence a q) :
  a 8 = 2 :=
sorry

end find_a8_l453_453025


namespace lune_area_l453_453295

noncomputable def area_of_lune : ℝ := π * (9 / 16)

theorem lune_area (small_dia large_dia : ℝ) (h1 : small_dia = 1.5) (h2 : large_dia = 3) : 
  area_of_lune = π * (9 / 16) :=
by
  rw [area_of_lune]
  sorry

end lune_area_l453_453295


namespace tangent_x_intercept_eq_neg_three_sevenths_l453_453198

-- Definitions based on conditions from a):
def f (x : ℝ) : ℝ := x^3 + 4 * x + 5
def f'(x : ℝ) : ℝ := 3 * x^2 + 4

-- The proof statement demonstrating the x-intercept is -3/7
theorem tangent_x_intercept_eq_neg_three_sevenths (x_tangent : ℝ) (hx : x_tangent = 1) :
  ∃ x_intercept : ℝ, (y : ℝ) (hy : y = f x_tangent.derive = 0) ((x_intercept * y) = - (3 / 7)) :=
sorry

end tangent_x_intercept_eq_neg_three_sevenths_l453_453198


namespace boys_passed_percentage_l453_453483

theorem boys_passed_percentage
  (total_candidates : ℝ)
  (total_girls : ℝ)
  (failed_percentage : ℝ)
  (girls_passed_percentage : ℝ)
  (boys_passed_percentage : ℝ) :
  total_candidates = 2000 →
  total_girls = 900 →
  failed_percentage = 70.2 →
  girls_passed_percentage = 32 →
  boys_passed_percentage = 28 :=
by
  sorry

end boys_passed_percentage_l453_453483


namespace probability_one_hits_correct_l453_453212

-- Define the probabilities for A hitting and B hitting
noncomputable def P_A : ℝ := 0.4
noncomputable def P_B : ℝ := 0.5

-- Calculate the required probability
noncomputable def probability_one_hits : ℝ :=
  P_A * (1 - P_B) + (1 - P_A) * P_B

-- Statement of the theorem
theorem probability_one_hits_correct :
  probability_one_hits = 0.5 := by 
  sorry

end probability_one_hits_correct_l453_453212


namespace solve_for_a_l453_453190

def quadratic_has_roots (a x1 x2 : ℝ) : Prop :=
  x1 + x2 = a ∧ x1 * x2 = -6 * a^2

theorem solve_for_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : quadratic_has_roots a x1 x2) (h3 : x2 - x1 = 10) : a = 2 :=
by
  sorry

end solve_for_a_l453_453190


namespace six_digit_number_theorem_l453_453427

noncomputable def six_digit_number (a b c d e f : ℕ) : ℕ :=
  10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f

noncomputable def rearranged_number (a b c d e f : ℕ) : ℕ :=
  10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + a

theorem six_digit_number_theorem (a b c d e f : ℕ) (h_a : a ≠ 0) 
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  (h4 : 0 ≤ d ∧ d ≤ 9) (h5 : 0 ≤ e ∧ e ≤ 9) (h6 : 0 ≤ f ∧ f ≤ 9) 
  : six_digit_number a b c d e f = 142857 ∨ six_digit_number a b c d e f = 285714 :=
by
  sorry

end six_digit_number_theorem_l453_453427


namespace not_exists_set_of_9_numbers_min_elements_l453_453238

theorem not_exists_set_of_9_numbers (s : Finset ℕ) 
  (h_len : s.card = 9) 
  (h_median : ∑ x in (s.filter (λ x, x ≤ 2)), 1 ≤ 5) 
  (h_other : ∑ x in (s.filter (λ x, x ≤ 13)), 1 ≤ 4) 
  (h_avg : ∑ x in s = 63) :
  False := sorry

theorem min_elements (n : ℕ) (h_nat: n ≥ 5) :
  ∃ s : Finset ℕ, s.card = 2 * n + 1 ∧
                  ∑ x in (s.filter (λ x, x ≤ 2)), 1 = n + 1 ∧ 
                  ∑ x in (s.filter (λ x, x ≤ 13)), 1 = n ∧
                  ∑ x in s = 14 * n + 7 := sorry

end not_exists_set_of_9_numbers_min_elements_l453_453238


namespace non_real_roots_interval_l453_453453

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l453_453453


namespace base_of_isosceles_triangle_l453_453185

theorem base_of_isosceles_triangle (a b side equil_perim iso_perim : ℕ) 
  (h1 : equil_perim = 60)
  (h2 : 3 * side = equil_perim)
  (h3 : iso_perim = 50)
  (h4 : 2 * side + b = iso_perim)
  : b = 10 :=
by
  sorry

end base_of_isosceles_triangle_l453_453185


namespace ducks_to_total_ratio_l453_453553

-- Definitions based on the given conditions
def totalBirds : ℕ := 15
def costPerChicken : ℕ := 2
def totalCostForChickens : ℕ := 20

-- Proving the desired ratio of ducks to total number of birds
theorem ducks_to_total_ratio : (totalCostForChickens / costPerChicken) + d = totalBirds → d = 15 - (totalCostForChickens / costPerChicken) → 
  (totalCostForChickens / costPerChicken) + d = totalBirds → d = totalBirds - (totalCostForChickens / costPerChicken) →
  d = 5 → (totalBirds - (totalCostForChickens / costPerChicken)) / totalBirds = 1 / 3 :=
by
  sorry

end ducks_to_total_ratio_l453_453553


namespace height_of_box_l453_453270

theorem height_of_box (h : ℝ) :
  (∃ (h : ℝ),
    (∀ (x y z : ℝ), (x = 3) ∧ (y = 3) ∧ (z = h / 2) → true) ∧
    (∀ (x y z : ℝ), (x = 1) ∧ (y = 1) ∧ (z = 1) → true) ∧
    h = 6) :=
sorry

end height_of_box_l453_453270


namespace Q_is_circumcenter_of_CDE_l453_453656

-- Definitions:
variable {l1 l2 : Line}
variable {ω ω1 ω2 : Circle}
variable {A B C D E Q : Point}

-- Conditions in the form of Lean definitions:
def condition1 : ω.Tangent_to_Line l1 ∧ ω.Tangent_to_Line l2 := sorry
def condition2 : ω1.Tangent_to_Point l1 A ∧ ω1.Tangent_to_Circle ω C := sorry
def condition3 : ω2.Tangent_to_Point l2 B ∧ ω2.Tangent_to_Circle ω D ∧ ω2.Tangent_to_Circle ω1 E := sorry
def condition4 : (Line_AD : Line) (Line_BC : Line), Line_AD.Contains A ∧ Line_AD.Contains D ∧ Line_BC.Contains B ∧ Line_BC.Contains C ∧ Intersection Line_AD Line_BC Q := sorry

-- The theorem to prove:
theorem Q_is_circumcenter_of_CDE 
(condition1 : ω.Tangent_to_Line l1 ∧ ω.Tangent_to_Line l2) 
(condition2 : ω1.Tangent_to_Point l1 A ∧ ω1.Tangent_to_Circle ω C) 
(condition3 : ω2.Tangent_to_Point l2 B ∧ ω2.Tangent_to_Circle ω D ∧ ω2.Tangent_to_Circle ω1 E) 
(condition4 : (Line_AD.Contains A ∧ Line_AD.Contains D ∧ Line_BC.Contains B ∧ Line_BC.Contains C ∧ Intersection Line_AD Line_BC Q)) : 
  (Q_is_Circumcenter_of_Triangle C D E) := 
  sorry

end Q_is_circumcenter_of_CDE_l453_453656


namespace opposite_arithmetic_sqrt_81_l453_453597

theorem opposite_arithmetic_sqrt_81 : -(Real.sqrt 81) = -9 :=
by 
  have h1 : Real.sqrt 81 = 9 := by
    sorry
  rw [h1]
  simp

end opposite_arithmetic_sqrt_81_l453_453597


namespace find_a5_l453_453049

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ ∀ n, a (n + 1) = 2 * a n - 3

theorem find_a5 (a : ℕ → ℤ) (h : sequence a) : a 5 = -61 :=
sorry

end find_a5_l453_453049


namespace fourth_even_integer_l453_453363

theorem fourth_even_integer (n : ℤ) (h : (n-2) + (n+2) = 92) : n + 4 = 50 := by
  -- This will skip the proof steps and assume the correct answer
  sorry

end fourth_even_integer_l453_453363


namespace greatest_possible_red_points_l453_453137

-- Definition of the problem in Lean 4
theorem greatest_possible_red_points (points : Finset ℕ) (red blue : Finset ℕ)
  (h_total_points : points.card = 25)
  (h_disjoint : red ∩ blue = ∅)
  (h_union : red ∪ blue = points)
  (h_segment : ∀ (r ∈ red), ∃! b ∈ blue, true) :
  red.card ≤ 13 :=
begin
  -- We assert that the greatest number of red points is at most 13.
  sorry
end

end greatest_possible_red_points_l453_453137


namespace marbles_problem_l453_453355
open ProbabilityTheory

theorem marbles_problem :
  let total_marbles := 36,
      prob_black_black := 13 / 18 in
  ∃ (m n : ℕ), Nat.coprime m n ∧ (∃ a b : ℕ, a + b = total_marbles ∧
  let prob_white_white := 1 / 9 in
  prob_white_white = (m : ℝ) / (n : ℝ) ∧ (m + n = 10)) :=
by
  let total_marbles := 36
  let prob_black_black := 13 / 18
  have h1 : sorry
  have h2 : sorry
  have h3 : sorry
  use 1, 9
  split
  · exact Nat.coprime_one_right 9
  use 1, 35
  split
  · linarith
  split
  · exact show (1 : ℝ) / 9 = 1 / 9 by norm_num
  · exact show 1 + 9 = 10 by linarith

end marbles_problem_l453_453355


namespace cube_face_problem_l453_453695

theorem cube_face_problem (n : ℕ) (h : 0 < n) :
  ((6 * n^2) : ℚ) / (6 * n^3) = 1 / 3 → n = 3 :=
by
  sorry

end cube_face_problem_l453_453695


namespace sum_of_roots_l453_453402

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 1 else f (x - 1) + 1

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, i

theorem sum_of_roots (n : ℕ) (hn : 0 < n) :
  S n = (n * (n - 1)) / 2 :=
by
  sorry

end sum_of_roots_l453_453402


namespace area_of_square_l453_453186

theorem area_of_square 
  {x1 y1 x2 y2 : ℝ} 
  (h1 : (x1, y1) = (1, 3)) 
  (h2 : (x2, y2) = (-2, 7)) 
  (adj : (x1, y1) ≠ (x2, y2)) :
  let side_len := Math.sqrt ((x1 - x2)^2 + (y1 - y2)^2)
  in (side_len)^2 = 25 := 
sorry

end area_of_square_l453_453186


namespace blue_paint_cans_l453_453551

noncomputable def ratio_of_blue_to_green := 4 / 1
def total_cans := 50
def fraction_of_blue := 4 / (4 + 1)
def number_of_blue_cans := fraction_of_blue * total_cans

theorem blue_paint_cans : number_of_blue_cans = 40 := by
  sorry

end blue_paint_cans_l453_453551


namespace symmetric_circles_common_chord_l453_453071

theorem symmetric_circles_common_chord (x y : ℝ) :
  let circle1 := x^2 + y^2 = 8,
      circle2 := x^2 + y^2 + 4 * x - 4 * y = 0
  in is_axis_of_symmetry circle1 circle2 (common_chord circle1 circle2) :=
by
  sorry

end symmetric_circles_common_chord_l453_453071


namespace travel_time_l453_453161

theorem travel_time (distance speed : ℝ) (h1 : distance = 300) (h2 : speed = 60) : 
  distance / speed = 5 := 
by
  sorry

end travel_time_l453_453161


namespace least_subtract_for_divisibility_l453_453625

theorem least_subtract_for_divisibility (n : ℕ) (hn : n = 427398) : 
  (∃ m : ℕ, n - m % 10 = 0 ∧ m = 2) :=
by
  sorry

end least_subtract_for_divisibility_l453_453625


namespace greatest_possible_red_points_l453_453135

theorem greatest_possible_red_points (R B : ℕ) (h1 : R + B = 25)
    (h2 : ∀ r1 r2, r1 < R → r2 < R → r1 ≠ r2 → ∃ (n : ℕ), (∃ b1 : ℕ, b1 < B) ∧ ¬∃ b2 : ℕ, b2 < B) :
  R ≤ 13 :=
by {
  sorry
}

end greatest_possible_red_points_l453_453135


namespace second_year_students_in_sample_l453_453287

theorem second_year_students_in_sample (students_first_year : ℕ) 
    (students_second_year : ℕ) (students_third_year : ℕ) 
    (total_sample_size : ℕ) (total_students : ℕ)
    (h_total_students : students_first_year + students_second_year + students_third_year = total_students) :
  let sample_second_year := (students_second_year * total_sample_size) / total_students in
  sample_second_year = 64 := 
by
  let students_first_year := 400
  let students_second_year := 320
  let students_third_year := 280
  let total_sample_size := 200
  let total_students := students_first_year + students_second_year + students_third_year
  sorry

end second_year_students_in_sample_l453_453287


namespace find_sale_in_fourth_month_l453_453285

variable (sale1 sale2 sale3 sale5 sale6 : ℕ)
variable (TotalSales : ℕ)
variable (AverageSales : ℕ)

theorem find_sale_in_fourth_month (h1 : sale1 = 6335)
                                   (h2 : sale2 = 6927)
                                   (h3 : sale3 = 6855)
                                   (h4 : sale5 = 6562)
                                   (h5 : sale6 = 5091)
                                   (h6 : AverageSales = 6500)
                                   (h7 : TotalSales = AverageSales * 6) :
  ∃ sale4, TotalSales = sale1 + sale2 + sale3 + sale4 + sale5 + sale6 ∧ sale4 = 7230 :=
by
  sorry

end find_sale_in_fourth_month_l453_453285


namespace angle_BIM_right_angle_l453_453097

-- Given definitions and conditions
variables {A B C I M N : Type}
variables [Incenter I A B C] [Midpoint M B C] [Midpoint N A C]
variables (h1 : ∠ AIN = 90°)

-- Required to prove
theorem angle_BIM_right_angle :
  ∠ BIM = 90° :=
sorry

end angle_BIM_right_angle_l453_453097


namespace octagon_rectangles_area_l453_453677

theorem octagon_rectangles_area (H : RegularOctagon.side_length = 1) (P : forall p : Parallelogram, p ∈ RegularOctagon.cut_into_parallelograms p):
  ∃ r1 r2 : Rectangle, 
  r1 ∈ RegularOctagon.cut_into_parallelograms r1 ∧ 
  r2 ∈ RegularOctagon.cut_into_parallelograms r2 ∧ 
  r1 ≠ r2 ∧
  (∑ r in {r1, r2}, Rectangle.area r) = 2 :=
by
  sorry

end octagon_rectangles_area_l453_453677


namespace parallelepiped_edges_parallel_to_axes_l453_453081

theorem parallelepiped_edges_parallel_to_axes 
  (V : ℝ) (a b c : ℝ) 
  (integer_coords : ∀ (x y z : ℝ), x = a ∨ x = 0 ∧ y = b ∨ y = 0 ∧ z = c ∨ z = 0) 
  (volume_cond : V = 2011) 
  (volume_def : V = a * b * c) 
  (a_int : a ∈ ℤ) 
  (b_int : b ∈ ℤ) 
  (c_int : c ∈ ℤ) : 
  a = 1 ∧ b = 1 ∧ c = 2011 ∨ 
  a = 1 ∧ b = 2011 ∧ c = 1 ∨ 
  a = 2011 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end parallelepiped_edges_parallel_to_axes_l453_453081


namespace smallest_prime_divisor_of_sum_of_powers_l453_453629

theorem smallest_prime_divisor_of_sum_of_powers :
  ∃ p, Prime p ∧ p = Nat.gcd (3 ^ 25 + 11 ^ 19) 2 := by
  sorry

end smallest_prime_divisor_of_sum_of_powers_l453_453629


namespace min_value_expression_l453_453765

theorem min_value_expression : ∃ x : ℝ, x = 300 ∧ ∀ y : ℝ, (y^2 - 600*y + 369) ≥ (300^2 - 600*300 + 369) := by
  use 300
  sorry

end min_value_expression_l453_453765


namespace least_entries_to_make_sums_unique_l453_453700

def matrix : list (list ℕ) := [[5, 10, 3], [7, 2, 8], [4, 6, 9]]

def row_sums (m : list (list ℕ)) := m.map list.sum

def column_sums (m : list (list ℕ)) : list ℕ :=
(list.zip_with list.sum (list.transpose m) (repeat 0))

def least_altered_entries (m : list (list ℕ)) : ℕ :=
  (row_sums m ++ column_sums m).nunique - 6

theorem least_entries_to_make_sums_unique :
  least_altered_entries matrix = 1 := sorry

end least_entries_to_make_sums_unique_l453_453700


namespace sufficient_but_not_necessary_l453_453408

variable (a : ℝ)

def p := a > 1
def q := ∀ x : ℝ, deriv (λ x : ℝ, a * x - sin x) x ≥ 0

theorem sufficient_but_not_necessary : (p a → q a) ∧ (q a → ¬p a) :=
by 
  sorry

end sufficient_but_not_necessary_l453_453408


namespace minimum_value_ln_squared_div_x_l453_453364

noncomputable def f (x : ℝ) : ℝ := (Real.log x) ^ 2 / x

theorem minimum_value_ln_squared_div_x :
  ∃ x : ℝ, x > 0 ∧ f x = 0 :=
by
  use 1
  split
  · linarith
  · sorry

end minimum_value_ln_squared_div_x_l453_453364


namespace greatest_possible_red_points_l453_453134

theorem greatest_possible_red_points (R B : ℕ) (h1 : R + B = 25)
    (h2 : ∀ r1 r2, r1 < R → r2 < R → r1 ≠ r2 → ∃ (n : ℕ), (∃ b1 : ℕ, b1 < B) ∧ ¬∃ b2 : ℕ, b2 < B) :
  R ≤ 13 :=
by {
  sorry
}

end greatest_possible_red_points_l453_453134


namespace find_a_l453_453532

variable {a b c : ℝ}

theorem find_a (h1 : a^3 / b^2 = 4) (h2 : b^4 / c^3 = 8) (h3 : c^5 / a^4 = 32) : 
  a = 2^(12/4.6) :=
sorry

end find_a_l453_453532


namespace minimize_segment_length_l453_453815

variable {AC BD : Ray}
variable {M N P : Point}

/-- Given two intersecting rays AC and BD, and points M and N chosen on these rays, respectively, 
such that AM = BN, then the length of segment MN is minimized if and only if PB is perpendicular to AP. -/
theorem minimize_segment_length (h_intersect : AC ≠ BD) 
  (h_M_on_AC : M ∈ AC) (h_N_on_BD : N ∈ BD) (h_AM_eq_BN: dist A M = dist B N) 
  (h_P_on_line : P = line_through M parallel_to BN) : 
  (dist M N) = min_length ↔ is_perpendicular PB AP := 
by
  sorry

end minimize_segment_length_l453_453815


namespace tan_B_in_right_triangle_l453_453914

theorem tan_B_in_right_triangle
  (A B C F G : Type)
  [right_triangle A B C]
  [point_on_line F (segment A B)]
  [point_on_line G (segment A B)]
  [trisect_angle F C G A]
  (h_ratio : FG / BG = 3 / 7) :
  ∃ B_tangent : Real,
      B_tangent = (Real.sqrt (4 * (58 - 21 * Real.sqrt 3) + 48 - (Real.sqrt (58 - 21 * Real.sqrt 3) + 3) ^ 2)) /
                  (Real.sqrt (58 - 21 * Real.sqrt 3) + 3) :=
sorry

end tan_B_in_right_triangle_l453_453914


namespace constant_term_expansion_l453_453792

noncomputable def a : ℝ := ∫ x in -1..(1 : ℝ), (1 + Real.sqrt(1 - x^2))

theorem constant_term_expansion:
  let expr := (λ x : ℝ, (a - Real.pi / 2) * x - 1 / x) in
  a = 2 + Real.pi / 2 →
  (expr 1) ^ 6 = -160 :=
by
  sorry

end constant_term_expansion_l453_453792


namespace distance_between_homes_is_50_l453_453890

noncomputable def distance_between_homes : ℝ :=
  let J_speed := 5 // János's walking speed in km/hour
  let L_speed := 4.5 // Lajos's walking speed in km/hour
  let car_speed := 30 // Speed of the car in km/hour
  let L_fraction_by_car := (3 / 5) -- Fraction of the route Lajos travels by car
  let J_fraction_by_car := (3 / 5) -- Fraction of the route János travels by car
  let L_fraction_by_walk := 1 - L_fraction_by_car
  let J_fraction_by_walk := 1 - J_fraction_by_car
  let J_distance := 1.5 * L_distance
  let J_time := (J_distance * J_fraction_by_walk / J_speed) + (J_distance * J_fraction_by_car / car_speed)
  let L_distance := L_distance -- unknown distance variable for Lajos's route to Magyarfalu
  let L_time := (L_distance * L_fraction_by_walk / L_speed) + (L_distance * L_fraction_by_car / car_speed)
  let time_difference := 4 / 60 -- Time difference between arrival in hours
  
  L_distance + 1.5 * L_distance = 50 / (L_distance + J_distance)

theorem distance_between_homes_is_50 : distance_between_homes = 50 := 
  by 
  sorry

end distance_between_homes_is_50_l453_453890


namespace xiao_hua_correct_questions_l453_453475

-- Definitions of the problem conditions
def n : Nat := 20
def p_correct : Int := 5
def p_wrong : Int := -2
def score : Int := 65

-- Theorem statement to prove the number of correct questions
theorem xiao_hua_correct_questions : 
  ∃ k : Nat, k = ((n : Int) - ((n * p_correct - score) / (p_correct - p_wrong))) ∧ 
               k = 15 :=
by
  sorry

end xiao_hua_correct_questions_l453_453475


namespace num_three_digit_powers_of_three_l453_453826

theorem num_three_digit_powers_of_three : 
  ∃ n1 n2 : ℕ, 100 ≤ 3^n1 ∧ 3^n1 ≤ 999 ∧ 100 ≤ 3^n2 ∧ 3^n2 ≤ 999 ∧ n1 ≠ n2 ∧ 
  (∀ n : ℕ, 100 ≤ 3^n ∧ 3^n ≤ 999 → n = n1 ∨ n = n2) :=
sorry

end num_three_digit_powers_of_three_l453_453826


namespace max_min_on_interval_l453_453750

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 + 5

theorem max_min_on_interval : 
  (∃ x ∈ set.Icc (1 : ℝ) 3, ∀ y ∈ set.Icc (1 : ℝ) 3, f y ≤ f x ∧ f x = 5) ∧
  (∃ x ∈ set.Icc (1 : ℝ) 3, ∀ y ∈ set.Icc (1 : ℝ) 3, f x ≤ f y ∧ f x = 1) := 
by 
  sorry

end max_min_on_interval_l453_453750


namespace probability_A_hits_B_misses_probability_A_B_equal_hits_l453_453123

-- Define the conditions
def Pr_A : ℝ := 3/4
def Pr_B : ℝ := 4/5
def Pr_not_B : ℝ := 1 - Pr_B

-- Define the probabilities for multiple shots
def comb (n k : ℕ) : ℕ := Nat.choose n k

def Pr_Ak (k : ℕ) : ℝ := comb 2 k * (Pr_A ^ k) * ((1 - Pr_A) ^ (2 - k))
def Pr_Bl (l : ℕ) : ℝ := comb 2 l * (Pr_B ^ l) * ((1 - Pr_B) ^ (2 - l))

-- Problem (Ⅰ): Probability that A hits while B does not
theorem probability_A_hits_B_misses : Pr_A * Pr_not_B = 3 / 20 := by
  sorry

-- Problem (Ⅱ): Probability that A and B hit the target an equal number of times in two shots
theorem probability_A_B_equal_hits : (Pr_Ak 0 * Pr_Bl 0) + (Pr_Ak 1 * Pr_Bl 1) + (Pr_Ak 2 * Pr_Bl 2) = 0.37 := by
  sorry

end probability_A_hits_B_misses_probability_A_B_equal_hits_l453_453123


namespace sign_up_ways_l453_453614

theorem sign_up_ways : (3 ^ 4) = 81 :=
by
  sorry

end sign_up_ways_l453_453614


namespace inequality_min_value_l453_453762

theorem inequality_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m : ℝ, (x + 2 * y) * (2 / x + 1 / y) ≥ m ∧ m ≤ 8 :=
by
  sorry

end inequality_min_value_l453_453762


namespace Cheyenne_earnings_l453_453331

-- Given conditions
def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

-- Calculations
def cracked_pots : ℕ := (cracked_fraction * total_pots).toNat
def good_pots : ℕ := total_pots - cracked_pots
def total_earnings : ℕ := good_pots * price_per_pot

-- Theorem statement
theorem Cheyenne_earnings : total_earnings = 1920 := by
  sorry

end Cheyenne_earnings_l453_453331


namespace non_real_roots_b_range_l453_453438

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l453_453438


namespace students_taking_french_only_l453_453476

/-- In a group of 28 junior high school students, some take French, 10 take Spanish, 
and 4 take both languages. The students taking both French and Spanish are not counted 
with the ones taking French or the 10 taking Spanish. There are 13 students not taking 
either French or Spanish. Prove that the number of students taking French only is 1. -/
theorem students_taking_french_only (total : ℕ) (spanish_only : ℕ) (both_languages : ℕ) (neither : ℕ) :
  total = 28 → spanish_only = 10 → both_languages = 4 → neither = 13 → 
  ∃ (french_only : ℕ), 
  french_only + spanish_only + both_languages + neither = total ∧ french_only = 1 :=
by
  intros h1 h2 h3 h4
  use 1
  split
  · rw [h1, h2, h3, h4]
    norm_num
  · rfl

end students_taking_french_only_l453_453476


namespace proof_problem_l453_453015

variables (R : Type*) [Real R]

def p : Prop := ∃ x : R, Real.sin x < 1
def q : Prop := ∀ x : R, Real.exp (abs x) ≥ 1

theorem proof_problem : p ∧ q := 
by 
  sorry

end proof_problem_l453_453015


namespace correct_order_of_x_y_z_l453_453795

noncomputable def x (a b : ℝ) : ℝ := a^b
noncomputable def y (a b : ℝ) : ℝ := b^a
noncomputable def z (a b : ℝ) : ℝ := Real.log a b

theorem correct_order_of_x_y_z (a b : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (x a b < y a b) ∧ (y a b < z a b) :=
by sorry

end correct_order_of_x_y_z_l453_453795


namespace yellow_daisies_percentage_l453_453855

noncomputable def total_flowers : ℕ := 120
noncomputable def percentage_of_tulips : ℝ := 0.3
noncomputable def half : ℝ := 0.5
noncomputable def three_fifths : ℝ := 0.6

theorem yellow_daisies_percentage :
  let tulips := percentage_of_tulips * total_flowers,
      red_tulips := half * tulips,
      yellow_tulips := tulips - red_tulips,
      daisies := total_flowers - tulips,
      yellow_daisies := three_fifths * daisies
  in (yellow_daisies / total_flowers * 100) = 40 :=
by
  sorry

end yellow_daisies_percentage_l453_453855


namespace minimum_set_size_l453_453242

theorem minimum_set_size (n : ℕ) :
  (2 * n + 1) ≥ 11 :=
begin
  have h1 : 7 * (2 * n + 1) ≤ 15 * n + 2,
  sorry,
  have h2 : 14 * n + 7 ≤ 15 * n + 2,
  sorry,
  have h3 : n ≥ 5,
  sorry,
  show 2 * n + 1 ≥ 11,
  from calc
    2 * n + 1 = 2 * 5 + 1 : by linarith
          ... ≥ 11 : by linarith,
end

end minimum_set_size_l453_453242


namespace staff_member_pays_l453_453282

variable (d : ℝ) (h1 : 0 < d)

theorem staff_member_pays (d : ℝ) (h1 : 0 < d) : 
    let price_after_25 := d * 0.75
    let price_after_staff := price_after_25 * 0.80
    let price_after_coupon := price_after_staff * 0.90
    let price_including_tax := price_after_coupon * 1.08 
in price_including_tax = 0.5832 * d := by
    sorry

end staff_member_pays_l453_453282


namespace smoothie_price_l453_453150

variable (x : ℝ)
variable (price_per_cup : ℝ) (cakes : ℝ) (total_revenue : ℝ)

def condition1 : price_per_cup = x := sorry
def condition2 : cakes = 2 := sorry
def condition3 : total_revenue = 156 := sorry
def condition4 : 40 * x + 18 * cakes = total_revenue := sorry

theorem smoothie_price : x = 3 :=
by
  assume condition1
  assume condition2
  assume condition3
  assume condition4
  sorry

end smoothie_price_l453_453150


namespace general_term_a_sum_b_n_l453_453511

noncomputable def S (n : ℕ) : ℚ := (finset.range n).sum (λi, a (i + 1))

noncomputable def a : ℕ → ℚ
| 1       := 1
| (n + 2) := -2 / ((2 * (n + 2) - 1) * (2 * (n + 1) - 1))

noncomputable def b (n : ℕ) : ℚ := S (n + 1) / (2 * n + 1)

noncomputable def T (n : ℕ) : ℚ := (finset.range n).sum (λi, b (i + 1))

theorem general_term_a (n : ℕ) : 
  a n = if n = 1 then 1 else -2 / ((2 * n - 1) * (2 * n - 3)) :=
sorry

theorem sum_b_n (n : ℕ) : 
  T n = n / (2 * n + 1) :=
sorry

end general_term_a_sum_b_n_l453_453511


namespace residue_S_mod_2016_l453_453509

theorem residue_S_mod_2016 :
  let S := (Finset.range 2016).sum (λ n, if n % 2 = 0 then - (n + 1) / 2 else (n + 1) / 2)
  in S % 2016 = 1008 :=
by
  let S := (Finset.range 2016).sum (λ n, if n % 2 = 0 then - (n + 1) / 2 else (n + 1) / 2)
  have h : S = -1008 := sorry
  have mod_eq := Int.modEq_iff_dvd.mpr (sorry : ↑(S + 1008) % 2016 = 0)
  show S % 2016 = 1008, from (Int.modEq_iff_dvd.mp mod_eq).symm

end residue_S_mod_2016_l453_453509


namespace projection_of_b_onto_a_l453_453418

variables (a b : ℝ^3) -- Assuming vectors in a 3-dimensional real space

-- Given conditions
def magnitude_a : Prop := ∥a∥ = 3
def magnitude_b : Prop := ∥b∥ = 2 * Real.sqrt 3
def dot_product : Prop := a • b = -9

-- Projection formula derived from the given conditions
def projection : ℝ := (a • b) / ∥a∥

-- The theorem to be proved
theorem projection_of_b_onto_a
  (h1 : magnitude_a a)
  (h2 : magnitude_b b)
  (h3 : dot_product a b) :
  projection a b = -3 :=
sorry

end projection_of_b_onto_a_l453_453418


namespace sampling_methods_correct_l453_453293

-- Definitions of the conditions:
def is_simple_random_sampling (method : String) : Prop := 
  method = "random selection of 24 students by the student council"

def is_systematic_sampling (method : String) : Prop := 
  method = "selection of students numbered from 001 to 240 whose student number ends in 3"

-- The equivalent math proof problem:
theorem sampling_methods_correct :
  is_simple_random_sampling "random selection of 24 students by the student council" ∧
  is_systematic_sampling "selection of students numbered from 001 to 240 whose student number ends in 3" :=
by
  sorry

end sampling_methods_correct_l453_453293


namespace solution_count_l453_453764

theorem solution_count (a : ℝ) :
  (∀ x : ℝ, ¬ (lg (2 - x^2) / lg (x - a) = 2)) ↔ 
    (a ≤ -2 ∨ a = 0 ∨ a ≥ sqrt 2) ∨
  (∃! x : ℝ, lg (2 - x^2) / lg (x - a) = 2) ↔ 
    (a < -sqrt 2 ∧ a > -2) ∨
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ lg (2 - x^2) / lg (x - a) = 2) ↔ 
    (a > -sqrt 2 ∧ a < 0 ∨ 0 < a ∧ a < sqrt 2) :=
sorry

end solution_count_l453_453764


namespace count_three_digit_numbers_power_of_three_l453_453836

theorem count_three_digit_numbers_power_of_three :
  { n : ℕ | 100 ≤ 3^n ∧ 3^n ≤ 999 }.toFinset.card = 2 := by
  sorry

end count_three_digit_numbers_power_of_three_l453_453836


namespace log_equation_solution_l453_453428

theorem log_equation_solution (x : ℝ) (h : Real.logBase 36 (x - 5) = 1 / 2) : 
  1 / Real.logBase x 4 = Real.log 11 / Real.log 4 := by
sorry

end log_equation_solution_l453_453428


namespace range_of_b_l453_453410

section TheoremProof

variable (x a b : ℝ)

def SetA : Set ℝ := { x | Real.log (x + 2) / Real.log (1 / 2) < 0 }
def SetB : Set ℝ := { x | (x - a) * (x - b) < 0 }

theorem range_of_b (h : a = -3) : ¬ (SetA ∩ SetB = ∅) ↔ b > -1 := 
  sorry

end TheoremProof

end range_of_b_l453_453410


namespace tangent_line_eq_l453_453586

-- Define necessary conditions and the problem
def curve (x : ℝ) : ℝ := Real.sin x + Real.exp x

def deriv_curve (x : ℝ) : ℝ := Real.cos x + Real.exp x

/-- Show that the tangent line to the curve y = sin x + e^x at x = 0 is y = 2x + 1 -/
theorem tangent_line_eq :
  let y := curve
  let y' := deriv_curve
  let slope := y' 0
  let point_of_tangency := (0 : ℝ, y 0)
  let (x0, y0) := point_of_tangency
  ∀ x, y0 + slope * (x - x0) = 2 * x + 1 := 
by
  sorry

end tangent_line_eq_l453_453586


namespace count_three_digit_numbers_power_of_three_l453_453833

theorem count_three_digit_numbers_power_of_three :
  { n : ℕ | 100 ≤ 3^n ∧ 3^n ≤ 999 }.toFinset.card = 2 := by
  sorry

end count_three_digit_numbers_power_of_three_l453_453833


namespace solve_equation_1_solve_equation_2_l453_453947

theorem solve_equation_1 (x : ℝ) : x^2 - 3 * x = 4 ↔ x = 4 ∨ x = -1 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by
  sorry

end solve_equation_1_solve_equation_2_l453_453947


namespace holder_inequality_l453_453559

theorem holder_inequality (n : ℕ) (a b c : Fin n → ℝ) (h_pos_a : ∀ i, 0 < a i) (h_pos_b : ∀ i, 0 < b i) (h_pos_c : ∀ i, 0 < c i)  :
  (∑ i, (a i) * (b i) * (c i)) ≤ (∑ i, (a i)^3)^(1/3) * (∑ i, (b i)^3)^(1/3) * (∑ i, (c i)^3)^(1/3) := by
  sorry

end holder_inequality_l453_453559


namespace no_set_of_9_numbers_l453_453253

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l453_453253


namespace cherry_tree_leaves_l453_453685

theorem cherry_tree_leaves (original_plan : ℕ) (multiplier : ℕ) (leaves_per_tree : ℕ) 
  (h1 : original_plan = 7) (h2 : multiplier = 2) (h3 : leaves_per_tree = 100) : 
  (original_plan * multiplier * leaves_per_tree = 1400) :=
by
  sorry

end cherry_tree_leaves_l453_453685


namespace c_positive_when_others_negative_l453_453463

variables {a b c d e f : ℤ}

theorem c_positive_when_others_negative (h_ab_cdef_lt_0 : a * b + c * d * e * f < 0)
  (h_a_neg : a < 0) (h_b_neg : b < 0) (h_d_neg : d < 0) (h_e_neg : e < 0) (h_f_neg : f < 0) 
  : c > 0 :=
sorry

end c_positive_when_others_negative_l453_453463


namespace range_of_a_l453_453781

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 3*x + a
noncomputable def g (x : ℝ) : ℝ := 2^x - x^2

theorem range_of_a (a : ℝ) : (∀ x ∈ Icc 0 1, f (g x) a ≥ 0) ↔ a ≥ -2 :=
by
  sorry

end range_of_a_l453_453781


namespace crackers_eaten_l453_453542

theorem crackers_eaten : ∃ x : ℕ, (x + 2*x = 48) ∧ x = 16 :=
by {
  use 16,
  split,
  { norm_num, },
  { refl }
}

end crackers_eaten_l453_453542


namespace smallest_digit_for_divisibility_by_9_l453_453367

theorem smallest_digit_for_divisibility_by_9 :
  ∃ d : ℕ, (∑ x in [5, 8, 6, d, 1, 7], x) % 9 = 0 ∧ d = 0 :=
by
  use 0
  sorry

end smallest_digit_for_divisibility_by_9_l453_453367


namespace price_of_cashew_nuts_l453_453666

theorem price_of_cashew_nuts 
  (C : ℝ)  -- price per kilo of cashew nuts
  (P_p : ℝ := 130)  -- price per kilo of peanuts
  (cashew_kilos : ℝ := 3)  -- kilos of cashew nuts bought
  (peanut_kilos : ℝ := 2)  -- kilos of peanuts bought
  (total_kilos : ℝ := 5)  -- total kilos of nuts bought
  (total_price_per_kilo : ℝ := 178)  -- total price per kilo of all nuts
  (h_total_cost : cashew_kilos * C + peanut_kilos * P_p = total_kilos * total_price_per_kilo) :
  C = 210 :=
sorry

end price_of_cashew_nuts_l453_453666


namespace product_area_perimeter_square_EFGH_l453_453545

theorem product_area_perimeter_square_EFGH:
  let E := (5, 5)
  let F := (5, 1)
  let G := (1, 1)
  let H := (1, 5)
  let side_length := 4
  let area := side_length * side_length
  let perimeter := 4 * side_length
  area * perimeter = 256 :=
by
  sorry

end product_area_perimeter_square_EFGH_l453_453545


namespace real_solutions_count_l453_453820

theorem real_solutions_count :
  ∃ n : ℕ, n = 2 ∧ ∀ x : ℝ, |x + 1| = |x - 3| + |x - 4| → x = 2 ∨ x = 8 :=
by
  sorry

end real_solutions_count_l453_453820


namespace coeff_of_expansion_l453_453376

noncomputable def expand_coeff (n : ℕ) (a b c : ℝ) : ℝ :=
  if h : n = 5 then 
    let term := ((2 : ℝ) * a) ^ 2 * ((-3 : ℝ) * b) ^ 1 * (c ^ (n - 3)) in
    ((nat.factorial n) / (nat.factorial 2 * nat.factorial (n - 2) * nat.factorial 1)) * term
  else 0

theorem coeff_of_expansion :
  expand_coeff 5 a b c = -4320 := 
by
  sorry

end coeff_of_expansion_l453_453376


namespace Lena_walking_hours_l453_453127

theorem Lena_walking_hours (Masha_time_store : ℝ) (Masha_time_ice_cream : ℝ) (Masha_time_meeting : ℝ)
    (Masha_Lena_same_time_store : Bool) :
  Masha_time_store = 12 ∧ Masha_time_ice_cream = 2 ∧ Masha_time_meeting = 2 ∧ Masha_Lena_same_time_store →
  let total_time_meeting := Masha_time_store + Masha_time_ice_cream + Masha_time_meeting in
  let Lena_distance_covered := 1 - (Masha_time_meeting / Masha_time_store) in
  let Lena_total_distance_covered_time := total_time_meeting / Lena_distance_covered in
  Lena_total_distance_covered_time = 19.2 :=
sorry

end Lena_walking_hours_l453_453127


namespace distance_probability_at_least_sqrt2_over_2_l453_453512

noncomputable def prob_dist_at_least : ℝ := 
  let T := ((0,0), (1,0), (0,1))
  -- Assumes conditions incorporated through identifying two random points within the triangle T.
  let area_T : ℝ := 0.5
  let valid_area : ℝ := 0.5 - (Real.pi * (Real.sqrt 2 / 2)^2 / 8 + ((Real.sqrt 2 / 2)^2 / 2) / 2)
  valid_area / area_T

theorem distance_probability_at_least_sqrt2_over_2 :
  prob_dist_at_least = (4 - π) / 8 :=
by
  sorry

end distance_probability_at_least_sqrt2_over_2_l453_453512


namespace probability_lucy_picks_from_mathematics_l453_453069

open Rat

theorem probability_lucy_picks_from_mathematics :
  let alphabet_size := 26
  let unique_letters_in_mathematics := 8
  let probability := mk_p nat.gcd 8 alphabet_size 
  let simplified_probability := 4 / 13
  probability = simplified_probability :=
by
  sorry

end probability_lucy_picks_from_mathematics_l453_453069


namespace min_max_values_l453_453754

noncomputable def myFunction (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem min_max_values :
  let f := myFunction in 
  isMinimum (set.range (fun (x : ℝ) => if (1 ≤ x ∧ x ≤ 3) then myFunction x else 0)) 1 ∧
  isMaximum (set.range (fun (x : ℝ) => if (1 ≤ x ∧ x ≤ 3) then myFunction x else 0)) 5 :=
sorry

end min_max_values_l453_453754


namespace min_real_roots_l453_453515

noncomputable def g (x : ℝ) : ℝ → ℝ := sorry  -- Defining g(x) polynomial; actual construction of g(x) is omitted

theorem min_real_roots (g : ℝ → ℝ) (h : polynomial ℝ)
  (h_deg : h.degree = 3000)
  (h_real_coeff : h.coeff ∈ ℝ)
  (h_magnitudes : ∃ (s : fin 3000 → ℝ), (∀ i, h.roots i = s i) ∧ 
                  (card (finset.image abs (finset.univ.image s))) = 1500) :
  ∃ (n : ℕ), n = 3 ∧ (∑ i in finset.univ, if (h.roots i).im = 0 then 1 else 0) = n :=
begin
  sorry
end

end min_real_roots_l453_453515


namespace speed_with_current_l453_453671

theorem speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h1 : current_speed = 2.8) 
  (h2 : against_current_speed = 9.4) 
  (h3 : against_current_speed = v - current_speed) 
  : (v + current_speed) = 15 := by
  sorry

end speed_with_current_l453_453671


namespace spectacular_number_exists_l453_453536

def is_prime_or_has_no_more_than_four_prime_factors (n : ℕ) : Prop :=
  Nat.Prime n ∨ (∃ p : ℕ, ∃ a b c d : ℕ, p * a * b * c * d = n ∧ Nat.Prime p ∧ Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d)

theorem spectacular_number_exists : 
  ∀ (n : ℕ), 100 ≤ n → n ≤ 993 → ∃ (x y : ℕ), x = n ∨ x = n + 1 ∨ x = n + 2 ∨ x = n + 3 ∨ x = n + 4 ∨ x = n + 5 ∨ x = n + 6 ∧ (\(y = x + 1)\) ∧ is_prime_or_has_no_more_than_four_prime_factors (1000 * x + y) sorry :=
begin
  sorry
end

end spectacular_number_exists_l453_453536


namespace not_exists_set_of_9_numbers_min_elements_l453_453240

theorem not_exists_set_of_9_numbers (s : Finset ℕ) 
  (h_len : s.card = 9) 
  (h_median : ∑ x in (s.filter (λ x, x ≤ 2)), 1 ≤ 5) 
  (h_other : ∑ x in (s.filter (λ x, x ≤ 13)), 1 ≤ 4) 
  (h_avg : ∑ x in s = 63) :
  False := sorry

theorem min_elements (n : ℕ) (h_nat: n ≥ 5) :
  ∃ s : Finset ℕ, s.card = 2 * n + 1 ∧
                  ∑ x in (s.filter (λ x, x ≤ 2)), 1 = n + 1 ∧ 
                  ∑ x in (s.filter (λ x, x ≤ 13)), 1 = n ∧
                  ∑ x in s = 14 * n + 7 := sorry

end not_exists_set_of_9_numbers_min_elements_l453_453240


namespace chris_did_not_get_A_l453_453477

variable (A : Prop) (MC_correct : Prop) (Essay80 : Prop)

-- The condition provided by professor
axiom condition : A ↔ (MC_correct ∧ Essay80)

-- The theorem we need to prove based on the statement (B) from the solution
theorem chris_did_not_get_A 
    (h : ¬ A) : ¬ MC_correct ∨ ¬ Essay80 :=
by sorry

end chris_did_not_get_A_l453_453477


namespace harry_friday_speed_l453_453057

theorem harry_friday_speed :
  let monday_speed := 10
  let tuesday_thursday_speed := monday_speed + monday_speed * (50 / 100)
  let friday_speed := tuesday_thursday_speed + tuesday_thursday_speed * (60 / 100)
  friday_speed = 24 :=
by
  sorry

end harry_friday_speed_l453_453057


namespace smallest_number_lemma_l453_453644

noncomputable def smallest_number (a b c d : ℕ) (h : lcm (lcm a b) (lcm c d) = 3675) : ℕ :=
LCM - 7

theorem smallest_number_lemma : smallest_number 25 49 15 21 (by sorry) = 3668 := 
sorry

end smallest_number_lemma_l453_453644


namespace proof_problem_l453_453018

variables (R : Type*) [Real R]

def p : Prop := ∃ x : R, Real.sin x < 1
def q : Prop := ∀ x : R, Real.exp (abs x) ≥ 1

theorem proof_problem : p ∧ q := 
by 
  sorry

end proof_problem_l453_453018


namespace find_x_l453_453786

theorem find_x (a x : ℝ) (h1 : 4^a = 2) (h2 : log a x = 2 * a) : x = 1 / 2 :=
by 
  sorry

end find_x_l453_453786


namespace exponential_graph_passes_through_point_l453_453972

variable (a : ℝ) (hx1 : a > 0) (hx2 : a ≠ 1)

theorem exponential_graph_passes_through_point :
  ∃ y : ℝ, (y = a^0 + 1) ∧ (y = 2) :=
sorry

end exponential_graph_passes_through_point_l453_453972


namespace binary_arithmetic_l453_453341

theorem binary_arithmetic : 
  (0b1011 + 0b0101 - 0b1100 + 0b1101 = 0b10001) :=
by 
  -- Definition of the binary numbers (conditions)
  let a := 0b1011
  let b := 0b0101
  let c := 0b1100
  let d := 0b1101
  
  -- We have to prove:
  -- a + b - c + d = 0b10001
  have h1 : a + b = 0b10000 := by sorry
  have h2 : -(c : Int) + d = 0b1 := by sorry
  have h3 : 0b10000 + 0b1 = 0b10001 := by sorry

  -- Thus:
  calc
  a + b - c + d = (a + b) - (c - d) : by sorry
              ... = 0b10000 - 0b1     : by sorry
              ... = 0b10001           : by sorry

end binary_arithmetic_l453_453341


namespace quadratic_non_real_roots_l453_453434

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l453_453434


namespace subset_relation_l453_453411

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x + 2}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (x - 4) / Real.log 2}

-- State the proof problem
theorem subset_relation : N ⊆ M := 
sorry

end subset_relation_l453_453411


namespace smallest_period_cos_l453_453987

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem smallest_period_cos (x : ℝ) : 
  smallest_positive_period (λ x => 2 * (Real.cos x)^2 + 1) Real.pi := 
by 
  sorry

end smallest_period_cos_l453_453987


namespace length_of_hypotenuse_l453_453678

theorem length_of_hypotenuse (a b : ℝ) (h1 : a = 15) (h2 : b = 21) : 
hypotenuse_length = Real.sqrt (a^2 + b^2) :=
by
  rw [h1, h2]
  sorry

end length_of_hypotenuse_l453_453678


namespace sum_fibonacci_minus_f2018_f2019_minus_sum_even_fibonacci_l453_453164

def fibonacci : ℕ → ℕ 
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem sum_fibonacci_minus_f2018 : 
  (∑ i in Finset.range 2017, fibonacci (i+1)) - fibonacci 2018 = -1 := 
by 
  sorry

theorem f2019_minus_sum_even_fibonacci : 
  fibonacci 2019 - (∑ k in Finset.range 1010, fibonacci (2 * k)) = 1 :=
by
  sorry

end sum_fibonacci_minus_f2018_f2019_minus_sum_even_fibonacci_l453_453164


namespace other_root_l453_453029

theorem other_root (x : ℚ) (h : 48 * x^2 + 29 = 35 * x + 12) : x = 3 / 4 ∨ x = 1 / 3 := 
by {
  -- Proof can be filled in here
  sorry
}

end other_root_l453_453029


namespace min_value_inv_sum_dist_lemma_l453_453788

noncomputable def min_value_inv_sum_dist (P: (ℝ × ℝ)) (F1 F2: (ℝ × ℝ)) : ℝ :=
  if hP : P.1^2 / 4 + P.2^2 = 1 then
    (1 / real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) + 
    (1 / real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))
  else 0

theorem min_value_inv_sum_dist_lemma : 
  ∃ (F1 F2 : ℝ × ℝ), ∀ (P : ℝ × ℝ), 
    (P.1^2 / 4 + P.2^2 = 1 → 
     min_value_inv_sum_dist P F1 F2 ≥ 1) := 
sorry

end min_value_inv_sum_dist_lemma_l453_453788


namespace Perimeter_gt_3_times_Diameter_l453_453508

noncomputable def Perimeter (M : Polyhedron) : ℝ := ∑ edge in M.edges, edge.length

noncomputable def Diameter (M : Polyhedron) : ℝ :=
  ⨆ (v1 v2 : M.vertices), dist v1 v2

def convex_polyhedron (M : Polyhedron) : Prop := 
  convex M

theorem Perimeter_gt_3_times_Diameter (M : Polyhedron) (hM : convex_polyhedron M) :
  Perimeter M > 3 * Diameter M :=
sorry

end Perimeter_gt_3_times_Diameter_l453_453508


namespace option_C_is_quadratic_l453_453226

-- Definitions based on conditions
def option_A (x : ℝ) : Prop := x^2 + (1/x^2) = 0
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (x - 1) * (x + 2) = 1
def option_D (x y : ℝ) : Prop := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Statement to prove option C is a quadratic equation in one variable.
theorem option_C_is_quadratic : ∀ x : ℝ, (option_C x) → (∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0) :=
by
  intros x hx
  -- To be proven
  sorry

end option_C_is_quadratic_l453_453226


namespace mary_has_34_lambs_l453_453126

def mary_lambs (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) (traded_lambs : ℕ) (found_lambs : ℕ): ℕ :=
  initial_lambs + (lambs_with_babies * babies_per_lamb) - traded_lambs + found_lambs

theorem mary_has_34_lambs :
  mary_lambs 12 4 3 5 15 = 34 :=
by
  -- This line is in place of the actual proof.
  sorry

end mary_has_34_lambs_l453_453126


namespace apple_distribution_l453_453311

theorem apple_distribution : ∃ N : ℕ, N = 253 ∧ 
  ∃ a b c : ℕ, a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 ∧ 
  (N = (nat.choose (30 - 3*3 + 3 - 1) (3 - 1))) := by
  sorry

end apple_distribution_l453_453311


namespace tan_OPQ_l453_453275

theorem tan_OPQ (x : ℝ) (angle_OPQ : ℝ) (OQ : ℝ) (PQ QR : ℝ) 
  (h1 : ∠ POQ = 90) 
  (h2 : ∠ QOR = 30) 
  (h3 : ∠ OPQ + ∠ R = 60)
  (h4 : QR = 2 * PQ)
  (h5 : OQ = 2 * x * real.sin (60 - angle_OPQ))
  (h6 : OQ = x * real.sin angle_OPQ) :
  real.tan angle_OPQ = 2 * real.sqrt(3) / 3 := 
sorry

end tan_OPQ_l453_453275


namespace min_value_expression_l453_453903

theorem min_value_expression (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 
  = 5^(5/4) - 10 * Real.sqrt (5^(1/4)) + 5 := 
sorry

end min_value_expression_l453_453903


namespace extreme_points_of_function_l453_453406

theorem extreme_points_of_function (f : ℝ → ℝ) (a : ℝ) (h : f = λ x, a * x^3 - (1 / 2) * x^2 + x - x * real.log x)
    (h' : ∀ x, deriv f x = 3 * a * x^2 - x - real.log x) : 
    (0 < a ∧ a < 1 / 3) ↔ ∃ x1 x2, x1 ≠ x2 ∧ deriv f x1 = 0 ∧ deriv f x2 = 0 :=
begin
  sorry
end

end extreme_points_of_function_l453_453406


namespace seq_strictly_monotone_decreasing_l453_453785

open Real

variables (a b : ℝ) (h : b > a ∧ a > 1)

noncomputable def seq (n : ℕ) : ℝ :=
  2^n * (real.rpow b (2⁻¹ : ℝ ^ (n : ℝ)) - real.rpow a (2⁻¹ : ℝ ^ (n : ℝ)))

theorem seq_strictly_monotone_decreasing :
  ∀ n : ℕ, seq a b n > seq a b (n + 1) :=
sorry

end seq_strictly_monotone_decreasing_l453_453785


namespace Cheyenne_earnings_l453_453332

-- Given conditions
def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

-- Calculations
def cracked_pots : ℕ := (cracked_fraction * total_pots).toNat
def good_pots : ℕ := total_pots - cracked_pots
def total_earnings : ℕ := good_pots * price_per_pot

-- Theorem statement
theorem Cheyenne_earnings : total_earnings = 1920 := by
  sorry

end Cheyenne_earnings_l453_453332


namespace common_chord_central_angle_l453_453957

theorem common_chord_central_angle:
  let circle1 := (x y : ℝ) → (x - 2)^2 + y^2 = 4
  let circle2 := (x y : ℝ) → x^2 + (y - 2)^2 = 4
  let center_M := (2, 0)
  let center_N := (0, 2)
  let radius := 2
  let MN := Real.sqrt ((2 - 0)^2 + (0 - 2)^2)
  let d := MN / 2
  let cos_theta := d / radius
  let theta := Real.acos cos_theta
  in 2 * theta = Real.pi / 2 :=
by
  sorry -- Proof to be completed

end common_chord_central_angle_l453_453957


namespace expression_evaluation_l453_453356

noncomputable def eval_expression : ℝ :=
    real.sqrt 5 * 5^(1/2) + 16 / 4 * 2 - 8^(3 / 2)

theorem expression_evaluation :
    eval_expression = 5 + 8 - 16 * real.sqrt 2 :=
by sorry

end expression_evaluation_l453_453356


namespace harry_friday_speed_l453_453058

theorem harry_friday_speed :
  let monday_speed := 10
  let tuesday_thursday_speed := monday_speed + monday_speed * (50 / 100)
  let friday_speed := tuesday_thursday_speed + tuesday_thursday_speed * (60 / 100)
  friday_speed = 24 :=
by
  sorry

end harry_friday_speed_l453_453058


namespace three_digit_powers_of_three_l453_453821

theorem three_digit_powers_of_three : 
  {n : ℤ | 100 ≤ 3^n ∧ 3^n ≤ 999}.finset.card = 2 :=
by
  sorry

end three_digit_powers_of_three_l453_453821


namespace true_propositions_count_l453_453984

theorem true_propositions_count
  (a b c : ℝ)
  (h : a > b) :
  ( (a > b → a * c^2 > b * c^2) ∧
    (a * c^2 > b * c^2 → a > b) ∧
    (a ≤ b → a * c^2 ≤ b * c^2) ∧
    (a * c^2 ≤ b * c^2 → a ≤ b) 
  ) ∧ 
  (¬(a > b → a * c^2 > b * c^2) ∧
   ¬(a * c^2 ≤ b * c^2 → a ≤ b)) →
  (a * c^2 > b * c^2 → a > b) ∧
  (a ≤ b → a * c^2 ≤ b * c^2) ∨
  (a > b → a * c^2 > b * c^2) ∨
  (a * c^2 ≤ b * c^2 → a ≤ b) :=
sorry

end true_propositions_count_l453_453984


namespace sum_of_reciprocals_l453_453604

theorem sum_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 2 := 
by
  sorry

end sum_of_reciprocals_l453_453604


namespace edge_length_of_cube_l453_453278

theorem edge_length_of_cube (total_cubes : ℕ) (box_edge_length_m : ℝ) (box_edge_length_cm : ℝ) 
  (conversion_factor : ℝ) (edge_length_cm : ℝ) : 
  total_cubes = 8 ∧ box_edge_length_m = 1 ∧ box_edge_length_cm = box_edge_length_m * conversion_factor ∧ conversion_factor = 100 ∧ 
  edge_length_cm = box_edge_length_cm / 2 ↔ edge_length_cm = 50 := 
by 
  sorry

end edge_length_of_cube_l453_453278


namespace part1_part2_l453_453407

noncomputable def line_equation (m x y : ℝ) := (m + 2) * x - (2 * m + 1) * y - 3

theorem part1 (m : ℝ) : line_equation m 2 1 = 0 :=
by
  sorry

def point_P : ℝ × ℝ := (-1, -2)

def point_A (a m : ℝ) : ℝ × ℝ := (a, 0)

def point_B (b m : ℝ) : ℝ × ℝ := (0, b)

def vector_PA (a : ℝ) : ℝ × ℝ := (a + 1, 2)

def vector_PB (b : ℝ) : ℝ × ℝ := (1, b + 2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem part2 (m : ℝ) : 
  let a := 4 in
  let b := 2 in
  let vPA := vector_PA a in
  let vPB := vector_PB b in
  dot_product vPA vPB = a + 2 * b + 5 ∧ m = -(5 / 4) :=
by
  sorry

end part1_part2_l453_453407


namespace problem1_problem2_l453_453158

-- Problem 1: Prove values of cos α and tan α given sin α = 5/13
theorem problem1 (α : ℝ) (h1 : sin α = 5 / 13) : cos α = 12 / 13 ∨ cos α = -12 / 13 ∧ (tan α = 5 / 12 ∨ tan α = -5 / 12) := 
sorry

-- Problem 2: Prove the value of the given expression is 1 given tan α = 2
theorem problem2 (α : ℝ) (h2 : tan α = 2) : 1 / (2 * sin α * cos α + cos α ^ 2) = 1 := 
sorry

end problem1_problem2_l453_453158


namespace intersection_of_A_and_B_l453_453895

-- Definitions of sets A and B
def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

-- The theorem we want to prove
theorem intersection_of_A_and_B : A ∩ B = { x | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_B_l453_453895


namespace mult_469158_9999_l453_453257

theorem mult_469158_9999 : 469158 * 9999 = 4691176842 := 
by sorry

end mult_469158_9999_l453_453257


namespace lewis_earns_at_least_388_per_week_l453_453125

theorem lewis_earns_at_least_388_per_week
    (weeks : ℕ)
    (weekly_rent : ℕ)
    (total_rent : ℕ)
    (h_weeks : weeks = 1359)
    (h_weekly_rent : weekly_rent = 388)
    (h_total_rent : total_rent = 527292) :
    weekly_rent * weeks = total_rent :=
by
  rw [h_weeks, h_weekly_rent, Nat.mul_eq_mul_right_iff]
  right
  exact h_total_rent

#check lewis_earns_at_least_388_per_week

end lewis_earns_at_least_388_per_week_l453_453125


namespace min_elements_l453_453247

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l453_453247


namespace fraction_of_number_subtract_l453_453623

-- Define the fraction, the number and the value to subtract
def fraction : ℚ := 3 / 4
def number : ℕ := 48
def subtractValue : ℕ := 12

-- The main proof statement demonstrating the equality
theorem fraction_of_number_subtract (fraction : ℚ) (number subtractValue : ℕ) :
  (fraction.toReal * number.toReal - subtractValue.toReal) = 24 :=
by
  sorry

end fraction_of_number_subtract_l453_453623


namespace jenny_coins_value_l453_453495

theorem jenny_coins_value (n d : ℕ) (h1 : d = 30 - n) (h2 : 150 + 5 * n = 300 - 5 * n + 120) :
  (300 - 5 * n : ℚ) / 100 = 1.65 := 
by
  sorry

end jenny_coins_value_l453_453495


namespace angles_terminal_side_equiv_l453_453313

-- Define the equivalence of terminal sides of angles modulo 360°
def same_terminal_side (α β : ℕ) : Prop :=
  (α % 360) = (β % 360)

-- The main statement to prove:
theorem angles_terminal_side_equiv :
  same_terminal_side (-300) 60 :=
sorry

end angles_terminal_side_equiv_l453_453313


namespace range_of_f_l453_453804

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - π / 3) * Real.cos (2 * x + π / 6)

theorem range_of_f : ∀ x ∈ Icc (0 : ℝ) (π / 3), f x ∈ Icc (-3 / 4) 0 :=
by
  sorry

end range_of_f_l453_453804


namespace c_is_integer_l453_453504

theorem c_is_integer (c : ℝ) (hc : c > 0)
  (h : ∀ n : ℕ, ∃ k ≤ n, k ≠ 0 ∧ (k ^ ℕ.ceil (real.to_nnreal c)) % 1 = 0) :
  ∃ m : ℤ, c = m := 
sorry

end c_is_integer_l453_453504


namespace trig_identity_part_1_trig_identity_part_2_l453_453326

section Problems

-- Problem 1
theorem trig_identity_part_1 : 
  (Real.cos (2 * Real.pi / 3) - Real.tan (-Real.pi / 4) + 3 / 4 * Real.tan (Real.pi / 6) - Real.sin (-31 * Real.pi / 6)) = Real.sqrt 3 / 4 :=
by sorry

-- Problem 2
variable (α : ℝ)

theorem trig_identity_part_2 : 
  (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.cos (-α + 3 * Real.pi / 2)) / 
  (Real.cos (Real.pi / 2 - α) * Real.sin (-Real.pi - α)) = -Real.cos α :=
by sorry

end Problems

end trig_identity_part_1_trig_identity_part_2_l453_453326


namespace not_possible_to_partition_into_groups_of_5_with_remainder_3_l453_453370

theorem not_possible_to_partition_into_groups_of_5_with_remainder_3 (m : ℤ) :
  ¬ (m^2 % 5 = 3) :=
by sorry

end not_possible_to_partition_into_groups_of_5_with_remainder_3_l453_453370


namespace option_D_has_square_root_l453_453636

theorem option_D_has_square_root : 
  let A := real.cbrt (-8)
  let B := -(3^2)
  let C := -real.sqrt 4
  let D := (-2)^2
  D ≥ 0 :=
by
  let A := real.cbrt (-8)
  let B := -(3^2)
  let C := -real.sqrt 4
  let D := (-2)^2
  show D ≥ 0
  sorry

end option_D_has_square_root_l453_453636


namespace rounding_precision_l453_453939

theorem rounding_precision :
  ∃ n : ℕ, (Real.round (1.003 ^ 4 * 10^n) / 10^n) = 1.012 ∧ n = 3 :=
by
  sorry

end rounding_precision_l453_453939


namespace find_cosB_l453_453393

theorem find_cosB
  (A B C : ℝ)
  (a b c : ℝ)
  (hA : A = 45)
  (h1 : c * sin B = sqrt 3 * b * cos C)
  (h2 : a = opposite_side_of_angle_in_triangle A)
  (h3 : b = opposite_side_of_angle_in_triangle B)
  (h4 : c = opposite_side_of_angle_in_triangle C) :
  cos B = (sqrt 6 - sqrt 2) / 4 := 
  sorry

end find_cosB_l453_453393


namespace range_of_h_eq_l453_453347

noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 3 * x^2)

theorem range_of_h_eq (a b : ℝ) (h_range : Ioo a b = set.range h) : a + b = 3 :=
sorry

end range_of_h_eq_l453_453347


namespace valid_passwords_count_l453_453314

-- Define the total number of unrestricted passwords
def total_passwords : ℕ := 10000

-- Define the number of restricted passwords (ending with 6, 3, 9)
def restricted_passwords : ℕ := 10

-- Define the total number of valid passwords
def valid_passwords := total_passwords - restricted_passwords

theorem valid_passwords_count : valid_passwords = 9990 := 
by 
  sorry

end valid_passwords_count_l453_453314


namespace correct_sum_relation_l453_453075

-- The relevant conditions
def hundreds_digit (T : ℕ) : ℕ := 2 * T
def units_digit (T : ℕ) : ℕ := T - 1
def sum_of_digits (T : ℕ) : ℕ := hundreds_digit T + T + units_digit T

theorem correct_sum_relation (T : ℕ ) : sum_of_digits  T = 18 
:= begin
sorry
end 

end correct_sum_relation_l453_453075


namespace simplify_fraction_l453_453945

theorem simplify_fraction :
  (18 / 462) + (35 / 77) = 38 / 77 := 
by sorry

end simplify_fraction_l453_453945


namespace min_pos_int_M_l453_453661

def smallest_M (M : ℕ) : Prop := 
  ∃ (M : ℕ), M > 0 ∧ 3 * M = 15 ∧ 5 * M = 15

theorem min_pos_int_M : smallest_M 5 :=
by {
  use 5,
  simp,
  split,
  norm_num,
  split,
  norm_num,
  norm_num,
}

end min_pos_int_M_l453_453661


namespace circle_tangent_to_x_axis_l453_453658

noncomputable def circle_equation (r : ℝ) : Prop :=
  x^2 + (y - r)^2 = r^2

theorem circle_tangent_to_x_axis
  (x y : ℝ)
  (h1 : ∀ r, (0, r) is the center and radial distance to (3, 1) is r)
  (h2 : r > 0):
  (x^2 + (y - r)^2 = r^2) :=
by
  sorry

end circle_tangent_to_x_axis_l453_453658


namespace books_per_customer_l453_453486

theorem books_per_customer (total_books : ℕ) (unsold_books sold_books customers books_per_customer : ℕ) 
  (h1 : total_books = 40) 
  (h2 : unsold_books = 4) 
  (h3 : sold_books = total_books - unsold_books)
  (h4 : customers = 4) 
  (h5 : books_per_customer = sold_books / customers) :
  books_per_customer = 9 :=
by {
  rw [h1, h2] at h3,
  norm_num at h3,
  rw [h3, h4]
}


end books_per_customer_l453_453486


namespace max_xy_l453_453115

variable {x y : ℝ}

theorem max_xy (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 * x + 8 * y = 48) : x * y ≤ 24 :=
sorry

end max_xy_l453_453115


namespace find_roots_l453_453744

noncomputable def polynomial_roots (x : ℂ) : Prop :=
  3 * x^4 + x^3 - 6 * x^2 + x + 3 = 0

theorem find_roots :
  ∃ x1 x2 x3 x4 : ℂ, 
    (polynomial_roots x1 ∧ polynomial_roots x2 ∧ polynomial_roots x3 ∧ polynomial_roots x4) ∧
    (∀ y : ℂ, y = (x1 + x2 + x3 + x4) / 4 → 
               ∃ a b : ℂ, (a = -1 + complex.sqrt 145 / 6 ∨ a = -1 - complex.sqrt 145 / 6) ∧
                          (b = -1 - complex.sqrt 145 / 6 ∨ b = -1 + complex.sqrt 145 / 6) ∧
                          (x1 = complex.cosh a ∨ x2 = complex.cosh b ∨ x3 = complex.cosh a ∨ x4 = complex.cosh b)) :=
by
  sorry

end find_roots_l453_453744


namespace remainder_when_divided_by_x_minus_2_l453_453220

-- We define the polynomial f(x)
def f (x : ℝ) := x^4 - 6 * x^3 + 11 * x^2 + 20 * x - 8

-- We need to show that the remainder when f(x) is divided by (x - 2) is 44
theorem remainder_when_divided_by_x_minus_2 : f 2 = 44 :=
by {
  -- this is where the proof would go
  sorry
}

end remainder_when_divided_by_x_minus_2_l453_453220


namespace quadratic_non_real_roots_l453_453448

theorem quadratic_non_real_roots (b : ℝ) : 
  let a : ℝ := 1 
  let c : ℝ := 16 in
  (b^2 - 4 * a * c < 0) ↔ (-8 < b ∧ b < 8) :=
sorry

end quadratic_non_real_roots_l453_453448


namespace find_sisters_dolls_l453_453421

variable (H S : ℕ)

-- Conditions
def hannah_has_5_times_sisters_dolls : Prop :=
  H = 5 * S

def total_dolls_is_48 : Prop :=
  H + S = 48

-- Question: Prove S = 8
theorem find_sisters_dolls (h1 : hannah_has_5_times_sisters_dolls H S) (h2 : total_dolls_is_48 H S) : S = 8 :=
sorry

end find_sisters_dolls_l453_453421


namespace tetrahedron_bisector_plane_area_l453_453956

-- Given a tetrahedron ABCD with specific face areas and a dihedral angle, 
-- we need to prove the area formed by the intersection of the bisector plane.
theorem tetrahedron_bisector_plane_area
  (A B C D : Type)
  (area_ABC : ℝ)
  (area_ADC : ℝ)
  (dihedral_angle : ℝ) :
  ∃ (S : ℝ), S = (2 * area_ABC * area_ADC * Real.cos (dihedral_angle / 2)) / (area_ABC + area_ADC) :=
by 
  let P := area_ABC
  let Q := area_ADC
  let α := dihedral_angle
  use (2 * P * Q * Real.cos (α / 2)) / (P + Q)
  sorry

end tetrahedron_bisector_plane_area_l453_453956


namespace triangle_problem_proof_l453_453776

-- Given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
variables (h2 : c = Real.sqrt 7)
variables (area : ℝ := 3 * Real.sqrt 3 / 2)

-- Prove angle C = π / 3 and perimeter of triangle
theorem triangle_problem_proof 
(h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
(h2 : c = Real.sqrt 7)
(area_condition : (1 / 2) * a * b * (Real.sin C) = area) :
  (C = Real.pi / 3) ∧ (a + b + c = 5 + Real.sqrt 7) := 
by
  sorry

end triangle_problem_proof_l453_453776


namespace Madeline_score_is_28_l453_453853

theorem Madeline_score_is_28 (leo_mistakes brent_mistakes : ℕ) 
  (h1 : 2 = leo_mistakes / 2)
  (h2 : 25 = 25)
  (h3 : brent_mistakes = leo_mistakes + 1) :
  let perfect_score := 25 + brent_mistakes,
      madeline_score := perfect_score - 2 in
  madeline_score = 28 := by
have h_leo : leo_mistakes = 4 := by sorry
have h_brent : brent_mistakes = 5 := by sorry
have h_perfect : perfect_score = 30 := by sorry
have h_madeline : madeline_score = 28 := by sorry
exact h_madeline

end Madeline_score_is_28_l453_453853


namespace projection_proof_l453_453131

variables (a b : ℝ × ℝ)
variable (θ : ℝ)
variable (π : ℝ)

-- Conditions
def non_zero (v : ℝ × ℝ) : Prop := v ≠ (0,0) 
def b_value : Prop := b = (Real.sqrt 3, 1)
def angle_condition : Prop := θ = π / 3
def orthogonal_condition : Prop := (a.1 - b.1, a.2 - b.2).1 * a.1 + (a.1 - b.1, a.2 - b.2).2 * a.2 = 0

-- To prove
theorem projection_proof (h1 : non_zero a) (h2 : non_zero b) (h3 : b_value b) (h4 : angle_condition θ π) (h5 : orthogonal_condition a b):
  let dot_product := a.1 * b.1 + a.2 * b.2 in
  let b_norm_sq := b.1 * b.1 + b.2 * b.2 in
  let projection := (dot_product / b_norm_sq) * b in
  projection = (1 / 4) * b := sorry

end projection_proof_l453_453131


namespace expression_of_odd_function_on_R_l453_453455

   variable {R : Type*} [LinearOrderedField R]

   def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

   def piecewise_function (f : R → R) (g h : R → R) : Prop :=
     (∀ x : R, 0 ≤ x → f x = g x) ∧ (∀ x : R, x < 0 → f x = h x)

   noncomputable def given_function (x : R) : R :=
     if 0 ≤ x then x^2 - 2 * x else -x^2 - 2 * x

   theorem expression_of_odd_function_on_R (f : R → R) (h_odd : odd_function f)
     (h_def : piecewise_function f (λ x, x^2 - 2 * x) (λ x, -x^2 - 2 * x)) :
     ∀ x : R, f x = x * (|x| - 2) :=
   by 
     sorry
   
end expression_of_odd_function_on_R_l453_453455


namespace wheel_radius_l453_453308

theorem wheel_radius (distance_covered_in_600_revolutions : ℝ) (number_of_revolutions : ℝ)
    (h_distance : distance_covered_in_600_revolutions = 844.8) (h_revolutions : number_of_revolutions = 600) :
    let circumference := distance_covered_in_600_revolutions / number_of_revolutions
    let radius := circumference / (2 * Real.pi) in
    radius = 0.224 :=
by
  sorry

end wheel_radius_l453_453308


namespace not_possible_consecutive_integers_l453_453616

theorem not_possible_consecutive_integers 
  (n : ℕ) 
  (nums : List ℤ) 
  (h : nums.length = 2 * n)
  : ∀ (t : ℕ), ¬ (List.nodup (List.concatMap (λ x : (ℤ × ℤ), [x.fst + x.snd, x.fst - x.snd]) 
                        (List.pairwise nums))) = (List.range (2 * n)) := 
by
  sorry

end not_possible_consecutive_integers_l453_453616


namespace tangent_line_at_1_l453_453743

noncomputable def f : ℝ → ℝ := λ x, x * Real.log x

theorem tangent_line_at_1 : tangent_line f 1 = λ x, x - 1 := by
  sorry

end tangent_line_at_1_l453_453743


namespace total_fishes_l453_453231

theorem total_fishes (Will_catfish : ℕ) (Will_eels : ℕ) (Henry_multiplier : ℕ) (Henry_return_fraction : ℚ) :
  Will_catfish = 16 → Will_eels = 10 → Henry_multiplier = 3 → Henry_return_fraction = 1 / 2 →
  (Will_catfish + Will_eels) + (Henry_multiplier * Will_catfish - (Henry_multiplier * Will_catfish / 2)) = 50 := 
by
  intros h1 h2 h3 h4
  sorry

end total_fishes_l453_453231


namespace minimum_amount_to_buy_11_items_l453_453092

-- Define the prices list
def prices : List ℕ := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]

-- Define a function to calculate the minimum cost given the promotion
def minimum_cost (items : List ℕ) : ℕ :=
  let groups := items.chunk 3 -- Group the items into groups of 3
  groups.foldl (fun acc group => acc + group.sum - group.minimumOrZero) 0

-- Define the main theorem
theorem minimum_amount_to_buy_11_items : minimum_cost prices = 4800 :=
  sorry

end minimum_amount_to_buy_11_items_l453_453092


namespace parabola_vertex_coord_l453_453579

theorem parabola_vertex_coord : ∀ (x : ℝ), 
  (∃ k : ℝ, k = 3*x^2 - 6*x + 5) → 
  ∃ (h k : ℝ), h = 1 ∧ k = 2 :=
begin
  sorry
end

end parabola_vertex_coord_l453_453579


namespace all_chameleons_blue_l453_453647

def chameleon_colors := {c : Nat // 1 ≤ c ∧ c ≤ 5 }

def initial_state (n : Nat) (color : chameleon_colors) := 
  if color.val = 1 then n else 0

def color_change_rule (biter: chameleon_colors) (bitten: chameleon_colors) : chameleon_colors :=
  if biter.val = 5 then 
    ⟨5, by norm_num⟩
  else if biter.val = bitten.val ∧ bitten.val < 5 then 
    ⟨bitten.val + 1, by omega⟩
  else 
    bitten

theorem all_chameleons_blue (n: Nat) (h: n ≥ 2023) : 
  ∃ f : ∀ t : Nat, list (chameleon_colors × chameleon_colors), 
  (∀ k : chameleon_colors, color_change_rule (k.1) (k.2).val = 5) :=
  sorry

end all_chameleons_blue_l453_453647


namespace range_e2_minus_e1_l453_453778

variables {a b m n : ℝ}
def e1 := sqrt(a^2 - b^2) / a
def e2 := sqrt(m^2 + n^2) / m

theorem range_e2_minus_e1 (h1 : a > b > 0) (h2 : m > 0) (h3 : n > 0) (h4 : a = m + 2 * sqrt(a^2 - b^2)) :
  set_of (λ (x : ℝ), ∃ e1 e2, (sqrt(a^2 - b^2) / a = e1) ∧ (sqrt(m^2 + n^2) / m = e2) ∧ x = e2 - e1)
  = {y | y > 2/3} :=
by {
  sorry
}

end range_e2_minus_e1_l453_453778


namespace range_of_a_l453_453969

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Ioo a (a + 1), ∃ f' : ℝ → ℝ, ∀ x, f' x = (x * Real.exp x) * (x + 2) ∧ f' x = 0) ↔ 
  a ∈ Set.Ioo (-3 : ℝ) (-2) ∪ Set.Ioo (-1) (0) := 
sorry

end range_of_a_l453_453969


namespace student_first_subject_percentage_l453_453680

variable (P : ℝ)

theorem student_first_subject_percentage 
  (H1 : 80 = 80)
  (H2 : 75 = 75)
  (H3 : (P + 80 + 75) / 3 = 75) :
  P = 70 :=
by
  sorry

end student_first_subject_percentage_l453_453680


namespace sine_ratio_cos_B_l453_453053

-- Definitions and conditions
variables {α : Type*} [linear_ordered_field α] {a b c : α}
variables {A B C : real} -- Note that angles are usually represented using real numbers
variables (h1 : vector.of_fn [b-a, 0, 0] ⬝ vector.of_fn [c-a, 0, 0] + 2 * vector.of_fn [a-b, 0, 0] ⬝ vector.of_fn [a-c, 0, 0] = vector.of_fn [a-c, 0, 0] ⬝ vector.of_fn [a-b, 0, 0])
variables (h2 : 2 * a * real.cos C = 2 * b - c)

-- Question 1: Proving the ratio of sines
theorem sine_ratio (h1 : vector.of_fn [b-a, 0, 0] ⬝ vector.of_fn [c-a, 0, 0] + 2 * vector.of_fn [a-b, 0, 0] ⬝ vector.of_fn [a-c, 0, 0] = vector.of_fn [a-c, 0, 0] ⬝ vector.of_fn [a-b, 0, 0])
  (a b c : ℝ) (A C : real) : real.sin A / real.sin C = real.sqrt 2 :=
begin
  sorry -- proof is omitted
end

-- Question 2: Proving the value of cos B
theorem cos_B (h2 : 2 * a * real.cos C = 2 * b - c) (a b c : ℝ) (A B C : real) : real.cos B = (3 * real.sqrt 2 - real.sqrt 10) / 8 :=
begin
  sorry -- proof is omitted
end

end sine_ratio_cos_B_l453_453053


namespace no_set_of_9_numbers_l453_453252

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l453_453252


namespace fraction_Cal_to_Anthony_l453_453919

-- definitions for Mabel, Anthony, Cal, and Jade's transactions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)
def Jade_transactions : ℕ := 85
def Cal_transactions : ℕ := Jade_transactions - 19

-- goal: prove the fraction Cal handled compared to Anthony is 2/3
theorem fraction_Cal_to_Anthony : (Cal_transactions : ℚ) / (Anthony_transactions : ℚ) = 2 / 3 :=
by
  sorry

end fraction_Cal_to_Anthony_l453_453919


namespace compute_f_iterated_l453_453500

def f : ℝ → ℝ :=
λ x, if x ≥ 0 then -x^3 else x + 9

theorem compute_f_iterated : f (f (f (f (f 3)))) = 0 :=
by
  sorry

end compute_f_iterated_l453_453500


namespace triangle_altitudes_perfect_square_l453_453717

theorem triangle_altitudes_perfect_square
  (a b c : ℤ)
  (h : (2 * (↑a * ↑b * ↑c )) = (2 * (↑a * ↑c ) + 2 * (↑a * ↑b))) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end triangle_altitudes_perfect_square_l453_453717


namespace three_digit_numbers_form_3_pow_l453_453832

theorem three_digit_numbers_form_3_pow (n : ℤ) : 
  ∃! (n : ℤ), 100 ≤ 3^n ∧ 3^n ≤ 999 :=
by {
  use [5, 6],
  sorry
}

end three_digit_numbers_form_3_pow_l453_453832


namespace line_plane_parallel_or_intersects_l453_453032

-- Definitions for line a and plane α
variable (a : Type) [Linear a]
variable (α : Type) [Plane α]

-- Condition that line a is not contained within plane α
variable (contains : α → a → Prop)
axiom not_contained : ¬contains α a

-- Theorem stating that if a line is not contained within a plane,
-- then it is either parallel to or intersects the plane
noncomputable def line_plane_relation : Prop :=
  ∀ (a : Type) [Linear a] (α : Type) [Plane α] (contains : α → a → Prop),
  ¬contains α a → (parallel a α ∨ intersects a α)

theorem line_plane_parallel_or_intersects :
  ∀ (a : Type) [Linear a] (α : Type) [Plane α] (contains : α → a → Prop),
  ¬contains α a → (parallel a α ∨ intersects a α) :=
by
  intro a
  sorry

end line_plane_parallel_or_intersects_l453_453032


namespace calc_result_l453_453323

noncomputable def expMul := (-0.25)^11 * (-4)^12

theorem calc_result : expMul = -4 := 
by
  -- Sorry is used here to skip the proof as instructed.
  sorry

end calc_result_l453_453323


namespace triangle_DEF_area_l453_453192

open Complex Real

noncomputable def minimum_triangle_area : Real :=
  let z := fun k : ℕ => -4 + 2 ^ (1 / 5) * exp (2 * π * I * k / 10)
  sorry

theorem triangle_DEF_area {D E F : ℕ → Complex} :
    (∀ k, D k = -4 + 2 ^ (1 / 5) * exp (2 * π * I * k / 10)) →
    (∀ k, E k = -4 + 2 ^ (1 / 5) * exp (2 * π * I * (k + 1) / 10)) →
    (∀ k, F k = -4 + 2 ^ (1 / 5) * exp (2 * π * I * (k + 2) / 10)) →
    ∃ k, area_of_triangle (D k) (E k) (F k) = 2 ^ (2 / 5) * (√5 - 1) / 8 :=
by
  sorry

open_locale complex_conjugate

end triangle_DEF_area_l453_453192


namespace min_distance_value_l453_453067

theorem min_distance_value (x1 x2 y1 y2 : ℝ) 
  (h1 : (e ^ x1 + 2 * x1) / (3 * y1) = 1 / 3)
  (h2 : (x2 - 1) / y2 = 1 / 3) :
  ((x1 - x2)^2 + (y1 - y2)^2) = 8 / 5 :=
by
  sorry

end min_distance_value_l453_453067


namespace f_log2_3_eq_1_div_24_l453_453766

-- Define the piecewise function f
def f : ℝ → ℝ := λ x, if x < 4 then f (x + 1) else (1 / 2) ^ x

-- Define the value to prove: f(log2 3) = 1 / 24
theorem f_log2_3_eq_1_div_24 : f (Real.log 3 / Real.log 2) = 1 / 24 :=
by sorry

end f_log2_3_eq_1_div_24_l453_453766


namespace percentage_markup_l453_453184

/--
The owner of a furniture shop charges his customer a certain percentage more than the cost price.
A customer paid Rs. 8587 for a computer table, and the cost price of the computer table was Rs. 6925.
Prove that the percentage markup on the cost price is approximately 23.99%.
-/
theorem percentage_markup (selling_price cost_price markup : ℝ)
  (h1 : selling_price = 8587)
  (h2 : cost_price = 6925)
  (h3 : markup = selling_price - cost_price) :
  (markup / cost_price) * 100 ≈ 23.99 :=
begin
  -- conditions
  have h_selling_price : selling_price = 8587 := h1,
  have h_cost_price : cost_price = 6925 := h2,
  have h_markup : markup = selling_price - cost_price := h3,
  have h_calc : (markup / cost_price) * 100 = (1662 / 6925) * 100,
  have h_res : (1662 / 6925) * 100 ≈ 23.99 := sorry,
  exact h_res,
end

end percentage_markup_l453_453184


namespace radius_of_smaller_circle_l453_453172

-- Definitions from the conditions
def larger_circle_radius : ℝ := 1
def smaller_circle_radius (r : ℝ) : Prop :=
  let P := ⟨0, 0⟩
  let Q := ⟨1 + r, 0⟩
  let R := ⟨0, 1 + r⟩
  let S := ⟨1 + r, 1 + r⟩
  (1 + r) ^ 2 + (1 + r) ^ 2 = 2 ^ 2

theorem radius_of_smaller_circle (r : ℝ) (h : smaller_circle_radius r) : r = Real.sqrt 2 - 1 :=
sorry

end radius_of_smaller_circle_l453_453172


namespace find_ratio_MN_FN_l453_453811
-- First, import the necessary Lean library

-- Define the initial conditions
def parabola : Set (ℝ × ℝ) := {p | p.snd ^ 2 = 4 * p.fst}
def focus : (ℝ × ℝ) := (1, 0)
def point_A : (ℝ × ℝ) := (0, -2)
def directrix_x : ℝ := -1

-- Define a structure for the proof problem and the statement
theorem find_ratio_MN_FN :
  ∃ M N : (ℝ × ℝ), 
    M ∈ parabola ∧
    (∃ k : ℝ, k ≠ 0 ∧ line_through focus point_A ∧ on_line k focus M) ∧
    (∃ L : Set (ℝ × ℝ), is_directrix L directrix_x ∧ ∃ P, (L ∧ line_through focus point_A ∧ on_line k focus N)) ∧
    line_perpendicular_to_directrix M ∧
    let PM := distance M (project_on_directrix M directrix_x) in
    let MN := distance M N in
    let FN := distance F N in
    MN / FN = sqrt(5) / (1 + sqrt(5)) :=
  sorry

end find_ratio_MN_FN_l453_453811


namespace triangle_eq_cos_sin_l453_453493

theorem triangle_eq_cos_sin {a b c : ℝ} {A B C : ℝ}
(h : a + b + c = 180) 
(ha : a = (b * sin A) / sin B) 
(hb : b = (c * sin B) / sin C) 
(hc : c = (a * sin C) / sin A) : 
(a + b) / c = (cos ((A - B) / 2)) / (sin (C / 2)) :=
sorry

end triangle_eq_cos_sin_l453_453493


namespace incorrect_conclusion_among_options_l453_453935

-- Define the sales data
def sales_data : List ℕ := [11, 10, 11, 13, 11, 13, 15]

-- Define properties to check:
def mode_is_11 : Prop := (List.mode sales_data = some 11)
def mean_is_12 : Prop := (List.mean sales_data = some 12)
def variance_is_18_over_7 : Prop := (List.variance sales_data = some (18 / 7))
def median_is_13 : Prop := (List.median sales_data = some 13)

theorem incorrect_conclusion_among_options : median_is_13 = false :=
by
  -- Note: proof intentionally skipped as instructed
  sorry

end incorrect_conclusion_among_options_l453_453935


namespace num_three_digit_powers_of_three_l453_453828

theorem num_three_digit_powers_of_three : 
  ∃ n1 n2 : ℕ, 100 ≤ 3^n1 ∧ 3^n1 ≤ 999 ∧ 100 ≤ 3^n2 ∧ 3^n2 ≤ 999 ∧ n1 ≠ n2 ∧ 
  (∀ n : ℕ, 100 ≤ 3^n ∧ 3^n ≤ 999 → n = n1 ∨ n = n2) :=
sorry

end num_three_digit_powers_of_three_l453_453828


namespace estimate_A_l453_453523

def total_teams : ℕ := 689
def correct_teams : ℕ := 175

noncomputable def proportion_a : ℚ := correct_teams / total_teams

noncomputable def scaled_value : ℤ := floor (10000 * proportion_a : ℚ)

theorem estimate_A : scaled_value = 2539 :=
by {
  -- proof to be provided
  sorry
}

end estimate_A_l453_453523


namespace negation_red_cards_in_deck_l453_453595

variable (Deck : Type) (is_red : Deck → Prop) (is_in_deck : Deck → Prop)

theorem negation_red_cards_in_deck :
  (¬ ∃ x : Deck, is_red x ∧ is_in_deck x) ↔ (∃ x : Deck, is_red x ∧ is_in_deck x) :=
by {
  sorry
}

end negation_red_cards_in_deck_l453_453595


namespace alice_password_prob_correct_l453_453697

noncomputable def password_probability : ℚ :=
  let even_digit_prob := 5 / 10
  let valid_symbol_prob := 3 / 5
  let non_zero_digit_prob := 9 / 10
  even_digit_prob * valid_symbol_prob * non_zero_digit_prob

theorem alice_password_prob_correct :
  password_probability = 27 / 100 := by
  rfl

end alice_password_prob_correct_l453_453697


namespace non_real_roots_of_quadratic_l453_453442

theorem non_real_roots_of_quadratic (b : ℝ) : 
  (¬ ∃ x1 x2 : ℝ, x1^2 + bx1 + 16 = 0 ∧ x2^2 + bx2 + 16 = 0 ∧ x1 = x2) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
by {
  sorry
}

end non_real_roots_of_quadratic_l453_453442


namespace beverage_price_function_l453_453601

theorem beverage_price_function (box_price : ℕ) (bottles_per_box : ℕ) (bottles_purchased : ℕ) (y : ℕ) :
  box_price = 55 →
  bottles_per_box = 6 →
  y = (55 * bottles_purchased) / 6 := 
sorry

end beverage_price_function_l453_453601


namespace seating_arrangements_l453_453996

/-- There are 5 chairs in a row. Persons A and B must sit next to each other, and 
    the three people cannot all sit next to each other. Prove that the number of 
    different seating arrangements for A, B, and one other person is 12. -/
theorem seating_arrangements : 
  ∃ (chair : Fin 5 → Prop) (A B : Fin 5), 
    (∀ i : Fin 5, chair i → ∃ j : Fin 5, chair j ∧ j ≠ i ∧ 
    (i = j.pred ∧ j ≠ 0 ∨ j = i + 1 ∧ i ≠ 4)) ∧
    (∀ k l m : Fin 5, k ≠ l ∧ k ≠ m ∧ l ≠ m → ¬(chair k ∧ chair l ∧ chair m)) ∧ 
    (set.card {i | chair i} = 2 ∧ 
    let arrangements := (finset.univ.filter (λ p : Fin 5 × Fin 5, 
      chair p.1 ∧ chair p.2 ∧ p.1 ≠ p.2)) in 
    finset.card arrangements = 12) :=
by
  sorry

end seating_arrangements_l453_453996


namespace calc_result_l453_453322

noncomputable def expMul := (-0.25)^11 * (-4)^12

theorem calc_result : expMul = -4 := 
by
  -- Sorry is used here to skip the proof as instructed.
  sorry

end calc_result_l453_453322


namespace necessary_and_sufficient_condition_l453_453839

theorem necessary_and_sufficient_condition {a b : ℝ} (h : a|a + b| < |a|(a + b)) : a < 0 ∧ b > -a :=
sorry

end necessary_and_sufficient_condition_l453_453839


namespace probability_both_heads_l453_453206

theorem probability_both_heads (tosses : List Bool) (h : tosses.length = 2) :
  (↑(List.countp (λ t, t = tt) tosses) / (tosses.length : ℝ) = 1 / 4) :=
sorry

end probability_both_heads_l453_453206


namespace simplify_polynomial_l453_453155

variable (y : ℤ)

theorem simplify_polynomial :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 6 * y^10 + 2 * y^9 + 4) = 
  15 * y^13 - y^12 + 12 * y^11 - 6 * y^10 - 4 * y^9 + 12 * y - 8 :=
by
  sorry

end simplify_polynomial_l453_453155


namespace arithmetic_sequence_function_positive_l453_453801

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_function_positive
  {f : ℝ → ℝ} {a : ℕ → ℝ}
  (hf_odd : is_odd f)
  (hf_mono : is_monotonically_increasing f)
  (ha_arith : is_arithmetic_sequence a)
  (ha3_pos : a 3 > 0) : 
  f (a 1) + f (a 3) + f (a 5) > 0 := 
sorry

end arithmetic_sequence_function_positive_l453_453801


namespace total_shaded_area_of_square_carpet_l453_453298

theorem total_shaded_area_of_square_carpet :
  ∀ (S T : ℝ),
    (9 / S = 3) →
    (S / T = 3) →
    (8 * T^2 + S^2 = 17) :=
by
  intros S T h1 h2
  sorry

end total_shaded_area_of_square_carpet_l453_453298


namespace Cheyenne_earnings_l453_453330

-- Given conditions
def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

-- Calculations
def cracked_pots : ℕ := (cracked_fraction * total_pots).toNat
def good_pots : ℕ := total_pots - cracked_pots
def total_earnings : ℕ := good_pots * price_per_pot

-- Theorem statement
theorem Cheyenne_earnings : total_earnings = 1920 := by
  sorry

end Cheyenne_earnings_l453_453330


namespace count_powers_of_three_not_powers_of_nine_l453_453424

theorem count_powers_of_three_not_powers_of_nine :
  {n : nat | n < 500000 ∧ ∃ k : nat, n = 3 ^ k ∧ ∀ m : nat, n ≠ 9 ^ m}.to_finset.card = 6 := 
sorry

end count_powers_of_three_not_powers_of_nine_l453_453424


namespace car_pass_time_l453_453098

-- Definitions for the conditions
def length_of_car : ℝ := 10
def speed_in_kmph : ℝ := 36
def conversion_factor : ℝ := 1000 / 3600
def speed_in_mps : ℝ := speed_in_kmph * conversion_factor

-- The theorem we need to prove
theorem car_pass_time : (length_of_car / speed_in_mps) = 1 := by
  sorry

end car_pass_time_l453_453098


namespace find_purchase_price_l453_453560

-- Definitions
def purchase_price_mobile : ℕ := 8000
def profit_mobile_percent : ℕ := 10
def loss_refrigerator_percent : ℕ := 5
def overall_profit : ℕ := 50

-- Variable for the purchase price of the refrigerator
variable {R : ℕ}

-- The Lean version of the proof problem.
theorem find_purchase_price (hR : -0.05 * R = -750) : R = 15000 :=
by
  have h₁ : (-0.05 * R) = (-750) := hR
  sorry

end find_purchase_price_l453_453560


namespace linear_term_coefficient_is_neg_two_l453_453577

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Define the specific quadratic equation
def specific_quadratic_eq (x : ℝ) : Prop :=
  quadratic_eq 1 (-2) (-1) x

-- The statement to prove the coefficient of the linear term
theorem linear_term_coefficient_is_neg_two : ∀ x : ℝ, specific_quadratic_eq x → ∀ a b c : ℝ, quadratic_eq a b c x → b = -2 :=
by
  intros x h_eq a b c h_quadratic_eq
  -- Proof is omitted
  sorry

end linear_term_coefficient_is_neg_two_l453_453577


namespace no_rational_roots_l453_453141

theorem no_rational_roots (n : ℕ) (h : n > 1) :
  ∀ (x : ℚ), (∑ k in finset.range (n + 1), x ^ k / (Nat.factorial k : ℚ)) + 1 ≠ 0 :=
by
  sorry

end no_rational_roots_l453_453141


namespace number_of_solutions_l453_453350

noncomputable def cos_equation (x : ℝ) : ℝ :=
  3 * (Real.cos x) ^ 3 - 7 * (Real.cos x) ^ 2 + 3 * (Real.cos x)

theorem number_of_solutions :
  (∑ x in Finset.filter (λ x, 0 ≤ Real.cos x ∧ Real.cos x ≤ 1) 
      (Finset.range (Real.pi.toReal + 1)), 
    if cos_equation x = 0 then 1 else 0 
  ) = 2 := sorry

end number_of_solutions_l453_453350


namespace find_p_l453_453740

noncomputable def satisfies_polynomial (p : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧
  (x1^4 + 2*p*x1^3 + (p+1)*x1^2 + 2*p*x1 + 1 = 0) ∧
  (x2^4 + 2*p*x2^3 + (p+1)*x2^2 + 2*p*x2 + 1 = 0)

theorem find_p (p : ℝ) : p ∈ set.Ioc (3/5) ∞ ↔ satisfies_polynomial p := sorry

end find_p_l453_453740


namespace collinear_intersections_of_diagonals_l453_453587

-- Define points and their properties including intersection points
variables {A B C D P Q E F : Type}
variables [AffineSpace A] [AffineSpace B] [AffineSpace C] [AffineSpace D] [AffineSpace E] [AffineSpace F]
variables [Intersect A B P] [Intersect C D P] [Intersect B C Q] [Intersect A D Q]
variables [OnLine E P] [OnLine E B] [OnLine F P] [OnLine F D]

-- Define the collinearity of intersection points
theorem collinear_intersections_of_diagonals 
  (h1 : ∃ (inter1 : A B C D) (inter2 : A B E F) (inter3 : C D F E), 
      collinear [inter1, inter2, inter3]) :
  collinear [inter1, Q, inter3] :=
sorry

end collinear_intersections_of_diagonals_l453_453587


namespace three_digit_powers_of_three_l453_453824

theorem three_digit_powers_of_three : 
  {n : ℤ | 100 ≤ 3^n ∧ 3^n ≤ 999}.finset.card = 2 :=
by
  sorry

end three_digit_powers_of_three_l453_453824


namespace max_value_sin_cos_func_l453_453756

theorem max_value_sin_cos_func : 
  (∃ x : ℝ, y = 3*sin x - 3*sqrt 3*cos x) → 
  ∀ y, y ≤ 6 := 
by
  sorry

end max_value_sin_cos_func_l453_453756


namespace distribution_probability_l453_453726

theorem distribution_probability :
  let total_ways := 4 ^ 5
  let favorable_ways := Nat.choose 5 2 * (4!)
  let probability := (favorable_ways : ℚ) / total_ways
  in probability = 15 / 64 :=
by
  let total_ways := 4 ^ 5
  let favorable_ways := Nat.choose 5 2 * (4!)
  let probability := (favorable_ways : ℚ) / total_ways
  show probability = 15 / 64
  sorry

end distribution_probability_l453_453726


namespace units_digit_of_5_pow_150_plus_7_l453_453630

theorem units_digit_of_5_pow_150_plus_7 : (5^150 + 7) % 10 = 2 := by
  sorry

end units_digit_of_5_pow_150_plus_7_l453_453630


namespace find_m_l453_453037

theorem find_m (m : ℝ) (h : (2 + complex.i) * (m + 2 * complex.i) = (m + 4) * complex.i) : m = 1 :=
by
  -- leaving the proof as sorry for now
  sorry

end find_m_l453_453037


namespace shift_to_right_by_pi_over_3_l453_453619

def f (x : ℝ) : ℝ := sin (2 * x - π / 3)
def g (x : ℝ) : ℝ := - sin (2 * x)

theorem shift_to_right_by_pi_over_3 :
  ∀ x, g x = f (x - π / 3) :=
sorry

end shift_to_right_by_pi_over_3_l453_453619


namespace verify_exponentiation_l453_453638

theorem verify_exponentiation : (x : ℝ) → (y : ℝ) → ((x ^ (-3)) ^ (-2) = x ^ 6) :=
by
  intros x y
  have h1 : (x ^ (-3)) ^ (-2) = x ^ (-3 * -2) := by sorry
  have h2 : x ^ (-3 * -2) = x ^ 6 := by sorry
  exact Eq.trans h1 h2

end verify_exponentiation_l453_453638


namespace min_max_values_l453_453753

noncomputable def myFunction (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem min_max_values :
  let f := myFunction in 
  isMinimum (set.range (fun (x : ℝ) => if (1 ≤ x ∧ x ≤ 3) then myFunction x else 0)) 1 ∧
  isMaximum (set.range (fun (x : ℝ) => if (1 ≤ x ∧ x ≤ 3) then myFunction x else 0)) 5 :=
sorry

end min_max_values_l453_453753


namespace set_statements_correctness_l453_453170

open Set

theorem set_statements_correctness :
  ¬ (\{2,3\} ≠ \{3,2\}) ∧
  ¬ (\{\left(x:y\right) \;|\; x + y = 1\} = \{y \;|\; x + y = 1\}) ∧
  (\{x \;|\; x > 1\} = \{y \;|\; y > 1\}) ∧
  (\{x \;|\; x = 2k + 1, k ∈ ℤ\} = \{x \;|\; x = 2k - 1, k ∈ ℤ\}). 
Proof. sorry


end set_statements_correctness_l453_453170


namespace distance_center_line_eq_sqrt2_l453_453173

-- Define the circle in polar coordinates
def circle_polar (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the line in polar coordinates
def line_polar (θ : ℝ) : Prop := Real.tan (θ + Real.pi / 2) = 1

-- Define the center of the circle in Cartesian coordinates
def center_circle : ℝ × ℝ := (2, 0)

-- Define the line in Cartesian coordinates
def line_cart (x y : ℝ) : Prop := x - y = 0

-- Calculate the distance from the center of the circle to the line
def distance_from_center_to_line : ℝ := Real.abs (2 - 0) / Real.sqrt 2

-- The statement to be proven
theorem distance_center_line_eq_sqrt2 : distance_from_center_to_line = Real.sqrt 2 :=
by
  -- Proof steps would go here
  sorry

end distance_center_line_eq_sqrt2_l453_453173


namespace invisible_dots_48_l453_453371

theorem invisible_dots_48 (visible : Multiset ℕ) (hv : visible = [1, 2, 3, 3, 4, 5, 6, 6, 6]) :
  let total_dots := 4 * (1 + 2 + 3 + 4 + 5 + 6)
  let visible_sum := visible.sum
  total_dots - visible_sum = 48 :=
by
  sorry

end invisible_dots_48_l453_453371


namespace tangent_line_at_f_2_l453_453913

def g : ℝ → ℝ := sorry
def f (x : ℝ) : ℝ := g (x / 2) + x^2

def g_prime (x : ℝ) : ℝ := sorry

theorem tangent_line_at_f_2 :
  (∀ x : ℝ, 9 * x + g x + 1 = 0 → g 1 = -8 ∧ g_prime 1 = -9) →
  (f 2 = -4 ∧ (1 / 2) * g_prime 1 + 4 = -1 / 2) →
  ∃ m b : ℝ, (y = m * x + b) ∧ (x, y) = (2, -4) ∧ m = -1 / 2 ∧ b + 4 = -1 / 2 →
  x + 2 * y + 6 = 0 := 
sorry

end tangent_line_at_f_2_l453_453913


namespace find_length_of_tunnel_l453_453304

-- Given conditions
def length_of_train : ℝ := 800
def speed_of_train_kmh : ℝ := 78
def time_to_cross_tunnel_min : ℝ := 1

-- Derived condition: converting speed to meters per second
def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)

-- Derived condition: converting time to seconds
def time_to_cross_tunnel_s : ℝ := time_to_cross_tunnel_min * 60

-- Goal: length of the tunnel
def length_of_tunnel : ℝ := 500.2

-- The proof goal
theorem find_length_of_tunnel :
  length_of_tunnel = (speed_of_train_ms * time_to_cross_tunnel_s) - length_of_train :=
sorry

end find_length_of_tunnel_l453_453304


namespace area_of_triangle_ADM_l453_453868

/-- Given an isosceles right triangle ABC with AB = AC = 2√2 and ∠BAC = 90°, 
    M is the midpoint of AC, and D is a point on AB such that AD = 1.5 * DB,
    prove the area of triangle ADM is 1.2. -/
theorem area_of_triangle_ADM (A B C M D : Point) (h_triangle : is_isosceles_right_triangle A B C (2*real.sqrt 2) (2*real.sqrt 2) (π / 2))
  (h_midpoint : is_midpoint M A C) (h_AD_DB_ratio : dist A D = 1.5 * dist D B) : 
  area (Triangle.mk A D M) = 1.2 := by
  sorry

end area_of_triangle_ADM_l453_453868


namespace hyperbola_vertex_distance_l453_453361

theorem hyperbola_vertex_distance :
  let eq := (16 * x^2 - 64 * x - 4 * y^2 + 8 * y + 100 = 0)
  in ∃ d : ℝ, d = sqrt 10 ∧
  (distance_between_vertices eq d) :=
by
  sorry

end hyperbola_vertex_distance_l453_453361


namespace triple_count_eq_4_l453_453365

theorem triple_count_eq_4 : 
  (∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (a * b + b * c = 56) ∧ (a * c + b * c = 35)) ↔ 4 := 
sorry

end triple_count_eq_4_l453_453365


namespace total_time_taken_l453_453989

def v_boat : ℝ := 22        -- Speed of boat in standing water in kmph
def v_stream : ℝ := 4       -- Speed of the stream in kmph
def d : ℝ := 10080          -- Distance one way in km

-- Proof statement
theorem total_time_taken :
  let speed_downstream := v_boat + v_stream,
      speed_upstream := v_boat - v_stream,
      time_downstream := d / speed_downstream,
      time_upstream := d / speed_upstream,
      total_time := time_downstream + time_upstream
  in total_time = 947.6923 :=
by
  sorry

end total_time_taken_l453_453989


namespace no_set_of_9_numbers_l453_453256

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l453_453256


namespace x_y_n_sum_l453_453519

theorem x_y_n_sum (x y n : ℕ) (h1 : 10 ≤ x ∧ x ≤ 99) (h2 : 10 ≤ y ∧ y ≤ 99) (h3 : y = (x % 10) * 10 + (x / 10)) (h4 : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end x_y_n_sum_l453_453519


namespace combined_area_l453_453694

theorem combined_area (radius : ℝ) (length : ℝ) (height : ℝ) (breadth : ℝ)
  (h_radius_square : radius = real.sqrt 1600)
  (h_length : length = (2 / 5) * radius)
  (h_height : height = 3 * radius)
  (h_breadth : breadth = 10)
  (h_base : base = length) :
  let area_triangle := (1 / 2) * base * height,
      area_rectangle := length * breadth,
      combined_area := area_triangle + area_rectangle
  in combined_area = 1120 :=
by
  sorry

end combined_area_l453_453694


namespace tan_square_B_eq_tan_A_tan_C_range_l453_453799

theorem tan_square_B_eq_tan_A_tan_C_range (A B C : ℝ) (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) 
  (h_tan : Real.tan B * Real.tan B = Real.tan A * Real.tan C) : (π / 3) ≤ B ∧ B < (π / 2) :=
by
  sorry

end tan_square_B_eq_tan_A_tan_C_range_l453_453799


namespace quadratic_non_real_roots_l453_453432

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l453_453432


namespace total_area_is_correct_l453_453472

noncomputable def total_area_of_molds : ℝ :=
  let area_circle := Real.pi * (3 : ℝ)^2,
      area_rectangle := 8 * 4,
      area_triangle := (5 * 3) / 2 in
  area_circle + area_rectangle + area_triangle

theorem total_area_is_correct : total_area_of_molds = 67.77433 :=
  by
  let area_circle := Real.pi * (3 : ℝ)^2
  let area_rectangle := 8 * 4
  let area_triangle := (5 * 3) / 2
  have h1 : area_circle = 28.274333882308138 := by sorry
  have h2 : area_rectangle = 32 := by sorry
  have h3 : area_triangle = 7.5 := by sorry
  show total_area_of_molds = 67.77433
  calc
    total_area_of_molds = area_circle + area_rectangle + area_triangle := by rfl
                      ... = 28.27433 + 32 + 7.5 := by
                        rw [h1, h2, h3]

end total_area_is_correct_l453_453472


namespace gcd_product_square_l453_453528

theorem gcd_product_square {x y z : ℕ} (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : 
  ∃ n : ℕ, n^2 = Nat.gcd x y z * x * y * z := 
sorry

end gcd_product_square_l453_453528


namespace finiteness_of_triples_l453_453911

theorem finiteness_of_triples (x : ℚ) : ∃! (a b c : ℤ), a < 0 ∧ b^2 - 4*a*c = 5 ∧ (a*x^2 + b*x + c > 0) := sorry

end finiteness_of_triples_l453_453911


namespace donut_distribution_l453_453317

theorem donut_distribution (n k: ℕ) (hn: n = 8) (hk: k = 5) :
  (nat.choose ((n - k) + k - 1) (k - 1)) = 35 :=
by
  rw [hn, hk]
  -- n = 8, k = 5
  -- (n-k) + k - 1 = 7
  have h1 : (8 - 5) + 5 - 1 = 7 := by norm_num
  rw [h1]
  -- k - 1 = 4
  have h2 : 5 - 1 = 4 := by norm_num
  rw [h2]
  -- result is binomial coefficient 7 choose 4
  exact nat.choose_self 35
  sorry -- proof not completed

end donut_distribution_l453_453317


namespace fraction_oil_is_correct_l453_453138

noncomputable def fraction_oil_third_bottle (C : ℚ) (oil1 : ℚ) (oil2 : ℚ) (water1 : ℚ) (water2 : ℚ) := 
  (oil1 + oil2) / (oil1 + oil2 + water1 + water2)

theorem fraction_oil_is_correct (C : ℚ) (hC : C > 0) :
  let oil1 := C / 2
  let oil2 := C / 2
  let water1 := C / 2
  let water2 := 3 * C / 4
  fraction_oil_third_bottle C oil1 oil2 water1 water2 = 4 / 9 := by
  sorry

end fraction_oil_is_correct_l453_453138


namespace initial_profit_is_40_l453_453302

variable (P S : ℝ)   -- Define cost price P and initial selling price S

-- Define the given condition: if sold at double the initial selling price, profit percentage is 180%
def condition : Prop := 1.8 = (2 * S - P) / P

-- Define the initial profit percentage calculation 
def initial_profit_percentage : ℝ := ((S - P) / P) * 100

-- The statement to prove: the initial profit percentage is 40%
theorem initial_profit_is_40 (h : condition P S) : initial_profit_percentage P S = 40 := 
by 
  sorry

end initial_profit_is_40_l453_453302


namespace cos_value_of_angle_l453_453429

theorem cos_value_of_angle (α : ℝ) (h : sin (π / 3 + α) = 1 / 3) : cos (5 * π / 6 + α) = -1 / 3 :=
sorry

end cos_value_of_angle_l453_453429


namespace no_primes_divisible_by_45_l453_453062

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_primes_divisible_by_45 : 
  ∀ p, is_prime p → ¬ (45 ∣ p) := 
by
  sorry

end no_primes_divisible_by_45_l453_453062


namespace shaded_area_of_inscribed_rectangles_l453_453210

theorem shaded_area_of_inscribed_rectangles :
  (let width := 10
   let height := 24
   let diameter := Real.sqrt ((width ^ 2) + (height ^ 2))
   let radius := diameter / 2
   let area_circle := Real.pi * (radius ^ 2)
   let area_overlap := 10 * 10
   let area_rectangle := width * height
   let total_area_rectangles := 2 * area_rectangle
   in area_circle - total_area_rectangles + area_overlap) = 169 * Real.pi - 380 := 
by
  sorry

end shaded_area_of_inscribed_rectangles_l453_453210


namespace number_of_assignment_methods_l453_453205

theorem number_of_assignment_methods {teachers questions : ℕ} (h_teachers : teachers = 5) (h_questions : questions = 3) (h_at_least_one_teacher : ∀ q, q ∈ finset.range questions → ∃ t, t ∈ finset.range teachers) : 
  ∃ n, n = 150 :=
by 
  sorry

end number_of_assignment_methods_l453_453205


namespace train_speed_l453_453303

def length_of_train : ℝ := 130
def length_of_bridge : ℝ := 245.03
def crossing_time : ℝ := 30

def total_distance : ℝ := length_of_train + length_of_bridge
def speed_m_per_s : ℝ := total_distance / crossing_time
def speed_km_per_hr : ℝ := speed_m_per_s * 3.6

theorem train_speed :
  speed_km_per_hr = 45.0036 := 
sorry

end train_speed_l453_453303


namespace intersection_of_A_and_B_l453_453896

-- Definitions of sets A and B
def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

-- The theorem we want to prove
theorem intersection_of_A_and_B : A ∩ B = { x | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_B_l453_453896


namespace domain_of_v_l453_453216

noncomputable def v (x : ℝ) : ℝ := 1 / (x^(1/3))

theorem domain_of_v : {x : ℝ | ∃ y, y = v x} = {x : ℝ | x ≠ 0} := by
  sorry

end domain_of_v_l453_453216


namespace solution_set_g_x_less_g_1_minus_2x_l453_453001

variable {ℝ : Type*}
variables (f : ℝ → ℝ) (g : ℝ → ℝ)

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x 

def derivative_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → (x / 2) * (deriv f x) + f x ≤ 0

theorem solution_set_g_x_less_g_1_minus_2x
  (hf_even : even_function f)
  (hf_deriv_cond : derivative_condition f) :
  ∀ x, (x > 1 / 3 ∧ x < 1) ↔ (g x < g (1 - 2 * x)) :=
sorry

end solution_set_g_x_less_g_1_minus_2x_l453_453001


namespace converse_l453_453960

theorem converse (x y : ℝ) (h : x + y ≥ 5) : x ≥ 2 ∧ y ≥ 3 := 
sorry

end converse_l453_453960


namespace geometric_sequence_common_ratio_l453_453875

theorem geometric_sequence_common_ratio (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : a_n 3 = a_n 2 * q) 
  (h2 : a_n 2 * q - 3 * a_n 2 = 2) 
  (h3 : 5 * a_n 4 = (12 * a_n 3 + 2 * a_n 5) / 2) : 
  q = 3 := 
by
  sorry

end geometric_sequence_common_ratio_l453_453875


namespace arithmetic_problem_l453_453223

theorem arithmetic_problem : 1357 + 3571 + 5713 - 7135 = 3506 :=
by
  sorry

end arithmetic_problem_l453_453223


namespace minimum_interval_l453_453046

noncomputable def f (x : ℝ) := 1 + x - (x^2)/2 + (x^3)/3
noncomputable def g (x : ℝ) := 1 - x + (x^2)/2 - (x^3)/3
noncomputable def F (x : ℝ) := f x * g x

theorem minimum_interval : 
  (∀ x : ℝ, F(x) = 0 → -1 ≤ x ∧ x ≤ 2) → 3 = 3 :=
by sorry

end minimum_interval_l453_453046


namespace proposition_truth_value_l453_453013

-- Definitions of the propositions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (|x|) ≥ 1

-- The proof problem statement
theorem proposition_truth_value : (p ∧ q) ∧ ¬ (¬p ∧ q) ∧ ¬ (p ∧ ¬q) ∧ ¬ (¬ (p ∨ q)) :=
by
  sorry

end proposition_truth_value_l453_453013


namespace quadratic_function_properties_l453_453395

variable {f : ℝ → ℝ}

def quadratic_vertex_form : Prop :=
  ∃ a, ∀ x, f(x) = a * (x - 2) ^ 2 - 4

def graph_passes_origin : Prop :=
  f(0) = 0

def monotonically_increasing (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

def quadratic_expression_correct : Prop :=
  f = (λ x, x ^ 2 - 4 * x)

def m_range_correct (m : ℝ) : Prop :=
  4 ≤ m ∧ m < 6

theorem quadratic_function_properties :
  quadratic_vertex_form ∧ graph_passes_origin →
  (∃ g, ∀ x, f x = g x ∧ quadratic_expression_correct) ∧
  (∀ m, monotonically_increasing (Set.Icc (m / 2) 3) ↔ m_range_correct m) :=
by { sorry }

end quadratic_function_properties_l453_453395


namespace point_X_on_AO_l453_453207

open Finset
open Set

variables {α : Type*} [Plane α]
variables {A B C D E F G K L X O : α}
variables {Ω Γ : Circle α}

-- Conditions
variables (ABC_tri_circumcenter : is_circumcenter ABC O)
variables (Gamma_center : is_center A Γ)
variables (D_on_segment : collinear B D E C)
variables (F_G_intersections :
  ∃ (F G : α), F ≠ G ∧ ∈_circle F Γ ∧ ∈_circle G Γ ∧ ∈_circle F Ω ∧ ∈_circle G Ω ∧
    cyclic {A, F, B, C, G})
variables (K_point :
  ∃ (circBDF : Circle α), circumcircle B F Δ ∧ segment_between K ∈_segment B A)
variables (L_point :
  ∃ (circCGE : Circle α), circumcircle C G E Δ ∧ segment_between L ∈_segment C A)
variables (FK_GL_intersect : FK_intersection L =  GL_intersection L ∧ FK_intersection ≠ GL_intersection)

-- Main statement we need to prove
theorem point_X_on_AO : collinear A X O := sorry

end point_X_on_AO_l453_453207


namespace exists_word_D_l453_453484

variable {α : Type} [Inhabited α] [DecidableEq α]

def repeats (D : List α) (w : List α) : Prop :=
  ∃ k : ℕ, w = List.join (List.replicate k D)

theorem exists_word_D (A B C : List α)
  (h : (A ++ A ++ B ++ B) = (C ++ C)) :
  ∃ D : List α, repeats D A ∧ repeats D B ∧ repeats D C :=
sorry

end exists_word_D_l453_453484


namespace sum_of_digits_infinite_repeat_l453_453648

-- Define polynomial with integer coefficients
def is_poly_with_int_coeff (w : ℤ → ℤ) : Prop :=
  ∃ (a : ℕ → ℤ) (k : ℕ), ∀ x : ℤ, w x = ∑ i in range (k+1), a i * x^i

-- Define sum of digits
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.sum

-- Define the sequence p_n
def p (w : ℤ → ℤ) (n : ℕ) : ℕ := sum_of_digits (w n).natAbs

-- The theorem statement to be proven
theorem sum_of_digits_infinite_repeat (w : ℤ → ℤ) (hw : is_poly_with_int_coeff w) :
  ∃ m : ℕ, ∀ N : ℕ, ∃ n : ℕ, N ≤ n ∧ p w n = m :=
sorry

end sum_of_digits_infinite_repeat_l453_453648


namespace proof_inequality_l453_453782

noncomputable def proof_problem (a b c d : ℝ) (h_ab : a * b + b * c + c * d + d * a = 1) : Prop :=
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1 / 3

theorem proof_inequality (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_ab : a * b + b * c + c * d + d * a = 1) : 
  proof_problem a b c d h_ab := 
by
  sorry

end proof_inequality_l453_453782


namespace mothers_full_time_job_fraction_l453_453481

noncomputable def fraction_mothers_with_full_time_jobs (P : ℝ) (M : ℝ) : ℝ :=
  if 0.175 * P = P - M * 0.60 * P - 0.30 * P then M else 0

theorem mothers_full_time_job_fraction :
  ∀ (P : ℝ), P > 0 →
  (fraction_mothers_with_full_time_jobs P 0.875) = 0.875 :=
by
  intros P hP
  rw fraction_mothers_with_full_time_jobs
  have h1 : 0.825 * P = P - 0.175 * P := by linarith
  split_ifs
  · exact rfl
  · sorry

end mothers_full_time_job_fraction_l453_453481


namespace find_income_function_probability_exceeds_180_min_value_m_l453_453235

def daily_income (x : ℕ) : ℝ :=
if x ≤ 150 then 1.2 * x else 1.5 * x - 45

noncomputable def probability_income_exceeds (xs : List ℕ) (threshold : ℝ) : ℝ :=
(xst.filter (λ x, daily_income x > threshold)).length / xs.length

def median (xs : List ℕ) : ℝ :=
let sorted_xs := xs.qsort (≤)
let n := sorted_xs.length
match n % 2 with
| 0 => (sorted_xs.get ((n / 2) - 1) + sorted_xs.get (n / 2)) / 2
| _ => sorted_xs.get (n / 2)

variable (xs : List ℕ := [130, 140, 140, 140, 150, 160, 160, 160, 160, 180])

theorem find_income_function (x : ℕ) (h : x > 0) : 
  daily_income x = if x ≤ 150 then 1.2 * x else 1.5 * x - 45 :=
sorry

theorem probability_exceeds_180 (xs : List ℕ) : 
  probability_income_exceeds xs 180 = 0.5 :=
sorry

theorem min_value_m (xs : List ℕ) (m : ℕ) (h : m > 155) :
  let new_data := m :: xs
  median new_data > 155 :=
sorry

end find_income_function_probability_exceeds_180_min_value_m_l453_453235


namespace range_and_extreme_values_of_f_l453_453041

noncomputable def f (x : ℝ) := (Real.log x / Real.log 4)^2 + Real.log (Real.sqrt x) / Real.log (1/2) - 3

theorem range_and_extreme_values_of_f :
  let I := set.Icc 1 8 in
  let R := (set.image f I) in
  (R = set.Icc (-13/4 : ℝ) (-9/4 : ℝ)) ∧
  (f 2 = -13/4) ∧
  (f 8 = -9/4) :=
by
  -- proof steps here
  sorry

end range_and_extreme_values_of_f_l453_453041


namespace find_x_perpendicular_l453_453518

open Real

theorem find_x_perpendicular (x : ℝ) (ha : (x, 1)) (hb : (1, -2)) (h : (x, 1).fst * (1, -2).fst + (x, 1).snd * (1, -2).snd = 0) : x = 2 := by
  sorry

end find_x_perpendicular_l453_453518


namespace overall_average_cost_per_trip_is_approx_1532_l453_453924

def totalCost : Nat -> Nat -> Nat -> Nat := fun (season_pass cost_per_visit visits : Nat) => season_pass + (cost_per_visit * visits)

noncomputable def averageCostPerTrip : (List (Nat × Nat × Nat)) -> Float := fun costs => 
  let totalVisits := (costs.map (fun (_, _, visits) => visits)).foldr (· + ·) 0
  let totalCosts := (costs.map (fun (season_pass, cost_per_visit, visits) => totalCost season_pass cost_per_visit visits)).foldr (· + ·) 0
  totalCosts.toFloat / totalVisits.toFloat

def problemData : List (Nat × Nat × Nat) := [(100, 12, 35), (90, 15, 25), (80, 10, 20), (70, 8, 15)]

theorem overall_average_cost_per_trip_is_approx_1532 : averageCostPerTrip problemData ≈ 15.32 := 
by sorry

end overall_average_cost_per_trip_is_approx_1532_l453_453924


namespace ratio_sum_l453_453079

-- Definitions for the rectangle and points
structure Rectangle where
  A B C D : Point
  AB BC CD DA : ℝ
  AB_length : AB = 8
  BC_length : BC = 4
  -- Assuming rectangle properties
  right_angle : ∀ {X Y Z : Point}, (X = A ∧ Y = B ∧ Z = D) → angle X Y Z = 90

structure Point where
  x y : ℝ

abbreviation Segment (P Q : Point) := ℝ

def E (B C : Point) : Point := {(x = (2*B.x + C.x) / 3, y = B.y)} -- Simplified one-third point
def F (B C : Point) : Point := {(x = (B.x + 2*C.x) / 3, y = B.y)} -- Simplified two-thirds point

def G (C D : Point) : Point := {(x = D.x, y = C.y)} -- Assume \(G\) divides \(CD\)

-- Points of intersection
def P (A E B D : Point) : Point := sorry
def Q (A F B D : Point) : Point := sorry
def R (A G B D : Point) : Point := sorry

def distance (P Q : Point) : ℝ := sorry -- Placeholder for distance calculation

-- Calculate the ratios
noncomputable def ratio (P E F Q R D : Point) : ℝ × ℝ × ℝ × ℝ :=
  let bp := distance B P
  let pq := distance P Q
  let qr := distance Q R
  let rd := distance R D
  (bp, pq, qr, rd)

theorem ratio_sum (rect : Rectangle) 
  (E_def : E rect.B rect.C) 
  (F_def : F rect.B rect.C) 
  (G_def : G rect.C rect.D) 
  (P_def : P rect.A E_def rect.B rect.D) 
  (Q_def : Q rect.A F_def rect.B rect.D) 
  (R_def : R rect.A G_def rect.B rect.D) :
  let (bp, pq, qr, rd) := ratio rect.B P_def Q_def R_def rect.D in
  bp + pq + qr + rd = 17 := sorry

end ratio_sum_l453_453079


namespace kite_area_l453_453581

theorem kite_area (a b : ℝ) (h_intersect: 
  (∀ x : ℝ, (5 * a * x^2 - 10 = 0) → (8 - (5 * b * x^2) = 0))
  ∧
  ∃ p1 p2 p3 p4 : ℝ×ℝ, 
  (p1 = (⟨0, -10⟩) ∨ p1 = (⟨0, 8⟩) ∨ p1 = (⟨√(2/a), 0⟩) ∨ p1 = (⟨-√(2/a), 0⟩))
  ∧
  (p2 = (⟨0, -10⟩) ∨ p2 = (⟨0, 8⟩) ∨ p2 = (⟨√(8/(5*b)), 0⟩) ∨ p2 = (⟨-√(8/(5*b)), 0⟩))
  ∧
  p1 ≠ p2 ∧ 
  let d1 := (2 * max (√(2/a)) (√(8/(5*b))) - min (√(2/a)) (√(8/(5*b)))) in 
  let d2 := abs ((-10) - 8) in 
  24 = 0.5 * d1 * d2
):
a + (4/5) * a = 2.025 :=
sorry

end kite_area_l453_453581


namespace no_such_three_digit_number_exists_l453_453354

open Nat

def all_digits_different_and_ascending (n : Nat) : Prop :=
  let digits := n.digits
  digits.nodup ∧ digits == digits.sort (· ≤ ·)

theorem no_such_three_digit_number_exists :
  ¬ ∃ n : Nat, 100 ≤ n ∧ n ≤ 999 ∧
  all_digits_different_and_ascending n ∧ 
  all_digits_different_and_ascending (n^2) ∧ 
  all_digits_different_and_ascending (n^3) := by
  sorry

end no_such_three_digit_number_exists_l453_453354


namespace cone_volume_l453_453642

-- Definitions of given conditions
def base_area : ℝ := 30 -- cm²
def height : ℝ := 6    -- cm

-- Defining the formula for the volume of a cone
def volume_of_cone (A h : ℝ) : ℝ := (1/3) * A * h

-- Mathematically equivalent proof problem stating the volume of the cone
theorem cone_volume : volume_of_cone base_area height = 60 := 
by
  -- The actual proof steps would go here
  sorry

end cone_volume_l453_453642


namespace cos_alpha_of_point_on_terminal_side_l453_453842

theorem cos_alpha_of_point_on_terminal_side (x y : ℝ) (h : (x, y) = (3, -4)) :
  Real.cos (Real.atan2 y x) = 3 / 5 :=
by
  sorry

end cos_alpha_of_point_on_terminal_side_l453_453842


namespace cheyenne_earnings_l453_453338

def total_pots := 80
def cracked_fraction := (2 : ℕ) / 5
def price_per_pot := 40

def cracked_pots (total_pots : ℕ) (fraction : ℚ) : ℕ :=
  (fraction * total_pots).toNat

def remaining_pots (total_pots : ℕ) (cracked_pots : ℕ) : ℕ :=
  total_pots - cracked_pots

def total_earnings (remaining_pots : ℕ) (price_per_pot : ℕ) : ℕ :=
  remaining_pots * price_per_pot

theorem cheyenne_earnings :
  total_earnings (remaining_pots total_pots (cracked_pots total_pots cracked_fraction)) price_per_pot = 1920 :=
by
  sorry

end cheyenne_earnings_l453_453338


namespace power_multiplication_equals_result_l453_453324

theorem power_multiplication_equals_result : 
  (-0.25)^11 * (-4)^12 = -4 := 
sorry

end power_multiplication_equals_result_l453_453324


namespace min_max_values_l453_453755

noncomputable def myFunction (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem min_max_values :
  let f := myFunction in 
  isMinimum (set.range (fun (x : ℝ) => if (1 ≤ x ∧ x ≤ 3) then myFunction x else 0)) 1 ∧
  isMaximum (set.range (fun (x : ℝ) => if (1 ≤ x ∧ x ≤ 3) then myFunction x else 0)) 5 :=
sorry

end min_max_values_l453_453755


namespace twenty_eight_is_seventy_percent_of_what_number_l453_453621

theorem twenty_eight_is_seventy_percent_of_what_number (x : ℝ) (h : 28 / x = 70 / 100) : x = 40 :=
by
  sorry

end twenty_eight_is_seventy_percent_of_what_number_l453_453621


namespace radius_comparison_l453_453955

theorem radius_comparison 
  {A B C D E F : Point} 
  (ABC_triangle : Triangle A B C)
  (AD_bisector : is_angle_bisector A D (∠ B A C))
  (CE_bisector : is_angle_bisector C E (∠ A C B))
  (F_intersect : intersection_point A D C E F)
  (BDEF_cyclic : cyclic_quad B D E F)
  (r : ℝ := incircle_radius ABC_triangle)
  (r1 : ℝ := circumcircle_radius B D E F) :
  r1 ≥ r := 
sorry

end radius_comparison_l453_453955


namespace thirteenth_term_geometric_sequence_l453_453189

theorem thirteenth_term_geometric_sequence 
  (a : ℕ → ℕ) 
  (r : ℝ)
  (h₁ : a 7 = 7) 
  (h₂ : a 10 = 21)
  (h₃ : ∀ (n : ℕ), a (n + 1) = a n * r) : 
  a 13 = 63 := 
by
  -- proof needed
  sorry

end thirteenth_term_geometric_sequence_l453_453189


namespace smallest_m_plus_n_l453_453964

theorem smallest_m_plus_n (m n : ℕ) (h1 : 1 < m) 
(h2 : ∀ x : ℝ, -1 ≤ Real.log x * (n / Real.log m.to_real) ∧ Real.log x * (n / Real.log m.to_real) ≤ 1) 
(h3 : ((m^2 - 1) / (m * n) = (1 / 403))) : m + n = 5237 :=
sorry

end smallest_m_plus_n_l453_453964


namespace ironman_age_greater_than_16_l453_453203

variable (Ironman_age : ℕ)
variable (Thor_age : ℕ := 1456)
variable (CaptainAmerica_age : ℕ := Thor_age / 13)
variable (PeterParker_age : ℕ := CaptainAmerica_age / 7)

theorem ironman_age_greater_than_16
  (Thor_13_times_CaptainAmerica : Thor_age = 13 * CaptainAmerica_age)
  (CaptainAmerica_7_times_PeterParker : CaptainAmerica_age = 7 * PeterParker_age)
  (Thor_age_given : Thor_age = 1456) :
  Ironman_age > 16 :=
by
  sorry

end ironman_age_greater_than_16_l453_453203


namespace general_term_max_value_of_m_l453_453510

open Nat

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions
variable (h1 : a 2 + a 12 = 24)
variable (h2 : S 11 = 121)
variable (h3 : ∀ n, b n = 1 / (a (n+1) * a (n+2)))
variable (h4 : ∀ n, T n = ∑ i in range n, b i)

-- Outputs
variable (a_general : ∀ n, a n = n + 5)
variable (m_max : ∀ n, 24 * T n - 3 / 7 ≥ 0)

-- Proof problem statements
theorem general_term {a : ℕ → ℝ} (h1 : a 2 + a 12 = 24) (h2 : S 11 = 121) :
  ∀ n, a n = n + 5 := sorry

theorem max_value_of_m {a : ℕ → ℝ} (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : a 2 + a 12 = 24) (h2 : S 11 = 121)
  (h3 : ∀ n, b n = 1 / (a (n+1) * a (n+2)))
  (h4 : ∀ n, T n = ∑ i in range n, b i) :
  ∀ n, 24 * T n - 3 / 7 ≥ 0 := sorry

end general_term_max_value_of_m_l453_453510


namespace percentage_of_apples_sold_l453_453284

variables (A P : ℝ) 

theorem percentage_of_apples_sold :
  (A = 700) →
  (A * (1 - P / 100) = 420) →
  (P = 40) :=
by
  intros h1 h2
  sorry

end percentage_of_apples_sold_l453_453284


namespace part_1_part_2_l453_453400

def f (a x : ℝ) : ℝ := Real.log x + a * x / (x + 1)

def f_prime (a x : ℝ) : ℝ := (1 / x) + a / (x + 1)^2

theorem part_1 (a : ℝ) : (∀ x ∈ Ioo 0 4, f_prime a x ≥ 0) → a ≥ -4 := sorry

theorem part_2 (a x_0 : ℝ) (h_f_prime : f_prime a x_0 = 2) (h_f : f a x_0 = 2 * x_0) : a = 4 := sorry

end part_1_part_2_l453_453400


namespace number_of_integers_satisfying_conditions_l453_453723

theorem number_of_integers_satisfying_conditions :
  let f :=
    λ (j : ℕ), ∑ d in (finset.filter (λ x, x ∣ j) (finset.range (j+1))), d in
  finset.filter (λ (j : ℕ), 1 ≤ j ∧ j ≤ 5000 ∧ f j = 1 + j + (j : ℝ).cbrt)
  (finset.range 5001).card = 7 :=
by
  let f :=
    λ (j : ℕ), ∑ d in (finset.filter (λ x, x ∣ j) (finset.range (j+1))), d
  have h : finset.filter (λ (j : ℕ), 1 ≤ j ∧ j ≤ 5000 ∧ f j = 1 + j + (j : ℝ).cbrt)
    (finset.range 5001) = {1, 8, 27, 64, 125, 343, 1331, 2197}
    -- These values correspond to 1^3, 2^3, 3^3, 4^3, ..., 13^3, 17^3.
  sorry -- Proof omitted


end number_of_integers_satisfying_conditions_l453_453723


namespace find_radius_of_omega_l453_453099

-- Define the circles and their properties
variables {ω ω₁ ω₂ : Type}
variables [Circle ω] [Circle ω₁] [Circle ω₂]

-- Define the points of intersection and tangency
variables (K L M N : Point)
variables (O O₁ O₂ : Point) -- Centers of the circles
variables (rω rω₁ rω₂ : ℝ) -- Radii of the circles

-- Declare the radii for ω₁ and ω₂ to be 3 and 5, respectively
axiom radius_ω₁ : rω₁ = 3
axiom radius_ω₂ : rω₂ = 5

-- Declare the collinearity of points K, M, and N
axiom collinear_KMN : Collinear K M N

-- Declare the tangency conditions
axiom tangency_ω_ω₁_M : Tangent ω ω₁ M
axiom tangency_ω_ω₂_N : Tangent ω ω₂ N

-- Declare Circle definitions
class Circle (s : Type) :=
  (center : Point)
  (radius : ℝ)

-- Define Point and Circle properties for usage
variables {Point : Type} [point_ops : Circle.Point_ops Point]

-- The theorem to prove
theorem find_radius_of_omega (O : Point) (rω : ℝ) (O₁ O₂ : Point)
  (K L M N : Point) :
  Circle ω → Circle ω₁ → Circle ω₂ →
  radius_ω₁ = 3 → radius_ω₂ = 5 →
  Tangent ω ω₁ M → Tangent ω ω₂ N →
  Collinear K M N →
  rω = 8 :=
by
  sorry -- The proof is omitted as per the requirements.

end find_radius_of_omega_l453_453099


namespace incorrect_permutations_hello_l453_453709

theorem incorrect_permutations_hello :
  ∃ n, n = 5 ∧ 
       (∃ m, m = 2 ∧
        (∃ t, t = (Nat.factorial 5) / (Nat.factorial 2) - 1 ∧ t = 59)) :=
by
  exists 5
  exists 2
  exists ((Nat.factorial 5) / (Nat.factorial 2) - 1)
  sorry

end incorrect_permutations_hello_l453_453709


namespace non_consecutive_seating_l453_453864

theorem non_consecutive_seating :
  (10.factorial - (7.factorial * 4.factorial)) = 3507840 :=
by sorry

end non_consecutive_seating_l453_453864


namespace soap_brands_l453_453672

theorem soap_brands :
  ∀ (total_households neither only_A: ℕ)
    (three_times_both: ℕ → ℕ),
    total_households = 240 →
    neither = 80 →
    only_A = 60 →
    (∀ X, three_times_both X = 3 * X) →
    ∃ X, 80 + 60 + X + (three_times_both X) = 240 ∧ X = 25 :=
by {
  intros total_households neither only_A three_times_both h_total h_neither h_only_A h_three_times_both,
  use 25,
  have h: 80 + 60 + 25 + three_times_both 25 = 240, sorry,
  split,
  { exact h },
  { refl }
}

end soap_brands_l453_453672


namespace solve_system_eq_l453_453570

theorem solve_system_eq :
  ∃ x y : ℝ, (x + y = 13) ∧ (log 4 x + log 4 y = 1 + log 4 10) ∧
  ((x = 5 ∧ y = 8) ∨ (x = 8 ∧ y = 5)) :=
sorry

end solve_system_eq_l453_453570


namespace train_interval_correct_l453_453922

noncomputable def expected_interval_between_trains : ℝ := sorry

theorem train_interval_correct :
  let p := 7 / 12
  let Y : ℝ := 1.25
  let T := 3 in
  (
    -- Northern route duration
    ∀ (northern_route: ℝ), northern_route = 17 ->
    -- Southern route duration
    ∀ (southern_route: ℝ), southern_route = 11 ->
    -- Average arrival time difference
    ∀ (arrival_diff: ℝ), arrival_diff = 1.25 ->
    -- Commute time difference
    ∀ (commute_diff: ℝ), commute_diff = 1 ->
    -- Calculation for the interval between successive trains
    T * (1 - p) = arrival_diff
  ) :=
sorry

end train_interval_correct_l453_453922


namespace find_a_b_sum_pos_solution_l453_453269

theorem find_a_b_sum_pos_solution :
  ∃ (a b : ℕ), (∃ (x : ℝ), x^2 + 16 * x = 100 ∧ x = Real.sqrt a - b) ∧ a + b = 172 :=
by
  sorry

end find_a_b_sum_pos_solution_l453_453269


namespace ratio_of_areas_of_concentric_circles_l453_453211

theorem ratio_of_areas_of_concentric_circles (C1 C2 : ℝ) (h1 : (60 / 360) * C1 = (45 / 360) * C2) :
  (C1 / C2) ^ 2 = (9 / 16) := by
  sorry

end ratio_of_areas_of_concentric_circles_l453_453211


namespace slope_of_line_AB_l453_453986

-- Define the points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (2, 4)

-- State the proposition that we need to prove
theorem slope_of_line_AB :
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 5 / 2 := by
  sorry

end slope_of_line_AB_l453_453986


namespace more_bottle_caps_than_wrappers_l453_453345

namespace DannyCollection

def bottle_caps_found := 50
def wrappers_found := 46

theorem more_bottle_caps_than_wrappers :
  bottle_caps_found - wrappers_found = 4 :=
by
  -- We skip the proof here with "sorry"
  sorry

end DannyCollection

end more_bottle_caps_than_wrappers_l453_453345


namespace probability_top_king_of_hearts_l453_453299

def deck_size : ℕ := 52

def king_of_hearts_count : ℕ := 1

def probability_king_of_hearts_top_card (n : ℕ) (k : ℕ) : ℚ :=
  if n ≠ 0 then k / n else 0

theorem probability_top_king_of_hearts : 
  probability_king_of_hearts_top_card deck_size king_of_hearts_count = 1 / 52 :=
by
  -- Proof omitted
  sorry

end probability_top_king_of_hearts_l453_453299


namespace celebration_day_is_monday_l453_453128

/-- Define the days of the week in a cyclic group modulo 7 --/
inductive WeekDay
| Friday
| Saturday
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday

open WeekDay

/-- Define an operation to move a given day n days forward --/
def move_days : WeekDay → ℕ → WeekDay
| Friday, n      => WeekDay.fromNat ((WeekDay.toNat Friday + n) % 7)
| Saturday, n    => WeekDay.fromNat ((WeekDay.toNat Saturday + n) % 7)
| Sunday, n      => WeekDay.fromNat ((WeekDay.toNat Sunday + n) % 7)
| Monday, n      => WeekDay.fromNat ((WeekDay.toNat Monday + n) % 7)
| Tuesday, n     => WeekDay.fromNat ((WeekDay.toNat Tuesday + n) % 7)
| Wednesday, n   => WeekDay.fromNat ((WeekDay.toNat Wednesday + n) % 7)
| Thursday, n    => WeekDay.fromNat ((WeekDay.toNat Thursday + n) % 7)

noncomputable def WeekDay.fromNat : ℕ → WeekDay
| 0 => Friday
| 1 => Saturday
| 2 => Sunday
| 3 => Monday
| 4 => Tuesday
| 5 => Wednesday
| 6 => Thursday
| _ => Friday  -- This case should never happen by the nature of modulo operation

noncomputable def WeekDay.toNat : WeekDay → ℕ
| Friday => 0
| Saturday => 1
| Sunday => 2
| Monday => 3
| Tuesday => 4
| Wednesday => 5
| Thursday => 6

theorem celebration_day_is_monday : move_days Friday 1200 = Monday :=
  by
    sorry

end celebration_day_is_monday_l453_453128


namespace power_multiplication_equals_result_l453_453325

theorem power_multiplication_equals_result : 
  (-0.25)^11 * (-4)^12 = -4 := 
sorry

end power_multiplication_equals_result_l453_453325


namespace cherry_trees_leaves_l453_453689

-- Define the original number of trees
def original_num_trees : ℕ := 7

-- Define the number of trees actually planted
def actual_num_trees : ℕ := 2 * original_num_trees

-- Define the number of leaves each tree drops
def leaves_per_tree : ℕ := 100

-- Define the total number of leaves that fall
def total_leaves : ℕ := actual_num_trees * leaves_per_tree

-- Theorem statement for the problem
theorem cherry_trees_leaves : total_leaves = 1400 := by
  sorry

end cherry_trees_leaves_l453_453689


namespace sides_and_diagonals_of_polygons_l453_453194

def diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem sides_and_diagonals_of_polygons :
  ∃ (n1 n2 : ℕ), n1 + n2 = 24 ∧ 
    diagonals n1 + diagonals n2 = 109 ∧ 
    ((n1 = 13 ∧ n2 = 11) ∨ (n1 = 11 ∧ n2 = 13)) :=
begin
  sorry
end

end sides_and_diagonals_of_polygons_l453_453194


namespace area_of_region_l453_453503

noncomputable def T := 516

def region (x y : ℝ) : Prop :=
  |x| - |y| ≤ T - 500 ∧ |y| ≤ T - 500

theorem area_of_region :
  (4 * (T - 500)^2 = 1024) :=
  sorry

end area_of_region_l453_453503


namespace domain_of_f_l453_453965

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x ≤ t then 2^x else real.log_base (1 / 2) x

theorem domain_of_f (t : ℝ) : 
  (∀ x ∈ set.Icc (1/16:ℝ) 2, x ≠ 4 → (∃ y, y ∈ set.Ioc (0:ℝ) 2^t ∪ set.Ioc (-∞) (real.log_base 2 t) ∧ (if x ≤ t then 2^x else real.log_base (1/2) x) = y)) :=
by
  sorry

end domain_of_f_l453_453965


namespace option_C_is_quadratic_l453_453227

-- Definitions based on conditions
def option_A (x : ℝ) : Prop := x^2 + (1/x^2) = 0
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (x - 1) * (x + 2) = 1
def option_D (x y : ℝ) : Prop := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Statement to prove option C is a quadratic equation in one variable.
theorem option_C_is_quadratic : ∀ x : ℝ, (option_C x) → (∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0) :=
by
  intros x hx
  -- To be proven
  sorry

end option_C_is_quadratic_l453_453227


namespace part1_S1_part1_S2_part2_const_l453_453812

noncomputable def a_n (n : ℕ) : ℝ := (1 / Real.sqrt 5) * ((1 + Real.sqrt 5) / 2) ^ n - (1 / Real.sqrt 5) * ((1 - Real.sqrt 5) / 2) ^ n

def S_n (n : ℕ) : ℝ := (Finset.range (n + 1)).sum (λ i, Nat.choose n i * a_n i)

theorem part1_S1: S_n 1 = 1 := sorry
theorem part1_S2: S_n 2 = 3 := sorry
theorem part2_const (n : ℕ) (hn : 1 ≤ n): 
  (S_n (n + 2) + S_n n) / S_n (n + 1) = 3 := sorry

end part1_S1_part1_S2_part2_const_l453_453812


namespace side_length_of_regular_decagon_l453_453736

-- Definitions and conditions
variables (R AB : ℝ)
def central_angle := 72 * (Mathlib.π / 180)
def bisected_angle := 36 * (Mathlib.π / 180)

theorem side_length_of_regular_decagon (R : ℝ) (h : 0 < R) :
  ∃ AB : ℝ, AB = (sqrt 5 - 1) * R / 2 :=
begin
  use (sqrt 5 - 1) * R / 2,
  sorry
end

end side_length_of_regular_decagon_l453_453736


namespace number_of_officer_assignments_l453_453003

theorem number_of_officer_assignments : 
  let members := ["Alice", "Bob", "Carol", "Dave"]
  let roles := ["president", "secretary", "treasurer"]
  (Finset.perm 4 3) = 24 := 
by
  sorry

end number_of_officer_assignments_l453_453003


namespace luke_fish_fillets_l453_453731

def fish_per_day : ℕ := 2
def days : ℕ := 30
def fillets_per_fish : ℕ := 2

theorem luke_fish_fillets : fish_per_day * days * fillets_per_fish = 120 := 
by
  sorry

end luke_fish_fillets_l453_453731


namespace exists_convex_polyhedron_with_edges_equal_to_face_diagonals_of_cube_l453_453727

theorem exists_convex_polyhedron_with_edges_equal_to_face_diagonals_of_cube :
  ∃ (P : Type) [convex_set P], has_edges P (12) → 
  (∀ (e : edge P), length e = face_diagonal_length_of_cube ∧ parallel_to_face_diagonal_of_cube e) :=
by
  sorry

end exists_convex_polyhedron_with_edges_equal_to_face_diagonals_of_cube_l453_453727


namespace circles_intersect_twice_l453_453721

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

noncomputable def circle2 (x y : ℝ) : Prop :=
  x^2 + (y - 1.5)^2 = 9 / 4

theorem circles_intersect_twice : 
  (∃ (p : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2) ∧ 
  (∀ (p q : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2 ∧ circle1 q.1 q.2 ∧ circle2 q.1 q.2 → (p = q ∨ p ≠ q)) →
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧
    circle1 p1.1 p1.2 ∧ circle2 p1.1 p1.2 ∧
    circle1 p2.1 p2.2 ∧ circle2 p2.1 p2.2 := 
by {
  sorry
}

end circles_intersect_twice_l453_453721


namespace find_xyz_l453_453840

theorem find_xyz (x y z : ℝ) 
  (h1 : x * (y + z) = 180) 
  (h2 : y * (z + x) = 192) 
  (h3 : z * (x + y) = 204) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z) : 
  x * y * z = 168 * Real.sqrt 6 :=
sorry

end find_xyz_l453_453840


namespace no_intersection_of_parabola_with_x_axis_l453_453979

theorem no_intersection_of_parabola_with_x_axis :
  let y (x : ℝ) := 3 * x^2 + 2 * x + 1 in
  ∀ x : ℝ, y x ≠ 0 :=
by sorry

end no_intersection_of_parabola_with_x_axis_l453_453979


namespace relationship_depends_on_b_l453_453199

theorem relationship_depends_on_b (a b : ℝ) : 
  (a + b > a - b ∨ a + b < a - b ∨ a + b = a - b) ↔ (b > 0 ∨ b < 0 ∨ b = 0) :=
by
  sorry

end relationship_depends_on_b_l453_453199


namespace irreducible_fraction_l453_453236

theorem irreducible_fraction (a : ℤ) : Nat.gcd (a^3 + 2*a) (a^4 + 3*a^2 + 1) = 1 := by
  sorry

end irreducible_fraction_l453_453236


namespace shortest_chord_value_of_m_l453_453048

theorem shortest_chord_value_of_m :
  (∃ m : ℝ,
      (∀ x y : ℝ, mx + y - 2 * m - 1 = 0) ∧
      (∀ x y : ℝ, x ^ 2 + y ^ 2 - 2 * x - 4 * y = 0) ∧
      (mx + y - 2 * m - 1 = 0 → ∃ x y : ℝ, (x, y) = (2, 1))
  ) → m = -1 :=
by
  sorry

end shortest_chord_value_of_m_l453_453048


namespace range_of_a_for_lim_deriv_pos_l453_453403

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * exp (a * x)

def lim_deriv_pos (a : ℝ) : Prop :=
∀ x : ℝ, 0 < lim (h → 0) (f a (x + h) - f a x) / h

theorem range_of_a_for_lim_deriv_pos :
  ∀ (a : ℝ), lim_deriv_pos a → 1 < a ∧ a ≤ sqrt 2 :=
by 
  sorry

end range_of_a_for_lim_deriv_pos_l453_453403


namespace random_events_l453_453698

/-- Definition of what constitutes a random event --/
def is_random_event (e : String) : Prop :=
  e = "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)" ∨
  e = "Forgetting the last digit of a phone number, randomly pressing and it is correct" ∨
  e = "Winning the first prize in a sports lottery"

/-- Define the specific events --/
def event_1 := "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)"
def event_2 := "Forgetting the last digit of a phone number, randomly pressing and it is correct"
def event_3 := "Opposite electric charges attract each other"
def event_4 := "Winning the first prize in a sports lottery"

/-- Lean 4 statement for the proof problem --/
theorem random_events :
  (is_random_event event_1) ∧
  (is_random_event event_2) ∧
  ¬(is_random_event event_3) ∧
  (is_random_event event_4) :=
by 
  sorry

end random_events_l453_453698


namespace number_of_blocks_used_l453_453882

def length : ℕ := 14
def width : ℕ := 12
def height : ℕ := 6
def wall_thickness : ℕ := 1

def external_volume : ℕ := length * width * height

def internal_length : ℕ := length - 2 * wall_thickness
def internal_width : ℕ := width - 2 * wall_thickness
def internal_height : ℕ := height - wall_thickness

def internal_volume : ℕ := internal_length * internal_width * internal_height

def partition_thickness : ℕ := 1
def partition_volume : ℕ := partition_thickness * internal_width * internal_height

def total_wall_volume : ℕ := external_volume - internal_volume + partition_volume

theorem number_of_blocks_used : total_wall_volume = 458 :=
  sorry

end number_of_blocks_used_l453_453882


namespace proof_example_l453_453008

open Real

theorem proof_example (p q : Prop) :
  (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  have p : ∃ x : ℝ, sin x < 1 := ⟨0, by norm_num⟩
  have q : ∀ x : ℝ, exp (abs x) ≥ 1 := by
    intro x
    have : abs x ≥ 0 := abs_nonneg x
    exact exp_pos (abs x)
  exact ⟨p, q⟩

end proof_example_l453_453008


namespace exist_congruent_triangle_l453_453305

noncomputable def triangle : Type := ℝ × ℝ × ℝ

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

axiom triangle_ABC : triangle := (sqrt 7, sqrt 13, sqrt 19)
axiom circle_A : Circle := ⟨(0, 0), 1/3⟩  -- Assuming center A at (0,0) for simplicity
axiom circle_B : Circle := ⟨(sqrt 7, 0), 2/3⟩  -- Similarly, place B on x-axis
axiom circle_C : Circle := ⟨(sqrt 7 / 2, sqrt (19 - (sqrt 7 / 2)^2)), 1⟩  -- Placements derived from distances

theorem exist_congruent_triangle :
  ∃ (A' B' C' : ℝ × ℝ),
    (A'.fst - circle_A.center.fst)^2 + (A'.snd - circle_A.center.snd)^2 = circle_A.radius^2 ∧
    (B'.fst - circle_B.center.fst)^2 + (B'.snd - circle_B.center.snd)^2 = circle_B.radius^2 ∧
    (C'.fst - circle_C.center.fst)^2 + (C'.snd - circle_C.center.snd)^2 = circle_C.radius^2 ∧
    let ⟨xa, ya⟩ := A', ⟨xb, yb⟩ := B', ⟨xc, yc⟩ := C', ⟨ab, bc, ca⟩ := triangle_ABC in
    (xa - xb)^2 + (ya - yb)^2 = ab ∧
    (xb - xc)^2 + (yb - yc)^2 = bc ∧
    (xc - xa)^2 + (yc - ya)^2 = ca := sorry

end exist_congruent_triangle_l453_453305


namespace quadratic_inequality_solution_l453_453191

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end quadratic_inequality_solution_l453_453191


namespace determine_C_of_triangle_l453_453469

-- Define the sides and angles
variables {a b c : ℝ} {A B C : ℝ}

-- Given conditions in the problem
def is_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a^2 = 3 * b^2 + 3 * c^2 - 2 * real.sqrt 3 * b * c * real.sin A

theorem determine_C_of_triangle :
  is_triangle a b c A → C = π / 6 :=
by
  sorry

end determine_C_of_triangle_l453_453469


namespace line_through_point_equal_intercepts_l453_453362

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (hP : P = (1, 1)) :
  (∀ x y : ℝ, (x - y = 0 ∨ x + y - 2 = 0) → ∃ k : ℝ, k = 1 ∧ k = 2) :=
by
  sorry

end line_through_point_equal_intercepts_l453_453362


namespace proposition_p_and_q_l453_453022

-- Define the propositions as per given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (|x|) ≥ 1

-- The theorem to be proved
theorem proposition_p_and_q : p ∧ q :=
by
  sorry

end proposition_p_and_q_l453_453022


namespace cherry_tree_leaves_l453_453686

theorem cherry_tree_leaves (original_plan : ℕ) (multiplier : ℕ) (leaves_per_tree : ℕ) 
  (h1 : original_plan = 7) (h2 : multiplier = 2) (h3 : leaves_per_tree = 100) : 
  (original_plan * multiplier * leaves_per_tree = 1400) :=
by
  sorry

end cherry_tree_leaves_l453_453686


namespace difference_in_girls_and_boys_l453_453985

theorem difference_in_girls_and_boys (x : ℕ) (h1 : 3 + 4 = 7) (h2 : 7 * x = 49) : 4 * x - 3 * x = 7 := by
  sorry

end difference_in_girls_and_boys_l453_453985


namespace optionC_is_quadratic_l453_453225

-- Define what it means to be a quadratic equation in one variable.
def isQuadraticInOneVariable (eq : Expr) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ eq = a * x^2 + b * x + c = 0

-- Define the given options
def optionA : Expr := x^2 + 1 / x^2 = 0
def optionB (a b c : ℝ) : Expr := a * x^2 + b * x + c = 0
def optionC : Expr := (x - 1) * (x + 2) = 1 
def optionD (y : ℝ) : Expr := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Define the proof problem
theorem optionC_is_quadratic :
  isQuadraticInOneVariable optionC :=
sorry

end optionC_is_quadratic_l453_453225


namespace root_of_quadratic_eq_l453_453467

theorem root_of_quadratic_eq (k : ℝ) : (2^2 - 5*2 + k = 0) → k = 6 := by
  assume h1 : (2^2 - 5*2 + k = 0)
  sorry

end root_of_quadratic_eq_l453_453467


namespace area_triangle_qrs_proof_l453_453899

def area_triangle_qrs (u v w : ℝ) (q r s : ℝ)
  (h1 : q * r = 4 * u) 
  (h2 : r * s = 4 * v) 
  (h3 : q * s = 4 * w) : ℝ :=
√(u ^ 2 + v ^ 2 + w ^ 2)

theorem area_triangle_qrs_proof
  {P Q R S : ℝ × ℝ × ℝ}
  (h_perpendicular : (P ≠ Q) ∧ (P ≠ R) ∧ (P ≠ S) ∧
                     (Q ≠ R) ∧ (Q ≠ S) ∧ (R ≠ S))
  (h_area_pqr : ∃ (u : ℝ), 2 * u = (1 / 2) * |((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1))|)
  (h_area_prs : ∃ (v : ℝ), 2 * v = (1 / 2) * |((R.1 - P.1) * (S.2 - P.2) - (R.2 - P.2) * (S.1 - P.1))|)
  (h_area_pqs : ∃ (w : ℝ), 2 * w = (1 / 2) * |((Q.1 - P.1) * (S.2 - P.2) - (Q.2 - P.2) * (S.1 - P.1))|) :
  let u := (classical.some h_area_pqr),
      v := (classical.some h_area_prs),
      w := (classical.some h_area_pqs) in
  √(u ^ 2 + v ^ 2 + w ^ 2) = area_triangle_qrs u v w sorry sorry sorry :=
sorry

end area_triangle_qrs_proof_l453_453899


namespace jokes_increase_factor_l453_453888

theorem jokes_increase_factor 
  (jessy_jokes_sat: ℕ) (alan_jokes_sat: ℕ) (total_jokes: ℕ) : 
  jessy_jokes_sat = 11 → 
  alan_jokes_sat = 7 → 
  total_jokes = 54 → 
  ∃ F : ℕ, (11 * F + 7 * F + 18 = 54) ∧ F = 2 :=
by
  intros h1 h2 h3
  use 2
  split
  { calc
    11 * 2 + 7 * 2 + 18
        = 22 + 14 + 18   : by norm_num
    ... = 54             : by norm_num },
  { refl }

end jokes_increase_factor_l453_453888


namespace price_of_uniform_is_200_l453_453670

-- Definitions based on the conditions
def total_amount_one_year (U : ℝ) : ℝ := 800 + U
def total_amount_nine_months (U : ℝ) : ℝ := (3 / 4) * 800

-- Predicate to capture the condition of nine months payment
def nine_months_payment (U : ℝ) : Prop :=
  total_amount_nine_months U = 400 + U

-- The theorem to prove
theorem price_of_uniform_is_200 (U : ℝ) (h : nine_months_payment U) : U = 200 :=
sorry

end price_of_uniform_is_200_l453_453670


namespace total_fishes_l453_453230

theorem total_fishes (Will_catfish : ℕ) (Will_eels : ℕ) (Henry_multiplier : ℕ) (Henry_return_fraction : ℚ) :
  Will_catfish = 16 → Will_eels = 10 → Henry_multiplier = 3 → Henry_return_fraction = 1 / 2 →
  (Will_catfish + Will_eels) + (Henry_multiplier * Will_catfish - (Henry_multiplier * Will_catfish / 2)) = 50 := 
by
  intros h1 h2 h3 h4
  sorry

end total_fishes_l453_453230


namespace sequence_20th_term_l453_453343

theorem sequence_20th_term :
  let seq (n : ℕ) := if n % 2 = 1 then real.sqrt (2 * n) else -real.sqrt (2 * n) in
  seq 20 = -2 * real.sqrt 10 :=
by
  sorry

end sequence_20th_term_l453_453343


namespace find_length_BI_l453_453479

-- Definitions of triangle properties
structure RightTriangle :=
  (A B C : ℝ)
  (AB BC : ℝ)
  (hypotenuse : ℝ)
  (angle_B : A = 90)

def triangle_ABC : RightTriangle :=
{
  A := 6,
  B := 8,
  C := 10,
  AB := 6,
  BC := 8,
  hypotenuse := Math.sqrt(36 + 64),
  angle_B := 90,
}

-- Calculation of the inradius 
def inradius (A : ℝ) (s : ℝ) : ℝ := (A / s)

-- Calculation of the length from vertex B to the incenter
noncomputable def length_BI (r : ℝ) : ℝ := (r * Math.sqrt(2))

-- The final proof of length BI
theorem find_length_BI (AB BC AC r : ℝ) (A s : ℝ) :
  AC = Math.sqrt (AB^2 + BC^2) →
  A = (1/2) * AB * BC →
  s = (AB + BC + AC) / 2 →
  r = inradius A s →
  length_BI r = 2 * Math.sqrt 2 :=
by
  intros,
  sorry

end find_length_BI_l453_453479


namespace stratified_sampling_group_l453_453660

-- Definitions of conditions
def female_students : ℕ := 24
def male_students : ℕ := 36
def selected_females : ℕ := 8
def selected_males : ℕ := 12

-- Total number of ways to select the group
def total_combinations : ℕ := Nat.choose female_students selected_females * Nat.choose male_students selected_males

-- Proof of the problem
theorem stratified_sampling_group :
  (total_combinations = Nat.choose 24 8 * Nat.choose 36 12) :=
by
  sorry

end stratified_sampling_group_l453_453660


namespace volume_of_substance_l453_453180

-- Definitions based on conditions
def mass_of_substance : ℝ := 200  -- mass in kg
def volume_of_one_gram_in_cm3 : ℝ := 5  -- volume of 1 gram in cubic centimeters
def gram_to_kilogram : ℝ := 0.001
def cm3_to_m3 : ℝ := 1e-6  -- 1 cubic centimeter to cubic meter conversion factor

-- Calculating the expected volume in cubic meters
theorem volume_of_substance (M : ℝ) (V : ℝ) (gk : ℝ) (cm3_m3 : ℝ) : V = 1 :=
  have V_g := V * cm3_m3,
  have V_kg := V_g * 1000,
  have V_200kg := V_kg * M,
  V_200kg = 1 := by sorry


end volume_of_substance_l453_453180


namespace total_students_in_class_l453_453854

def students_chorus := 18
def students_band := 26
def students_both := 2
def students_neither := 8

theorem total_students_in_class : 
  (students_chorus + students_band - students_both) + students_neither = 50 := by
  sorry

end total_students_in_class_l453_453854


namespace find_value_of_a_l453_453874

-- Let a, b, and c be different numbers from {1, 2, 4}
def a_b_c_valid (a b c : ℕ) : Prop := 
  (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
  (a = 1 ∨ a = 2 ∨ a = 4) ∧ 
  (b = 1 ∨ b = 2 ∨ b = 4) ∧ 
  (c = 1 ∨ c = 2 ∨ c = 4)

-- The condition that (a / 2) / (b / c) equals 4 when evaluated
def expr_eq_four (a b c : ℕ) : Prop :=
  (a / 2 : ℚ) / (b / c : ℚ) = 4

-- Given the above conditions, prove that the value of 'a' is 4
theorem find_value_of_a (a b c : ℕ) (h_valid : a_b_c_valid a b c) (h_expr : expr_eq_four a b c) : a = 4 := 
  sorry

end find_value_of_a_l453_453874


namespace midpoint_product_of_given_segment_l453_453219

-- Define the points
def A : ℝ × ℝ := (-4, 7)
def B : ℝ × ℝ := (-10, -3)

-- Midpoint formula and product of coordinates
def midpoint_product (A B : ℝ × ℝ) : ℝ :=
  let M_x := (A.1 + B.1) / 2
  let M_y := (A.2 + B.2) / 2
  M_x * M_y

-- Prove that the product of the coordinates of the midpoint is -14
theorem midpoint_product_of_given_segment : midpoint_product A B = -14 :=
by
  -- The proof goes here, replacing sorry
  sorry

end midpoint_product_of_given_segment_l453_453219


namespace ratio_XP_XU_l453_453881

variables (X Y Z M N U P : Point)
variables (XM MY XN NZ XP XU : ℝ)
variables (h1 : OnSegment M X Y)
variables (h2 : OnSegment N X Z)
variables (h3 : Bisector X U P M N)
variables (h4 : XM = 2)
variables (h5 : MY = 6)
variables (h6 : XN = 3)
variables (h7 : NZ = 9)

theorem ratio_XP_XU (X Y Z M N U P : Point) 
  (XM MY XN NZ XP XU : ℝ)
  (h1 : OnSegment M X Y)
  (h2 : OnSegment N X Z)
  (h3 : Bisector X U P M N)
  (h4 : XM = 2)
  (h5 : MY = 6)
  (h6 : XN = 3)
  (h7 : NZ = 9) :
  XP / XU = 1 / 4 :=
sorry

end ratio_XP_XU_l453_453881


namespace perimeter_of_figure_l453_453572

-- Define the problem conditions
def square_side_length : ℕ := 2

def rectangle_rows : ℕ := 2
def rectangle_columns : ℕ := 3

def l_shape_horizontal_extension : ℕ := 2
def l_shape_vertical_extension : ℕ := 1

-- Define the goal: Determine the perimeter of the described figure
theorem perimeter_of_figure : 
  let horizontal_segments := (rectangle_columns + l_shape_horizontal_extension) * square_side_length + rectangle_columns * square_side_length in
  let vertical_segments := rectangle_rows * square_side_length + (rectangle_rows * square_side_length + l_shape_vertical_extension * square_side_length) in
  horizontal_segments + vertical_segments = 26 := 
by
  sorry

end perimeter_of_figure_l453_453572


namespace files_remaining_l453_453817

def initial_music_files : ℕ := 27
def initial_video_files : ℕ := 42
def initial_doc_files : ℕ := 12
def compression_ratio_music : ℕ := 2
def compression_ratio_video : ℕ := 3
def files_deleted : ℕ := 11

def compressed_music_files : ℕ := initial_music_files * compression_ratio_music
def compressed_video_files : ℕ := initial_video_files * compression_ratio_video
def total_compressed_files : ℕ := compressed_music_files + compressed_video_files + initial_doc_files

theorem files_remaining : total_compressed_files - files_deleted = 181 := by
  -- we skip the proof for now
  sorry

end files_remaining_l453_453817


namespace value_of_expression_l453_453065

-- Define the hypothesis and the goal
theorem value_of_expression (x y : ℝ) (h : 3 * y - x^2 = -5) : 6 * y - 2 * x^2 - 6 = -16 := by
  sorry

end value_of_expression_l453_453065


namespace minimum_set_size_l453_453243

theorem minimum_set_size (n : ℕ) :
  (2 * n + 1) ≥ 11 :=
begin
  have h1 : 7 * (2 * n + 1) ≤ 15 * n + 2,
  sorry,
  have h2 : 14 * n + 7 ≤ 15 * n + 2,
  sorry,
  have h3 : n ≥ 5,
  sorry,
  show 2 * n + 1 ≥ 11,
  from calc
    2 * n + 1 = 2 * 5 + 1 : by linarith
          ... ≥ 11 : by linarith,
end

end minimum_set_size_l453_453243


namespace total_apples_l453_453562

theorem total_apples (boxes : ℕ) (apples_per_box : ℕ) (h1 : boxes = 7) (h2 : apples_per_box = 7) : boxes * apples_per_box = 49 := by
  rw [h1, h2]
  norm_num
  sorry

end total_apples_l453_453562


namespace vertices_colored_with_k_plus_one_l453_453153

theorem vertices_colored_with_k_plus_one (V : Type) [Fintype V] (E : V → V → Prop) (k : ℕ)
  (h_deg : ∀ v : V, Fintype.card {v' : V // E v v'} ≤ k) :
  ∃ (f : V → Fin (k+1)), ∀ v v' : V, E v v' → f v ≠ f v' := 
sorry

end vertices_colored_with_k_plus_one_l453_453153


namespace yuan_representation_l453_453088

-- Define the essential conditions and numeric values
def receiving (amount : Int) : Int := amount
def spending (amount : Int) : Int := -amount

-- The main theorem statement
theorem yuan_representation :
  receiving 80 = 80 ∧ spending 50 = -50 → receiving (-50) = spending 50 :=
by
  intros h
  sorry

end yuan_representation_l453_453088


namespace lines_are_perpendicular_l453_453582

theorem lines_are_perpendicular :
  let a : ℝ × ℝ × ℝ := (1, -3, -1)
  let b : ℝ × ℝ × ℝ := (8, 2, 2)
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0 :=
by
  let a : ℝ × ℝ × ℝ := (1, -3, -1)
  let b : ℝ × ℝ × ℝ := (8, 2, 2)
  calc
    a.1 * b.1 + a.2 * b.2 + a.3 * b.3
        = (1:ℝ) * 8 + (-3:ℝ) * 2 + (-1:ℝ) * 2 : by sorry
    ... = 0 : by sorry

end lines_are_perpendicular_l453_453582


namespace range_of_x_inequality_l453_453031

theorem range_of_x_inequality (a : ℝ) (x : ℝ)
  (h : -1 ≤ a ∧ a ≤ 1) : 
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end range_of_x_inequality_l453_453031


namespace fish_per_bowl_l453_453200

theorem fish_per_bowl (num_bowls num_fish : ℕ) (h1 : num_bowls = 261) (h2 : num_fish = 6003) :
  num_fish / num_bowls = 23 :=
by {
  sorry
}

end fish_per_bowl_l453_453200


namespace shaded_area_l453_453873

-- Define the setup as per the given conditions
theorem shaded_area (radius : ℝ) (O A B C D : ℝ × ℝ) (angle_COD_deg : ℝ)
  (radius_pos : radius = 6)
  (diameters_perpendicular : (∠ A O B = 90) ∧ (∠ C O D = 90))
  (center_O : O = (0, 0))
  (angle_COD : angle_COD_deg = 45) :
  let area_triangle := (1 / 2) * radius * radius in
  let area_sector := (angle_COD_deg / 360) * π * radius^2 in
  area_triangle + area_sector = 18 + 4.5 * π := 
by {
  sorry
}

end shaded_area_l453_453873


namespace non_real_roots_interval_l453_453452

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l453_453452


namespace proof_example_l453_453009

open Real

theorem proof_example (p q : Prop) :
  (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  have p : ∃ x : ℝ, sin x < 1 := ⟨0, by norm_num⟩
  have q : ∀ x : ℝ, exp (abs x) ≥ 1 := by
    intro x
    have : abs x ≥ 0 := abs_nonneg x
    exact exp_pos (abs x)
  exact ⟨p, q⟩

end proof_example_l453_453009


namespace base_measurement_zions_house_l453_453639

-- Given conditions
def height_zion_house : ℝ := 20
def total_area_three_houses : ℝ := 1200
def num_houses : ℝ := 3

-- Correct answer
def base_zion_house : ℝ := 40

-- Proof statement (question translated to lean statement)
theorem base_measurement_zions_house :
  ∃ base : ℝ, (height_zion_house = 20 ∧ total_area_three_houses = 1200 ∧ num_houses = 3) →
  base = base_zion_house :=
by
  sorry

end base_measurement_zions_house_l453_453639


namespace a_n_formula_sum_c_n_l453_453490

noncomputable theory

-- Definitions for the sequences based on conditions
def a_n : ℕ → ℝ
| 1 := 0
| 2 := 1 -- inferred from a1 and a3
| 3 := 2
| n := n - 1

def b_n (n : ℕ) : ℝ := 2 * a_n n + 1

def c_n (n : ℕ) : ℝ := a_n n * b_n n

-- Sum of the first n terms of sequence c_n
def S_n (n : ℕ) : ℝ := ∑ i in (finset.range n).map finset.nat.cast, c_n (i + 1)

-- Theorem statements
theorem a_n_formula (n : ℕ) : a_n n = n - 1 := by
  sorry

theorem sum_c_n (n : ℕ) : S_n n = 4 + (n - 2) * 2^(n + 1) := by
  sorry

end a_n_formula_sum_c_n_l453_453490


namespace four_digit_numbers_count_l453_453757

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 672 ∧ 
  (∀ (d1 d2 d3 d4 : ℕ), 
    d1 ∈ {2, 4, 6, 8} ∧
    (d1 + d4) % 3 = 0 ∧
    1 ≤ d2 ∧ d2 ≤ 9 ∧
    1 ≤ d3 ∧ d3 ≤ 9 ∧
    1 ≤ d4 ∧ d4 ≤ 9 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 → 
    true) := sorry

end four_digit_numbers_count_l453_453757


namespace part1_part2_l453_453091

noncomputable def vec_a : ℝ × ℝ := (3, 2)
noncomputable def vec_b : ℝ × ℝ := (-1, 2)
noncomputable def vec_c : ℝ × ℝ := (4, 1)

def d_coordinates (λ: ℝ) : ℝ × ℝ := 
  ((5*λ/8) * vec_a.1 + (7*λ/8) * vec_b.1, (5*λ/8) * vec_a.2 + (7*λ/8) * vec_b.2)

def vec_d_norm_eq_sqrt_10 (λ: ℝ) : Prop := 
  let d := d_coordinates λ
  (d.1^2 + d.2^2) = 10

def parallel (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.2 = v1.2 * v2.1

def a_kc_parallel_2b_a (k : ℝ) : Prop := 
  parallel (vec_a.1 + k * vec_c.1, vec_a.2 + k * vec_c.2) (2 * vec_b.1 - vec_a.1, 2 * vec_b.2 - vec_a.2)

theorem part1 (λ : ℝ) (h : vec_d_norm_eq_sqrt_10 λ) : d_coordinates λ = (1, 3) ∨ d_coordinates λ = (-1, -3) :=
sorry

theorem part2 (k : ℝ) : a_kc_parallel_2b_a k -> k = -16 / 13 :=
sorry

end part1_part2_l453_453091


namespace intersection_of_sets_l453_453051

theorem intersection_of_sets :
  let A := {x : ℝ | 0 < x ∧ x < 2},
      B := {y : ℝ | y > 1},
      C := {x : ℝ | 1 < x ∧ x < 2}
  in A ∩ B = C :=
by sorry

end intersection_of_sets_l453_453051


namespace minor_premise_is_2_l453_453880

theorem minor_premise_is_2
    (premise1: ∀ (R: Type) (rect: R → Prop) (parallelogram: R → Prop), ∀ x, rect x → parallelogram x)
    (premise2: ∀ (T: Type) (triangle: T → Prop) (parallelogram: T → Prop), ∀ x, triangle x → ¬ parallelogram x)
    (premise3: ∀ (T: Type) (triangle: T → Prop) (rect: T → Prop), ∀ x, triangle x → ¬ rect x):
    (premise2) := 
sorry

end minor_premise_is_2_l453_453880


namespace circle_tangent_line_l453_453657

-- Define the conditions
def is_tangent (c : ℝ × ℝ) (r : ℝ) (line : ℝ → ℝ) : Prop :=
  let center := c
  let radius := Real.sqrt r
  let distance := abs ((line center.1) + center.2 - r) / Real.sqrt 2
  radius = distance

-- Problem statement in Lean
theorem circle_tangent_line (r : ℝ) (h_pos : r > 0) 
  (h_tangent : is_tangent (1, 1) r (λ x, r - x)) : 
  r = 3 + Real.sqrt 5 :=
sorry

end circle_tangent_line_l453_453657


namespace calculation_correct_l453_453706

theorem calculation_correct :
  (-1 : ℤ) ^ 53 + 2 ^ (4 ^ 3 + 5 ^ 2 - 7 ^ 2) = 1099511627775 :=
by
  have h1 : (-1 : ℤ) ^ 53 = -1 := by sorry
  have h2 : 4 ^ 3 = 64 := by norm_num
  have h3 : 5 ^ 2 = 25 := by norm_num
  have h4 : 7 ^ 2 = 49 := by norm_num
  have h5 : 2 ^ 40 = 1099511627776 := by norm_num
  calc
    (-1 : ℤ) ^ 53 + 2 ^ (4 ^ 3 + 5 ^ 2 - 7 ^ 2)
        = -1 + 2 ^ (64 + 25 - 49) : by rw [h1, h2, h3, h4]
    ... = -1 + 2 ^ 40 : by norm_num
    ... = -1 + 1099511627776 : by rw h5
    ... = 1099511627775 : by norm_num

end calculation_correct_l453_453706


namespace initial_population_l453_453982

/-- The population of a town decreases annually at the rate of 20% p.a.
    Given that the population of the town after 2 years is 19200,
    prove that the initial population of the town was 30,000. -/
theorem initial_population (P : ℝ) (h : 0.64 * P = 19200) : P = 30000 :=
sorry

end initial_population_l453_453982


namespace problem_statement_l453_453793

variable {f : ℝ → ℝ}

theorem problem_statement 
  (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
  (h_cond : ∀ x : ℝ, f x > deriv f x) :
  f 2013 < real.exp 2013 * f 0 :=
sorry

end problem_statement_l453_453793


namespace tenth_term_arithmetic_sequence_l453_453608

theorem tenth_term_arithmetic_sequence (a d : ℤ) 
  (h1 : a + 2 * d = 23) (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
  by
    sorry

end tenth_term_arithmetic_sequence_l453_453608


namespace smallest_value_4x_plus_3y_l453_453396

-- Define the condition as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

-- Prove the smallest possible value of 4x + 3y given the condition
theorem smallest_value_4x_plus_3y : ∃ x y : ℝ, circle_eq x y ∧ (4 * x + 3 * y = -40) :=
by
  -- Placeholder for the proof
  sorry

end smallest_value_4x_plus_3y_l453_453396


namespace arithmetic_sequence_modulo_sum_l453_453627

theorem arithmetic_sequence_modulo_sum (a_1 d : ℕ) (n : ℕ) 
  (h1 : a_1 = 3) (h2 : d = 8) 
  (h3 : n = 36) 
  (h4 : ∀ k, 0 ≤ k ∧ k < n → (a_1 + k * d) % 8 = 3) : 
  (∑ i in Finset.range n, (a_1 + i * d)) % 8 = 4 :=
by
  sorry

end arithmetic_sequence_modulo_sum_l453_453627


namespace find_n_l453_453937

noncomputable def region_area_eq {A B P : Type} (r : ℝ) (s PA PB arc_ab : ℝ) : Prop :=
  PA = r ∧ PB = s ∧ arc_ab = 1 / 8 - (s * (2 ^ (1/2) / PA))

theorem find_n :
  let A1 A2 A3 A4 A5 A6 A7 A8 : Type := sorry,
  let P : Type := sorry,
  let circle_area : ℝ := 1,
  let region_eq_17 : region_area_eq 1 (1 / 7) (1 / sqrt π) (sqrt 2 / 4π) := sorry,
  let region_eq_19 : region_area_eq 1 (1 / 9) (1 / sqrt π) (sqrt 2 / 4π) := sorry,
  ∃ (n : ℕ), n = 504 := 
sorry

end find_n_l453_453937


namespace range_of_a_l453_453809

noncomputable def f (x : ℝ) := (-4 * x + 5) / (x + 1)

noncomputable def g (a x : ℝ) := a * Real.sin (π / 3 * x) - 2 * a + 2

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ ∈ Set.Icc (0 : ℝ) 2, f x₁ = g a x₂) → a > 0 ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l453_453809


namespace probability_of_air_quality_good_next_day_l453_453650

variable (P_A : ℝ) (P_A_and_B : ℝ)
def P_B_given_A : ℝ := P_A_and_B / P_A

-- Given conditions
axiom h1 : P_A = 0.8
axiom h2 : P_A_and_B = 0.6

-- Prove that P(B | A) = 0.75
theorem probability_of_air_quality_good_next_day : P_B_given_A P_A P_A_and_B = 0.75 :=
by
  rw [h1, h2]
  -- proof steps
  sorry

end probability_of_air_quality_good_next_day_l453_453650


namespace f_increasing_on_interval_solve_inequality_determine_m_range_l453_453388

section
variables {f : ℝ → ℝ} {x a b m : ℝ}

-- Condition: f is an odd function
axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x

-- Condition: f(1) = 1
axiom f_one : f 1 = 1

-- Condition: ∀ a, b ∈ [-1, 1], a + b ≠ 0 → (f(a) + f(b)) / (a + b) > 0
axiom pos_frac (f : ℝ → ℝ) (a b : ℝ) : a ∈ Icc (-1 : ℝ) 1 → b ∈ Icc (-1 : ℝ) 1 → a + b ≠ 0 → (f a + f b) / (a + b) > 0

-- Problem 1: Prove that f is increasing on [-1, 1]
theorem f_increasing_on_interval : 
  (∀ x1 x2 ∈ Icc (-1 : ℝ) 1, x1 < x2 → f x1 < f x2) := sorry

-- Problem 2: Solve the inequality
theorem solve_inequality (x : ℝ) : 
  f (x + 1/2) < f (1 / (x - 1)) ↔ x ∈ Icc (-3/2 : ℝ) (-1: ℝ) := sorry

-- Problem 3: Determine the range of m
theorem determine_m_range : 
  (∀ x a ∈ Icc (-1 : ℝ) 1, f x ≤ m^2 - 2 * a * m + 1) → (m ≤ -2 ∨ m ≥ 2 ∨ m = 0) := sorry

end

end f_increasing_on_interval_solve_inequality_determine_m_range_l453_453388


namespace exists_m_divisible_by_2005_l453_453526

def f (x : ℕ) : ℕ := 3 * x + 2

noncomputable def f_iter (n : ℕ) : ℕ → ℕ
| 0     => id
| (k+1) => λ x, f (f_iter k x)

theorem exists_m_divisible_by_2005 : ∃ (m : ℕ), f_iter 100 m % 2005 = 0 := 
by
  sorry

end exists_m_divisible_by_2005_l453_453526


namespace incorrect_statement_D_l453_453932

def data := [11, 10, 11, 13, 11, 13, 15]

noncomputable def mode (l : List ℕ) : ℕ :=
l.groupBy id
  |>.maxBy (λ g => g.length)
  |>.head!
  
noncomputable def mean (l : List ℕ) : ℚ :=
(float.ofNat l.sum / float.ofNat l.length).approx

noncomputable def variance (l : List ℕ) : ℚ :=
let μ := mean l
(float.ofNat (l.map (λ x => (float.ofNat x - μ) ^ 2).sum) / float.ofNat l.length).approx 

noncomputable def median (l : List ℕ) : ℕ :=
let sorted := l.sorted
sorted.nth (sorted.length / 2)

theorem incorrect_statement_D : median data ≠ 13 := by sorry

end incorrect_statement_D_l453_453932


namespace travel_time_l453_453162

theorem travel_time (distance speed : ℝ) (h1 : distance = 300) (h2 : speed = 60) : 
  distance / speed = 5 := 
by
  sorry

end travel_time_l453_453162


namespace intersection_of_A_and_B_l453_453898

-- Define the sets A and B
def A := {x : ℝ | x ≥ 1}
def B := {x : ℝ | -1 < x ∧ x < 2}

-- Define the expected intersection
def expected_intersection := {x : ℝ | 1 ≤ x ∧ x < 2}

-- The proof problem statement
theorem intersection_of_A_and_B :
  A ∩ B = expected_intersection := by
  sorry

end intersection_of_A_and_B_l453_453898


namespace domain_is_interval_l453_453583

noncomputable def domain_of_function := { x : ℝ | ∃ y : ℝ, y = real.sqrt (real.log (3 * x - 2) / real.log (0.5)) }

theorem domain_is_interval :
  domain_of_function = { x : ℝ | (2 / 3 : ℝ) < x ∧ x ≤ 1 } :=
by
  sorry

end domain_is_interval_l453_453583


namespace increasing_on_interval_min_interval_length_l453_453805

-- first part: proving the range for omega
theorem increasing_on_interval (ω : ℝ) (h : ω > 0) :
  (2 * Real.sin (ω * -3 * Real.pi / 4) < 2 * Real.sin (ω * Real.pi / 3)) ↔ (0 < ω ∧ ω ≤ 2/3) :=
sorry

-- second part: proving minimum length for interval
theorem min_interval_length :
  ∃ (a b : ℝ), a < b ∧ (∀ x ∈ (set.Icc a b), 2 * Real.sin (2 * x + Real.pi / 3) + 1 = 0) ∧ 
  (b - a = (28 * Real.pi) / 3) :=
sorry

end increasing_on_interval_min_interval_length_l453_453805


namespace fraction_of_shaded_area_l453_453489

-- Define the conditions and constants
def h_triangle : ℝ := 1
def r_semicircle : ℝ := 1
def side_length_of_triangle : ℝ := (2 : ℝ) * Real.sqrt 3 / 3

-- Define the area calculations
def area_triangle : ℝ := (Real.sqrt 3 / 4) * (side_length_of_triangle ^ 2)
def r_inscribed_circle : ℝ := (side_length_of_triangle * Real.sqrt 3) / 6
def area_inscribed_circle : ℝ := Real.pi * (r_inscribed_circle ^ 2)
def area_semicircle : ℝ := (1 / 2) * Real.pi * (r_semicircle ^ 2)

-- Define the shaded area and the fraction of the shaded area
def A_shaded : ℝ := area_semicircle - area_triangle - area_inscribed_circle
def fraction_shaded : ℝ := A_shaded / area_semicircle

-- The proof statement
theorem fraction_of_shaded_area : fraction_shaded = 2 / 9 := 
by {
  -- The proof goes here
  sorry
}

end fraction_of_shaded_area_l453_453489


namespace win_sector_area_l453_453659

theorem win_sector_area (r : ℝ) (p : ℝ) (area : ℝ) (win_area : ℝ) :
  r = 8 → p = 3 / 8 → area = π * r ^ 2 → win_area = p * area → win_area = 24 * π :=
by
  intros h_r h_p h_area h_win_area
  rw [h_r, h_p] at *
  rw [←h_area, ←h_win_area]
  sorry

end win_sector_area_l453_453659


namespace not_exists_set_of_9_numbers_min_elements_l453_453239

theorem not_exists_set_of_9_numbers (s : Finset ℕ) 
  (h_len : s.card = 9) 
  (h_median : ∑ x in (s.filter (λ x, x ≤ 2)), 1 ≤ 5) 
  (h_other : ∑ x in (s.filter (λ x, x ≤ 13)), 1 ≤ 4) 
  (h_avg : ∑ x in s = 63) :
  False := sorry

theorem min_elements (n : ℕ) (h_nat: n ≥ 5) :
  ∃ s : Finset ℕ, s.card = 2 * n + 1 ∧
                  ∑ x in (s.filter (λ x, x ≤ 2)), 1 = n + 1 ∧ 
                  ∑ x in (s.filter (λ x, x ≤ 13)), 1 = n ∧
                  ∑ x in s = 14 * n + 7 := sorry

end not_exists_set_of_9_numbers_min_elements_l453_453239


namespace general_term_of_geometric_sequence_l453_453390

theorem general_term_of_geometric_sequence 
  (positive_terms : ∀ n : ℕ, 0 < a_n) 
  (h1 : a_1 = 1) 
  (h2 : ∃ a : ℕ, a_2 = a + 1 ∧ a_3 = 2 * a + 5) : 
  ∃ q : ℕ, ∀ n : ℕ, a_n = q^(n-1) :=
by
  sorry

end general_term_of_geometric_sequence_l453_453390


namespace correct_statement_of_option_C_l453_453229

def frequency_distribution_table : Type := sorry
def frequency : Type := sorry
def sample_size : Type := sorry

noncomputable def frequency_of_group (group_frequency : frequency) (total_sample : sample_size) : frequency := 
  group_frequency / total_sample

theorem correct_statement_of_option_C (group_frequency : frequency) (total_sample : sample_size) :
  frequency_of_group group_frequency total_sample = group_frequency / total_sample := by
  sorry

end correct_statement_of_option_C_l453_453229


namespace minimum_value_abs_sum_l453_453530

theorem minimum_value_abs_sum (α β γ : ℝ) (h1 : α + β + γ = 2) (h2 : α * β * γ = 4) : 
  |α| + |β| + |γ| ≥ 6 :=
by
  sorry

end minimum_value_abs_sum_l453_453530


namespace travel_time_l453_453160

def speed : ℝ := 60  -- Speed of the car in miles per hour
def distance : ℝ := 300  -- Distance to the campground in miles

theorem travel_time : distance / speed = 5 := by
  sorry

end travel_time_l453_453160


namespace classrooms_students_guinea_pigs_difference_l453_453732

theorem classrooms_students_guinea_pigs_difference :
  let students_per_classroom := 22
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 5
  let total_students := students_per_classroom * number_of_classrooms
  let total_guinea_pigs := guinea_pigs_per_classroom * number_of_classrooms
  total_students - total_guinea_pigs = 95 :=
  by
    sorry

end classrooms_students_guinea_pigs_difference_l453_453732


namespace log_sum_of_sequence_l453_453380

theorem log_sum_of_sequence :
  (∃ x : ℕ → ℝ, (∀ n : ℕ, n > 0 → log (x (n + 1)) = 1 + log (x n)) ∧ (∑ k in finset.range 10, x (k + 1)) = 100) →
  log (∑ k in finset.range 10, x (k + 11)) = 12 :=
sorry

end log_sum_of_sequence_l453_453380


namespace quadratic_non_real_roots_l453_453449

theorem quadratic_non_real_roots (b : ℝ) : 
  let a : ℝ := 1 
  let c : ℝ := 16 in
  (b^2 - 4 * a * c < 0) ↔ (-8 < b ∧ b < 8) :=
sorry

end quadratic_non_real_roots_l453_453449


namespace find_principal_amount_compound_interest_l453_453682

theorem find_principal_amount_compound_interest 
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℝ) (t : ℕ)
  (hA : A = 1_000_000)
  (hr : r = 0.08)
  (hn : n = 4)
  (ht : t = 5)
  (hcompound : A = P * (1 + r / n) ^ (n * t)) :
  P ≈ 673_014.35 :=
by
  sorry

end find_principal_amount_compound_interest_l453_453682


namespace exponential_logarithmic_order_l453_453790

theorem exponential_logarithmic_order (a π b : ℝ) (h_a : a > π) (h_π : π > b) (h_b : b > 1) (c : ℝ) (h_c : 1 > c) (h_c0 : c > 0) :
    (let x := a^(1/π) in
     let y := log b / log π in
     let z := log π / log c in
     x > y ∧ y > z) := 
by 
    let x := a^(1/π)
    let y := log b / log π
    let z := log π / log c
    have h1 : x > 1, from sorry
    have h2 : 0 < y ∧ y < 1, from sorry
    have h3 : z < 0, from sorry
    exact h1 ∧ h2.1 ∧ sorry

end exponential_logarithmic_order_l453_453790


namespace quadratic_inequality_solution_l453_453182

theorem quadratic_inequality_solution (x : ℝ) : (2 * x^2 - 5 * x - 3 < 0) ↔ (-1/2 < x ∧ x < 3) :=
by
  sorry

end quadratic_inequality_solution_l453_453182


namespace find_extrema_l453_453749

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem find_extrema :
  let I := set.Icc (1 : ℝ) (3 : ℝ) in
  (∀ x ∈ I, f x ≥ 1) ∧ (∀ x ∈ I, 1 ≤ f x) ∧
  (∀ x ∈ I, f x ≤ 5) ∧ (∀ x ∈ I, f x < 5) :=
by {
  sorry,
}

end find_extrema_l453_453749


namespace min_elements_l453_453250

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l453_453250


namespace innings_question_l453_453272

theorem innings_question (n : ℕ) (runs_in_inning : ℕ) (avg_increase : ℕ) (new_avg : ℕ) 
  (h_runs_in_inning : runs_in_inning = 88) 
  (h_avg_increase : avg_increase = 3) 
  (h_new_avg : new_avg = 40)
  (h_eq : 37 * n + runs_in_inning = new_avg * (n + 1)): n + 1 = 17 :=
by
  -- Proof to be filled in here
  sorry

end innings_question_l453_453272


namespace union_of_A_and_B_l453_453843

open Set

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > -1} :=
by sorry

end union_of_A_and_B_l453_453843


namespace hexagon_surface_area_ratio_l453_453149

theorem hexagon_surface_area_ratio :
  let r := 1 / 2
  let R := 1
  let a := 1
  let F1 := 2 * (π * (r * r) / 4) + 2 * (π * (R + r) * a)
  let sqrt3 := Real.sqrt 3
  let r' := sqrt3 / 2
  let m := sqrt3
  let F2 := 2 * (π * r' * a) + π * r' * m
  (F1 / F2) ≈ 1.0104 := 
by
  sorry

end hexagon_surface_area_ratio_l453_453149


namespace initial_men_colouring_l453_453070

theorem initial_men_colouring (M : ℕ) : 
  (∀ m : ℕ, ∀ d : ℕ, ∀ l : ℕ, m * d = 48 * 2 → 8 * 0.75 = 6 → M = 4) :=
by
  sorry

end initial_men_colouring_l453_453070


namespace collinear_points_l453_453201

theorem collinear_points (a b c d : ℝ) (d_ne_zero : d ≠ 0) :
  let p1 := (2, 0, a)
      p2 := (b, 2, 0)
      p3 := (0, c, 2)
      p4 := (8 * d, 4 * d, -2 * d) in
  collinear_points p1 p2 p3 p4 :=
sorry

end collinear_points_l453_453201


namespace tan_x_minus_pi_over_4_parallel_range_of_f_l453_453052

-- Definitions
def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 4)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Conditions
def is_parallel (x : ℝ) : Prop :=
  vec_a x = (k * vec_b x) for some k ∈ ℝ

-- Problem 1
theorem tan_x_minus_pi_over_4_parallel (x : ℝ) (h : is_parallel x) : 
  Real.tan (x - Real.pi / 4) = -7 :=
sorry

-- Problem 2
def f (x : ℝ) : ℝ :=
  let a := vec_a x
      b := vec_b x
  2 * ((a + b).1 * b.1 + (a + b).2 * b.2)

theorem range_of_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  -1 / 2 ≤ f x ∧ f x ≤ Real.sqrt 2 - 1 / 2 :=
sorry

end tan_x_minus_pi_over_4_parallel_range_of_f_l453_453052


namespace range_of_a_l453_453779

theorem range_of_a {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y + 4 = 2 * x * y) (h2 : ∀ (x y : ℝ), x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) :
  a ≤ 17/4 := sorry

end range_of_a_l453_453779


namespace unique_cube_arrangements_l453_453234

/-- There are exactly 2 unique ways to arrange the integers 1 to 6 on the faces of a cube, such that
    any pair of consecutive numbers (including 6 and 1) are on adjacent faces, considering the arrangements
    are identical if they can be transformed into one another by rotation, reflection, or cyclic permutations. -/
theorem unique_cube_arrangements : ∃ (arrangements : Fin 6 → Fin 6) (unique_count : ℕ), unique_count = 2 ∧ 
  ∀ i j : Fin 6, (j = i + 1 ∨ (i = 5 ∧ j = 0)) → adjacent_faces (arrangements i) (arrangements j)
  -- Define "adjacent_faces" which checks if two faces are adjacent on a cube
  sorry

end unique_cube_arrangements_l453_453234


namespace tangency_symmetry_iff_incenters_line_l453_453715

-- Definitions of the given problem
variables {A B C D U V H : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variables [AddCommGroup U] [AddCommGroup V] [AddCommGroup H]

-- Let (I) and (J) be the incircles of triangles ΔDAB and ΔBCD respectively, and let BD be a diagonal
variables (I J : Type) [AddCommGroup I] [AddCommGroup J]
variables (DAB BCD : Type) [AddCommGroup DAB] [AddCommGroup BCD]
variables (BD AC : Type) [AddCommGroup BD] [AddCommGroup AC]

-- Define the midpoint function
noncomputable def midpoint (x y : A) : A := sorry

-- Define symmetry with respect to the midpoint
def symmetrical (u v m : A) : Prop := sorry

-- Define the problem statement
theorem tangency_symmetry_iff_incenters_line
  (h1 : symmetrical U V (midpoint B D))
  (h2 : (line I J) ∩ BD = H)
  (h3 : (line AC) ∩ BD = H) :
  (U + BD - D = CD + DB - BC) ↔ (line_of_incenters IJ) ∩ BD = diagonals_crossing_point AC BD :=
sorry

end tangency_symmetry_iff_incenters_line_l453_453715


namespace product_simplifies_l453_453710

theorem product_simplifies :
  (6 * (1/2) * (3/4) * (1/5) = (9/20)) :=
by
  sorry

end product_simplifies_l453_453710


namespace cheyenne_earnings_l453_453334

-- Define constants and main conditions
def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

-- Number of cracked pots
def cracked_pots : ℕ := (cracked_fraction * total_pots).toNat
-- Number of pots that are good for sale
def sellable_pots : ℕ := total_pots - cracked_pots
-- Total money earned
def total_earnings : ℕ := sellable_pots * price_per_pot

-- Theorem statement
theorem cheyenne_earnings : total_earnings = 1920 := by
  sorry

end cheyenne_earnings_l453_453334


namespace find_extrema_l453_453748

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem find_extrema :
  let I := set.Icc (1 : ℝ) (3 : ℝ) in
  (∀ x ∈ I, f x ≥ 1) ∧ (∀ x ∈ I, 1 ≤ f x) ∧
  (∀ x ∈ I, f x ≤ 5) ∧ (∀ x ∈ I, f x < 5) :=
by {
  sorry,
}

end find_extrema_l453_453748


namespace fixed_point_l453_453593

theorem fixed_point (a : ℝ) : (a + 1) * (-4) - (2 * a + 5) * (-2) - 6 = 0 :=
by
  sorry

end fixed_point_l453_453593


namespace simplify_and_sum_coefficients_l453_453725

theorem simplify_and_sum_coefficients :
  (∃ A B C D : ℤ, (∀ x : ℝ, x ≠ D → (x^3 + 6 * x^2 + 11 * x + 6) / (x + 1) = A * x^2 + B * x + C) ∧ A + B + C + D = 11) :=
sorry

end simplify_and_sum_coefficients_l453_453725


namespace cost_of_gravelling_is_correct_l453_453290

-- Define the dimensions of the rectangular plot
def plot_length : ℝ := 150
def plot_width : ℝ := 85

-- Define the width of the gravel path
def path_width : ℝ := 3.5

-- Define the cost of gravelling per square meter in paise and convert it to rupees
def cost_per_sq_meter_paise : ℝ := 75
def cost_per_sq_meter : ℝ := cost_per_sq_meter_paise / 100

-- Compute the total dimensions including the path
def total_length : ℝ := plot_length + 2 * path_width
def total_width : ℝ := plot_width + 2 * path_width

-- Compute the area of the whole plot including the path
def total_area : ℝ := total_length * total_width

-- Compute the area of the grassy plot without the path
def grassy_area : ℝ := plot_length * plot_width

-- Compute the area of the gravel path
def path_area : ℝ := total_area - grassy_area

-- Compute the total cost of gravelling the path
def total_cost : ℝ := path_area * cost_per_sq_meter

-- Theorem statement
theorem cost_of_gravelling_is_correct : total_cost = 1270.5 :=
sorry

end cost_of_gravelling_is_correct_l453_453290


namespace ratio_quarters_lost_l453_453926

-- Define the starting number of quarters
def start_quarters := 50

-- Define the quarters after first doubling
def doubled_quarters := start_quarters * 2

-- Define the quarters collected each month in the second year
def collected_second_year := 3 * 12

-- Define the quarters collected every third month in the third year
def collected_third_year := 1 * (12 / 3)

-- Define the total quarters before losing any
def total_before_losing := doubled_quarters + collected_second_year + collected_third_year

-- Define the quarters left after losing some
def quarters_left := 105

-- Define the quarters lost
def quarters_lost := total_before_losing - quarters_left

-- Prove the ratio of lost quarters to the total before losing any is 1:4
theorem ratio_quarters_lost : 
  (quarters_lost.to_rat / total_before_losing.to_rat) = (1 / 4) := by
    sorry

end ratio_quarters_lost_l453_453926


namespace condition_neither_sufficient_nor_necessary_l453_453784

theorem condition_neither_sufficient_nor_necessary (p q : Prop) :
  (¬ (p ∧ q)) → (p ∨ q) → False :=
by sorry

end condition_neither_sufficient_nor_necessary_l453_453784


namespace lateral_surface_area_of_regular_triangular_pyramid_l453_453746

theorem lateral_surface_area_of_regular_triangular_pyramid 
  (S : ℝ) (a : ℝ) 
  (h_base : S = (a^2 * real.sqrt 3) / 4) 
  (h_height : S = 3 * (a^2 / 4)) 
  (h_dihedral : ∀ h, h = a / 2) :
  S * real.sqrt 3 = S * real.sqrt 3 :=
by sorry

end lateral_surface_area_of_regular_triangular_pyramid_l453_453746


namespace symmetry_about_point_0_1_l453_453045

noncomputable def h (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + Real.pi / 4)

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x - Real.pi / 4) + 2

theorem symmetry_about_point_0_1 :
  ∀ x, f(x) = 2 - h(-x) :=
sorry

end symmetry_about_point_0_1_l453_453045


namespace not_exists_set_of_9_numbers_min_elements_l453_453237

theorem not_exists_set_of_9_numbers (s : Finset ℕ) 
  (h_len : s.card = 9) 
  (h_median : ∑ x in (s.filter (λ x, x ≤ 2)), 1 ≤ 5) 
  (h_other : ∑ x in (s.filter (λ x, x ≤ 13)), 1 ≤ 4) 
  (h_avg : ∑ x in s = 63) :
  False := sorry

theorem min_elements (n : ℕ) (h_nat: n ≥ 5) :
  ∃ s : Finset ℕ, s.card = 2 * n + 1 ∧
                  ∑ x in (s.filter (λ x, x ≤ 2)), 1 = n + 1 ∧ 
                  ∑ x in (s.filter (λ x, x ≤ 13)), 1 = n ∧
                  ∑ x in s = 14 * n + 7 := sorry

end not_exists_set_of_9_numbers_min_elements_l453_453237


namespace lattice_points_on_line_segment_l453_453663

-- Definition of lattice points on a line segment
def is_lattice_point (x y : ℤ) : Prop := x ≠ 0 ∧ y ≠ 0

-- Coordinates of endpoints
def endpoint1 := (2 : ℤ, 10 : ℤ)
def endpoint2 := (47 : ℤ, 295 : ℤ)

-- Function to calculate the gcd
def gcd (a b : ℤ) : ℤ := a.gcd b

-- Statement of the theorem
theorem lattice_points_on_line_segment : 
  let count_lattice_points := -- function to count lattice points
    let ⟨x1, y1⟩ := endpoint1 in
    let ⟨x2, y2⟩ := endpoint2 in
    let Δx := x2 - x1 in
    let Δy := y2 - y1 in
    gcd (abs Δx) (abs Δy) + 1 in
  count_lattice_points = 16 :=
by
  sorry -- Proof goes here

end lattice_points_on_line_segment_l453_453663


namespace total_interest_correct_l453_453561

-- Definitions
def total_amount : ℝ := 3500
def P1 : ℝ := 1550
def P2 : ℝ := total_amount - P1
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Total interest calculation
noncomputable def interest1 : ℝ := P1 * rate1
noncomputable def interest2 : ℝ := P2 * rate2
noncomputable def total_interest : ℝ := interest1 + interest2

-- Theorem statement
theorem total_interest_correct : total_interest = 144 := 
by
  -- Proof steps would go here
  sorry

end total_interest_correct_l453_453561


namespace exists_m_divisible_by_2005_l453_453525

def f (x : ℤ) : ℤ := 3*x + 2

def f_iterate (n : ℕ) (x : ℤ) : ℤ :=
  (nat.iterate n f) x

theorem exists_m_divisible_by_2005 : ∃ m : ℕ, 2005 ∣ f_iterate 100 m :=
by
  sorry

end exists_m_divisible_by_2005_l453_453525


namespace smallest_x_palindrome_l453_453628

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem smallest_x_palindrome :
  ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 3456) ∧ x = 97 := 
by 
  sorry

end smallest_x_palindrome_l453_453628


namespace three_digit_numbers_form_3_pow_l453_453829

theorem three_digit_numbers_form_3_pow (n : ℤ) : 
  ∃! (n : ℤ), 100 ≤ 3^n ∧ 3^n ≤ 999 :=
by {
  use [5, 6],
  sorry
}

end three_digit_numbers_form_3_pow_l453_453829


namespace shifted_parabola_relationship_l453_453971

-- Step a) and conditions
def original_function (x : ℝ) : ℝ := -2 * x ^ 2 + 4

def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x => f (x + a)
def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := fun x => f x + b

-- Step c) encoding the proof problem
theorem shifted_parabola_relationship :
  (shift_up (shift_left original_function 2) 3 = fun x => -2 * (x + 2) ^ 2 + 7) :=
by
  sorry

end shifted_parabola_relationship_l453_453971


namespace find_joey_age_l453_453105

def ages : list ℕ := [4, 6, 8, 10, 12]

theorem find_joey_age (h_movies : ∃ a b, a + b = 18 ∧ a ∈ ages ∧ b ∈ ages)
                      (h_park : ∃ c d, c < 9 ∧ d < 9 ∧ c ∈ ages ∧ d ∈ ages)
                      (h_home : 6 ∈ ages) : 
                      ∃ joey_age, joey_age ∈ ages ∧ joey_age = 10 :=
by
  sorry

end find_joey_age_l453_453105


namespace tangent_line_at_0_1_l453_453742

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

theorem tangent_line_at_0_1 :
  let p := (0 : ℝ, 1 : ℝ)
  ∃ (m : ℝ), ∀ (x y : ℝ), y = f p.1 + m * (x - p.1)  → y = 3 * x + 1 :=
by
  sorry

end tangent_line_at_0_1_l453_453742


namespace min_rows_512_l453_453863

theorem min_rows_512 (n : ℕ) (table : ℕ → Fin 10 → ℕ)
  (h : ∀ (r : ℕ) (c1 c2 : Fin 10), c1 ≠ c2 → ∃ (r' : ℕ), r' ≠ r ∧ table r c1 ≠ table r' c1 ∧ table r c2 ≠ table r' c2 ∧ (∀ c ≠ c1 ∧ c ≠ c2, table r c = table r' c)) : 
  n ≥ 512 :=
sorry

end min_rows_512_l453_453863


namespace proper_subsets_of_union_l453_453023

def Z := Int
def N := {n : Nat // 0 < n}

def A : Set Z := { x | x^2 + x - 2 < 0 }
def B : Set N := { n | 0 ≤ Real.log (n + 1) / Real.log 2 ∧ Real.log (n + 1) / Real.log 2 < 2 }

def union_set : Set Z := A ∪ (B : Set Z)

def number_of_elements_union_set := Set.card union_set

theorem proper_subsets_of_union : (2^number_of_elements_union_set - 1) = 15 :=
sorry

end proper_subsets_of_union_l453_453023


namespace toy_store_fraction_l453_453422

theorem toy_store_fraction
  (allowance : ℝ) (arcade_fraction : ℝ) (candy_store_amount : ℝ)
  (h1 : allowance = 1.50)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : candy_store_amount = 0.40) :
  (0.60 - candy_store_amount) / (allowance - arcade_fraction * allowance) = 1 / 3 :=
by
  -- We're skipping the actual proof steps
  sorry

end toy_store_fraction_l453_453422


namespace solve_for_x_l453_453946

theorem solve_for_x : ∃ (x : ℤ), 7 * (2 * x + 3) - 5 = -3 * (2 - 5 * x) → x = 22 :=
by
  intro h
  use 22
  sorry

end solve_for_x_l453_453946


namespace factorization_l453_453948

-- Define the polynomial P(x)
def P (x : ℤ) : ℤ := x^5 + x^4 + 1

-- Define the condition that for all integer inputs from -10 to +10 except -1, 0, 1, P(x) yields a composite number
def condition : Prop :=
  ∀ x : ℤ, (-10 ≤ x ∧ x ≤ 10 ∧ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) → ¬ (∀ d : ℤ, 1 < d ∧ d < P(x) → P(x) % d ≠ 0)

-- Theorem: Polynomial P(x) can be factored into polynomials with integer coefficients
theorem factorization (h : condition) : 
  ∃ (A B : ℤ[X]), P = A * B :=
sorry

end factorization_l453_453948


namespace expand_fourier_series_l453_453357

-- Define the interval
def interval := set.Ioo 0 1

-- Define the function y = x + 1
def f (x : ℝ) := x + 1

-- Define the cosine system terms
def cosine_term (n : ℕ) (x : ℝ) := Real.cos (((2 * n + 1) * Real.pi * x) / 2)

-- State the theorem
theorem expand_fourier_series (x : ℝ) (h : x ∈ interval) :
  f x = (8 / Real.pi ^ 2) * ∑ n in Finset.range (n + 1), (Real.cos ((2 * n + 1 * Real.pi * x) / 2)) :=
sorry

end expand_fourier_series_l453_453357


namespace range_of_a_l453_453409

theorem range_of_a :
  (∀ x : ℝ, a - (3/2) > 0 ∧ a - (3/2) < 1 → (a - (3/2)) ^ x < (a - (3/2)) ^ (x + 1)) ∧
  (∀ x : ℝ, a > 0 ∧ 1 - 4 * a * (1/16) * a < 0 → ∀ x : ℝ, log 10 (a * x^2 - x + (1/16) * a) ∈ ℝ) →
  (¬ (a - (3/2) > 0 ∧ a - (3/2) < 1 ∧ a > 0 ∧ 1 - 4 * a * (1/16) * a < 0)) ∧
  ((0 < a - (3/2) < 1) ∨ a > 2)
  ↔ ((3/2 < a ∧ a ≤ 2) ∨ a ≥ 5/2) :=
sorry

end range_of_a_l453_453409


namespace limit_fn_eq_l453_453262

noncomputable theory
open Real

-- Define the sequence of functions using recurrence relation
def fn_seq : ℕ → (ℝ → ℝ)
| 0 := λ x, 1  -- f_1(x) = 1
| (n + 1) := 
  let fn := fn_seq n in
  λ x, Exp (∫ t in 0..x, fn t)  -- f_{n+1}(x) = exp (∫_0^x f_n(t) dt)

-- Define the limit function
def limit_fn (x : ℝ) : ℝ := 1 / (1 - x)

-- Prove that the limit function exists and is equal to 1 / (1 - x) for all x ∈ [0, 1)
theorem limit_fn_eq (x : ℝ) (hx : 0 ≤ x ∧ x < 1) :
  ∃ f : ℝ → ℝ, (∀ n, fn_seq n x = f x) ∧ f x = limit_fn x :=
sorry

end limit_fn_eq_l453_453262


namespace min_elements_l453_453248

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l453_453248


namespace strictly_decreasing_interval_l453_453745

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem strictly_decreasing_interval :
  ∀ x, (0 < x) ∧ (x < 2) → (deriv f x < 0) := by
sorry

end strictly_decreasing_interval_l453_453745


namespace fries_remaining_time_l453_453321

theorem fries_remaining_time (recommended_time_min : ℕ) (time_in_oven_sec : ℕ)
    (h1 : recommended_time_min = 5)
    (h2 : time_in_oven_sec = 45) :
    (recommended_time_min * 60 - time_in_oven_sec = 255) :=
by
  sorry

end fries_remaining_time_l453_453321


namespace find_angle_A_find_triangle_area_l453_453383

/-- Problem setup for finding angle A in a triangle given conditions --/
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : a * sin B + b * cos A = 0) (h_triangle : 0 < A ∧ A < π) : 
  A = 3 * π / 4 := sorry

/-- Problem setup for finding area of triangle given side lengths and angle --/
theorem find_triangle_area (a b c : ℝ) (A B C : ℝ) (h1 : a * sin B + b * cos A = 0) 
    (h2 : a = 2 * sqrt 5) (h3 : b = 2) (h4 : A = 3 * π / 4) :
  1/2 * b * (sqrt 2) * sin A = 2 := sorry

end find_angle_A_find_triangle_area_l453_453383


namespace hyperbola_condition_l453_453649

theorem hyperbola_condition (a b c : ℝ) : 
  (ax^2 + by^2 = c ∧ a*b < 0) → (∃ x y : ℝ, ax^2 + by^2 ≠ c) := sorry

end hyperbola_condition_l453_453649


namespace line_equation_l453_453034

noncomputable def line_intersects_at_point (a1 a2 b1 b2 c1 c2 : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 * a1 + p.2 * b1 = c1 ∧ p.1 * a2 + p.2 * b2 = c2

noncomputable def point_on_line (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

theorem line_equation
  (p : ℝ × ℝ)
  (h1 : line_intersects_at_point 3 2 2 3 5 5 p)
  (h2 : point_on_line 0 1 (-5) p)
  : ∃ a b c : ℝ,  a * p.1 + b * p.2 + (-5) = 0 :=
sorry

end line_equation_l453_453034


namespace num_three_digit_powers_of_three_l453_453827

theorem num_three_digit_powers_of_three : 
  ∃ n1 n2 : ℕ, 100 ≤ 3^n1 ∧ 3^n1 ≤ 999 ∧ 100 ≤ 3^n2 ∧ 3^n2 ≤ 999 ∧ n1 ≠ n2 ∧ 
  (∀ n : ℕ, 100 ≤ 3^n ∧ 3^n ≤ 999 → n = n1 ∨ n = n2) :=
sorry

end num_three_digit_powers_of_three_l453_453827


namespace edges_parallel_to_axes_l453_453083

theorem edges_parallel_to_axes (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ)
  (hx : x1 = 0 ∨ y1 = 0 ∨ z1 = 0)
  (hy : x2 = x1 + 1 ∨ y2 = y1 + 1 ∨ z2 = z1 + 1)
  (hz : x3 = x1 + 1 ∨ y3 = y1 + 1 ∨ z3 = z1 + 1)
  (hv : x4*y4*z4 = 2011) :
  (x2-x1 ∣ 2011) ∧ (y2-y1 ∣ 2011) ∧ (z2-z1 ∣ 2011) := 
sorry

end edges_parallel_to_axes_l453_453083


namespace minimum_black_cells_l453_453912

theorem minimum_black_cells (P : ℕ) : 
  (∀ (n : ℕ), (n ≤ (2007 * 2007) → n < P → ∃ (config : Matrix ℕ ℕ Bool), 
    (∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ i → ¬(config[i,j] ∧ config[j,k] ∧ config[k,i])))) ∧
  (∀ n : ℕ, n = 2007 * 2007 → n = 2007 * 2007 - P + 1) := 
by 
  sorry

end minimum_black_cells_l453_453912


namespace circle_centered_on_parabola_tangent_to_axis_and_directrix_l453_453967

open Real

noncomputable def circle_eq(center_x center_y radius : ℝ) : (ℝ × ℝ) → Prop :=
  λ p, (p.1 - center_x) ^ 2 + (p.2 - center_y) ^ 2 = radius ^ 2

theorem circle_centered_on_parabola_tangent_to_axis_and_directrix
  (center_x center_y : ℝ) (h1 : center_x = 1 / 2)
  (h2 : center_y ^ 2 = 2 * center_x) (h3 : center_y = 1 ∨ center_y = -1)
  (radius : ℝ) (h4 : radius = 1) :
  ∃ x y : ℝ, circle_eq center_x center_y radius (x, y) ∧
    (x - 1 / 2) ^ 2 + (y - 1) ^ 2 = 1 ∨ (x - 1 / 2) ^ 2 + (y + 1) ^ 2 = 1 :=
begin
  sorry
end

end circle_centered_on_parabola_tangent_to_axis_and_directrix_l453_453967


namespace kekai_money_left_l453_453106

theorem kekai_money_left
  (shirt_price : ℝ) (shirt_discount : ℝ) (num_shirts : ℕ)
  (pants_price : ℝ) (pants_discount : ℝ) (num_pants : ℕ)
  (hat_price : ℝ) (num_hats : ℕ)
  (shoe_price : ℝ) (shoe_discount : ℝ) (num_shoes : ℕ)
  (contribution_percentage : ℝ) :
  let
    total_money_shirts := num_shirts * (shirt_price * (1 - shirt_discount / 100)),
    total_money_pants := num_pants * (pants_price * (1 - pants_discount / 100)),
    total_money_hats := num_hats * hat_price,
    total_money_shoes := num_shoes * (shoe_price * (1 - shoe_discount / 100)),
    total_money := total_money_shirts + total_money_pants + total_money_hats + total_money_shoes,
    contribution := total_money * (contribution_percentage / 100),
    money_left := total_money - contribution
  in money_left = 26.32 :=
begin
  -- Sorry to skip proof for now
  sorry
end

#eval kekai_money_left 1 20 5 3 10 5 2 3 10 15 2 35 -- Will check the theorem assertion

end kekai_money_left_l453_453106


namespace find_line_equations_l453_453665

noncomputable def line_equations (A B : ℝ × ℝ) (C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  let circle_center : ℝ × ℝ := (2, 2)
  let circle_radius : ℝ := 1
  let circle_equation : ℝ × ℝ → ℝ := λ ⟨x, y⟩, (x - circle_center.1)^2 + (y - circle_center.2)^2 - circle_radius^2
  let reflected_circle_equation : ℝ × ℝ → ℝ := λ ⟨x, y⟩, (x - circle_center.1)^2 + (y + circle_center.2)^2 - circle_radius^2

  let point_of_reflection : ℝ × ℝ := (A.1, 0)
  let line_eqs : Set (ℝ × ℝ) := {(x, y) | 3 * x + 4 * y = 3 ∨ 4 * x + 3 * y = -3}
  line_eqs

theorem find_line_equations :
  let A := (-3, 3)
  let B := (2, 2)
  let C := (2, 2)
  line_equations A B C = {⟨3, 4⟩, ⟨4, -3⟩} := by
  sorry

end find_line_equations_l453_453665


namespace count_three_digit_numbers_power_of_three_l453_453835

theorem count_three_digit_numbers_power_of_three :
  { n : ℕ | 100 ≤ 3^n ∧ 3^n ≤ 999 }.toFinset.card = 2 := by
  sorry

end count_three_digit_numbers_power_of_three_l453_453835


namespace quadratic_non_real_roots_l453_453431

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l453_453431


namespace correct_operation_l453_453637

-- Define that m and n are elements of an arbitrary commutative ring
variables {R : Type*} [CommRing R] (m n : R)

theorem correct_operation : (m * n) ^ 2 = m ^ 2 * n ^ 2 := by
  sorry

end correct_operation_l453_453637


namespace solution_I_solution_II_l453_453044

def f (x : ℝ) := abs (2 * abs x - 1)

theorem solution_I : { x : ℝ | f x ≤ 1 } = set.Icc (-1) (1) :=
by sorry

theorem solution_II (m n : ℝ) (hm : m ∈ set.Icc (-1) 1) (hn : n ∈ set.Icc (-1) 1) : abs (m + n) ≤ m * n + 1 :=
by sorry

end solution_I_solution_II_l453_453044


namespace t_range_l453_453004

noncomputable def g (x : ℝ) : ℝ := log x + 3 / (4 * x) - 1 / 4 * x - 1
noncomputable def f (x t : ℝ) : ℝ := x^2 - 2 * t * x + 4

theorem t_range (t : ℝ) :
  (∀ x1 ∈ Set.Ioo 0 2, ∃ x2 ∈ Set.Icc 1 2, g x1 ≥ f x2 t) ↔
    t ∈ Set.Ici (17 / 8) :=
by
  sorry

end t_range_l453_453004


namespace unique_solution_of_log_equation_l453_453988

open Real

noncomputable def specific_log_equation (x : ℝ) : Prop := log (2 * x + 1) + log x = 1

theorem unique_solution_of_log_equation :
  ∀ x : ℝ, (x > 0) → (2 * x + 1 > 0) → specific_log_equation x → x = 2 := by
  sorry

end unique_solution_of_log_equation_l453_453988


namespace digging_project_length_l453_453655

theorem digging_project_length (L : ℝ) (V1 V2 : ℝ) (depth1 length1 depth2 breadth1 breadth2 : ℝ) 
  (h1 : depth1 = 100) (h2 : length1 = 25) (h3 : breadth1 = 30) (h4 : V1 = depth1 * length1 * breadth1)
  (h5 : depth2 = 75) (h6 : breadth2 = 50) (h7 : V2 = depth2 * L * breadth2) (h8 : V1 / V2 = 1) :
  L = 20 :=
by
  sorry

end digging_project_length_l453_453655


namespace remove_element_for_desired_average_l453_453631

noncomputable def list := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
noncomputable def original_sum := 91
noncomputable def desired_average := 7.25
noncomputable def remaining_sum := 12 * desired_average

theorem remove_element_for_desired_average :
  (∃ x ∈ list, original_sum - (remaining_sum + x) = 0) :=
begin
  use 4,
  split,
  { -- Show that 4 is in the list
    repeat { simp [list] <|> norm_num },
  },
  { -- Show that removing 4 results in the desired average
    have h1 : original_sum - remaining_sum = 4,
    { simp [original_sum, remaining_sum, desired_average], norm_num, },
    rw ←h1,
    norm_num,
  },
end

end remove_element_for_desired_average_l453_453631


namespace jackson_sandwiches_l453_453883

theorem jackson_sandwiches (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ)
    (h_weeks : weeks = 36) (h_missed_wednesdays : missed_wednesdays = 1) (h_missed_fridays : missed_fridays = 2) :
    let total_days := weeks * 2
    let missed_days := missed_wednesdays + missed_fridays
    total_days - missed_days = 69 :=
by
    sorry

end jackson_sandwiches_l453_453883


namespace luke_fish_fillets_l453_453728

theorem luke_fish_fillets : 
  (∃ (catch_rate : ℕ) (days : ℕ) (fillets_per_fish : ℕ), catch_rate = 2 ∧ days = 30 ∧ fillets_per_fish = 2 → 
  (catch_rate * days * fillets_per_fish = 120)) :=
by
  sorry

end luke_fish_fillets_l453_453728


namespace ants_on_icosahedron_probability_l453_453209

theorem ants_on_icosahedron_probability :
  let V := 12  -- Number of vertices (which is 12 for an icosahedron)
  let A := 12  -- Number of ants
  let choices := 5 -- Each ant has 5 choices
  
  -- The probability that no two ants land on the same vertex
  let favorable_outcomes := Nat.factorial A
  let total_probability := (choices : ℝ)^(A)
  
  let probability := favorable_outcomes / total_probability
  
  probability = (12! : ℝ) / (5^12 : ℝ) := 
by
  sorry

end ants_on_icosahedron_probability_l453_453209


namespace find_number_l453_453681

-- Define the number N
variable (N : ℚ) -- Using rational numbers for precision in fractional calculations.

-- Define the conditions
def correct_result := (4 / 5) * N
def student_result := (5 / 4) * N
def condition := student_result = correct_result + 27

-- The proof statement
theorem find_number (h : condition N) : N = 60 :=
by 
  sorry

end find_number_l453_453681


namespace next_unique_digits_date_l453_453470

-- Define the conditions
def is_after (d1 d2 : String) : Prop := sorry -- Placeholder, needs a date comparison function
def has_8_unique_digits (date : String) : Prop := sorry -- Placeholder, needs a function to check unique digits

-- Specify the problem and assertion
theorem next_unique_digits_date :
  ∀ date : String, is_after date "11.08.1999" → has_8_unique_digits date → date = "17.06.2345" :=
by
  sorry

end next_unique_digits_date_l453_453470


namespace car_average_speed_l453_453591

-- Define the given conditions
def total_time_hours : ℕ := 5
def total_distance_miles : ℕ := 200

-- Define the average speed calculation
def average_speed (distance time : ℕ) : ℕ :=
  distance / time

-- State the theorem to be proved
theorem car_average_speed :
  average_speed total_distance_miles total_time_hours = 40 :=
by
  sorry

end car_average_speed_l453_453591


namespace lines_are_coplanar_l453_453547

/- Define the parameterized lines -/
def L1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - k * s, 2 + 2 * k * s)
def L2 (t : ℝ) : ℝ × ℝ × ℝ := (2 + t, 7 + 3 * t, 1 - 2 * t)

/- Prove that k = 0 ensures the lines are coplanar -/
theorem lines_are_coplanar (k : ℝ) : k = 0 ↔ 
  ∃ (s t : ℝ), L1 s k = L2 t :=
by {
  sorry
}

end lines_are_coplanar_l453_453547


namespace race_length_l453_453858

noncomputable def solve_race_length (a b c d : ℝ) : Prop :=
  (d > 0) →
  (d / a = (d - 40) / b) →
  (d / b = (d - 30) / c) →
  (d / a = (d - 65) / c) →
  d = 240

theorem race_length : ∃ (d : ℝ), solve_race_length a b c d :=
by
  use 240
  sorry

end race_length_l453_453858


namespace zero_point_condition_l453_453959

theorem zero_point_condition (a : ℝ) :
  (a < -2) → 
  ∃ x : ℝ, x ∈ Icc (-1 : ℝ) 2 ∧ f a x = 0 ∧ (∀ y : ℝ, ∃ x : ℝ, x ∈ Icc (-1) 2 ∧ f a x = 0 → a ≤ -3 / 2 ∨ a ≥ 3) → ∃ x : ℝ, x ∈ Icc (-1 : ℝ) 2 ∧ f a x = 0 :=
by
  sorry

def f (a x : ℝ) : ℝ := a * x + 3

end zero_point_condition_l453_453959


namespace luke_fish_fillets_l453_453730

def fish_per_day : ℕ := 2
def days : ℕ := 30
def fillets_per_fish : ℕ := 2

theorem luke_fish_fillets : fish_per_day * days * fillets_per_fish = 120 := 
by
  sorry

end luke_fish_fillets_l453_453730


namespace find_b_l453_453599

theorem find_b (b p : ℝ) 
  (h1 : 3 * p + 15 = 0)
  (h2 : 15 * p + 3 = b) :
  b = -72 :=
by
  sorry

end find_b_l453_453599


namespace chessboard_covering_impossible_l453_453142

theorem chessboard_covering_impossible :
  ¬∃ (tiles : ℕ → (ℕ × ℕ × color) → Prop),
    (∀ (i : ℕ), (∃ k : Fin 4, (tiles i) = (λ r, (r = (a,b,black)) ∨ (r = (a,b,white)))) ∧
      covering a b ) ∧
    (coverable_with_tiles tiles) := sorry

end chessboard_covering_impossible_l453_453142


namespace largest_k_log_defined_l453_453195

def T : ℕ → ℕ
| 1       := 3
| (n + 1) := 3 ^ (T n)

def A : ℕ := (T 5) ^ (T 5)
def B : ℕ := (T 5) ^ A

theorem largest_k_log_defined : ∃ (k : ℕ), k = 6 ∧ (∀ i, i ≤ 6 →
  (∃ y : ℝ, y = nat.log 3 i)) :=
by sorry

end largest_k_log_defined_l453_453195


namespace sequence_arithmetic_l453_453775

open Nat

-- Definition of the sum of the first n terms S_n and the sequence {a_n}
noncomputable def Sn : ℕ+ → ℚ
| 1     => 1 / 2
| n+1   => sorry

noncomputable def an (n : ℕ+) : ℚ :=
if n = 1 then 1 / 2 else -2 * Sn n * Sn (n - 1)

-- Main theorem statement
theorem sequence_arithmetic (n : ℕ+) :
  ∀ (n : ℕ+), (n = 1 ∨ n ≥ 2) →
  (if n = 1 then 1 / Sn n = 2 else 1 / Sn n = 1 / Sn (n - 1) + 2)
  ∧ (Sn n = 1 / (2 * n))
  ∧ (an n = if n = 1 then 1 / 2 else -1 / (2 * n * (n - 1))) :=
sorry

end sequence_arithmetic_l453_453775


namespace megan_probability_l453_453917

noncomputable def probability_of_correct_dial : ℚ :=
  let first_digit_options := 3
  let last_digit_permutations := Nat.factorial 5
  let total_combinations := first_digit_options * last_digit_permutations
  let correct_number_combinations := 1
  correct_number_combinations / total_combinations

theorem megan_probability : probability_of_correct_dial = 1 / 360 := by
  let first_digit_options := 3
  let last_digit_permutations := Nat.factorial 5
  have factorial_five : Nat.factorial 5 = 120 := by sorry
  have total_combinations : first_digit_options * last_digit_permutations = 360 :=
    by sorry
  show probability_of_correct_dial = 1 / 360 from
    calc
      probability_of_correct_dial
        = correct_number_combinations / (first_digit_options * last_digit_permutations) := rfl
      ... = 1 / 360 := by
        rw [total_combinations]
        exact rfl
  sorry

end megan_probability_l453_453917


namespace log_expression_result_l453_453707

theorem log_expression_result :
  log 4 + log 9 + 2 * real.sqrt((log 6) ^ 2 - log 36 + 1) = 2 :=
by
  sorry

end log_expression_result_l453_453707


namespace stripe_area_correct_l453_453273

noncomputable def circumference (r : ℝ) := 2 * Real.pi * r

noncomputable def length_of_stripe (C : ℝ) (revolutions : ℕ) := revolutions * C

noncomputable def area_of_stripe (L : ℝ) (width : ℝ) := L * width

-- Given values
def diameter := 20
def radius := (diameter : ℝ) / 2
def height := 60
def stripe_width := 4
def revolutions := 3

def problem_statement : Prop :=
  let C := circumference radius in
  let L := length_of_stripe C revolutions in
  let area := area_of_stripe L stripe_width in
  area = (240 * Real.pi : ℝ)

theorem stripe_area_correct : problem_statement :=
by
  sorry

end stripe_area_correct_l453_453273


namespace harry_speed_on_friday_l453_453056

theorem harry_speed_on_friday :
  ∀ (speed_monday speed_tuesday_to_thursday speed_friday : ℝ)
  (ran_50_percent_faster ran_60_percent_faster: ℝ),
  speed_monday = 10 →
  ran_50_percent_faster = 0.50 →
  ran_60_percent_faster = 0.60 →
  speed_tuesday_to_thursday = speed_monday + (ran_50_percent_faster * speed_monday) →
  speed_friday = speed_tuesday_to_thursday + (ran_60_percent_faster * speed_tuesday_to_thursday) →
  speed_friday = 24 := by {
  intros speed_monday speed_tuesday_to_thursday speed_friday ran_50_percent_faster ran_60_percent_faster,
  intros h0 h1 h2 h3 h4,
  rw [h0, h1, h2, h3, h4],
  norm_num,
}

end harry_speed_on_friday_l453_453056


namespace exists_m_divisible_by_2005_l453_453527

def f (x : ℕ) : ℕ := 3 * x + 2

noncomputable def f_iter (n : ℕ) : ℕ → ℕ
| 0     => id
| (k+1) => λ x, f (f_iter k x)

theorem exists_m_divisible_by_2005 : ∃ (m : ℕ), f_iter 100 m % 2005 = 0 := 
by
  sorry

end exists_m_divisible_by_2005_l453_453527


namespace base4_number_divisible_by_19_l453_453167

theorem base4_number_divisible_by_19 (x : ℕ) (h : x ∈ {0, 1, 2, 3}) : 
  (8 * x + 146) % 19 = 0 ↔ x = 3 :=
by
  sorry

end base4_number_divisible_by_19_l453_453167


namespace dihedral_angle_sum_bounds_l453_453931

variable (α β γ : ℝ)

/-- The sum of the internal dihedral angles of a trihedral angle is greater than 180 degrees and less than 540 degrees. -/
theorem dihedral_angle_sum_bounds (hα: α < 180) (hβ: β < 180) (hγ: γ < 180) : 180 < α + β + γ ∧ α + β + γ < 540 :=
by
  sorry

end dihedral_angle_sum_bounds_l453_453931


namespace flower_shop_february_roses_l453_453588

/-- Given a pattern of incrementing the number of roses displayed at a flower shop each month,
     prove the number of roses displayed in February. The pattern starts with 100 roses in 
     October and increments by 2, 4, and 6 in the following months. -/
theorem flower_shop_february_roses :
  ∀ (oct nov dec jan feb : ℕ),
    oct = 100 →
    nov = oct + 2 →
    dec = nov + 4 →
    jan = dec + 6 →
    feb = jan + 8 →
    feb = 120 :=
by
  intros oct nov dec jan feb h_oct h_nov h_dec h_jan h_feb
  rw [h_oct, h_nov, h_dec, h_jan, h_feb]
  sorry

end flower_shop_february_roses_l453_453588


namespace coffee_shop_cups_l453_453916

variables (A B X Y : ℕ) (Z : ℕ)

theorem coffee_shop_cups (h1 : Z = (A * B * X) + (A * (7 - B) * Y)) : 
  Z = (A * B * X) + (A * (7 - B) * Y) := 
by
  sorry

end coffee_shop_cups_l453_453916


namespace area_of_isosceles_triangle_l453_453208

theorem area_of_isosceles_triangle
  (A B C P : Type)
  [IsNormalTri ABC]
  [Isosceles ABC AB AC]
  (BC_length : ℝ)
  (BC_length_eq : BC_length = 65)
  (dist_P_to_AB : ℝ)
  (dist_P_to_AB_eq : dist_P_to_AB = 24)
  (dist_P_to_AC : ℝ)
  (dist_P_to_AC_eq : dist_P_to_AC = 36)
  (area : ℝ)
  (h : area = 2535) :
  ∃ S : ℝ, S = area := sorry

end area_of_isosceles_triangle_l453_453208


namespace roots_of_unity_eq_roots_quadratic_l453_453349

theorem roots_of_unity_eq_roots_quadratic (c d : ℤ) (h1 : |c| ≤ 3) (h2 : d^2 ≤ 2) : 
  ∃ (z : ℂ) (n : ℕ), z^n = 1 ∧ (z^2 + (c : ℂ) * z + (d : ℂ) = 0) →
  (∃ (solutions : finset ℂ), solutions.card = 8 ∧ ∀ (ω : ℂ), ω ∈ solutions ↔ (∃ (n : ℕ), ω^n = 1)) :=
sorry

end roots_of_unity_eq_roots_quadratic_l453_453349


namespace quadratic_solution_1_quadratic_solution_2_l453_453569

theorem quadratic_solution_1 (x : ℝ) : x^2 - 8 * x + 12 = 0 ↔ x = 2 ∨ x = 6 := 
by
  sorry

theorem quadratic_solution_2 (x : ℝ) : (x - 3)^2 = 2 * x * (x - 3) ↔ x = 3 ∨ x = -3 := 
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l453_453569


namespace flat_fee_is_65_l453_453673

-- Define the problem constants
def George_nights : ℕ := 3
def Noah_nights : ℕ := 6
def George_cost : ℤ := 155
def Noah_cost : ℤ := 290

-- Prove that the flat fee for the first night is 65, given the costs and number of nights stayed.
theorem flat_fee_is_65 
  (f n : ℤ)
  (h1 : f + (George_nights - 1) * n = George_cost)
  (h2 : f + (Noah_nights - 1) * n = Noah_cost) :
  f = 65 := 
sorry

end flat_fee_is_65_l453_453673


namespace max_length_309_l453_453734

-- Definitions of the sequence
def sequence (x : ℕ) : ℕ → ℤ
| 0       := 500
| 1       := x
| (n + 2) := sequence n - sequence (n + 1)

-- Conditions for reaching 12 terms before encountering a negative term
def max_length_x (x : ℕ) : Prop :=
  17000 - 55 * x > 0 ∧ 34 * x - 10500 > 0 ∧
  ∀ n < 11, sequence x n > 0

-- Prove that the positive integer x that produces the sequence of maximum length is 309.
theorem max_length_309 : max_length_x 309 :=
sorry

end max_length_309_l453_453734


namespace circle_packing_line_division_l453_453857

theorem circle_packing_line_division :
  ∃ (a b c : ℕ), (a = 1) ∧ (b = 1) ∧ (c = 2) ∧ (a^2 + b^2 + c^2 = 6) ∧ Nat.gcd (Nat.gcd a b) c = 1 :=
begin
  use [1, 1, 2],
  split, refl,
  split, refl,
  split, refl,
  split,
  { norm_num },
  { exact Nat.gcd_one_right (Nat.gcd 1 1) }
end

end circle_packing_line_division_l453_453857


namespace num_non_congruent_triangles_l453_453080

-- Conditions
def points : List (ℝ × ℝ) := [(0,0), (1,0), (2,0), (0,0.5), (1,0.5), (2,0.5), (0,1), (1,1), (2,1), (1,0.5)]

-- Question and proof
theorem num_non_congruent_triangles : ∃ n, n = 4 ∧ (∀ (a b c : ℝ × ℝ), a ∈ points → b ∈ points → c ∈ points → 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c → (number_of_non_congruent_triangles_using (a, b, c) = n))) :=
sorry

end num_non_congruent_triangles_l453_453080


namespace rental_company_fixed_cost_l453_453307

theorem rental_company_fixed_cost :
  ∀ (F : ℝ),
    (F + 0.31 * 150 = 41.95 + 0.29 * 150) → F = 38.95 :=
by
  intro F
  intro h_eq
  linarith

end rental_company_fixed_cost_l453_453307


namespace regression_slope_interpretation_l453_453538

-- Define the variables and their meanings
variable {x y : ℝ}

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.8 * x + 4.6

-- Define the proof statement
theorem regression_slope_interpretation (hx : ∀ x, y = regression_line x) :
  ∀ delta_x : ℝ, delta_x = 1 → (regression_line (x + delta_x) - regression_line x) = 0.8 :=
by
  intros delta_x h_delta_x
  rw [h_delta_x, regression_line, regression_line]
  simp
  sorry

end regression_slope_interpretation_l453_453538


namespace sperner_lemma_l453_453990

def labeled_triangle (T : Type) := ∀ v : T, v = 0 ∨ v = 1 ∨ v = 2
def subdivision (Δ : Type) := ∀ t : Δ, ∃ v1 v2 v3 : labeled_triangle T, triangle v1 v2 v3

theorem sperner_lemma {T Δ : Type} (h : labeled_triangle T) (H : subdivision Δ) :
  ∃ t : Δ, ∃ v1 v2 v3 : T, v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3 ∧ labeled_triangle t :=
sorry

end sperner_lemma_l453_453990


namespace factor_expression_l453_453942

theorem factor_expression (x a b c : ℝ) :
  (x - a) ^ 2 * (b - c) + (x - b) ^ 2 * (c - a) + (x - c) ^ 2 * (a - b) = -(a - b) * (b - c) * (c - a) :=
by
  sorry

end factor_expression_l453_453942


namespace find_extrema_l453_453747

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem find_extrema :
  let I := set.Icc (1 : ℝ) (3 : ℝ) in
  (∀ x ∈ I, f x ≥ 1) ∧ (∀ x ∈ I, 1 ≤ f x) ∧
  (∀ x ∈ I, f x ≤ 5) ∧ (∀ x ∈ I, f x < 5) :=
by {
  sorry,
}

end find_extrema_l453_453747


namespace cycle_original_price_l453_453667

def originalPrice (S : ℝ) (lossPercentage : ℝ) : ℝ :=
  S / (1 - lossPercentage)

theorem cycle_original_price (S : ℝ) (lossPercentage : ℝ) (P : ℝ) :
  S = 680 → lossPercentage = 0.15 → originalPrice S lossPercentage = 800 :=
by
  intros hS hLoss
  rw [hS, hLoss]
  norm_num

#check cycle_original_price

end cycle_original_price_l453_453667


namespace area_of_ABCD_l453_453862

theorem area_of_ABCD 
  (AB CD DA: ℝ) (angle_CDA: ℝ) (a b c: ℕ) 
  (H1: AB = 10) 
  (H2: BC = 6) 
  (H3: CD = 13) 
  (H4: DA = 13) 
  (H5: angle_CDA = 45) 
  (H_area: a = 8 ∧ b = 30 ∧ c = 2) :

  ∃ (a b c : ℝ), a + b + c = 40 := 
by
  sorry

end area_of_ABCD_l453_453862


namespace minimum_value_is_three_l453_453846

noncomputable def f (x m : ℝ) := x + m / (x - 1)

theorem minimum_value_is_three (m : ℝ) (h₀ : m > 0)
  (h₁ : ∀ x : ℝ, 1 < x → f x m ≥ 3) : m = 1 :=
begin
  sorry
end

end minimum_value_is_three_l453_453846


namespace non_real_roots_b_range_l453_453435

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l453_453435


namespace transformed_avg_var_l453_453800

-- Definitions of original average and variance
variables {x : ℕ → ℝ} {n : ℕ}
def original_samples := λ i, x i + 2
def transformed_samples := λ i, 3 * (x i) + 2

axiom avg_original_samples_9 : (∑ i in finset.range n, original_samples i) / n = 9
axiom var_original_samples_3 : variance (finset.range n) original_samples = 3

noncomputable def avg_transformed_samples := (∑ i in finset.range n, transformed_samples i) / n
noncomputable def var_transformed_samples := variance (finset.range n) transformed_samples

theorem transformed_avg_var :
  avg_transformed_samples = 23 ∧ var_transformed_samples = 27 :=
by 
  -- leaving the proof as a placeholder
  sorry

end transformed_avg_var_l453_453800


namespace jackson_sandwiches_l453_453884

theorem jackson_sandwiches (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ)
    (h_weeks : weeks = 36) (h_missed_wednesdays : missed_wednesdays = 1) (h_missed_fridays : missed_fridays = 2) :
    let total_days := weeks * 2
    let missed_days := missed_wednesdays + missed_fridays
    total_days - missed_days = 69 :=
by
    sorry

end jackson_sandwiches_l453_453884


namespace num_common_tangents_of_circles_l453_453596

-- Definitions of the circles
def circle1 : Set (ℝ × ℝ) := {p | let x := p.1 in let y := p.2 in x^2 + y^2 - 2 * x = 0}
def circle2 : Set (ℝ × ℝ) := {p | let x := p.1 in let y := p.2 in x^2 + y^2 - 4 * x = 0}

-- Lean statement to prove the number of common tangents
theorem num_common_tangents_of_circles : 
  let O1_center := (1 : ℝ, 0 : ℝ)
  let O1_radius := 1
  let O2_center := (2 : ℝ, 0 : ℝ)
  let O2_radius := 2
  O1_center.dist O2_center = O2_radius - O1_radius →
  ∃ (tangents : Finset (Set (ℝ × ℝ))),
    tangents.card = 1 ∧
    ∀ t : Set (ℝ × ℝ), t ∈ tangents → 
      (∀ p ∈ t, p ∈ circle1) ∧ (∀ p ∈ t, p ∈ circle2) :=
by
  sorry

end num_common_tangents_of_circles_l453_453596


namespace non_real_roots_of_quadratic_l453_453440

theorem non_real_roots_of_quadratic (b : ℝ) : 
  (¬ ∃ x1 x2 : ℝ, x1^2 + bx1 + 16 = 0 ∧ x2^2 + bx2 + 16 = 0 ∧ x1 = x2) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
by {
  sorry
}

end non_real_roots_of_quadratic_l453_453440


namespace sin_bounds_l453_453265

theorem sin_bounds {x : ℝ} (h : 0 < x ∧ x < 1) : x - x^2 < sin x ∧ sin x < x :=
sorry

end sin_bounds_l453_453265


namespace relationship_l453_453904

-- Definitions according to conditions
def a : ℝ := Real.log 3 / Real.log 5
def b : ℝ := Real.exp (-1)
def c : ℝ := (Real.log 9 / Real.log 16) * (Real.log 8 / Real.log 27)

-- Statement of the proof problem
theorem relationship : b < c ∧ c < a :=
by
  -- Proof would go here
  sorry

end relationship_l453_453904


namespace isosceles_triangle_smallest_angle_l453_453702

def is_isosceles (angle_A angle_B angle_C : ℝ) : Prop := 
(angle_A = angle_B) ∨ (angle_B = angle_C) ∨ (angle_C = angle_A)

theorem isosceles_triangle_smallest_angle
  (angle_A angle_B angle_C : ℝ)
  (h_isosceles : is_isosceles angle_A angle_B angle_C)
  (h_angle_162 : angle_A = 162) :
  angle_B = 9 ∧ angle_C = 9 ∨ angle_A = 9 ∧ (angle_B = 9 ∨ angle_C = 9) :=
by
  sorry

end isosceles_triangle_smallest_angle_l453_453702


namespace unique_plants_count_l453_453618

open Set

-- Definitions based on conditions
variables (X Y Z : Set ℕ)
def |X| := 600
def |Y| := 500
def |Z| := 400
def |X ∩ Y| := 100
def |X ∩ Z| := 150
def |Y ∩ Z| := 0
def |X ∩ Y ∩ Z| := 0

-- Theorem statement
theorem unique_plants_count : 
  |X ∪ Y ∪ Z| = |X| + |Y| + |Z| - |X ∩ Y| - |X ∩ Z| - |Y ∩ Z| + |X ∩ Y ∩ Z| :=
by
  -- We substitute the provided counts into the theorem statement
  have hX : |X| = 600 := by rfl
  have hY : |Y| = 500 := by rfl
  have hZ : |Z| = 400 := by rfl
  have hXY : |X ∩ Y| = 100 := by rfl
  have hXZ : |X ∩ Z| = 150 := by rfl
  have hYZ : |Y ∩ Z| = 0 := by rfl
  have hXYZ : |X ∩ Y ∩ Z| = 0 := by rfl

  calc
    |X ∪ Y ∪ Z| = 600 + 500 + 400 - 100 - 150 - 0 + 0 : by rw [hX, hY, hZ, hXY, hXZ, hYZ, hXYZ]
              ... = 1250 : by norm_num

end unique_plants_count_l453_453618


namespace Ed_cats_l453_453733

variable (C F : ℕ)

theorem Ed_cats 
  (h1 : F = 2 * (C + 2))
  (h2 : 2 + C + F = 15) : 
  C = 3 := by 
  sorry

end Ed_cats_l453_453733


namespace cube_max_skew_lines_l453_453151

noncomputable def max_skew_lines_in_cube : ℕ := 4

theorem cube_max_skew_lines
    (edges_and_diags : set (ℝ × ℝ))
    (h1 : ∀ l₁ l₂ ∈ edges_and_diags, l₁ ≠ l₂ → are_skew_lines l₁ l₂)
    (h2 : edges_and_diags ⊆ possible_lines_in_cube)
    : ∃ k, k = max_skew_lines_in_cube ∧ k = |edges_and_diags| := 
sorry

end cube_max_skew_lines_l453_453151


namespace find_lambda_l453_453026

-- Definitions for unit vectors and orthogonal conditions
variables (e1 e2 : ℝ → ℝ → ℝ)
def is_unit_vector (v : ℝ → ℝ → ℝ) := v • v = 1
def is_orthogonal (v1 v2 : ℝ → ℝ → ℝ) := v1 • v2 = 0

-- Special vectors involved in the problem
def vec1 : ℝ → ℝ → ℝ := λ θ, sqrt(3) * e1 θ - e2 θ
def vec2 (λ : ℝ) : ℝ → ℝ → ℝ := λ θ, e1 θ + λ * e2 θ

-- Angle condition
def angle_condition (v1 v2 : ℝ → ℝ → ℝ) (θ : ℝ) :=
  v1 θ • v2 θ = (∥v1 θ∥ * ∥v2 θ∥ * real.cos (real.pi / 3))

-- Main theorem
theorem find_lambda (λ : ℝ) (h1 : is_unit_vector e1) (h2 : is_unit_vector e2) (h3 : is_orthogonal e1 e2) :
  angle_condition vec1 (vec2 λ) 60 → λ = sqrt(3) / 3 :=
by
  sorry

end find_lambda_l453_453026


namespace correct_statements_l453_453466

-- Conditions
variable {C : Type} [curve : curve_class C]
variable {F : ℝ → ℝ → Prop} -- The equation F(x, y) = 0

-- Definitions based on conditions
def satisfies_curve (x y : ℝ) : Prop := F x y = 0
def on_curve (x y : ℝ) : Prop := satisfies_curve x y

-- The mathematically equivalent proof problem
theorem correct_statements (x y : ℝ) :
  (on_curve x y ↔ satisfies_curve x y) 
  ∧ (¬ on_curve x y ↔ ¬ satisfies_curve x y) :=
by
  sorry

end correct_statements_l453_453466


namespace game_ends_after_22_rounds_l453_453288

-- Defining the initial tokens for players A, B, and C.
def initial_tokens : ℕ × ℕ × ℕ := (16, 15, 14)

-- Defining the rule of the game as a function.
-- This function considers a round in the game and returns the updated state of tokens.
noncomputable def game_round (tokens : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (a, b, c) := tokens in
  if a >= b ∧ a >= c then
    (a - 6, b + 2, c + 2)
  else if b >= a ∧ b >= c then
    (a + 2, b - 6, c + 2)
  else
    (a + 2, b + 2, c - 6)

-- Defining a function to iterate the game rounds a specified number of times.
noncomputable def iterate_game (rounds : ℕ) (tokens : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  Nat.iterate game_round rounds tokens

-- The theorem stating the game will end after 22 rounds.
theorem game_ends_after_22_rounds :
  ∃ rounds : ℕ, rounds = 22 ∧ (iterate_game rounds initial_tokens).fst = 0 ∨
                                        (iterate_game rounds initial_tokens).snd = 0 ∨
                                        (iterate_game rounds initial_tokens).snd.snd = 0 :=
by
  sorry

end game_ends_after_22_rounds_l453_453288


namespace total_highlighters_l453_453850

-- Definitions and conditions:
def yellow_highlighters : ℕ := 7
def pink_highlighters : ℕ := yellow_highlighters + 7
def blue_highlighters : ℕ := pink_highlighters + 5
def green_highlighters (o : ℕ) : ℕ := (21 - o)
def orange_highlighters : ℕ := 3 * green_highlighters / 7 -- Correct ratio

-- Main proof statement:
theorem total_highlighters : ∃ (o g : ℕ), o = (3 / 7 : ℝ) * g ∧ o + g = 21 ∧
  let y := yellow_highlighters in
  let p := pink_highlighters in
  let b := blue_highlighters in
  let o := 9 in
  let g := 12 in
  y + p + b + o + g = 61 :=
by {
  use (9),
  use (12),
  split,
  { norm_num },
  split,
  { norm_num },
  repeat { norm_num },
  sorry
}

end total_highlighters_l453_453850


namespace sum_of_reciprocals_l453_453193

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 55) :
  (1 / x + 1 / y) = 16 / 55 :=
by
  sorry

end sum_of_reciprocals_l453_453193


namespace fries_remaining_time_l453_453320

theorem fries_remaining_time (recommended_time_min : ℕ) (time_in_oven_sec : ℕ)
    (h1 : recommended_time_min = 5)
    (h2 : time_in_oven_sec = 45) :
    (recommended_time_min * 60 - time_in_oven_sec = 255) :=
by
  sorry

end fries_remaining_time_l453_453320


namespace leaves_fall_l453_453690

theorem leaves_fall (planned_trees : ℕ) (tree_multiplier : ℕ) (leaves_per_tree : ℕ) (h1 : planned_trees = 7) (h2 : tree_multiplier = 2) (h3 : leaves_per_tree = 100) :
  (planned_trees * tree_multiplier) * leaves_per_tree = 1400 :=
by
  rw [h1, h2, h3]
  -- Additional step suggestions for interactive proof environments, e.g.,
  -- Have: 7 * 2 = 14
  -- Goal: 14 * 100 = 1400
  sorry

end leaves_fall_l453_453690


namespace reginald_apples_sold_l453_453148

theorem reginald_apples_sold 
  (apple_price : ℝ) 
  (bike_cost : ℝ)
  (repair_percentage : ℝ)
  (remaining_fraction : ℝ)
  (discount_apples : ℕ)
  (free_apples : ℕ)
  (total_apples_sold : ℕ) : 
  apple_price = 1.25 → 
  bike_cost = 80 → 
  repair_percentage = 0.25 → 
  remaining_fraction = 0.2 → 
  discount_apples = 5 → 
  free_apples = 1 → 
  (∃ (E : ℝ), (125 = E ∧ total_apples_sold = 120)) → 
  total_apples_sold = 120 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end reginald_apples_sold_l453_453148


namespace painting_sections_l453_453353

theorem painting_sections (colors : Fin 4 → Nat) (sections : Fin 4) :
  ∃! perm : Equiv.Perm (Fin 4), ∀ i, colors (perm i) ≠ colors i :=
begin
  -- Assuming colors to be a function from sections to the color set [1,2,3,4].
  -- Each section gets a different primary color.
  -- The total number of permutations without repetition is factorial 4.
  use Equiv.Perm.fin_rotate 4,
  -- It is required to show that there exists a unique permutation that meets the criteria.
  sorry
end

end painting_sections_l453_453353


namespace exactly_two_inequalities_hold_l453_453814

-- Definitions
noncomputable def p1 (a b : ℝ) := (x : ℝ) → x ^ 2 + a * x + b
noncomputable def p2 (c d : ℝ) := (x : ℝ) → x ^ 2 + c * x + d
noncomputable def p3 (e f : ℝ) := (x : ℝ) → x ^ 2 + e * x + f

-- Assumptions about the roots
axiom root_common_p1_p2 : ∃ (x1 x2 x3 : ℝ), p1 (-(x1 + x2)) (x1 * x2) x1 = 0 ∧ p2 (-(x1 + x3)) (x1 * x3) x1 = 0
axiom root_common_p2_p3 : ∃ (x1 x2 x3 : ℝ), p2 (-(x1 + x3)) (x1 * x3) x2 = 0 ∧ p3 (-(x2 + x3)) (x2 * x3) x2 = 0
axiom root_common_p1_p3 : ∃ (x1 x2 x3 : ℝ), p1 (-(x1 + x2)) (x1 * x2) x3 = 0 ∧ p3 (-(x2 + x3)) (x2 * x3) x3 = 0
axiom not_common_root_all : ¬(∃ x : ℝ, p1 a b x = 0 ∧ p2 c d x = 0 ∧ p3 e f x = 0)

-- Proof statement
theorem exactly_two_inequalities_hold (a b c d e f : ℝ) :
  ∃ (x1 x2 x3 : ℝ),
    (p1 (-(x1 + x2)) (x1 * x2) x1 = 0 ∧ p2 (-(x1 + x3)) (x1 * x3) x2 = 0 ∧
     p3 (-(x2 + x3)) (x2 * x3) x3 = 0) →
    (not_common_root_all) →
    (2 = (if (a^2 + c^2 - e^2) / 4 > b + d - f then 1 else 0) +
         (if (c^2 + e^2 - a^2) / 4 > d + f - b then 1 else 0) +
         (if (e^2 + a^2 - c^2) / 4 > f + b - d then 1 else 0)) := sorry

end exactly_two_inequalities_hold_l453_453814


namespace license_plate_palindrome_probability_l453_453291

-- Define the two-letter palindrome probability
def prob_two_letter_palindrome : ℚ := 1 / 26

-- Define the four-digit palindrome probability
def prob_four_digit_palindrome : ℚ := 1 / 100

-- Define the joint probability of both two-letter and four-digit palindrome
def prob_joint_palindrome : ℚ := prob_two_letter_palindrome * prob_four_digit_palindrome

-- Define the probability of at least one palindrome using Inclusion-Exclusion
def prob_at_least_one_palindrome : ℚ := prob_two_letter_palindrome + prob_four_digit_palindrome - prob_joint_palindrome

-- Convert the probability to the form of sum of two integers
def sum_of_integers : ℕ := 5 + 104

-- The final proof problem
theorem license_plate_palindrome_probability :
  (prob_at_least_one_palindrome = 5 / 104) ∧ (sum_of_integers = 109) := by
  sorry

end license_plate_palindrome_probability_l453_453291


namespace parallel_line_through_point_l453_453624

theorem parallel_line_through_point :
  ∀ (x y : ℝ), (6 * x - 3 * y = 9) ∧ (x = 1) ∧ (y = -2) →
  (∃ b : ℝ, y = 2 * x + b) :=
begin
  sorry
end

end parallel_line_through_point_l453_453624


namespace algebraic_expression_value_l453_453412

-- Given system of equations and condition
def system_eq (x y m : ℝ) :=
  3 * x + 5 * y = m + 2 ∧ 2 * x + 3 * y = m ∧ x + y = -10

-- Statement to prove the value of the algebraic expression
theorem algebraic_expression_value :
  ∀ (x y m : ℝ), system_eq x y m → m^2 - 2 * m + 1 = 81 :=
by
  intros x y m h
  cases h with h1 h2
  cases h2 with h2 h3
  -- Formal proof goes here
  sorry

end algebraic_expression_value_l453_453412


namespace count_powers_of_three_not_powers_of_nine_l453_453423

theorem count_powers_of_three_not_powers_of_nine :
  {n : nat | n < 500000 ∧ ∃ k : nat, n = 3 ^ k ∧ ∀ m : nat, n ≠ 9 ^ m}.to_finset.card = 6 := 
sorry

end count_powers_of_three_not_powers_of_nine_l453_453423


namespace total_profit_l453_453271

theorem total_profit
  (initial_investment_A initial_investment_B : ℝ)
  (withdraw_A advance_B : ℝ)
  (months_1 months_2 : ℕ)
  (A_share : ℝ) : 
  let total_investment_A := (initial_investment_A * months_1) + (withdraw_A * months_2)
      total_investment_B := (initial_investment_B * months_1) + (advance_B * months_2)
      ratio_A := total_investment_A / (total_investment_A + total_investment_B)
  in (ratio_A * (total_investment_A + total_investment_B)) = A_share → (total_investment_A + total_investment_B) = 630 :=
by
  intros initial_investment_A initial_investment_B withdraw_A advance_B months_1 months_2 A_share
  let total_investment_A := (initial_investment_A * ↑months_1) + (withdraw_A * ↑months_2)
  let total_investment_B := (initial_investment_B * ↑months_1) + (advance_B * ↑months_2)
  let ratio_A := total_investment_A / (total_investment_A + total_investment_B)
  intro h
  let total_profit := total_investment_A + total_investment_B
  change (ratio_A * total_profit) = A_share at h
  have h1 : ratio_A * total_profit = A_share, from h
  sorry

end total_profit_l453_453271


namespace work_completion_time_l453_453277

theorem work_completion_time (A_rate B_rate : ℝ) (hA : A_rate = 1/60) (hB : B_rate = 1/20) :
  1 / (A_rate + B_rate) = 15 :=
by
  sorry

end work_completion_time_l453_453277


namespace area_of_tangency_triangle_l453_453292

theorem area_of_tangency_triangle (c a b T varrho : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_area : T = (1/2) * a * b) (h_inradius : varrho = (a + b - c) / 2) :
  (area_tangency : ℝ) = (varrho / c) * T :=
sorry

end area_of_tangency_triangle_l453_453292


namespace rhombus_diagonal_length_l453_453961

theorem rhombus_diagonal_length (d2 area : ℝ) (h1 : d2 = 12) (h2 : area = 180) : 
  ∃ d1, (area = (d1 * d2) / 2) ∧ d1 = 30 :=
by
  use 30
  split
  · simp [h1, h2]
  · rfl

# The above assumption ensures we include all mathematics required.
# The theorem proves the length of the unknown diagonal is 30 cm, given the conditions on area and another diagonal length of the rhombus.

end rhombus_diagonal_length_l453_453961


namespace sages_can_win_l453_453546

-- Define the basic parameters of the game
def sage_count : Nat := 100
def choices : Finset ℕ := {1, 2, 3}

-- Type representing the given numbers to a sage
structure SageChoice :=
(choices : Finset ℕ)
(h_distinct : ∀ a b ∈ choices, a ≠ b)

-- Type representing the chosen number by a sage
structure SageDecision :=
(choice : ℕ)
(h_in_choices : choice ∈ choices)

-- Define preliminary sage decisions convention
def N (i : ℤ) : ℕ :=
if i < 1 then 2 else sorry  -- This will be defined properly in the full proof

-- Define B function to get pairs from previous decisions
def B (i : ℕ) : (ℕ × ℕ) :=
(N (i - 2), N (i - 1))

-- Predicate to check if the game is a failure
def game_failure (decisions : Fin (sage_count) → SageDecision) : Prop :=
(finset.univ.sum (λ i, decisions i).choice) = 200

-- The main theorem to be proved
theorem sages_can_win (choices : Fin n → SageChoice) (n : ℕ) :
  ∃ decisions : Fin n → SageDecision, ¬ game_failure decisions :=
by
  sorry

end sages_can_win_l453_453546


namespace minimum_fertilizer_cost_l453_453713

-- Define the dimensions (areas) of the sections
def area_upper_left : ℕ := 3 * 1
def area_lower_right : ℕ := 4 * 2
def area_upper_right : ℕ := 6 * 2
def area_middle_center : ℕ := 2 * 3
def area_bottom_left : ℕ := 5 * 4

-- Define the cost per square meter for each vegetable
def cost_lettuce : ℕ := 2
def cost_spinach : ℕ := 2.5
def cost_carrots : ℕ := 3
def cost_beans : ℕ := 3.5
def cost_tomatoes : ℕ := 4

-- Total minimum cost is the crucial statement we want to prove
theorem minimum_fertilizer_cost : ℕ :=
  (area_upper_left * cost_tomatoes) +
  (area_middle_center * cost_beans) +
  (area_lower_right * cost_carrots) +
  (area_upper_right * cost_spinach) +
  (area_bottom_left * cost_lettuce) = 127 :=
  begin
    sorry
  end

end minimum_fertilizer_cost_l453_453713


namespace min_distance_A_E_l453_453761

noncomputable def AB := 12
noncomputable def BC := 5
noncomputable def CD := 3
noncomputable def DE := 2

theorem min_distance_A_E : ∃ AE, AE ≥ 0 ∧ AE = 2 := 
by {
  use 2,
  split,
  { linarith, },
  { rfl, },
  sorry
}

end min_distance_A_E_l453_453761


namespace arithmetic_sequence_tenth_term_l453_453609

theorem arithmetic_sequence_tenth_term (a d : ℤ) 
  (h1 : a + 2 * d = 23) 
  (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
by 
  -- proof goes here
  sorry

end arithmetic_sequence_tenth_term_l453_453609


namespace find_b_l453_453718

def f (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ (b : ℝ), f b = 3 :=
by
  use 2
  show f 2 = 3
  sorry

end find_b_l453_453718


namespace find_values_y_k_l453_453372

variables {k y : ℝ}
def v := (λ k y : ℝ, k • (⟨2, y⟩ : ℝ × ℝ))
def w := (⟨4, 6⟩ : ℝ × ℝ)
def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  ((v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)) • w

theorem find_values_y_k (hyp : proj (v k y) w = (⟨-6, -9⟩ : ℝ × ℝ)) :
  y = -23 ∧ k = (39 : ℝ) / 65 := by
  sorry

end find_values_y_k_l453_453372


namespace at_least_one_photograph_with_no_more_than_998_cameras_l453_453101

theorem at_least_one_photograph_with_no_more_than_998_cameras (n : ℕ) (h : n = 1000) 
(objects_captured : Π (i : fin n), set (fin n)) (h_angle : ∀ i j k : fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → measure_theory.angle (objects_captured i j) (objects_captured i k) ≤ 179) :
  ∃ i : fin n, (set.filter (λ j, j ≠ i) (objects_captured i)).card ≤ 998 :=
sorry

end at_least_one_photograph_with_no_more_than_998_cameras_l453_453101


namespace stolen_bones_is_two_l453_453497

/-- Juniper's initial number of bones -/
def initial_bones : ℕ := 4

/-- Juniper's bones after receiving more bones -/
def doubled_bones : ℕ := initial_bones * 2

/-- Juniper's remaining number of bones after theft -/
def remaining_bones : ℕ := 6

/-- Number of bones stolen by the neighbor's dog -/
def stolen_bones : ℕ := doubled_bones - remaining_bones

theorem stolen_bones_is_two : stolen_bones = 2 := sorry

end stolen_bones_is_two_l453_453497


namespace max_c1_minus_c2_l453_453181

theorem max_c1_minus_c2 (c1 c2 c3 : ℕ) (S : set ℕ)
  (h : S = {0, 1, 2, 4, 5, 6}) :
  (∀ x ∈ S, (x^2 - 6*x + c1) * (x^2 - 6*x + c2) * (x^2 - 6*x + c3) = 0) →
  (x^2 - 6*x + c1 = 0 ↔ x = 1 ∨ x = 5) →
  (x^2 - 6*x + c2 = 0 ↔ x = 2 ∨ x = 4) →
  (x^2 - 6*x + c3 = 0 ↔ x = 0 ∨ x = 6) →
  c1 - c2 = 8 :=
by
sorrry

end max_c1_minus_c2_l453_453181


namespace product_divisible_by_4_l453_453975

theorem product_divisible_by_4 (a b c d : ℤ) 
    (h : a^2 + b^2 + c^2 = d^2) : 4 ∣ (a * b * c) :=
sorry

end product_divisible_by_4_l453_453975


namespace rotate_A_180_l453_453927

section
variable (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)

-- Define the coordinates of the points
def pointA := (-4, 1 : ℝ)
def pointB := (-1, 4 : ℝ)
def pointC := (-1, 1 : ℝ)

-- Function to rotate a point by 180 degrees clockwise about the origin
def rotate180 (p : ℝ × ℝ) := (-p.1, -p.2)

-- The coordinates of the image of point A after a 180-degree rotation
def rotatedA := rotate180 pointA

-- Statement to be proved
theorem rotate_A_180 :
  rotatedA = (4, -1 : ℝ) :=
sorry

end

end rotate_A_180_l453_453927


namespace carla_initial_marbles_l453_453329

noncomputable def marble_initial := 
  let a := 1
  let b := 1
  (2778.65 - 489.35 : ℝ)

theorem carla_initial_marbles : 
  (\result : ℝ) -> 
  result = (marble_initial) ->
  result = 2289.3 := 
by 
  intro result 
  intro h 
  rw ← h
  simp
  sorry

end carla_initial_marbles_l453_453329


namespace triangle_angles_eq_60_80_40_l453_453521

theorem triangle_angles_eq_60_80_40
  (A B C P Q : Type)
  [triangle A B C]
  (angle_BAC : ∠BAC = 60)
  (AP_bisector : bisects AP ∠BAC)
  (BQ_bisector : bisects BQ ∠ABC)
  (P_on_BC : P ∈ line BC)
  (Q_on_AC : Q ∈ line AC)
  (congruence_condition : AB + BP = AQ + QB) :
  angles_of_triangle ABC = (60, 80, 40) :=
sorry

end triangle_angles_eq_60_80_40_l453_453521


namespace arrangement_equality_l453_453994

-- Given conditions
def students : ℕ := 48
def P : ℕ := factorial students
def Q : ℕ := factorial students

-- Our theorem to prove
theorem arrangement_equality : P = Q := by
  sorry

end arrangement_equality_l453_453994


namespace relationship_xyz_w_l453_453837

theorem relationship_xyz_w (x y z w : ℝ) (h : (x + y) / (y + z) = (2 * z + w) / (w + x)) :
  x = 2 * z - w := 
sorry

end relationship_xyz_w_l453_453837


namespace smallest_positive_period_axis_of_symmetry_range_of_f_in_interval_l453_453419

noncomputable def f (x : ℝ) : ℝ :=
  let m := (Real.cos x, -Real.sin x)
  let n := (Real.cos x, Real.sin x - 2 * Real.sqrt 3 * Real.cos x)
  m.1 * n.1 + m.2 * n.2

theorem smallest_positive_period (x : ℝ) : 
  ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x := 
by
  use Real.pi
  sorry

theorem axis_of_symmetry (x : ℝ) : 
  ∀ k : ℤ, x = (Real.pi / 6) + (k * (Real.pi / 2)) :=
by
  sorry

theorem range_of_f_in_interval : 
  Set.Icc (-Real.pi / 12) (Real.pi / 2) ⊆ 
  Set.Icc (-1) 2 :=
by
  sorry

end smallest_positive_period_axis_of_symmetry_range_of_f_in_interval_l453_453419


namespace find_l_find_C3_l453_453002

-- Circle definitions
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 7 = 0

-- Given line passes through common points of C1 and C2
theorem find_l (x y : ℝ) (h1 : C1 x y) (h2 : C2 x y) : x = 1 := by
  sorry

-- Circle C3 passes through intersection points of C1 and C2, and its center lies on y = x
def C3 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def on_line_y_eq_x (x y : ℝ) : Prop := y = x

theorem find_C3 (x y : ℝ) (hx : C3 x y) (hy : on_line_y_eq_x x y) : (x - 1)^2 + (y - 1)^2 = 1 := by
  sorry

end find_l_find_C3_l453_453002


namespace trigonometric_identity_l453_453027

theorem trigonometric_identity
  (θ : ℝ)
  (h1 : sin θ < 0)
  (h2 : tan θ > 0)
  : sqrt (1 - sin θ ^ 2) = -cos θ :=
by
  sorry

end trigonometric_identity_l453_453027


namespace incorrect_conclusion_among_options_l453_453934

-- Define the sales data
def sales_data : List ℕ := [11, 10, 11, 13, 11, 13, 15]

-- Define properties to check:
def mode_is_11 : Prop := (List.mode sales_data = some 11)
def mean_is_12 : Prop := (List.mean sales_data = some 12)
def variance_is_18_over_7 : Prop := (List.variance sales_data = some (18 / 7))
def median_is_13 : Prop := (List.median sales_data = some 13)

theorem incorrect_conclusion_among_options : median_is_13 = false :=
by
  -- Note: proof intentionally skipped as instructed
  sorry

end incorrect_conclusion_among_options_l453_453934


namespace angle_bpc_iff_ae_ap_pd_l453_453096

theorem angle_bpc_iff_ae_ap_pd 
  {A B C D E F P : Point}
  (h_incircle : Incircle (ABC) O D E F)
  (h_tangent_points : TangentPoints (O) D E F)
  (h_ad_intersects_p : LineIntersectsAtTwo (AD) (O) P)
  (h_connect_bp_cp : ConnectLines (BP, CP))
  (h_condition : AE + AP = PD) :
  ∠(B P C) = 90 ↔ AE + AP = PD :=
sorry

end angle_bpc_iff_ae_ap_pd_l453_453096


namespace reena_interest_paid_l453_453936

-- Define the conditions
def Principal : ℝ := 1200
def Rate : ℝ := 4
def Time : ℝ := 4

-- Define the simple interest formula
def SimpleInterest (P R T : ℝ) : ℝ := P * R * T / 100

-- Prove that the calculated simple interest is $192
theorem reena_interest_paid : SimpleInterest Principal Rate Time = 192 := by
    sorry

end reena_interest_paid_l453_453936


namespace paul_money_duration_l453_453925

theorem paul_money_duration (mowing_income weed_eating_income weekly_spending money_last: ℕ) 
    (h1: mowing_income = 44) 
    (h2: weed_eating_income = 28) 
    (h3: weekly_spending = 9) 
    (h4: money_last = 8) 
    : (mowing_income + weed_eating_income) / weekly_spending = money_last := 
by
  sorry

end paul_money_duration_l453_453925


namespace monotonicity_and_extrema_l453_453122

open Real

noncomputable def f (x : ℝ) : ℝ := log (2 * x + 3) + x ^ 2

theorem monotonicity_and_extrema :
  (∀ x, -3 / 2 < x ∧ x < -1 → 0 < deriv f x) ∧
  (∀ x, -1 < x ∧ x < -1 / 2 → deriv f x < 0) ∧
  (∀ x, -1 / 2 < x → 0 < deriv f x) ∧
  ∃ x₁ x₂ x₃ x₄,
    x₁ = -3 / 4 ∧ x₂ = -1 / 2 ∧ x₃ = 1 / 4 ∧
    f x₁ = log (3 / 2) + 9 / 16 ∧
    f x₂ = log 2 + 1 / 4 ∧
    f x₃ = log (7 / 2) + 1 / 16 ∧
    (f x₃ = max (f x₁) (max (f x₂) (f x))) ∧
    (f x₂ = min (f x₁) (min (f x₂) (f x)))
        :=
begin
  sorry
end

end monotonicity_and_extrema_l453_453122


namespace quadratic_non_real_roots_l453_453447

theorem quadratic_non_real_roots (b : ℝ) : 
  let a : ℝ := 1 
  let c : ℝ := 16 in
  (b^2 - 4 * a * c < 0) ↔ (-8 < b ∧ b < 8) :=
sorry

end quadratic_non_real_roots_l453_453447


namespace boat_speed_in_still_water_l453_453652

theorem boat_speed_in_still_water (V_b : ℝ) : 
    (∀ (stream_speed : ℝ) (travel_time : ℝ) (distance : ℝ), 
        stream_speed = 5 ∧ 
        travel_time = 5 ∧ 
        distance = 105 →
        distance = (V_b + stream_speed) * travel_time) → 
    V_b = 16 := 
by 
    intro h
    specialize h 5 5 105 
    have h1 : 105 = (V_b + 5) * 5 := h ⟨rfl, ⟨rfl, rfl⟩⟩
    sorry

end boat_speed_in_still_water_l453_453652


namespace sum_conditions_mod_3_l453_453085

def sum_rows (M : matrix (fin n) (fin n) ℤ) (k : fin n) : ℤ :=
  ∑ i, M i k

def sum_cols (M : matrix (fin n) (fin n) ℤ) (k : fin n) : ℤ :=
  ∑ i, M k i

theorem sum_conditions_mod_3 {n : ℕ} (M : matrix (fin n) (fin n) ℤ) :
  (∀ k : fin n, sum_cols M k = sum_rows M k - 1 ∨ sum_cols M k = sum_rows M k + 2) → n % 3 = 0 :=
begin
  sorry
end

end sum_conditions_mod_3_l453_453085


namespace sin_gamma_plus_delta_l453_453458

theorem sin_gamma_plus_delta (γ δ : ℝ) (hγ : Complex.exp (Complex.I * γ) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I)
                             (hδ : Complex.exp (Complex.I * δ) = (-5/13 : ℂ) + (12/13 : ℂ) * Complex.I) :
  Real.sin (γ + δ) = 33 / 65 :=
by
  sorry

end sin_gamma_plus_delta_l453_453458


namespace tangent_line_b_value_l453_453848

theorem tangent_line_b_value : 
  ∀ (k b x1 x2 : ℝ), 
  (y = k * x + b) → 
  (y = ln x + 2) → 
  (y = ln (x + 1)) → 
  (k = 2 ∧ x1 = 1 / 2 ∧ x2 = -1 / 2 ∧ b = 1 - ln 2) 
  := sorry

end tangent_line_b_value_l453_453848


namespace find_quantity_A_l453_453312

variable (x : ℝ)

-- Define the costs per pound for each type of coffee
def cost_A : ℝ := 4.60
def cost_B : ℝ := 5.95
def cost_C : ℝ := 6.80

-- Define the relationships between the quantities of coffee types
def quantity_B : ℝ := 2 * x
def quantity_C : ℝ := 3 * x

-- Define the equation representing the total cost
def total_cost : ℝ := cost_A * x + cost_B * quantity_B + cost_C * quantity_C

-- Define the given total cost
def given_total_cost : ℝ := 1237.20

-- The proof to be provided
theorem find_quantity_A : total_cost = given_total_cost → x = 33.54 :=
by
  sorry

end find_quantity_A_l453_453312


namespace non_real_roots_b_range_l453_453436

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l453_453436


namespace least_cans_needed_l453_453283

theorem least_cans_needed 
  (vol_maaza : ℕ) (vol_pepsi : ℕ) (vol_sprite : ℕ)
  (h_maaza : vol_maaza = 215) (h_pepsi : vol_pepsi = 547) (h_sprite : vol_sprite = 991) :
  let gcd_vol := Nat.gcd (Nat.gcd vol_maaza vol_pepsi) vol_sprite in
  gcd_vol = 1 → 
  ((vol_maaza / gcd_vol) + (vol_pepsi / gcd_vol) + (vol_sprite / gcd_vol)) = 1753 := 
by
  sorry

end least_cans_needed_l453_453283


namespace minor_premise_is_2_l453_453879

theorem minor_premise_is_2
    (premise1: ∀ (R: Type) (rect: R → Prop) (parallelogram: R → Prop), ∀ x, rect x → parallelogram x)
    (premise2: ∀ (T: Type) (triangle: T → Prop) (parallelogram: T → Prop), ∀ x, triangle x → ¬ parallelogram x)
    (premise3: ∀ (T: Type) (triangle: T → Prop) (rect: T → Prop), ∀ x, triangle x → ¬ rect x):
    (premise2) := 
sorry

end minor_premise_is_2_l453_453879


namespace sphere_radius_eq_cylinder_radius_l453_453643

theorem sphere_radius_eq_cylinder_radius 
  (h : ℝ) (d : ℝ) (r_sphere : ℝ) (r_cylinder : ℝ)
  (cylinder_height : d = 6) (cylinder_diameter : h = 6) :
  4 * π * r_sphere ^ 2 = 2 * π * r_cylinder * h → r_sphere = 3 := by
    intro h_eq
    have r_cylinder_eq : r_cylinder = 3 := by
      have : d / 2 = 3 := by sorry
      exact this
    sorry

end sphere_radius_eq_cylinder_radius_l453_453643


namespace sum_of_possible_values_of_k_l453_453116

open Complex

theorem sum_of_possible_values_of_k (x y z k : ℂ) (hxyz : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h : x / (1 - y + z) = k ∧ y / (1 - z + x) = k ∧ z / (1 - x + y) = k) : k = 1 :=
by
  sorry

end sum_of_possible_values_of_k_l453_453116


namespace greatest_integer_sum_l453_453901

def floor (x : ℚ) : ℤ := ⌊x⌋

theorem greatest_integer_sum :
  floor (2017 * 3 / 11) + 
  floor (2017 * 4 / 11) + 
  floor (2017 * 5 / 11) + 
  floor (2017 * 6 / 11) + 
  floor (2017 * 7 / 11) + 
  floor (2017 * 8 / 11) = 6048 :=
  by sorry

end greatest_integer_sum_l453_453901


namespace Vanya_bullets_l453_453622

theorem Vanya_bullets (initial_bullets : ℕ) (hits : ℕ) (shots_made : ℕ) (hits_reward : ℕ) :
  initial_bullets = 10 →
  shots_made = 14 →
  hits = shots_made / 2 →
  hits_reward = 3 →
  (initial_bullets + hits * hits_reward) - shots_made = 17 :=
by
  intros
  sorry

end Vanya_bullets_l453_453622


namespace area_between_tangent_and_curve_l453_453132

/-- The area of the region surrounded by the tangent line to the curve
    y = 2006x^3 - 12070102x^2 + ax + b at x = 2006 and the curve itself is 1003/6. -/
theorem area_between_tangent_and_curve (a b : ℝ) :
  let C := λ x : ℝ, 2006*x^3 - 12070102*x^2 + a*x + b in
  let tangent_line := λ x : ℝ, (a - 24208600556)*(x - 2006) + (-32332407600 + a*2006 + b) in
  ∫ x in 2005..2006, (C x - tangent_line x) = 1003 / 6 :=
begin
  sorry
end

end area_between_tangent_and_curve_l453_453132


namespace compute_sum_l453_453714

theorem compute_sum :
  (1 / 2^1984) * (Finset.sum (Finset.range 993) (λ n, (-3: ℝ)^n * (Nat.choose 1984 (2*n)))) = (-1/2: ℝ) :=
by
  sorry

end compute_sum_l453_453714


namespace min_value_of_c_in_triangle_ABC_l453_453468

theorem min_value_of_c_in_triangle_ABC 
  (A B C : ℝ) (a c : ℝ) (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : a > 0)
  (h5 : sin A * cos A > 0)
  (h6 : c * cos B + 2 * cos C = 4 * a * sin B * sin C) :
  c ≈ 2.25 :=
sorry

end min_value_of_c_in_triangle_ABC_l453_453468


namespace time_spent_on_bus_l453_453310

def start_time := 7 + 15 / 60 -- 7:15 a.m.
def end_time := 17 -- 5:00 p.m.
def classes := 7
def class_duration := 45 -- minutes
def lunch_duration := 40 -- minutes
def additional_activities_duration := 1.5 * 60 -- 90 minutes

theorem time_spent_on_bus : 
  ∃ (time_on_bus : ℕ), 
    time_on_bus = 140 ∧ 
    ((end_time - start_time) * 60 - 
    (classes * class_duration + lunch_duration + additional_activities_duration)) = time_on_bus := 
by
  sorry

end time_spent_on_bus_l453_453310


namespace simplify_trig_expression_l453_453944

theorem simplify_trig_expression (x : ℝ) :
  (sin x + (3 * sin x - 4 * (sin x)^3)) / (1 + cos x + (4 * (cos x)^3 - 3 * cos x)) =
  4 * sin x * (cos x)^2 / (1 - 4 * (cos x)^2) :=
by
  sorry

end simplify_trig_expression_l453_453944


namespace natural_number_195_is_solution_l453_453674

-- Define the conditions
def is_odd_digit (n : ℕ) : Prop :=
  n > 0 ∧ n % 2 = 1

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d : ℕ, n / 10 ^ d % 10 < 10 → is_odd_digit (n / 10 ^ d % 10)

-- Define the proof problem
theorem natural_number_195_is_solution :
  195 < 200 ∧ all_digits_odd 195 ∧ (∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 195) :=
by
  sorry

end natural_number_195_is_solution_l453_453674


namespace num_divisors_sum_of_divisors_l453_453061

theorem num_divisors (N : ℕ) (h : N = 86400000) : 
  ((factors N).map_with_count.prod $ λ _, Nat.succ).prod = 264 := sorry

theorem sum_of_divisors (N : ℕ) (h : N = 86400000) : 
  (List.finset (factors N)).sum = 319823280 := sorry

end num_divisors_sum_of_divisors_l453_453061


namespace circle_trajectory_and_point_on_xaxis_proof_l453_453415

noncomputable def circle_trajectory := 
  ∀ (M N : set (ℝ × ℝ)),
  (∀ (x y : ℝ), M (x, y) ↔ (x + 1)^2 + y^2 = 1) ∧ 
  (∀ (x y : ℝ), N (x, y) ↔ (x - 1)^2 + y^2 = 9) → 
  ∃ C : set (ℝ × ℝ), 
    (∀ (x y : ℝ), C (x, y) ↔ x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2)

noncomputable def point_on_xaxis :=
  ∀ (M N : set (ℝ × ℝ)),
  (∀ (x y : ℝ), M (x, y) ↔ (x + 1)^2 + y^2 = 1) ∧ 
  (∀ (x y : ℝ), N (x, y) ↔ (x - 1)^2 + y^2 = 9) →
  ∃ (T : ℝ × ℝ), T = (4, 0) ∧ 
    ∀ (R S : ℝ × ℝ) (k : ℝ),
    line_intersects_curve R S (λ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) (1, 0) k → 
    angle_eq_zero T R S k

axiom line_intersects_curve : 
  ∀ (R S : ℝ × ℝ), 
  (curve_eq : ℝ × ℝ → Prop) (p: ℝ × ℝ) (k: ℝ), Prop

axiom angle_eq_zero : 
  ∀ (T R S : ℝ × ℝ) (k : ℝ), Prop

/-- Proof that the trajectory of the center of circle P is an ellipse and the point T exists -/
theorem circle_trajectory_and_point_on_xaxis_proof :
  circle_trajectory ∧ point_on_xaxis := sorry

end circle_trajectory_and_point_on_xaxis_proof_l453_453415


namespace integer_solutions_count_l453_453819

theorem integer_solutions_count : 
  {x : ℤ | -3 ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ 8}.finite.toFinset.card = 4 := 
by sorry

end integer_solutions_count_l453_453819


namespace asymptote_equation_l453_453773

theorem asymptote_equation 
  (a b c : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 2 * b^2 = a^2 + c^2) 
  (h4 : c^2 = a^2 + b^2) : 
  ∀ x y, (y = (√2 / 2) * x ∨ y = -(√2 / 2) * x) :=
by
  sorry

end asymptote_equation_l453_453773


namespace carrie_harvests_9000_l453_453712

noncomputable def garden_area (length width : ℕ) := length * width
noncomputable def total_plants (plants_per_sqft sqft : ℕ) := plants_per_sqft * sqft
noncomputable def total_cucumbers (yield_plants plants : ℕ) := yield_plants * plants

theorem carrie_harvests_9000 :
  garden_area 10 12 = 120 →
  total_plants 5 120 = 600 →
  total_cucumbers 15 600 = 9000 :=
by sorry

end carrie_harvests_9000_l453_453712


namespace intersection_of_A_and_B_l453_453897

-- Define the sets A and B
def A := {x : ℝ | x ≥ 1}
def B := {x : ℝ | -1 < x ∧ x < 2}

-- Define the expected intersection
def expected_intersection := {x : ℝ | 1 ≤ x ∧ x < 2}

-- The proof problem statement
theorem intersection_of_A_and_B :
  A ∩ B = expected_intersection := by
  sorry

end intersection_of_A_and_B_l453_453897


namespace fruits_calculation_l453_453541

structure FruitStatus :=
  (initial_picked  : ℝ)
  (initial_eaten  : ℝ)

def apples_status : FruitStatus :=
  { initial_picked := 7.0 + 3.0 + 5.0, initial_eaten := 6.0 + 2.0 }

def pears_status : FruitStatus :=
  { initial_picked := 0, initial_eaten := 4.0 + 3.0 }  -- number of pears picked is unknown, hence 0

def oranges_status : FruitStatus :=
  { initial_picked := 8.0, initial_eaten := 8.0 }

def cherries_status : FruitStatus :=
  { initial_picked := 4.0, initial_eaten := 4.0 }

theorem fruits_calculation :
  (apples_status.initial_picked - apples_status.initial_eaten = 7.0) ∧
  (pears_status.initial_picked - pears_status.initial_eaten = 0) ∧  -- cannot be determined in the problem statement
  (oranges_status.initial_picked - oranges_status.initial_eaten = 0) ∧
  (cherries_status.initial_picked - cherries_status.initial_eaten = 0) :=
by {
  sorry
}

end fruits_calculation_l453_453541


namespace balls_in_boxes_l453_453063

theorem balls_in_boxes :
  let balls := 5
  let boxes := 4
  boxes ^ balls = 1024 :=
by
  sorry

end balls_in_boxes_l453_453063


namespace no_other_boundary_lattice_points_general_case_l453_453196

theorem no_other_boundary_lattice_points (n : ℕ) : 
  (∃ (triangle : ℝ × ℝ × ℝ) (vertices_are_lattice_points : ℤ × ℤ), (nat.count_lattice_points_inside triangle = n) 
  ∧ (nat.count_lattice_points_on_boundary_excluding_vertices triangle = 0) 
  → (triangle_area triangle = n + 1 / 2)) := sorry

theorem general_case (n m : ℕ) : 
  (∃ (triangle : ℝ × ℝ × ℝ) (vertices_are_lattice_points : ℤ × ℤ), (nat.count_lattice_points_inside triangle = n) 
  ∧ (nat.count_lattice_points_on_boundary_excluding_vertices triangle = m) 
  → (triangle_area triangle = n + m / 2 + 1 / 2)) := sorry

end no_other_boundary_lattice_points_general_case_l453_453196


namespace average_speed_correct_l453_453537

-- Define the two legs of the trip
def leg1_distance : ℝ := 450
def leg1_time : ℝ := 7 + 30 / 60  -- 7 hours 30 minutes

def leg2_distance : ℝ := 540
def leg2_time : ℝ := 8 + 15 / 60  -- 8 hours 15 minutes

-- Define the total distance and time
def total_distance : ℝ := leg1_distance + leg2_distance
def total_time : ℝ := leg1_time + leg2_time

-- Define the average speed
def average_speed : ℝ := total_distance / total_time

-- State that the average speed is approximately 62.86 mph
theorem average_speed_correct : abs (average_speed - 62.86) < 1e-2 := sorry

end average_speed_correct_l453_453537


namespace minyoung_division_l453_453633

theorem minyoung_division : 
  ∃ x : ℝ, 107.8 / x = 9.8 ∧ x = 11 :=
by
  use 11
  simp
  sorry

end minyoung_division_l453_453633


namespace initial_bales_l453_453617

theorem initial_bales (bales_initially bales_added bales_now : ℕ)
  (h₀ : bales_added = 26)
  (h₁ : bales_now = 54)
  (h₂ : bales_now = bales_initially + bales_added) :
  bales_initially = 28 :=
by
  sorry

end initial_bales_l453_453617


namespace part_a_l453_453505

variable {k n : ℕ} (a : ℕ → ℝ)

noncomputable def Q (x : ℝ) : ℝ :=
  x^k + ∑ i in Finset.range n, a i * x^(k+i+1)

theorem part_a (x : ℝ) (hx : 0 < |x| ∧ |x| < 1 / (1 + ∑ i in Finset.range n, |a i|)) :
  Q a x / x^k > 0 := sorry

end part_a_l453_453505


namespace line_through_foci_eq_l453_453584

noncomputable def parabola_focus : (ℝ × ℝ) := (0, 1)
noncomputable def hyperbola_focus : (ℝ × ℝ) := (3, 0)

theorem line_through_foci_eq :
  ∀ (p1 p2 : ℝ × ℝ), p1 = parabola_focus → p2 = hyperbola_focus → 
  ∃ a b c : ℝ, a * (p1.1) + b * (p1.2) + c = 0 ∧ a * (p2.1) + b * (p2.2) + c = 0 ∧ a = 1 ∧ b = 3 ∧ c = -3 :=
by
  assume p1 p2 hp1 hp2
  rw [hp1, hp2]
  use 1, 3, -3
  split
  -- point (0, 1)
  { simp }
  split
  -- point (3, 0)
  { simp }
  split
  -- a = 1
  { refl }
  split
  -- b = 3
  { refl }
  -- c = -3
  { refl }


end line_through_foci_eq_l453_453584


namespace max_points_no_triangle_max_points_with_triangle_l453_453662

namespace GraphTheory

noncomputable def is_valid_graph (G : SimpleGraph ℕ) (n : ℕ) : Prop :=
  ∀ v, G.degree v ≤ 3 ∧ ∀ u w, u ≠ w ∧ ¬G.Adj u w → ∃ z, G.Adj u z ∧ G.Adj w z

theorem max_points_no_triangle (G : SimpleGraph ℕ) (n : ℕ) :
  is_valid_graph G n → n ≤ 10 :=
by
  sorry

theorem max_points_with_triangle (G : SimpleGraph ℕ) (n : ℕ) :
  is_valid_graph G n ∧ ∃ (u v w: ℕ), G.Adj u v ∧ G.Adj v w ∧ G.Adj w u → n ≤ 8 :=
by
  sorry

end GraphTheory

end max_points_no_triangle_max_points_with_triangle_l453_453662


namespace area_of_curve_and_line_l453_453165

noncomputable def area_of_figure : ℝ :=
  ∫ y in -2..4, y + 4 - (y^2 / 2)

theorem area_of_curve_and_line : area_of_figure = 18 :=
sorry

end area_of_curve_and_line_l453_453165


namespace complex_sqrt_unique_l453_453163

theorem complex_sqrt_unique (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : (x - y * complex.I) ^ 2 = 15 - 20 * complex.I) : 
  x - y * complex.I = 5 - 2 * complex.I :=
sorry

end complex_sqrt_unique_l453_453163


namespace available_codes_correct_l453_453918

-- Define the set of valid hexadecimal digits
def hex_digits := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'}

-- Initial or original code
def original_code := ['0', 'A', '3']

-- Count the total number of three-digit hexadecimal codes
def total_codes : ℕ := 16 ^ 3

-- Count the number of codes that differ in only one position
def one_pos_diff_codes : ℕ := 3 * 15

-- Count the number of transposed codes (switching positions of two digits)
def transposed_codes : ℕ := 3

-- Count the original code itself
def original_code_itself : ℕ := 1

-- Calculate the number of available codes for Reckha
def available_codes : ℕ := total_codes - one_pos_diff_codes - transposed_codes - original_code_itself

-- Proving the final result using Lean
theorem available_codes_correct : available_codes = 4047 := by
  -- Skipping the proof state for now
  sorry

end available_codes_correct_l453_453918


namespace final_number_after_operations_l453_453921

theorem final_number_after_operations:
  (∀ d_k : ℕ, 1 ≤ d_k → ∃ n : ℕ, n = 1 ∧
  (S = (∑ k in (range 1988), k) - ∑ k in (range 1987), (1989 - k) * d_k)
  ∧ ∀ k, (S ≥ 0)) :=
sorry

end final_number_after_operations_l453_453921


namespace range_of_a_l453_453970

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_non_neg (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ y) → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f → increasing_on_non_neg f → f a ≤ f 2 → -2 ≤ a ∧ a ≤ 2 :=
by
  intro h_even h_increasing h_le
  sorry

end range_of_a_l453_453970


namespace profit_percentage_l453_453297

theorem profit_percentage (S C : ℝ) (hS : S = 100) (hC : C = 68.97) : 
  let P := S - C in
  let profit_percentage := (P / C) * 100 in
  profit_percentage ≈ 44.97 :=
  by
    -- Introduce the values
    have hP : P = S - C := rfl
    rw [hS, hC] at hP
    -- Calculate profit
    have : P ≈ 31.03 := by {
      linarith
    }
    -- Calculate profit percentage
    have : profit_percentage ≈ (31.03 / 68.97) * 100 := by {
      field_simp [hC, this],
      ring
    }
    have : (31.03 / 68.97) * 100 ≈ 44.97 := by {
      norm_num
    }
    exact this

end profit_percentage_l453_453297


namespace line_equation_through_circle_center_l453_453796

def line_through_center_with_slope (x y : ℝ) (C : ℂ) : Prop :=
  let line := √3 * x - y + √3 = 0 in
  ∃ x y, 
    ∃ (C : ℂ), 
      C = complex.mk (-1) 0 ∧  
      60 = 60 ∧ 
      line = line

theorem line_equation_through_circle_center
  (x y : ℝ) (C : ℂ)
  (h1 : x^2 + 2 * x + y^2 = 0)
  (h2 : C = complex.mk (-1) 0)
  (h3 : ∃ θ, θ = 60) 
  : line_through_center_with_slope x y C := 
sorry

end line_equation_through_circle_center_l453_453796


namespace max_non_multiples_of_3_l453_453696

theorem max_non_multiples_of_3 (a b c d e f : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h2 : a * b * c * d * e * f % 3 = 0) : 
  ¬ ∃ (count : ℕ), count > 5 ∧ (∀ x ∈ [a, b, c, d, e, f], x % 3 ≠ 0) :=
by
  sorry

end max_non_multiples_of_3_l453_453696


namespace sequence_general_term_l453_453073

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2) 
    (h_a₁ : S 1 = 1) (h_an : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) : 
  ∀ n, a n = 2 * n - 1 := 
by
  sorry

end sequence_general_term_l453_453073


namespace total_twin_functions_l453_453461

def twin_functions (f : ℝ → ℝ) (range : Set ℝ) (domain : Set (Set ℝ)) : Prop :=
  ∀ d ∈ domain, (∀ x ∈ d, f x ∈ range) ∧ (∀ x₁ x₂ ∈ d, f x₁ = f x₂ → x₁ = x₂)

noncomputable def example_function (x : ℝ) : ℝ := 2 * x^2 - 1

theorem total_twin_functions : 
  exists (domain : Set (Set ℝ)), twin_functions example_function {1, 7} domain ∧ domain.to_finset.card = 9 :=
sorry

end total_twin_functions_l453_453461


namespace integral_f_l453_453107

variable {f : ℝ → ℝ}

axiom functional_eq (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → (1 - f(x)) = f(1 - x))

theorem integral_f (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → (1 - f(x)) = f(1 - x)) :
  ∫ x in 0..1, f(x) = 1 / 2 :=
by
  sorry

end integral_f_l453_453107


namespace slips_with_3_is_10_l453_453574

open Real

-- Definitions based on the conditions provided
def total_slips : ℝ := 15
def expected_value : ℝ := 4.6
def slips_with_8 (y : ℝ) := total_slips - y
def probability_of_3 (y : ℝ) := y / total_slips
def probability_of_8 (y : ℝ) := (total_slips - y) / total_slips
def expected_value_expr (y : ℝ) := probability_of_3 y * 3 + probability_of_8 y * 8

-- Statement to prove
theorem slips_with_3_is_10 (y : ℝ) (hy : expected_value_expr y = expected_value) : y = 10 := by
  have eq1 : expected_value_expr y = (3 * y + 8 * (total_slips - y)) / total_slips := by sorry
  sorry

end slips_with_3_is_10_l453_453574


namespace sum_of_digits_Michael_age_l453_453540

def Michael_age_next_multiple_of_Tim (M S T : ℕ) := 
  ∃ n, M + n = (T + n) * k ∧ n > 0

theorem sum_of_digits_Michael_age :
  (S = 10) → (T = 2) → (M = S + 2) → 
  (4 ∃ n, n > 0 ∧ S + n % (T + n) = 0) →
  let m' := (∃ n, Michael_age_next_multiple_of_Tim M S T) → 
  let sum_digits := S := digit_sum (M + n) in 
  sum_digits = 2 :=
begin
  sorry
end

end sum_of_digits_Michael_age_l453_453540


namespace incorrect_statement_D_l453_453933

def data := [11, 10, 11, 13, 11, 13, 15]

noncomputable def mode (l : List ℕ) : ℕ :=
l.groupBy id
  |>.maxBy (λ g => g.length)
  |>.head!
  
noncomputable def mean (l : List ℕ) : ℚ :=
(float.ofNat l.sum / float.ofNat l.length).approx

noncomputable def variance (l : List ℕ) : ℚ :=
let μ := mean l
(float.ofNat (l.map (λ x => (float.ofNat x - μ) ^ 2).sum) / float.ofNat l.length).approx 

noncomputable def median (l : List ℕ) : ℕ :=
let sorted := l.sorted
sorted.nth (sorted.length / 2)

theorem incorrect_statement_D : median data ≠ 13 := by sorry

end incorrect_statement_D_l453_453933


namespace problem_part1_problem_part2_l453_453513

variables {A B C I O E : Type}
variables {R : ℝ}
variables {angle_A angle_B angle_C : ℝ}
variables {triangle : A × B × C}
variables (circumcircle_O : O)
variables (incenter_I : I)
variables (angle_B_eq_60 : angle_B = 60)
variables (angle_A_lt_angle_C : angle_A < angle_C)
variables (external_angle_bisector_A : E)
variables (intersection_E : E)

-- The first statement to prove: IO = AE
theorem problem_part1 (h1 : incenter_I = I)
                      (h2 : circumcircle_O = O)
                      (h3 : angle_B_eq_60 : angle_B = 60)
                      (h4 : angle_A_lt_angle_C : angle_A < angle_C)
                      (h5: external_angle_bisector_A = E)
                      (h6: intersection_E = E) : 
                      IO = AE := 
sorry

-- The second statement to prove: 2R < IO + IA + IC < (1 + sqrt(3))R
theorem problem_part2 (h1 : incenter_I = I)
                      (h2 : circumcircle_O = O)
                      (h3 : angle_B_eq_60 : angle_B = 60)
                      (h4 : angle_A_lt_angle_C : angle_A < angle_C)
                      (h5: external_angle_bisector_A = E)
                      (h6: intersection_E = E) :
                      2R < IO + IA + IC < (1 + sqrt(3))R :=
sorry

end problem_part1_problem_part2_l453_453513


namespace edges_parallel_to_axes_l453_453084

theorem edges_parallel_to_axes (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ)
  (hx : x1 = 0 ∨ y1 = 0 ∨ z1 = 0)
  (hy : x2 = x1 + 1 ∨ y2 = y1 + 1 ∨ z2 = z1 + 1)
  (hz : x3 = x1 + 1 ∨ y3 = y1 + 1 ∨ z3 = z1 + 1)
  (hv : x4*y4*z4 = 2011) :
  (x2-x1 ∣ 2011) ∧ (y2-y1 ∣ 2011) ∧ (z2-z1 ∣ 2011) := 
sorry

end edges_parallel_to_axes_l453_453084


namespace fixed_point_l453_453176

-- Define the function and conditions
def f (a : ℝ) (x : ℝ) := a^(2 * x - 1) - 2

variables (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)

-- Theorem statement
theorem fixed_point (h1 : a > 0) (h2 : a ≠ 1) : 
  f a (1/2) = -1 := by
  sorry

end fixed_point_l453_453176


namespace evaluate_g3_of_1_l453_453121

def g (x : ℝ) : ℝ :=
if x ≥ 3 then x^3 else 3 - x

theorem evaluate_g3_of_1 : g (g (g 1)) = 2 := by
  sorry

end evaluate_g3_of_1_l453_453121


namespace sin_ratio_equal_one_or_neg_one_l453_453514

theorem sin_ratio_equal_one_or_neg_one
  (a b : Real)
  (h1 : Real.cos (a + b) = 1/4)
  (h2 : Real.cos (a - b) = 3/4) :
  (Real.sin a) / (Real.sin b) = 1 ∨ (Real.sin a) / (Real.sin b) = -1 :=
sorry

end sin_ratio_equal_one_or_neg_one_l453_453514


namespace proof_equiv_l453_453949

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 6 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem proof_equiv (x : ℝ) : f (g x) - g (f x) = 6 * x ^ 2 - 12 * x + 9 := by
  sorry

end proof_equiv_l453_453949


namespace parabola_point_comparison_l453_453465

theorem parabola_point_comparison :
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  y1 < y2 :=
by
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  have h : y1 < y2 := by sorry
  exact h

end parabola_point_comparison_l453_453465


namespace age_of_new_person_l453_453166

theorem age_of_new_person 
    (n : ℕ) 
    (T : ℕ := n * 14) 
    (n_eq : n = 9) 
    (new_average : (T + A) / (n + 1) = 16) 
    (A : ℕ) : A = 34 :=
by
  sorry

end age_of_new_person_l453_453166


namespace modulus_of_z_l453_453794

-- Definitions given in the conditions
def i : ℂ := complex.I
def z : ℂ := (2 + i) / i

-- The theorem we need to prove
theorem modulus_of_z : complex.abs z = real.sqrt 5 := sorry

end modulus_of_z_l453_453794


namespace rainfall_march_l453_453471

variable (M A : ℝ)
variable (Hm : A = M - 0.35)
variable (Ha : A = 0.46)

theorem rainfall_march : M = 0.81 := by
  sorry

end rainfall_march_l453_453471


namespace range_of_b_l453_453768

noncomputable def f (x a b : ℝ) := (x - a)^2 * (x + b) * Real.exp x

theorem range_of_b (a b : ℝ) (h_max : ∃ δ > 0, ∀ x, |x - a| < δ → f x a b ≤ f a a b) : b < -a := sorry

end range_of_b_l453_453768


namespace find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l453_453382

-- Define the ellipse and parameters
variables (a b c : ℝ) (x y : ℝ)
-- Conditions
def ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)

-- Given conditions
def eccentricity (c a : ℝ) : Prop := c = a * (Real.sqrt 3 / 2)
def rhombus_area (a b : ℝ) : Prop := (1/2) * (2 * a) * (2 * b) = 4
def relation_a_b_c (a b c : ℝ) : Prop := a^2 = b^2 + c^2

-- Questions transformed into proof problems
def ellipse_equation (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def range_OA_OB (OA OB : ℝ) : Prop := OA * OB ∈ Set.union (Set.Icc (-(3/2)) 0) (Set.Ioo 0 (3/2))
def quadrilateral_area : ℝ := 4

-- Prove the results given the conditions
theorem find_equation_of_ellipse (a b c : ℝ) (h_ellipse : ellipse a b) (h_ecc : eccentricity c a) (h_area : rhombus_area a b) (h_rel : relation_a_b_c a b c) :
  ellipse_equation x y := by
  sorry

theorem find_range_OA_OB (OA OB : ℝ) (kAC kBD : ℝ) (h_mult : kAC * kBD = -(1/4)) :
  range_OA_OB OA OB := by
  sorry

theorem find_area_quadrilateral : quadrilateral_area = 4 := by
  sorry

end find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l453_453382


namespace integer_multiples_2017_l453_453894

theorem integer_multiples_2017 (x : Fin 1000 → ℤ)
  (h: ∀ k : ℕ, 0 < k ∧ k ≤ 672 → (∑ i, (x i) ^ k) % 2017 = 0) :
  ∀ i, x i % 2017 = 0 :=
by
  sorry

end integer_multiples_2017_l453_453894


namespace area_ratios_l453_453089

noncomputable def CEF_over_DBE (a b c d f e : Point) (A B C D F E : Triangle) : ℝ :=
  (area A E F) / (area D B E)

theorem area_ratios (A B C D E F : Point)
  (h_AB : A.dist B = 130) (h_AC : A.dist C = 130) (h_AD : A.dist D = 50) (h_CF : C.dist F = 100) :
  CEF_over_DBE A B C D F E = 5 / 23 :=
by
  sorry

end area_ratios_l453_453089


namespace find_line_equation_l453_453741

-- Define the first line equation
def line1 (x y : ℝ) : Prop := 2 * x - y - 5 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the parallel line equation with a variable constant term
def line_parallel (x y m : ℝ) : Prop := 3 * x + y + m = 0

-- State the intersection point
def intersect_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The desired equation of the line passing through the intersection point
theorem find_line_equation (x y : ℝ) (h : intersect_point x y) : ∃ m, line_parallel x y m := by
  sorry

end find_line_equation_l453_453741


namespace largest_sum_5x5_grid_l453_453981

theorem largest_sum_5x5_grid :
  let grid : List (List ℤ) := [[1, 2, 3, 4, 5],
                               [10, 9, 8, 7, 6],
                               [11, 12, 13, 14, 15],
                               [20, 19, 18, 17, 16],
                               [21, 22, 23, 24, 25]] in
  ∃ numbers : List ℤ, 
    (∀ (x : ℤ), x ∈ numbers → ∃ i j, grid[i]![j] = x) ∧
    (∀ (i j k l : ℕ), i ≠ k → j ≠ l → grid[i]![j] ∉ numbers ∨ grid[k]![l] ∉ numbers) ∧
    numbers.sum = 71 := sorry

end largest_sum_5x5_grid_l453_453981


namespace inequality_l453_453117

theorem inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 :=
sorry

end inequality_l453_453117


namespace exists_efficiency_gt_0_80_l453_453646

def efficiency (n : ℕ) : ℝ :=
  let φ := n * (1 - ∑ p in (nat.factors n).to_finset, (1 / p))
  1 - φ / n

theorem exists_efficiency_gt_0_80 :
  ∃ n : ℕ, efficiency n > 0.80 ∧ n = 30030 :=
by
  sorry

end exists_efficiency_gt_0_80_l453_453646


namespace inequality_ge_zero_l453_453143

theorem inequality_ge_zero (x y z : ℝ) : 
  4 * x * (x + y) * (x + z) * (x + y + z) + y^2 * z^2 ≥ 0 := 
sorry

end inequality_ge_zero_l453_453143


namespace min_interval_f_l453_453374

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := Real.exp x
def g (x : ℝ) : ℝ := Real.log x

-- Define the main statement
theorem min_interval_f (t s : ℝ) (h : f t = g s) : 
  ∃ a : ℝ, a > 0 ∧ t = Real.log a ∧ s = Real.exp a ∧ (f t) ∈ (Set.Ioo (1 / 2 : ℝ) (Real.log 2)) := 
sorry

end min_interval_f_l453_453374


namespace max_mag_value_is_sqrt_5_prob_max_mag_is_2_over_9_l453_453473

noncomputable def card_set : List ℕ := [1, 2, 3]

def B_coords (x y : ℕ) : ℤ × ℤ := (x - 2, x - y)

def mag_OB (coords : ℤ × ℤ) : ℝ :=
  real.sqrt (coords.1 ^ 2 + coords.2 ^ 2)

def max_mag_OB : ℝ :=
  real.sqrt 5

def favorable_outcomes : List (ℕ × ℕ) :=
[(1,3), (3,1)]

def probability_max_mag : ℚ :=
  (favorable_outcomes.length : ℚ) / (card_set.length ^ 2 : ℚ)

theorem max_mag_value_is_sqrt_5 :
  ∀ (x y : ℕ), x ∈ card_set → y ∈ card_set → mag_OB (B_coords x y) ≤ max_mag_OB := by
  sorry

theorem prob_max_mag_is_2_over_9 :
  probability_max_mag = 2 / 9 := by
  sorry

end max_mag_value_is_sqrt_5_prob_max_mag_is_2_over_9_l453_453473


namespace find_angle_ERF_l453_453094

noncomputable def angle_ERF (DEF : Triangle) (DP EQ : Line) (R : Point) (h1 : is_altitude DEF DP) (h2 : is_altitude DEF EQ) (intersection : intersect_at R DP EQ) (angle_DEF : angle DEF = 58) (angle_DFE : angle DFE = 67) : Prop :=
  angle E R F = 35

theorem find_angle_ERF (DEF : Triangle) (DP EQ : Line) (R : Point)
  (h1 : is_altitude DEF DP) (h2 : is_altitude DEF EQ) (intersection : intersect_at R DP EQ)
  (angle_DEF : angle DEF = 58) (angle_DFE : angle DFE = 67) :
  angle E R F = 35 :=
sorry

end find_angle_ERF_l453_453094


namespace segment_halving_1M_l453_453704

noncomputable def segment_halving_sum (k : ℕ) : ℕ :=
  3^k + 1

theorem segment_halving_1M : segment_halving_sum 1000000 = 3^1000000 + 1 :=
by
  sorry

end segment_halving_1M_l453_453704


namespace part1_part2_l453_453401

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - (a / 2) * x^2

-- Define the line l
noncomputable def l (k : ℤ) (x : ℝ) : ℝ := (k - 2) * x - k + 1

-- Theorem for part (1)
theorem part1 (x : ℝ) (a : ℝ) (h₁ : e ≤ x) (h₂ : x ≤ e^2) (h₃ : f a x > 0) : a < 2 / e :=
sorry

-- Theorem for part (2)
theorem part2 (k : ℤ) (h₁ : a = 0) (h₂ : ∀ (x : ℝ), 1 < x → f 0 x > l k x) : k ≤ 4 :=
sorry

end part1_part2_l453_453401


namespace jean_avg_call_handling_minutes_l453_453103

noncomputable def total_minutes_call_handling :=
  let monday_minutes := (35 / (1 - 0.15)) * 5
  let tuesday_minutes := (46 / (1 - 0.10)) * 4
  let wednesday_minutes := (27 / (1 - 0.20)) * 3
  let thursday_minutes := (61 / (1 - 0.05)) * 6
  let friday_minutes := (31 / (1 - 0)) * 5
  (monday_minutes + tuesday_minutes + wednesday_minutes + thursday_minutes + friday_minutes) / 5

theorem jean_avg_call_handling_minutes :
  abs (total_minutes_call_handling - 210.57) < 1 / 100 :=
by
  sorry

end jean_avg_call_handling_minutes_l453_453103


namespace rebecca_husband_slices_eaten_l453_453147

-- Conditions
def total_pies : ℕ := 2
def slices_per_pie : ℕ := 8
def total_slices := total_pies * slices_per_pie
def rebecca_slices_per_pie : ℕ := 1
def rebecca_total_slices := total_pies * rebecca_slices_per_pie
def remaining_slices := total_slices - rebecca_total_slices
def family_friends_percentage : ℚ := 0.5
def family_friends_slices := (family_friends_percentage * remaining_slices).natAbs
def slices_left_before_sunday := remaining_slices - family_friends_slices
def slices_left_after_sunday : ℕ := 5

-- Prove how many slices Rebecca and her husband ate on Sunday evening
theorem rebecca_husband_slices_eaten : slices_left_before_sunday - slices_left_after_sunday = 2 := by
  sorry

end rebecca_husband_slices_eaten_l453_453147


namespace limit_fraction_l453_453259

theorem limit_fraction :
  (filter.tendsto (λ x : ℝ, ((1 + x)^3 - (1 + 3 * x)) / (x + x^5)) (nhds 0) (nhds 0)) :=
by
  sorry

end limit_fraction_l453_453259


namespace solve_log_equation_l453_453567

theorem solve_log_equation :
  ∀ x : ℝ, 
  5 * Real.logb x (x / 9) + Real.logb (x / 9) x^3 + 8 * Real.logb (9 * x^2) (x^2) = 2
  → (x = 3 ∨ x = Real.sqrt 3) := by
  sorry

end solve_log_equation_l453_453567


namespace cards_one_common_number_l453_453139

theorem cards_one_common_number (cards : Fin 21 → Finset (Fin 20)) 
  (h₁ : ∀ i, (cards i).card = 3) 
  (h₂ : ∀ i j, i ≠ j → cards i ≠ cards j) : 
  ∃ i j, i ≠ j ∧ (cards i ∩ cards j).card = 1 :=
begin
  sorry
end

end cards_one_common_number_l453_453139


namespace polygon_same_color_bounds_l453_453576

variable (Π : Type) 

def isCheckeredPlane (Π : Type) : Prop := sorry

def area (Π : Type) : ℕ := sorry
def perimeter (Π : Type) : ℕ := sorry

theorem polygon_same_color_bounds 
  (Π : Type) 
  (h1 : isCheckeredPlane Π)
  (S : ℕ) (P : ℕ)
  (hA : area Π = S)
  (hP : perimeter Π = P) :
  (∃ n : ℕ, n ≤ S / 2 + P / 8 ∧ n ≥ S / 2 - P / 8) :=
sorry

end polygon_same_color_bounds_l453_453576


namespace M_gt_N_l453_453910

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2 * (x + y - 1)

theorem M_gt_N : M x y > N x y := sorry

end M_gt_N_l453_453910


namespace area_of_largest_medallion_is_314_l453_453202

noncomputable def largest_medallion_area_in_square (side: ℝ) (π: ℝ) : ℝ :=
  let diameter := side
  let radius := diameter / 2
  let area := π * radius^2
  area

theorem area_of_largest_medallion_is_314 :
  largest_medallion_area_in_square 20 3.14 = 314 := 
  sorry

end area_of_largest_medallion_is_314_l453_453202


namespace f_neg_t_eq_zero_l453_453808

def f (x : ℝ) : ℝ := 3 * x + Real.sin x + 1

-- Define the given condition: f(t) = 2
axiom f_t_eq_two (t : ℝ) : f t = 2

-- State the theorem: If f(t) = 2, then f(-t) = 0
theorem f_neg_t_eq_zero (t : ℝ) (h : f t = 2) : f (-t) = 0 :=
by {
  sorry
}

end f_neg_t_eq_zero_l453_453808


namespace swimming_club_total_members_l453_453683

def valid_total_members (total : ℕ) : Prop :=
  ∃ (J S V : ℕ),
    3 * S = 2 * J ∧
    5 * V = 2 * S ∧
    total = J + S + V

theorem swimming_club_total_members :
  valid_total_members 58 := by
  sorry

end swimming_club_total_members_l453_453683


namespace relationship_between_s_and_t_l453_453377

theorem relationship_between_s_and_t (s t : ℝ) 
  (h1 : 3^s + 13^t = 17^s)
  (h2 : 5^s + 7^t = 11^t) : t > s :=
by
  sorry

end relationship_between_s_and_t_l453_453377


namespace sum_of_digits_of_pow_minus_hundred_l453_453634

theorem sum_of_digits_of_pow_minus_hundred : 
  let n := 100 
  let x := 10^n
  let y := x - 100 
  (sum_digits y = 882) := 
by
  let n := 100
  let x := 10^n
  let y := x - 100
  have h1 : sum_digits y = sum_digits 999...99900 := by sorry -- 98 nines
  have h2 : sum_digits (999...99900) = 9 * 98 := by sorry -- 98 nines
  have h3 : 9 * 98 = 882 := by sorry
  rw h2 at h1
  rw h3 at h1
  exact h1
  -- the goal sum_digits y = 882 follows directly from the rewritings
  sorry

end sum_of_digits_of_pow_minus_hundred_l453_453634


namespace trigonometric_simplification_l453_453564

theorem trigonometric_simplification (α : ℝ) :
  (cos (5 * π / 2 - α) * cos (-α)) / (sin (3 * π / 2 + α) * cos (21 * π / 2 - α)) = -1 := by
  sorry

end trigonometric_simplification_l453_453564


namespace applicants_majored_in_political_science_l453_453550

theorem applicants_majored_in_political_science
  (total_applicants : ℕ)
  (gpa_above_3 : ℕ)
  (non_political_science_and_gpa_leq_3 : ℕ)
  (political_science_and_gpa_above_3 : ℕ) :
  total_applicants = 40 →
  gpa_above_3 = 20 →
  non_political_science_and_gpa_leq_3 = 10 →
  political_science_and_gpa_above_3 = 5 →
  ∃ P : ℕ, P = 15 :=
by
  intros
  sorry

end applicants_majored_in_political_science_l453_453550


namespace proof_problem_l453_453724

noncomputable def number_of_integers_condition : ℕ :=
  ∑ r in (Finset.range 39999), if (∃ n : ℤ, 
    2 + Int.floor (200 * n / 201) = Int.ceil (198 * n / 199) ∧ 
    n % 39999 = r) then 1 else 0

theorem proof_problem :
  number_of_integers_condition = 39999 := 
sorry

end proof_problem_l453_453724


namespace angle_ABC_is_27_degrees_l453_453174

-- The problem deals with angles in a regular pentagon and square.

-- Definitions of key conditions as given in the problem.
def interior_angle_pentagon : ℝ := 108
def interior_angle_square : ℝ := 90
def common_side_length : ℝ := 1  -- arbitrary but fixed side length for simplicity

-- Function to define the measure of angle ABC based on the given conditions
noncomputable def measure_angle_ABC : ℝ :=
  let BDC := interior_angle_pentagon - interior_angle_square in -- ∠BDC
  let x := (180 - BDC) / 2 in -- Base angles of isosceles triangle BCD
  interior_angle_pentagon - x -- ∠ABC = ∠ABD - ∠CBD

-- Assertion of the problem's query
theorem angle_ABC_is_27_degrees : measure_angle_ABC = 27 :=
by
  sorry

end angle_ABC_is_27_degrees_l453_453174


namespace FDI_rural_AP_l453_453309

theorem FDI_rural_AP (X U : ℝ)
  (h1 : 0.30 * X = FDI_Gujarat)
  (h2 : 0.20 * FDI_Gujarat = FDI_rural_Gujarat)
  (h3 : 0.80 * FDI_Gujarat = U)
  (h4 : 0.20 * X = FDI_AP)
  (h5 : 0.50 * FDI_AP = FDI_rural_AP)
  (h6 : 0.24 * X = U) : 
  FDI_rural_AP = (5 / 12) * U := 
by
  sorry

end FDI_rural_AP_l453_453309


namespace max_profit_l453_453281

noncomputable def profit (x : ℕ) : ℤ :=
  -x^2 + 19*x + 30

theorem max_profit : 
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 120 :=
begin
  sorry
end

end max_profit_l453_453281


namespace cos_inequality_l453_453852

open Real

-- Given angles of a triangle A, B, C

theorem cos_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hTriangle : A + B + C = π) :
  1 / (1 + cos B ^ 2 + cos C ^ 2) + 1 / (1 + cos C ^ 2 + cos A ^ 2) + 1 / (1 + cos A ^ 2 + cos B ^ 2) ≤ 2 :=
by
  sorry

end cos_inequality_l453_453852


namespace sum_of_coefficients_l453_453798

theorem sum_of_coefficients (x : ℚ) (n : ℕ) (h : n = 6) (H : ∀ k : ℕ, k ≠ 4 → (binomial n k * (x^(n-k) * (2 / x)^k)).Bignum.coe ≠ 4 → (binomial n 4 * (x^(n-4) * (2 / x)^4)).Bignum.coeff > (binomial n k * (x^(n-k) * (2 / x)^k)).Bignum.coeff) :
  (x+2/x)^n = 729 := by
  sorry

end sum_of_coefficients_l453_453798


namespace part1_solution_set_m1_part2_find_m_l453_453112

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (m+1) * x^2 - m * x + m - 1

theorem part1_solution_set_m1 :
  { x : ℝ | f x 1 > 0 } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 0.5 } :=
by
  sorry

theorem part2_find_m :
  (∀ x : ℝ, f x m + 1 > 0 ↔ x > 1.5 ∧ x < 3) → m = -9/7 :=
by
  sorry

end part1_solution_set_m1_part2_find_m_l453_453112


namespace isosceles_triangle_l453_453974

theorem isosceles_triangle
  (ABC : Type) [triangle ABC]
  (angle_BAC : ABC.angle A B C = 75)
  (height_BN : ∀ AC : ℝ, height_from B to AC = AC / 2) :
  is_isosceles ABC :=
sorry

end isosceles_triangle_l453_453974


namespace min_elements_l453_453249

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l453_453249


namespace intersection_point_P_line_l1_perpendicular_line_l2_equal_intercepts_l453_453394

theorem intersection_point_P :
  ∃ (P : ℝ × ℝ), (2 * P.1 - P.2 - 4 = 0) ∧ (P.1 - 2 * P.2 + 1 = 0) ∧ P = (3,2) :=
by
  existsi (3,2)
  split
  sorry
  split
  sorry
  sorry

theorem line_l1_perpendicular (P : ℝ × ℝ) (h1 : 2 * P.1 - P.2 - 4 = 0) (h2 : P.1 - 2 * P.2 + 1 = 0)
  (hP : P = (3,2)) :
  ∃ (A B C : ℝ), A = 4 ∧ B = -3 ∧ C = -6 ∧ (A * P.1 + B * P.2 + C = 0) :=
by
  existsi (4, -3, -6)
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

theorem line_l2_equal_intercepts (P : ℝ × ℝ) (h1 : 2 * P.1 - P.2 - 4 = 0) (h2 : P.1 - 2 * P.2 + 1 = 0)
  (hP : P = (3,2)) :
  ∃ (A B C : ℝ), (A = 2 ∧ B = -3 ∧ C = 0 ∧ (A * P.1 + B * P.2 + C = 0)) ∨
                  (A = 1 ∧ B = 1 ∧ C = -5 ∧ (A * P.1 + B * P.2 + C = 0)) :=
by
  existsi (2, -3, 0)
  left
  split
  sorry
  split
  sorry
  split
  sorry
  sorry
  <|> 
  existsi (1, 1, -5)
  right
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end intersection_point_P_line_l1_perpendicular_line_l2_equal_intercepts_l453_453394


namespace natural_number_range_l453_453738

theorem natural_number_range (n : ℕ) :
  (∀ k : ℕ, k > 0 → k^2 + (⌊n / k^2⌋ : ℕ) ≥ 1991) ↔ (1024 * 967 ≤ n ∧ n ≤ 1024 * 967 + 1023) :=
by
  sorry

end natural_number_range_l453_453738


namespace jackson_sandwiches_l453_453885

noncomputable def total_sandwiches (weeks : ℕ) (miss_wed : ℕ) (miss_fri : ℕ) : ℕ :=
  let total_wednesdays := weeks - miss_wed
  let total_fridays := weeks - miss_fri
  total_wednesdays + total_fridays

theorem jackson_sandwiches : total_sandwiches 36 1 2 = 69 := by
  sorry

end jackson_sandwiches_l453_453885


namespace tangent_line_at_one_l453_453585

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 2 * Real.exp x + x + Real.exp 1

theorem tangent_line_at_one : 
  let y := f 1 in
  ∀ x, y = f(1) -> (1 : ℝ) + 0 →  y = x :=
sorry

end tangent_line_at_one_l453_453585


namespace total_length_S_l453_453900

def S : Set (Real × Real) := { p | ∃ x y : Real, p = (x, y) ∧ 
  (abs (abs x - 3) - 2) + (abs (abs y - 3) - 2) = 2 }

theorem total_length_S : total_line_length S = 96 * Real.sqrt 2 :=
sorry

end total_length_S_l453_453900


namespace petri_dishes_count_l453_453087

def germs_total : ℕ := 5400000
def germs_per_dish : ℕ := 500
def petri_dishes : ℕ := germs_total / germs_per_dish

theorem petri_dishes_count : petri_dishes = 10800 := by
  sorry

end petri_dishes_count_l453_453087


namespace proof_example_l453_453010

open Real

theorem proof_example (p q : Prop) :
  (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  have p : ∃ x : ℝ, sin x < 1 := ⟨0, by norm_num⟩
  have q : ∀ x : ℝ, exp (abs x) ≥ 1 := by
    intro x
    have : abs x ≥ 0 := abs_nonneg x
    exact exp_pos (abs x)
  exact ⟨p, q⟩

end proof_example_l453_453010


namespace percentile_80_correct_l453_453188

noncomputable def scores : List ℚ := 
  [56, 70, 72, 78, 79, 80, 81, 83, 84, 86, 88, 90, 91, 94, 98]

noncomputable def percentile_position (scores : List ℚ) (percent : ℚ) : ℚ :=
  (scores.length : ℚ) * percent

noncomputable def percentile_value (scores : List ℚ) (p : ℚ) : ℚ :=
  let sorted_scores := scores.qsort (· < ·)
  let pos := percentile_position scores p
  if pos.fract = 0 then
    let idx := pos.to_nat - 1
    (sorted_scores[idx] + sorted_scores[idx + 1]) / 2
  else
    sorted_scores[pos.to_nat]

theorem percentile_80_correct : 
  percentile_value scores 0.80 = 90.5 := 
  by 
    sorry

end percentile_80_correct_l453_453188


namespace ellipse_eq_solution_trajectory_eq_solution_l453_453000

-- Define the equations of the ellipse and parabola
def ellipse_eq (a b : ℝ) (a_gt_b : a > b) :=
  ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

def parabola_eq : ℝ → ℝ → Prop := λ x y, y^2 = 4 * x

-- Define the focal point of the parabola
def parabola_focus : (ℝ × ℝ) := (1, 0)

-- Define the first problem: proving the equation of the ellipse
theorem ellipse_eq_solution (a b : ℝ) (a_gt_b : a > b) :
  ∃ a b : ℝ, a > b ∧ a^2 = 4 ∧ b^2 = 3 ∧
  ellipse_eq a b a_gt_b :=
by
  -- solution proof goes here
  sorry

-- Define the second problem: proving the equation of the trajectory of R
theorem trajectory_eq_solution :
  ∀ M N R A : ℝ × ℝ,
    A = (-1, 0) →
    ellipse_eq 2 sqrt(3 / 2) (by norm_num) M.1 M.2 →
    ellipse_eq 2 sqrt(3 / 2) (by norm_num) N.1 N.2 →
    let F := parabola_focus in
    let FM := (M.1 - F.1, M.2 - F.2) in
    let FN := (N.1 - F.1, N.2 - F.2) in
    let FR := (R.1 - F.1, R.2 - F.2) in
    FM + FN = FR →
    (4 * R.2^2 + 3 * (R.1^2 + 4 * R.1 + 3) = 0) :=
by
  -- solution proof goes here
  sorry

end ellipse_eq_solution_trajectory_eq_solution_l453_453000


namespace ball_selection_count_l453_453995

/-- There are 5 balls of each of the three colors: red, yellow, and blue, each marked 
with the letters A, B, C, D, E. If we take out 5 balls, requiring that each letter 
is different, then there are 243 ways to do this. -/
theorem ball_selection_count :
  ∃ (balls : Finset (Fin 3 × Fin 5)) (letters : Finset (Fin 5))
  (h : letters.card = 5)
  (h_distinct : ∀ (i : Fin 5) (j : Fin 5), i ≠ j → ∃ (c1 c2 : Fin 3), (c1, i) ∈ balls ∧ (c2, j) ∈ balls),
  balls.card = 5 ∧
  finset.card {x // (x.1, x.2) ∈ balls} = 243 :=
sorry

end ball_selection_count_l453_453995


namespace smallest_x_solution_l453_453760

theorem smallest_x_solution :
  (∃ x : ℚ, abs (4 * x + 3) = 30 ∧ ∀ y : ℚ, abs (4 * y + 3) = 30 → x ≤ y) ↔ x = -33 / 4 := by
  sorry

end smallest_x_solution_l453_453760


namespace minimum_number_of_printed_cards_l453_453615

theorem minimum_number_of_printed_cards : 
  let eligible_digits := {1, 6, 8, 9}
  let tens_place_digits := {0, 1, 6, 8, 9}
  let total_numbers := 900
  let eligible_numbers :=
    {d1, d2, d3 | d1 ∈ eligible_digits ∧ d2 ∈ tens_place_digits ∧ d3 ∈ eligible_digits}
  let palindromes :=
    {d | d.1 = d.3 ∧ d.1 ∈ eligible_digits ∧ d.2 ∈ tens_place_digits}
  let non_palindromes := eligible_numbers \ palindromes
  let min_printed_cards := (non_palindromes.card / 2) + palindromes.card
  in
  min_printed_cards = 34 := sorry

end minimum_number_of_printed_cards_l453_453615


namespace price_of_70_cans_l453_453258

noncomputable def regular_price_per_can : ℝ := 0.55
noncomputable def discount_rate_case : ℝ := 0.25
noncomputable def bulk_discount_rate : ℝ := 0.10
noncomputable def cans_per_case : ℕ := 24
noncomputable def total_cans_purchased : ℕ := 70

theorem price_of_70_cans :
  let discounted_price_per_can := regular_price_per_can * (1 - discount_rate_case)
  let discounted_price_for_cases := 48 * discounted_price_per_can
  let bulk_discount := if 70 >= 3 * cans_per_case then discounted_price_for_cases * bulk_discount_rate else 0
  let final_price_for_cases := discounted_price_for_cases - bulk_discount
  let additional_cans := total_cans_purchased % cans_per_case
  let price_for_additional_cans := additional_cans * discounted_price_per_can
  final_price_for_cases + price_for_additional_cans = 26.895 :=
by sorry

end price_of_70_cans_l453_453258


namespace Josanna_seventh_test_score_l453_453496

theorem Josanna_seventh_test_score (scores : List ℕ) (h_scores : scores = [95, 85, 75, 65, 90, 70])
                                   (average_increase : ℕ) (h_average_increase : average_increase = 5) :
                                   ∃ x, (List.sum scores + x) / (List.length scores + 1) = (List.sum scores) / (List.length scores) + average_increase := 
by
  sorry

end Josanna_seventh_test_score_l453_453496


namespace part1_max_value_part2_three_distinct_real_roots_l453_453039

def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem part1_max_value (m : ℝ) (h_max : ∀ x, f x m ≤ f 2 m) : m = 6 := by
  sorry

theorem part2_three_distinct_real_roots (a : ℝ) (h_m : (m = 6))
  (h_a : ∀ x₁ x₂ x₃ : ℝ, f x₁ m = a ∧ f x₂ m = a ∧ f x₃ m = a →
     x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) : 0 < a ∧ a < 32 := by
  sorry

end part1_max_value_part2_three_distinct_real_roots_l453_453039


namespace cheyenne_earnings_l453_453333

-- Define constants and main conditions
def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

-- Number of cracked pots
def cracked_pots : ℕ := (cracked_fraction * total_pots).toNat
-- Number of pots that are good for sale
def sellable_pots : ℕ := total_pots - cracked_pots
-- Total money earned
def total_earnings : ℕ := sellable_pots * price_per_pot

-- Theorem statement
theorem cheyenne_earnings : total_earnings = 1920 := by
  sorry

end cheyenne_earnings_l453_453333


namespace eggs_donated_l453_453482

theorem eggs_donated (collect_tues_thurs : ℕ)
  (deliver_market : ℕ) (deliver_mall : ℕ) (pie_sat : ℕ) :
  collect_tues_thurs * 2 - (deliver_market + deliver_mall + pie_sat) * 12 = 48 :=
by
  -- Conditions
  have h1 : collect_tues_thurs = 8,
  have h2 : deliver_market = 3,
  have h3 : deliver_mall = 5,
  have h4 : pie_sat = 4,
  sorry

end eggs_donated_l453_453482


namespace parallel_lines_of_isogonal_conjugate_l453_453554

variables {A B C M M' P Q R P' Q' R' E F G : Type}
variables [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C]

-- Define the isogonal conjugate relation
def isogonal_conjugate (M M' A B C : Type) : Prop := sorry

-- Define the perpendicular relation
def perpendicular (X Y Z : Type) : Prop := sorry

-- Define the intersection points
def intersect_at (X Y Z : Type) (P Z' : Type) : Prop := sorry

-- Main theorem statement
theorem parallel_lines_of_isogonal_conjugate 
  (isogonal : isogonal_conjugate M M' A B C)
  (hp : perpendicular M P B C)
  (hq : perpendicular M Q A C)
  (hr : perpendicular M R A B)
  (hp' : perpendicular M' P' B C)
  (hq' : perpendicular M' Q' A C)
  (hr' : perpendicular M' R' A B)
  (he : intersect_at Q R Q' R' E)
  (hf : intersect_at R P R' P' F)
  (hg : intersect_at P Q P' Q' G) :
  parallel EA FB GC :=
sorry

end parallel_lines_of_isogonal_conjugate_l453_453554


namespace new_number_formed_l453_453845

variable (a b : ℕ)

theorem new_number_formed (ha : a < 10) (hb : b < 10) : 
  ((10 * a + b) * 10 + 2) = 100 * a + 10 * b + 2 := 
by
  sorry

end new_number_formed_l453_453845


namespace minimum_amount_to_buy_11_items_l453_453093

-- Define the prices list
def prices : List ℕ := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]

-- Define a function to calculate the minimum cost given the promotion
def minimum_cost (items : List ℕ) : ℕ :=
  let groups := items.chunk 3 -- Group the items into groups of 3
  groups.foldl (fun acc group => acc + group.sum - group.minimumOrZero) 0

-- Define the main theorem
theorem minimum_amount_to_buy_11_items : minimum_cost prices = 4800 :=
  sorry

end minimum_amount_to_buy_11_items_l453_453093


namespace jorge_gifts_l453_453938

theorem jorge_gifts (gifts_from_emilio : ℕ) (gifts_from_pedro : ℕ) (total_gifts : ℕ) : 
  gifts_from_emilio = 11 → 
  gifts_from_pedro = 4 → 
  total_gifts = 21 → 
  (total_gifts - (gifts_from_emilio + gifts_from_pedro)) = 6 :=
by 
  intros he hp ht 
  simp [he, hp, ht]
  sorry

end jorge_gifts_l453_453938


namespace optionC_is_quadratic_l453_453224

-- Define what it means to be a quadratic equation in one variable.
def isQuadraticInOneVariable (eq : Expr) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ eq = a * x^2 + b * x + c = 0

-- Define the given options
def optionA : Expr := x^2 + 1 / x^2 = 0
def optionB (a b c : ℝ) : Expr := a * x^2 + b * x + c = 0
def optionC : Expr := (x - 1) * (x + 2) = 1 
def optionD (y : ℝ) : Expr := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Define the proof problem
theorem optionC_is_quadratic :
  isQuadraticInOneVariable optionC :=
sorry

end optionC_is_quadratic_l453_453224


namespace shorten_other_side_area_l453_453940

-- Assuming initial dimensions and given conditions
variable (length1 length2 : ℕ)
variable (new_length : ℕ)
variable (area1 area2 : ℕ)

-- Initial dimensions of the index card
def initial_dimensions (length1 length2 : ℕ) : Prop :=
  length1 = 3 ∧ length2 = 7

-- Area when one side is shortened to a specific new length
def shortened_area (length1 length2 new_length : ℕ) : ℕ :=
  if new_length = length1 - 1 then new_length * length2 else length1 * (length2 - 1)

-- Condition that the area is 15 square inches when one side is shortened
def condition_area_15 (length1 length2 : ℕ) : Prop :=
  (shortened_area length1 length2 (length1 - 1) = 15 ∨
   shortened_area length1 length2 (length2 - 1) = 15)

-- Area when the other side is shortened by 1 inch
def new_area (length1 new_length : ℕ) : ℕ :=
  new_length * (length1 - 1)

-- Proving the final area when the other side is shortened
theorem shorten_other_side_area :
  initial_dimensions length1 length2 →
  condition_area_15 length1 length2 →
  new_area length2 (length2 - 1) = 10 :=
by
  intros hdim hc15
  have hlength1 : length1 = 3 := hdim.1
  have hlength2 : length2 = 7 := hdim.2
  sorry

end shorten_other_side_area_l453_453940


namespace area_of_shaded_region_l453_453090

theorem area_of_shaded_region (ABCD : ℝ) (BC : ℝ) (IJ KL DO NM : ℝ)
  (h1 : BC = 2) (h2 : IJ = 1) (h3 : KL = 1) (h4 : DO = 1) (h5 : NM = 1) :
  let HQ := (BC - 2 * 1) / 2 in
  let HP := HQ in
  (HQ + HP = 2) → 
  let A_IJH := 1 / 2 * IJ * HQ in
  let A_KLH := 1 / 2 * KL * HQ in
  let A_DOH := 1 / 2 * DO * HP in
  let A_NMH := 1 / 2 * NM * HP in
  A_IJH + A_KLH + A_DOH + A_NMH = 2 :=
sorry

end area_of_shaded_region_l453_453090


namespace number_of_zeros_of_f_l453_453980

def f (x : ℝ) : ℝ :=
if x = 0 then 0 else x - (1 / x)

theorem number_of_zeros_of_f : 
  {x : ℝ | f x = 0}.finite.toFinset.card = 3 := 
by 
  sorry

end number_of_zeros_of_f_l453_453980


namespace final_value_l453_453777

-- Define the initial setup and replacement logic
def initial_array : List ℕ := List.range' 672 2016.succ

def replace (A : List ℕ) : List ℕ :=
  let (a, b, c) := (A.nth 0, A.nth 1, A.nth 2)
  let min_abc := (a.min b).min c
  ((min_abc / 3)::A.drop 3)

def replacement_process (steps : ℕ) (A : List ℕ) : List ℕ :=
  if steps = 0 then A 
  else replacement_process (steps - 1) (replace A)

theorem final_value :
  let A := initial_array
  let m := replacement_process 672 A |> List.head
  m = 672 * (1 / 3) ^ 672 ∧ 0 < m ∧ m < 1 := sorry

end final_value_l453_453777


namespace minimum_workers_for_profit_l453_453280

theorem minimum_workers_for_profit (n : ℕ) : n > 66 → 
  let daily_cost := 600 + 180 * n in
  let revenue := 189 * n in
  revenue > daily_cost :=
by
  — sorry

end minimum_workers_for_profit_l453_453280


namespace zeros_of_composed_function_l453_453807

def piecewise_function (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then log 2 x else a * x + 1

def composed_function (a : ℝ) (x : ℝ) : ℝ :=
  piecewise_function a (piecewise_function a x) - 1

theorem zeros_of_composed_function (a : ℝ) (ha : a > 0) :
  {x : ℝ | composed_function a x = 0}.finite.to_finset.card = 3 :=
by sorry

end zeros_of_composed_function_l453_453807


namespace prime_iff_satisfies_poly_condition_l453_453359

open polynomial

noncomputable def satisfies_poly_condition (n : ℕ) : Prop :=
n > 4 ∧ ∀ (A B : finset (fin n)), ∃ f : polynomial ℤ, 
  (A.image (λ x, polynomial.eval x f)).val = B.val ∨
  (B.image (λ x, polynomial.eval x f)).val = A.val

theorem prime_iff_satisfies_poly_condition (n : ℕ) : n > 4 → (nat.prime n ↔ satisfies_poly_condition n) :=
begin
  sorry
end

end prime_iff_satisfies_poly_condition_l453_453359


namespace not_exists_set_of_9_numbers_min_elements_l453_453241

theorem not_exists_set_of_9_numbers (s : Finset ℕ) 
  (h_len : s.card = 9) 
  (h_median : ∑ x in (s.filter (λ x, x ≤ 2)), 1 ≤ 5) 
  (h_other : ∑ x in (s.filter (λ x, x ≤ 13)), 1 ≤ 4) 
  (h_avg : ∑ x in s = 63) :
  False := sorry

theorem min_elements (n : ℕ) (h_nat: n ≥ 5) :
  ∃ s : Finset ℕ, s.card = 2 * n + 1 ∧
                  ∑ x in (s.filter (λ x, x ≤ 2)), 1 = n + 1 ∧ 
                  ∑ x in (s.filter (λ x, x ≤ 13)), 1 = n ∧
                  ∑ x in s = 14 * n + 7 := sorry

end not_exists_set_of_9_numbers_min_elements_l453_453241


namespace proof_problem_l453_453017

variables (R : Type*) [Real R]

def p : Prop := ∃ x : R, Real.sin x < 1
def q : Prop := ∀ x : R, Real.exp (abs x) ≥ 1

theorem proof_problem : p ∧ q := 
by 
  sorry

end proof_problem_l453_453017


namespace complex_logarithm_problem_l453_453780

open Complex

theorem complex_logarithm_problem
  {z1 z2 : ℂ}
  (h1 : abs z1 = 3)
  (h2 : abs (z1 + z2) = 3)
  (h3 : abs (z1 - z2) = 3 * Real.sqrt 3) :
  Real.log2 (abs ((z1 * conj z2) ^ 2000 + (conj z1 * z2) ^ 2000)) = 4000 := 
sorry

end complex_logarithm_problem_l453_453780


namespace problem_l453_453109

open_locale classical
noncomputable theory

-- Definitions
def Q_pos : Type := {q : ℚ // 0 < q}

variables (f : Q_pos -> ℝ)
variable (a : ℚ)
variable (h_a : 1 < a)
variable (h_fa : f ⟨a, by linarith⟩ = a)

-- Conditions
def condition1 := ∀ x y : Q_pos, f x * f y ≥ f ⟨x.val * y.val, mul_pos x.prop y.prop⟩
def condition2 := ∀ x y : Q_pos, f ⟨x.val + y.val, add_pos x.prop y.prop⟩ ≥ f x + f y

-- The theorem to prove
theorem problem (cond1 : condition1 f) (cond2 : condition2 f) (ha : h_fa) :
  ∀ x : Q_pos, f x = x.val :=
sorry

end problem_l453_453109


namespace cos_alpha_in_second_quadrant_l453_453391

theorem cos_alpha_in_second_quadrant 
  (alpha : ℝ) 
  (h1 : π / 2 < alpha ∧ alpha < π)
  (h2 : ∀ x y : ℝ, 2 * x + (Real.tan alpha) * y + 1 = 0 → 8 / 3 = -(2 / (Real.tan alpha))) :
  Real.cos alpha = -4 / 5 :=
by
  sorry

end cos_alpha_in_second_quadrant_l453_453391


namespace ara_height_is_60_l453_453941

/-- Definitions for the conditions -/
def shea_original_height : ℝ := 65 / 1.3
def shea_current_height : ℝ := 65
def ara_growth (shea_growth: ℝ) : ℝ := shea_growth - 5
def ara_current_height : ℝ := shea_original_height + 10

/-- The theorem we want to prove -/
theorem ara_height_is_60 :
  (shea_original_height * 1.3 = shea_current_height) →
  (shea_growth : ℝ) = (shea_original_height * 0.3) →
  (ara_current_height = shea_original_height + (shea_growth - 5)) →
  ara_current_height = 60 := 
sorry

end ara_height_is_60_l453_453941


namespace perpendicular_lines_a_value_l453_453352

theorem perpendicular_lines_a_value :
  (∃ (a : ℝ), ∀ (x y : ℝ), (3 * y + x + 5 = 0) ∧ (4 * y + a * x + 3 = 0) → a = -12) :=
by
  sorry

end perpendicular_lines_a_value_l453_453352


namespace area_of_midpoints_l453_453571

theorem area_of_midpoints (side_length : ℝ) (segment_length : ℝ) (m : ℝ) : 
  side_length = 4 → 
  segment_length = 3 → 
  abs (m - (16 - (9 * real.pi / 4))) < 0.01 → 100 * m = 1429 :=
by 
  intros hs hl happrox
  sorry

end area_of_midpoints_l453_453571


namespace more_pie_eaten_l453_453703

theorem more_pie_eaten (erik_pie : ℝ) (frank_pie : ℝ)
  (h_erik : erik_pie = 0.6666666666666666)
  (h_frank : frank_pie = 0.3333333333333333) :
  erik_pie - frank_pie = 0.3333333333333333 :=
by
  sorry

end more_pie_eaten_l453_453703


namespace quadratic_non_real_roots_l453_453446

theorem quadratic_non_real_roots (b : ℝ) : 
  let a : ℝ := 1 
  let c : ℝ := 16 in
  (b^2 - 4 * a * c < 0) ↔ (-8 < b ∧ b < 8) :=
sorry

end quadratic_non_real_roots_l453_453446


namespace value_of_b_l453_453119

theorem value_of_b (b : ℝ) (h : 0 < b) 
  (hg : ∀ x, g x = b * x ^ 3 - sqrt 3) 
  (hgg : g (g (sqrt 3)) = -sqrt 3) : 
  b = 1 / 3 := 
by
  sorry

end value_of_b_l453_453119


namespace leaves_fall_l453_453692

theorem leaves_fall (planned_trees : ℕ) (tree_multiplier : ℕ) (leaves_per_tree : ℕ) (h1 : planned_trees = 7) (h2 : tree_multiplier = 2) (h3 : leaves_per_tree = 100) :
  (planned_trees * tree_multiplier) * leaves_per_tree = 1400 :=
by
  rw [h1, h2, h3]
  -- Additional step suggestions for interactive proof environments, e.g.,
  -- Have: 7 * 2 = 14
  -- Goal: 14 * 100 = 1400
  sorry

end leaves_fall_l453_453692


namespace smallest_period_monotonic_interval_l453_453399

def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 + Real.cos (2 * x - Real.pi / 3)

theorem smallest_period (x : ℝ) : ∃ T > 0, T = Real.pi :=
sorry

theorem monotonic_interval (x : ℝ) : ∀ a b : ℝ, 0 < a → a ≤ b → b ≤ Real.pi / 3 → 
  (∀ x ∈ Icc a b, deriv f x > 0) :=
sorry

end smallest_period_monotonic_interval_l453_453399


namespace infinite_series_sum_l453_453531

theorem infinite_series_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) :
  (∑' n in ℕ, (1 / (((2 * n - 3) * a - (n - 2) * b) * (2 * n * a - (2 * n - 1) * b)))) = (1 / ((a - b) * b)) :=
sorry

end infinite_series_sum_l453_453531


namespace find_inverse_of_f_l453_453392

noncomputable def f (x : ℝ) : ℝ := 2^(4-x) + 1

theorem find_inverse_of_f :
  ∀ (x : ℝ), 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + 4) = f x) ∧ (∀ x : ℝ, 4 - x ∈ (Icc 4 6) → f (4 - x) = 2^(4 - x) + 1) ∧ (x ∈ (Icc 4 6) → f x = 2^x + 1) →
  f⁻¹ 19 = log 8 9 := 
sorry

end find_inverse_of_f_l453_453392


namespace rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l453_453300

variable (a : ℂ)

theorem rationalize (h : a = 1 / (Real.sqrt 2 - 1)) : a = Real.sqrt 2 + 1 := by
  sorry

theorem value_of_a2_minus_2a (h : a = Real.sqrt 2 + 1) : a ^ 2 - 2 * a = 1 := by
  sorry

theorem value_of_2a3_minus_4a2_minus_1 (h : a = Real.sqrt 2 + 1) : 2 * a ^ 3 - 4 * a ^ 2 - 1 = 2 * Real.sqrt 2 + 1 := by
  sorry

end rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l453_453300


namespace not_divisible_by_2006_l453_453460

theorem not_divisible_by_2006 (k : ℤ) : ¬ ∃ m : ℤ, k^2 + k + 1 = 2006 * m :=
sorry

end not_divisible_by_2006_l453_453460


namespace determine_d_l453_453716

variable {c d : ℤ} -- c and d are integers
variable (g : ℤ → ℤ) (g_inv : ℤ → ℤ) -- g and its inverse

-- The definition of the function g
def g := λ x : ℤ, 4 * x + c

-- The given condition that (-4, d) is on the function g
def cond1 : g (-4) = d := by
  dsimp [g]
  sorry

-- The inverse function definition:
def g_inv := λ y : ℤ, (y - c) / 4

-- The given condition that (-4, d) is on the inverse function g_inv
def cond2 : g_inv (-4) = d := by
  dsimp [g_inv]
  sorry

-- The final theorem to prove
theorem determine_d : d = -4 :=
 by
  -- Here we would use the conditions cond1 and cond2 
  -- to prove the result
  sorry

end determine_d_l453_453716


namespace proposition_p_and_q_l453_453021

-- Define the propositions as per given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (|x|) ≥ 1

-- The theorem to be proved
theorem proposition_p_and_q : p ∧ q :=
by
  sorry

end proposition_p_and_q_l453_453021


namespace average_error_diff_l453_453301

theorem average_error_diff (n : ℕ) (total_data_pts : ℕ) (error_data1 error_data2 : ℕ)
  (h_n : n = 30) (h_total_data_pts : total_data_pts = 30)
  (h_error_data1 : error_data1 = 105) (h_error_data2 : error_data2 = 15)
  : (error_data1 - error_data2) / n = 3 :=
sorry

end average_error_diff_l453_453301


namespace sum_of_powers_mod_5_l453_453626

theorem sum_of_powers_mod_5 :
  (∑ i in Finset.range 2008, i^5) % 5 = 3 :=
by
  have h : ∀ i, i % 5 = (i^5 % 5),
  from λ i, by rw [Finset.nat.mod_eq_of_lt (nat.lt_of_lt_of_le (nat.lt_add_one_iff.mpr (nat.mul_lt_mul_of_pos_left (by norm_num : 1 < 5) (nat.succ_pos _))) (nat.le_refl _))]
  calc
    (∑ i in Finset.range 2008, i^5) % 5
      = (∑ i in Finset.range 2008, i) % 5 : by rw [Finset.sum_congr rfl, h]
    ... = 3 : sorry

end sum_of_powers_mod_5_l453_453626


namespace proof_neg_q_l453_453385

variable (f : ℝ → ℝ)
variable (x : ℝ)

def proposition_p (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

def proposition_q : Prop := ∃ x : ℝ, (deriv fun y => 1 / y) x > 0

theorem proof_neg_q : ¬ proposition_q := 
by
  intro h
  -- proof omitted for brevity
  sorry

end proof_neg_q_l453_453385


namespace angle_ABC_l453_453474

theorem angle_ABC (O A B C : Type) (h : O \triangle ABC is_circle_center) 
  (angleBOC : angle B O C = 110)
  (angleAOB : angle A O B = 150) :
  angle A B C = 50 := 
  sorry

end angle_ABC_l453_453474


namespace proposition_p_and_q_l453_453019

-- Define the propositions as per given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (|x|) ≥ 1

-- The theorem to be proved
theorem proposition_p_and_q : p ∧ q :=
by
  sorry

end proposition_p_and_q_l453_453019


namespace lucas_theorem_l453_453113

theorem lucas_theorem (p n k : ℕ) [fact p.prime] (n_digits k_digits : ℕ → ℕ) (d : ℕ)
  (hn : n = ∑ i in finset.range (d + 1), n_digits i * p ^ i)
  (hk : k = ∑ i in finset.range (d + 1), k_digits i * p ^ i)
  (h_le : k ≤ n) :
  (nat.choose n k) % p = ∏ i in finset.range (d + 1), (nat.choose (n_digits i) (k_digits i)) % p :=
sorry

end lucas_theorem_l453_453113


namespace count_three_digit_numbers_power_of_three_l453_453834

theorem count_three_digit_numbers_power_of_three :
  { n : ℕ | 100 ≤ 3^n ∧ 3^n ≤ 999 }.toFinset.card = 2 := by
  sorry

end count_three_digit_numbers_power_of_three_l453_453834


namespace smallest_positive_integer_l453_453221

-- Given integers m and n, prove the smallest positive integer of the form 2017m + 48576n
theorem smallest_positive_integer (m n : ℤ) : 
  ∃ m n : ℤ, 2017 * m + 48576 * n = 1 := by
sorry

end smallest_positive_integer_l453_453221


namespace greatest_possible_red_points_l453_453136

-- Definition of the problem in Lean 4
theorem greatest_possible_red_points (points : Finset ℕ) (red blue : Finset ℕ)
  (h_total_points : points.card = 25)
  (h_disjoint : red ∩ blue = ∅)
  (h_union : red ∪ blue = points)
  (h_segment : ∀ (r ∈ red), ∃! b ∈ blue, true) :
  red.card ≤ 13 :=
begin
  -- We assert that the greatest number of red points is at most 13.
  sorry
end

end greatest_possible_red_points_l453_453136


namespace cube_divided_into_parts_l453_453580

-- Define the cube C
def C : set (ℝ × ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.3 ∧ p.3 ≤ 1}

-- Define the planes
def plane1 : set (ℝ × ℝ × ℝ) := {p | p.1 = p.2}
def plane2 : set (ℝ × ℝ × ℝ) := {p | p.2 = p.3}
def plane3 : set (ℝ × ℝ × ℝ) := {p | p.3 = p.1}

-- Define the division of the cube
def divided_parts : ℕ := -- We will calculate the number of parts

-- The goal is to prove that the number of resulting parts is 8
theorem cube_divided_into_parts : divided_parts = 8 := 
sorry

end cube_divided_into_parts_l453_453580


namespace initial_books_eq_41_l453_453653

-- Definitions and conditions
def books_sold : ℕ := 33
def books_added : ℕ := 2
def books_remaining : ℕ := 10

-- Proof problem
theorem initial_books_eq_41 (B : ℕ) (h : B - books_sold + books_added = books_remaining) : B = 41 :=
by
  sorry

end initial_books_eq_41_l453_453653


namespace factorization_problem_l453_453369

theorem factorization_problem (p q : ℝ) :
  (∃ a b c : ℝ, 
    x^4 + p * x^2 + q = (x^2 + 2 * x + 5) * (a * x^2 + b * x + c)) ↔
  p = 6 ∧ q = 25 := 
sorry

end factorization_problem_l453_453369


namespace triangle_angle_C_pi_half_proof_triangle_isosceles_BPC_l453_453492

theorem triangle_angle_C_pi_half_proof (a b c A B C : ℝ) :
  a = real.sqrt 5 →
  (c * real.sin A = real.sqrt 2 * real.sin ((A + B) / 2)) →
  ∠A + ∠B + ∠C = real.pi →
  ∠C = real.pi / 2 := 
by {
  intros h1 h2 Hsum,
  sorry
}

theorem triangle_isosceles_BPC (a c PA angle_APC : ℝ) (A B C P : Point ℝ) :
  a = real.sqrt 5 →
  A.dist P = 1 →
  AC = real.sqrt 5 →
  invertedTriangleAngle (triangleAngle P A C angle_APC = 3 * real.pi / 4) →
  P ∈ InteriorTriangle A B C →
  ∠C = real.pi / 2 →
  isosceles B P C :=
by {
  intros h1 h2 h3 h4 h5 h6,
  sorry
}

end triangle_angle_C_pi_half_proof_triangle_isosceles_BPC_l453_453492


namespace cheyenne_earnings_l453_453335

-- Define constants and main conditions
def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

-- Number of cracked pots
def cracked_pots : ℕ := (cracked_fraction * total_pots).toNat
-- Number of pots that are good for sale
def sellable_pots : ℕ := total_pots - cracked_pots
-- Total money earned
def total_earnings : ℕ := sellable_pots * price_per_pot

-- Theorem statement
theorem cheyenne_earnings : total_earnings = 1920 := by
  sorry

end cheyenne_earnings_l453_453335


namespace area_AEB_correct_l453_453869

noncomputable def area_of_triangle_AEB (AB BC DF GC : ℝ) (hAB : AB = 7) (hBC : BC = 4) (hDF : DF = 2)
  (hGC : GC = 3) : ℝ :=
by
  let CD := AB
  let FG := DF + GC
  have hCD : CD = 7 := hAB
  have hFG : FG = 5 := (show 2 + 3 = 5 by norm_num)
  let EH := (8 / 5)
  have hEH : EH = 8 / 5 := by norm_num
  let AEB_area := (1 / 2) * AB * EH
  have hAEB_area : AEB_area = (1 / 2) * 7 * (8 / 5) := by norm_num
  exact (1 / 2) * 7 * (8 / 5)

theorem area_AEB_correct : area_of_triangle_AEB 7 4 2 3 7.refl 4.refl 2.refl 3.refl = 28 / 5 := by
  simp only [area_of_triangle_AEB, noncomputable]
  exact rfl

end area_AEB_correct_l453_453869


namespace find_x_l453_453417

variables {x : ℝ}
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (x, 2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_x
  (h : dot_product a b = magnitude a) : x = -1 :=
sorry

end find_x_l453_453417


namespace g_is_odd_function_l453_453722

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = - g x := by
  sorry

end g_is_odd_function_l453_453722


namespace total_mustard_bottles_l453_453705

theorem total_mustard_bottles : 
  let table1 : ℝ := 0.25
  let table2 : ℝ := 0.25
  let table3 : ℝ := 0.38
  table1 + table2 + table3 = 0.88 :=
by
  sorry

end total_mustard_bottles_l453_453705


namespace sum_of_set_of_numbers_l453_453998

noncomputable def sum_of_five_numbers := 
  let a1 := 7 in
  let a2 := a1 + 5 in
  let a3 := a2 + 5 in
  let a4 := a3 + 5 in
  let a5 := a4 + 5 in
  a1 + a2 + a3 + a4 + a5

theorem sum_of_set_of_numbers (S : Set ℕ) : 
  {7, 12, 17, 22, 27} ⊆ S → (∀ n ∈ S, n > 0) → (∀ n ∈ S, ∃ k ∈ S, (n = k + 20) ∨ (n = k - 20) ∨ (n = k + 5)) →
  sum_of_five_numbers = 85 := 
by
  sorry

end sum_of_set_of_numbers_l453_453998


namespace general_term_arithmetic_sequence_sequence_becomes_negative_l453_453802

variables (a_n : ℕ → ℝ) (a_1 d : ℝ)

def arithmetic_sequence :=
  ∀ n : ℕ, a_n n = a_1 + (n - 1) * d

theorem general_term_arithmetic_sequence :
  a_3 = 9 ∧ a_9 = 3 →
  (a_1 = 15 ∧ d = -3/2 ∧ ∀ n, a_n n = 15 - (3/2) * (n - 1)) :=
begin
  intro h,
  cases h with ha3 ha9,
  sorry
end

theorem sequence_becomes_negative :
  a_3 = 9 ∧ a_9 = 3 →
  (∀ n, a_n n < 0 ↔ n > 12) :=
begin
  intro h,
  cases h with ha3 ha9,
  sorry
end

end general_term_arithmetic_sequence_sequence_becomes_negative_l453_453802


namespace limit_to_e_neg_33_l453_453328

noncomputable def limit_expression (n : ℕ) : ℝ := (n - 10) / (n + 1)
noncomputable def exponent_expression (n : ℕ) : ℝ := 3 * n + 1

theorem limit_to_e_neg_33 :
  tendsto (λ n : ℕ, (limit_expression n)^(exponent_expression n)) at_top (𝓝 (Real.exp (-33))) :=
sorry

end limit_to_e_neg_33_l453_453328


namespace polynomial_constant_or_increasing_l453_453606

theorem polynomial_constant_or_increasing {n k: ℕ} (hkn: 1 ≤ k ∧ k ≤ 2^n) :
  ∀ (polynomials: ℕ → ℕ → ℕ → ℝ → ℝ),
  (∀ i, polynomials n i k = (polynomials n i k)) →  
  (∃ f : ℝ → ℝ, 
    (∀ x ∈ set.Icc (0 : ℝ) 1, (f x = f x ∨ (∀ x1 x2, (0 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 1) → f x1 ≤ f x2)))) := 
sorry

end polynomial_constant_or_increasing_l453_453606


namespace rotation_matrix_is_correct_l453_453348

noncomputable def rotation_matrix_150 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![real.cos (150 * real.pi / 180), -real.sin (150 * real.pi / 180)],
    ![real.sin (150 * real.pi / 180), real.cos (150 * real.pi / 180)]]

theorem rotation_matrix_is_correct :
  rotation_matrix_150 = ![
    ![-(real.sqrt 3 / 2), -1 / 2],
    ![1 / 2, -(real.sqrt 3 / 2)]
  ] :=
by
  sorry

end rotation_matrix_is_correct_l453_453348


namespace trapezoid_area_is_correct_l453_453861

noncomputable def trapezoid_area : ℝ :=
  let r₁ := 1       -- Radius of the first (inscribed) circle
  let r₂ := 1 / 2   -- Radius of the second circle which touches the first circle
  let height := 2 * (r₁ + r₂)  -- Height of the trapezoid
  let base₁ := 4 * √2 * r₁
  let base₂ := √2 / 2 * r₁
  (base₁ + base₂) * height / 2  -- Area of the trapezoid

theorem trapezoid_area_is_correct :
  trapezoid_area = 9 * √2 / 2 := sorry

end trapezoid_area_is_correct_l453_453861


namespace top_cube_white_faces_l453_453346

theorem top_cube_white_faces
  (painted_faces : ℕ → bool)  -- A function representing whether each face of a cube is painted (true if painted, false if not)
  (n : ℕ)  -- 10 distinct cubes
  (different_coloring : ∀ i j, i ≠ j → ∀ f, painted_faces f i ≠ painted_faces f j)  -- different colorings for every cube
  (top_cube_downward_gray : painted_faces 5 0 = true)  -- the bottom face (index 0) is gray
  (remaining_faces_white : ∀ f, f ≠ 0 → painted_faces f 0 = false)  -- all other faces of the top cube are white
  : ∀ white_faces, white_faces = 5 :=
by
  assume white_faces
  sorry

end top_cube_white_faces_l453_453346


namespace percentage_of_loss_l453_453668

theorem percentage_of_loss (CP SP : ℝ) (h₁ : CP = 1200) (h₂ : SP = 960) : 
  ((CP - SP) / CP) * 100 = 20 :=
by 
  -- We'll directly state the assumptions, and then proceed with the proof.
  rw [h₁, h₂],
  -- Now calculate and simplify.
  sorry

end percentage_of_loss_l453_453668


namespace water_formed_from_reaction_l453_453360

theorem water_formed_from_reaction 
    (moles_NaHCO3 : ℝ) 
    (moles_CH3COOH : ℝ) 
    (molar_mass_H2O : ℝ) 
    (balanced_eq : moles_NaHCO3 = moles_CH3COOH)
    (moles_Water_per_reactant : ℝ := 1)
    : moles_NaHCO3 = 3 → moles_CH3COOH = 3 → molar_mass_H2O = 18.015 → moles_Water_per_reactant * moles_NaHCO3 * molar_mass_H2O = 54.045 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end water_formed_from_reaction_l453_453360


namespace solve_fabric_price_l453_453086

-- Defining the variables
variables (x y : ℕ)

-- Conditions as hypotheses
def condition1 := 7 * x = 9 * y
def condition2 := x - y = 36

-- Theorem statement to prove the system of equations
theorem solve_fabric_price (h1 : condition1 x y) (h2 : condition2 x y) :
  (7 * x = 9 * y) ∧ (x - y = 36) :=
by
  -- No proof is provided
  sorry

end solve_fabric_price_l453_453086


namespace eigenvalues_of_A_l453_453739

-- Defining the problem parameters
def A : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.vecCons (Matrix.vecCons 3 (-6)) (Matrix.vecCons 4 (-1))

def eigenvalue_condition (k : ℝ) : Prop :=
  ∃ (v : Vector (Fin 2) ℝ), v ≠ 0 ∧ A.mul_vec v = k • v

-- The main statement to prove
theorem eigenvalues_of_A : {k : ℝ | eigenvalue_condition k} = {6, -4} := sorry

end eigenvalues_of_A_l453_453739


namespace A_difference_cube_B_difference_sum_of_cubes_l453_453118

def a (i d : ℕ) : ℕ := 1 + 2 * (i - 1) * d
def b (i d : ℕ) : ℕ := 1 + (i - 1) * d

def s (k d : ℕ) : ℕ := ∑ i in (Finset.range k).map Nat.succ, a i d
def t (k d : ℕ) : ℕ := ∑ i in (Finset.range k).map Nat.succ, b i d

def A (n d : ℕ) : ℕ := s (t n d) d
def B (n d : ℕ) : ℕ := t (s n d) d

theorem A_difference_cube (d n : ℕ) : ∃ k : ℕ, A (n + 1) d - A n d = k^3 :=
sorry

theorem B_difference_sum_of_cubes (d n : ℕ) : ∃ k l : ℕ, B (n + 1) d - B n d = k^3 + l^3 :=
sorry

end A_difference_cube_B_difference_sum_of_cubes_l453_453118


namespace bricks_per_course_l453_453889

theorem bricks_per_course : 
  ∃ B : ℕ, (let initial_courses := 3
            let additional_courses := 2
            let total_courses := initial_courses + additional_courses
            let last_course_half_removed := B / 2
            let total_bricks := B * total_courses - last_course_half_removed
            total_bricks = 1800) ↔ B = 400 :=
by {sorry}

end bricks_per_course_l453_453889


namespace tangent_AR_l453_453529

variables (A B C S X Y P Q R : Point)

theorem tangent_AR (h_circle_abc : Circle A B C)
  (h_s_midpoint : S = midpointArc A B C)
  (h_xy_parallel : parallel XY BC)
  (h_p_second_intersection : P = secondIntersection (line_through S X) h_circle_abc)
  (h_q_second_intersection : Q = secondIntersection (line_through S Y) h_circle_abc)
  (h_r_intersection : R = intersection (line_through P Q) XY):
  tangentAtCircle A R h_circle_abc := 
sorry

end tangent_AR_l453_453529


namespace estimate_students_with_high_score_proof_l453_453279

noncomputable def estimate_students_with_high_score : ℕ :=
  let students := 50
  let mean := 110
  let std_dev := 10
  let prob_interval := 0.34
  let prob_above_120 := 0.16
  let num_students_above_120 := prob_above_120 * students
  num_students_above_120

theorem estimate_students_with_high_score_proof : estimate_students_with_high_score = 8 := by
  sorry

end estimate_students_with_high_score_proof_l453_453279


namespace seating_arrangements_l453_453866

theorem seating_arrangements (n : ℕ) (fixed_group_size : ℕ) (total_people : ℕ) :
  n = 10 → fixed_group_size = 4 → total_people = 10 → 
  factorial total_people - factorial (total_people - fixed_group_size + 1) * factorial fixed_group_size = 3507840 :=
by {
  intros h_total h_group h_people,
  rw [h_total, h_group, h_people],
  simp only [factorial],
  sorry
}

end seating_arrangements_l453_453866


namespace acute_angle_at_3_25_l453_453217

def degrees_per_hour_mark : ℝ := 360 / 12
def degrees_per_minute : ℝ := 360 / 60

def hour_hand_angle_at (hours : ℝ) (minutes : ℝ) : ℝ :=
  hours * degrees_per_hour_mark + (minutes / 60) * degrees_per_hour_mark

def minute_hand_angle_at (minutes : ℝ) : ℝ := minutes * degrees_per_minute

def angle_between_hands (hour_angle : ℝ) (minute_angle : ℝ) : ℝ :=
  abs (minute_angle - hour_angle)

def angle_between_hands_at_3_25 : ℝ :=
  let hour_angle := hour_hand_angle_at 3 25
  let minute_angle := minute_hand_angle_at 25
  angle_between_hands hour_angle minute_angle

theorem acute_angle_at_3_25 : angle_between_hands_at_3_25 = 47.5 := by
  sorry

end acute_angle_at_3_25_l453_453217


namespace pink_roses_in_garden_l453_453851

theorem pink_roses_in_garden : 
  ∀ (rows roses_per_row : ℕ)
  (red_ratio white_ratio purple_ratio : ℚ)
  (remaining_after_red remaining_after_white remaining_after_purple : ℕ),
  rows = 30 → roses_per_row = 50 →
  red_ratio = 2 / 5 → white_ratio = 1 / 4 → purple_ratio = 3 / 8 →
  remaining_after_red = roses_per_row - (roses_per_row * red_ratio).to_nat - 1 →
  remaining_after_white = remaining_after_red - (remaining_after_red * white_ratio).to_nat - 2 →
  remaining_after_purple = remaining_after_white - (remaining_after_white * purple_ratio).to_nat - 3 →
  (remaining_after_purple * rows = 300) := 
by
  intros rows roses_per_row red_ratio white_ratio purple_ratio remaining_after_red remaining_after_white remaining_after_purple 
  assume h1 : rows = 30
  assume h2 : roses_per_row = 50
  assume h3 : red_ratio = 2 / 5
  assume h4 : white_ratio = 1 / 4
  assume h5 : purple_ratio = 3 / 8
  assume h6 : remaining_after_red = roses_per_row - (roses_per_row * red_ratio).to_nat - 1
  assume h7 : remaining_after_white = remaining_after_red - (remaining_after_red * white_ratio).to_nat - 2
  assume h8 : remaining_after_purple = remaining_after_white - (remaining_after_white * purple_ratio).to_nat - 3
  sorry

end pink_roses_in_garden_l453_453851


namespace calculate_x_l453_453860

-- Define the problem setup
variables {A B C D E F G H J : Type} [EquilateralTriangle A B C]
variables (BD DE EC AG GF HJ FC : ℝ)
variables (a x b : ℝ)

-- Conditions
axiom collinear_segments : AG = 3 ∧ GF = 18 ∧ HJ = 8 ∧ FC = 2
axiom segment_sum : AG + GF + HJ + FC = 31

-- The goal is to prove the value of x
theorem calculate_x (h : BD = a ∧ DE = x ∧ EC = b) :
  x = 9.5 :=
sorry

end calculate_x_l453_453860


namespace range_of_x_l453_453844

theorem range_of_x (x : ℝ) : 
  (sqrt (5 / (2 - 3 * x))).val.real = true ↔ x < 2 / 3 :=
by 
  sorry

end range_of_x_l453_453844


namespace symmetric_points_range_l453_453806

theorem symmetric_points_range {a : ℝ} (f : ℝ → ℝ)
  (h_f_positive: ∀ x, 0 < x → f x = 2 * x^2 - 3 * x)
  (h_f_negative: ∀ x, x < 0 → f x = a / real.exp x)
  (h_symmetric: ∃ x > 0, f x = f (-x))
  : a ∈ set.Icc (-(real.exp (-1 / 2))) (9 * real.exp (-3)) :=
sorry

end symmetric_points_range_l453_453806


namespace maximal_colors_l453_453533

open Finset

variable {n : ℕ}
-- Define the conditions
def is_valid_cube (n : ℕ) (colors : Finset ℕ) (coloring : Fin (n^3) → ℕ) : Prop :=
  ∀ (i j k : Fin n),
    let x_box := (Finset.univ : Finset (Fin n × Fin n))
    let y_box := (Finset.univ : Finset (Fin n × Fin n))
    let z_box := (Finset.univ : Finset (Fin n × Fin n))
    coloring (⟨i.val * n^2 + j.val * n + k.val, _⟩) ∈ x_box ∧
    coloring (⟨i.val * n^2 + j.val * n + k.val, _⟩) ∈ y_box ∧
    coloring (⟨i.val * n^2 + j.val * n + k.val, _⟩) ∈ z_box

-- Define a theorem for the maximal number of colors
theorem maximal_colors (h : n > 1) (colors : Finset ℕ) (coloring : Fin (n^3) → ℕ)
  (valid_cube : is_valid_cube n colors coloring) :
  colors.card ≤ 2 * n := sorry

end maximal_colors_l453_453533


namespace area_of_quadrilateral_ABCD_l453_453555

theorem area_of_quadrilateral_ABCD
  (A B C D : Point)
  (hB : ∠ B = 90)
  (hD : ∠ D = 90)
  (hAB_BC : dist A B = dist B C)
  (hAD_DC : dist A D + dist D C = 1) :
  area (quadrilateral A B C D) = 1 / 4 := 
sorry

end area_of_quadrilateral_ABCD_l453_453555


namespace profit_percentage_l453_453171

theorem profit_percentage (C S : ℝ) (h : 30 * C = 24 * S) :
  (S - C) / C * 100 = 25 :=
by sorry

end profit_percentage_l453_453171


namespace math_group_equals_orange_group_l453_453552

-- Define the groups, ensuring they appear as conditions in the problem statement
variables (MathOrange MathPurple PhysicsOrange PhysicsPurple : ℕ)

-- Condition from the problem
axiom condition : PhysicsOrange = MathPurple

-- Define the total participants in each group
def TotalMath : ℕ := MathOrange + MathPurple
def TotalOrange : ℕ := MathOrange + PhysicsOrange

-- Theorem statement to be proved
theorem math_group_equals_orange_group (MathOrange MathPurple PhysicsOrange PhysicsPurple : ℕ)
  (h : condition) :
  TotalMath MathOrange MathPurple = TotalOrange MathOrange PhysicsOrange :=
begin
  rw h, -- use the given condition PhysicsOrange = MathPurple
  refl, -- conclude equality
end

end math_group_equals_orange_group_l453_453552


namespace parabola_equation_fixed_point_on_xaxis_l453_453928

-- Defining the conditions
def parabola_focus := (2 : ℝ, 0 : ℝ)
def parabola_directrix := (x : ℝ) = -2

-- Condition 1: Parabola Equation
theorem parabola_equation :
  parabola E (2, 0) -2 ∀ M, M ∈ E ↔ (y : ℝ) ∃ x, y^2 = 8 * (x - 2) :=
sorry

-- Condition 2: Fixed Point on x-axis
theorem fixed_point_on_xaxis :
  ∃ (M N O : ℝ × ℝ), (M ≠ O ∧ N ≠ O) → 
  (dist M (2, 0) = dist M x - 2 - 1) → (dist O M ⊥ dist O N) →  
  ∃ P : ℝ × ℝ, P ∈ line MN ∧ P ∈ x_axis :=
sorry

end parabola_equation_fixed_point_on_xaxis_l453_453928


namespace compound_interest_comparison_l453_453261

theorem compound_interest_comparison (r : ℝ) (h : r = 5 / 100) :
  1 + r < (1 + r / 12) ^ 12 :=
by {
  rw h,
  sorry
}

end compound_interest_comparison_l453_453261


namespace square_brush_ratio_l453_453480

theorem square_brush_ratio (s w : ℝ) (h_area : s^2 = 49) (h_painted_area : s * (s * sqrt 2 * w - w^2) = (s^2) / 3) : 
  s / w = 3 * sqrt 2 :=
  sorry

end square_brush_ratio_l453_453480


namespace non_real_roots_interval_l453_453450

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l453_453450


namespace emily_savings_future_value_l453_453632

theorem emily_savings_future_value :
  let P := 2500
  let r := 0.015
  let n := 21 * 4
  let A := P * (1 + r) ^ n
  A = 8465.73 :=
by
  let P := 2500
  let r := 0.015
  let n := 21 * 4
  let A := P * (1 + r) ^ n
  have h1 : A = 2500 * (1.015) ^ 84 := by rfl
  sorry

end emily_savings_future_value_l453_453632


namespace quadratic_non_real_roots_l453_453430

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l453_453430


namespace x_plus_inv_x_eq_8_then_power_4_l453_453068

theorem x_plus_inv_x_eq_8_then_power_4 (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end x_plus_inv_x_eq_8_then_power_4_l453_453068


namespace jason_books_l453_453887

theorem jason_books (books_per_shelf : ℕ) (num_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 45 → num_shelves = 7 → total_books = books_per_shelf * num_shelves → total_books = 315 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jason_books_l453_453887


namespace measure_of_other_angle_l453_453414

def angles_parallel {A B : Type} (α β : A → B → Prop) : Prop := 
  ∀ a1 a2 b1 b2, α a1 b1 → α a2 b2 → Parallel a1 a2 → Parallel b1 b2 → Prop

def angle_measure {A : Type} {B : Type} (α : A → B → Prop) (m : B → ℕ) : Prop :=
  ∀ a1 b, α a1 b → ∃ n, m b = n

theorem measure_of_other_angle (α β : ∀ {A B : Type}, A → B → Prop) {A B : Type} (m : B → ℕ) :
  ∀ a1 a2 b1 b2, angles_parallel α β → angle_measure α m →
  α a1 b1 → m b1 = 40 → α a2 b2 → (m b2 = 40 ∨ m b2 = 140) := by
  sorry

end measure_of_other_angle_l453_453414


namespace problem1_problem2_l453_453404

section
variable (f : ℝ → ℝ) (a k : ℝ)

def f_def : ℝ → ℝ := λ x, (1 + Real.log x) / x

-- Problem 1: If f(x) has an extremum in (a, a + 1/2), then 1/2 < a < 1
theorem problem1 (h : (∃ x ∈ Ioo a (a + 1/2), ( ∃ y ∈ Ioo a (a + 1/2), f y < f x ∨ f y > f x ))) : 
  1/2 < a ∧ a < 1 :=
sorry

-- Problem 2: If for x ≥ 1, f(x) ≥ k / (x + 1), then k ≤ 2
theorem problem2 (h : ∀ x, x ≥ 1 → f_def x ≥ k / (x + 1)) : k ≤ 2 :=
sorry

end

end problem1_problem2_l453_453404


namespace simple_interest_rate_is_3_96_percent_l453_453066

theorem simple_interest_rate_is_3_96_percent :
  ∀ (P : ℝ) (r_compound : ℝ) (n : ℕ) (t : ℕ),
  P = 5000 → 
  r_compound = 0.04 → 
  n = 2 → 
  t = 1 →
  let A_compound := P * (1 + r_compound / n) ^ (n * t) in
  let I_simple_target := A_compound - P - 4 in
  ∀ (r_simple : ℝ),
  I_simple_target = P * r_simple * t →
  r_simple = 0.0396 :=
begin
  intros P r_compound n t hp hr hn ht A_compound I_simple_target r_simple hI_target,
  sorry
end

end simple_interest_rate_is_3_96_percent_l453_453066


namespace solve_roles_l453_453797

-- Define the roles and people
inductive Role
| soldier
| worker
| farmer

inductive Person
| A
| B
| C

open Role Person

-- Define the condition that B is older than the farmer
axiom B_older_than_farmer : ∀ r, r ≠ farmer → Person.B ≠ r

-- Define the condition that C's age is different from the worker's age
axiom C_diff_worker_age : ∀ r, r ≠ worker → Person.C ≠ r

-- Define the condition that the worker is younger than A
axiom worker_younger_than_A : ∀ r, r = worker → ∀ p, p = Person.A → r ≠ p

theorem solve_roles : 
  (Person.A = soldier ∧ Person.B = worker ∧ Person.C = farmer) :=
by
  -- proof to be implemented
  sorry

end solve_roles_l453_453797


namespace non_real_roots_b_range_l453_453437

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l453_453437


namespace exists_m_divisible_by_2005_l453_453524

def f (x : ℤ) : ℤ := 3*x + 2

def f_iterate (n : ℕ) (x : ℤ) : ℤ :=
  (nat.iterate n f) x

theorem exists_m_divisible_by_2005 : ∃ m : ℕ, 2005 ∣ f_iterate 100 m :=
by
  sorry

end exists_m_divisible_by_2005_l453_453524


namespace no_prime_satisfies_polynomial_l453_453950

theorem no_prime_satisfies_polynomial :
  ∀ p : ℕ, p.Prime → p^3 - 6*p^2 - 3*p + 14 ≠ 0 := by
  sorry

end no_prime_satisfies_polynomial_l453_453950


namespace quadratic_trinomial_value_at_6_l453_453590

theorem quadratic_trinomial_value_at_6 {p q : ℝ} 
  (h1 : ∃ r1 r2, r1 = q ∧ r2 = 1 + p + q ∧ r1 + r2 = -p ∧ r1 * r2 = q) : 
  (6^2 + p * 6 + q) = 31 :=
by
  sorry

end quadratic_trinomial_value_at_6_l453_453590


namespace max_a_plus_ab_plus_abc_l453_453111

noncomputable def f (a b c: ℝ) := a + a * b + a * b * c

theorem max_a_plus_ab_plus_abc (a b c : ℝ) (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h2 : a + b + c = 1) :
  ∃ x, (f a b c ≤ x) ∧ (∀ y, f a b c ≤ y → y = 1) :=
sorry

end max_a_plus_ab_plus_abc_l453_453111


namespace trains_clear_time_l453_453204

def length_train_1 : ℝ := 200 -- in meters
def length_train_2 : ℝ := 280 -- in meters
def length_train_3 : ℝ := 350 -- in meters

def speed_train_1 : ℝ := 42 * 1000 / 3600 -- in m/s
def speed_train_2 : ℝ := 30 * 1000 / 3600 -- in m/s
def speed_train_3 : ℝ := 36 * 1000 / 3600 -- in m/s

noncomputable def max_relative_speed : ℝ := max (speed_train_1 + speed_train_2) (max (speed_train_1 + speed_train_3) (speed_train_2 + speed_train_3))

noncomputable def total_length : ℝ := length_train_1 + length_train_2 + length_train_3

noncomputable def time_to_clear : ℝ := total_length / max_relative_speed

theorem trains_clear_time :
  abs (time_to_clear - 38.29) < 1e-2 := 
  sorry

end trains_clear_time_l453_453204


namespace number_of_pupils_l453_453289

theorem number_of_pupils (T n : ℕ) (h1 : 65 ≤ 73) (h2 : (T + 8) / n = T / n + 0.5) : n = 16 := 
by
  sorry

end number_of_pupils_l453_453289


namespace no_set_of_9_numbers_l453_453254

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l453_453254


namespace parallelogram_area_inside_triangle_l453_453556

theorem parallelogram_area_inside_triangle (T : Triangle) (P : Parallelogram) 
(hP_inside_T : P ⊆ T) : 
  area P ≤ (1 / 2) * area T := 
sorry

end parallelogram_area_inside_triangle_l453_453556


namespace no_set_of_9_numbers_l453_453255

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l453_453255


namespace max_min_on_interval_l453_453752

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 + 5

theorem max_min_on_interval : 
  (∃ x ∈ set.Icc (1 : ℝ) 3, ∀ y ∈ set.Icc (1 : ℝ) 3, f y ≤ f x ∧ f x = 5) ∧
  (∃ x ∈ set.Icc (1 : ℝ) 3, ∀ y ∈ set.Icc (1 : ℝ) 3, f x ≤ f y ∧ f x = 1) := 
by 
  sorry

end max_min_on_interval_l453_453752


namespace processing_is_for_assignment_and_calculation_l453_453078

inductive ProgramBlock
| terminal
| input_output
| processing
| decision

def hasAssignmentAndCalculationFunction : ProgramBlock → Prop
| ProgramBlock.processing := True
| _ := False

theorem processing_is_for_assignment_and_calculation : hasAssignmentAndCalculationFunction ProgramBlock.processing = True :=
by 
  -- statement only, proof is not required
  sorry

end processing_is_for_assignment_and_calculation_l453_453078


namespace minimum_set_size_l453_453246

theorem minimum_set_size (n : ℕ) :
  (2 * n + 1) ≥ 11 :=
begin
  have h1 : 7 * (2 * n + 1) ≤ 15 * n + 2,
  sorry,
  have h2 : 14 * n + 7 ≤ 15 * n + 2,
  sorry,
  have h3 : n ≥ 5,
  sorry,
  show 2 * n + 1 ≥ 11,
  from calc
    2 * n + 1 = 2 * 5 + 1 : by linarith
          ... ≥ 11 : by linarith,
end

end minimum_set_size_l453_453246


namespace min_elements_l453_453251

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l453_453251


namespace has_smallest_prime_factor_in_set_C_l453_453152

def C : Set ℕ := {93, 97, 100, 103, 109}

theorem has_smallest_prime_factor_in_set_C : (∃ n ∈ C, ∀ m ∈ C, Nat.minFac n ≤ Nat.minFac m) ∧ (∃ n ∈ C, Nat.minFac n = 2) :=
by
  use 100
  split
  { intros n hn,
    repeat { sorry } }
  { use 100,
    repeat { sorry } }

end has_smallest_prime_factor_in_set_C_l453_453152


namespace isosceles_triangle_and_projections_l453_453507

theorem isosceles_triangle_and_projections
  (ABC : Triangle)
  (A B C M : Point)
  (Γ : Circle)
  (E F D : Point)
  (h_iso : Isosceles ABC A)
  (h_tangent_AB : Tangent Γ AB B)
  (h_tangent_AC : Tangent Γ AC C)
  (h_M_on_arc : OnArc M Γ)
  (h_E_proj : Projection M AC E)
  (h_F_proj : Projection M AB F)
  (h_D_proj : Projection M BC D) :
  MD^2 = ME * MF := 
sorry

end isosceles_triangle_and_projections_l453_453507


namespace pumacd_probability_m_n_l453_453923

-- Define the conditions
def conditions := (5 : ℕ) -- number of menu items

-- Define the statement to prove the probability and m+n
theorem pumacd_probability_m_n (h: ∀ m n : ℕ, m + n = 157 ∧ gcd m n = 1) : ∃ (m n : ℕ), 
  m = 32 ∧ n = 125 ∧ m + n = 157 :=
by {
  existsi 32,
  existsi 125,
  simp,
  sorry
}

end pumacd_probability_m_n_l453_453923


namespace range_of_k_not_monotonic_l453_453640

theorem range_of_k_not_monotonic (k : ℝ) :
  (∀ x y ∈ (k - 1, k + 1), (2 * x - 1) * (2 * y - 1) < 0) ↔ -1 < k ∧ k < 1 :=
by
sory

end range_of_k_not_monotonic_l453_453640


namespace rotten_oranges_found_l453_453129

def initial_oranges : ℕ := 7 * 12
def reserved_oranges : ℕ := initial_oranges / 4
def remaining_oranges_after_reserving : ℕ := initial_oranges - reserved_oranges
def sold_yesterday : ℕ := (3 * remaining_oranges_after_reserving) / 7
def remaining_oranges_after_selling : ℕ := remaining_oranges_after_reserving - sold_yesterday
def oranges_left_today : ℕ := 32

theorem rotten_oranges_found :
  (remaining_oranges_after_selling - oranges_left_today) = 4 :=
by
  sorry

end rotten_oranges_found_l453_453129


namespace parallelepiped_edges_parallel_to_axes_l453_453082

theorem parallelepiped_edges_parallel_to_axes 
  (V : ℝ) (a b c : ℝ) 
  (integer_coords : ∀ (x y z : ℝ), x = a ∨ x = 0 ∧ y = b ∨ y = 0 ∧ z = c ∨ z = 0) 
  (volume_cond : V = 2011) 
  (volume_def : V = a * b * c) 
  (a_int : a ∈ ℤ) 
  (b_int : b ∈ ℤ) 
  (c_int : c ∈ ℤ) : 
  a = 1 ∧ b = 1 ∧ c = 2011 ∨ 
  a = 1 ∧ b = 2011 ∧ c = 1 ∨ 
  a = 2011 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end parallelepiped_edges_parallel_to_axes_l453_453082


namespace electrical_appliance_supermarket_l453_453654

-- Define the known quantities and conditions
def purchase_price_A : ℝ := 140
def purchase_price_B : ℝ := 100
def week1_sales_A : ℕ := 4
def week1_sales_B : ℕ := 3
def week1_revenue : ℝ := 1250
def week2_sales_A : ℕ := 5
def week2_sales_B : ℕ := 5
def week2_revenue : ℝ := 1750
def total_units : ℕ := 50
def budget : ℝ := 6500
def profit_goal : ℝ := 2850

-- Define the unknown selling prices
noncomputable def selling_price_A : ℝ := 200
noncomputable def selling_price_B : ℝ := 150

-- Define the constraints
def cost_constraint (m : ℕ) : Prop := 140 * m + 100 * (50 - m) ≤ 6500
def profit_exceeds_goal (m : ℕ) : Prop := (200 - 140) * m + (150 - 100) * (50 - m) > 2850

-- The main theorem stating the results
theorem electrical_appliance_supermarket :
  (4 * selling_price_A + 3 * selling_price_B = week1_revenue)
  ∧ (5 * selling_price_A + 5 * selling_price_B = week2_revenue)
  ∧ (∃ m : ℕ, m ≤ 37 ∧ cost_constraint m)
  ∧ (∃ m : ℕ, m > 35 ∧ m ≤ 37 ∧ profit_exceeds_goal m) :=
sorry

end electrical_appliance_supermarket_l453_453654


namespace least_clock_equivalent_to_square_greater_than_4_l453_453544

theorem least_clock_equivalent_to_square_greater_than_4 : 
  ∃ (x : ℕ), x > 4 ∧ (x^2 - x) % 12 = 0 ∧ ∀ (y : ℕ), y > 4 → (y^2 - y) % 12 = 0 → x ≤ y :=
by
  -- The proof will go here
  sorry

end least_clock_equivalent_to_square_greater_than_4_l453_453544


namespace complex_magnitude_l453_453516

theorem complex_magnitude (m : ℝ)
  (z1 : ℂ) (z2 : ℂ) (h1 : z1 = 1 + 2 * Complex.I)
  (h2 : z2 = m + 3 * Complex.I)
  (h3 : ∃ im, z1 * conj z2 = im * Complex.I) :
  Complex.abs (z1 + z2) = 5 * Real.sqrt 2 := 
begin
  sorry
end

end complex_magnitude_l453_453516


namespace vector_dot_product_l453_453030

variable (a b : Vector ℝ) (theta : ℝ)

def magnitude_a : ℝ := 4
def magnitude_b : ℝ := Real.sqrt 2
def angle_between_a_b : ℝ := Real.pi * (135 / 180)

/- Definition of the dot product involving vectors a and b and their magnitudes/angles -/
noncomputable def dot_product : ℝ :=
let a_dot_a := magnitude_a ^ 2
let a_dot_b := magnitude_a * magnitude_b * Real.cos angle_between_a_b
a_dot_a + a_dot_b

theorem vector_dot_product :
  dot_product a b (angle_between_a_b) = 12 := by
  sorry

end vector_dot_product_l453_453030


namespace cherry_trees_leaves_l453_453688

-- Define the original number of trees
def original_num_trees : ℕ := 7

-- Define the number of trees actually planted
def actual_num_trees : ℕ := 2 * original_num_trees

-- Define the number of leaves each tree drops
def leaves_per_tree : ℕ := 100

-- Define the total number of leaves that fall
def total_leaves : ℕ := actual_num_trees * leaves_per_tree

-- Theorem statement for the problem
theorem cherry_trees_leaves : total_leaves = 1400 := by
  sorry

end cherry_trees_leaves_l453_453688


namespace angle_of_inclination_perpendicular_l453_453033

theorem angle_of_inclination_perpendicular 
  (M N : ℝ × ℝ)
  (l_perpendicular_to_MN : ∀ {p q : ℝ × ℝ}, p = M ∨ p = N → q = M ∨ q = N → 
    (p ≠ q) → (l.slope * (q.snd - p.snd) / (q.fst - p.fst) = -1)) : 
  angle_of_inclination l = 45 :=
sorry

end angle_of_inclination_perpendicular_l453_453033


namespace non_consecutive_seating_l453_453865

theorem non_consecutive_seating :
  (10.factorial - (7.factorial * 4.factorial)) = 3507840 :=
by sorry

end non_consecutive_seating_l453_453865


namespace polynomial_symmetric_roots_sum_of_squares_eq_12_l453_453114

variables {p q r s : ℝ}

theorem polynomial_symmetric_roots_sum_of_squares_eq_12
  (hroots_real : ∀ z : ℂ, z + -z ∈ [0, 0] → z ∈ ℝ)
  (hpoly : ∀ z1 z2 z3 z4 : ℝ, 
           (z1 + z2 + z3 + z4 = 0 ∧ z1z2z3z4 = s ∧
            (z1*z2 + z1*z3 + z1*z4 + z2*z3 + z2*z4 + z3*z4) = q ∧
            (z1*z2*z3 + z1*z2*z4 + z1*z3*z4 + z2*z3*z4 = r) ∧
            z1^2 + z2^2 + z3^2 + z4^2 = 12)) :
  q = 12 :=
sorry

end polynomial_symmetric_roots_sum_of_squares_eq_12_l453_453114


namespace parabola_line_intersection_l453_453598

theorem parabola_line_intersection :
  ∀ (x y : ℝ), 
  (y = 20 * x^2 + 19 * x) ∧ (y = 20 * x + 19) →
  y = 20 * x^3 + 19 * x^2 :=
by sorry

end parabola_line_intersection_l453_453598


namespace problem_solution_l453_453228

def statement_A_holds (k b : ℝ) : Prop :=
  (k < 0) ∧ (b > 0)

noncomputable def statement_B_holds (p : ℝ × ℝ) : Prop :=
  p = (3, -2) → ∀ a : ℝ, x + y = a

def statement_C_holds (x₁ y₁ m : ℝ) : Prop :=
  (m = - real.sqrt 3) ∧ (x₁ = 2) ∧ (y₁ = -1) → y + 1 = - real.sqrt 3 * (x - 2)

def statement_D_fails (m b : ℝ) : Prop :=
  (m = -2) ∧ (b = 3) → (y = -2*x + b) ∧ (y ≠ -2*x ± 3)

theorem problem_solution :
  statement_A_holds k b ∧ ¬ statement_B_holds (3, -2) ∧ statement_C_holds 2 -1 (- real.sqrt 3) ∧ ¬ statement_D_fails -2 3 :=
by
  sorry

end problem_solution_l453_453228


namespace justin_reads_pages_l453_453498

theorem justin_reads_pages (x : ℕ) 
  (h1 : 130 = x + 6 * (2 * x)) : x = 10 := 
sorry

end justin_reads_pages_l453_453498


namespace find_number_l453_453952

theorem find_number (x : ℕ) (h : x + 20 + x + 30 + x + 40 + x + 10 = 4100) : x = 1000 := 
by
  sorry

end find_number_l453_453952


namespace necessary_but_not_sufficient_condition_l453_453783

variables (p q : Prop)

theorem necessary_but_not_sufficient_condition
  (h : ¬p → q) (hn : ¬q → p) : 
  (p → ¬q) ∧ ¬(¬q → p) :=
sorry

end necessary_but_not_sufficient_condition_l453_453783


namespace proof_problem_l453_453892

variables {a b c x y z : ℝ} -- Declare the variables as real numbers

-- Use noncomputables if necessary because the problem involves real number properties
noncomputable def problem_statement : Prop :=
  (x + y + z = a + b + c) ∧
  (xyz = abc) ∧
  (a ≤ x ∧ x < y ∧ y < z ∧ z ≤ c) ∧
  (a < b ∧ b < c) ∧
  (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z) →
  (a = x ∧ b = y ∧ c = z)

theorem proof_problem : problem_statement := 
by 
  sorry -- Proof is not required, as per instructions

end proof_problem_l453_453892


namespace hotel_R_greater_than_G_l453_453958

variables (R G P : ℝ)

def hotel_charges_conditions :=
  P = 0.50 * R ∧ P = 0.80 * G

theorem hotel_R_greater_than_G :
  hotel_charges_conditions R G P → R = 1.60 * G :=
by
  sorry

end hotel_R_greater_than_G_l453_453958


namespace men_in_second_group_l453_453651

theorem men_in_second_group (W : ℝ)
  (h1 : W = 18 * 20)
  (h2 : W = M * 30) :
  M = 12 :=
by
  sorry

end men_in_second_group_l453_453651


namespace claudia_candle_choices_l453_453340

-- Claudia can choose 4 different candles
def num_candles : ℕ := 4

-- Claudia can choose 8 out of 9 different flowers
def num_ways_to_choose_flowers : ℕ := Nat.choose 9 8

-- The total number of groupings is given as 54
def total_groupings : ℕ := 54

-- Prove the main theorem using the conditions
theorem claudia_candle_choices :
  num_ways_to_choose_flowers = 9 ∧ num_ways_to_choose_flowers * C = total_groupings → C = 6 :=
by
  sorry

end claudia_candle_choices_l453_453340


namespace at_least_n_minus_2_real_roots_l453_453384

/-- A polynomial representing a non-constant linear function -/
structure NonConstantLinearFunction where
  (a b : ℝ)
  (a_nonzero : a ≠ 0)

namespace NonConstantLinearFunction
/-- Evaluate the non-constant linear function at a point -/
def eval (f : NonConstantLinearFunction) (x : ℝ) : ℝ := f.a * x + f.b
end NonConstantLinearFunction

open NonConstantLinearFunction

/-- Define the polynomial q_i -/
noncomputable def q (n : ℕ) (p : ℕ → NonConstantLinearFunction) (i : ℕ) : ℝ → ℝ :=
  fun x => (∏ j in (Finset.range n).filter (≠ i), (p j).eval x) + (p i).eval x

/-- Statement of the proof -/
theorem at_least_n_minus_2_real_roots (n : ℕ) (p : ℕ → NonConstantLinearFunction) :
  (Finset.card ((Finset.range n).filter (λ i => ∃ x, (q n p i) x = 0))) ≥ n - 2 := 
sorry

end at_least_n_minus_2_real_roots_l453_453384


namespace number_of_permutations_l453_453060

open Nat

theorem number_of_permutations (digits : Finset ℕ)
  (h_digits : digits = {3, 3, 3, 5, 5, 7, 7, 7, 2}) :
  (∃ n : ℕ, n = 1120 ∧ digits = {3, 3, 3, 5, 5, 7, 7, 7, 2} ∧ n = (factorial 8) / ((factorial 3) * (factorial 3))) :=
begin
  sorry
end

end number_of_permutations_l453_453060


namespace infinitely_many_coprime_pairs_divide_sum_of_exponentials_l453_453154

theorem infinitely_many_coprime_pairs_divide_sum_of_exponentials :
  ∃∞ (a b : ℤ), (a > 1 ∧ b > 1) ∧ (Int.gcd a b = 1) ∧ (a + b ∣ a^b + b^a) := sorry

end infinitely_many_coprime_pairs_divide_sum_of_exponentials_l453_453154


namespace solve_equation_correctly_l453_453605

theorem solve_equation_correctly : 
  ∀ x : ℝ, (x - 1) / 2 - 1 = (2 * x + 1) / 3 → x = -11 :=
by
  intro x h
  sorry

end solve_equation_correctly_l453_453605


namespace triangle_problem_l453_453074

theorem triangle_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : Real.Tan A = 7 * Real.Tan B)
  (h2 : (a^2 - b^2) / c = 3)
  (h3 : a = (Real.sin A) / (Real.cos A))
  (h4 : b = (Real.sin B) / (Real.cos B))
  :
  c = 4 :=
sorry

end triangle_problem_l453_453074


namespace remaining_fuel_relation_l453_453620

-- Define the car's travel time and remaining fuel relation
def initial_fuel : ℝ := 100

def fuel_consumption_rate : ℝ := 6

def remaining_fuel (t : ℝ) : ℝ := initial_fuel - fuel_consumption_rate * t

-- Prove that the remaining fuel after t hours is given by the linear relationship Q = 100 - 6t
theorem remaining_fuel_relation (t : ℝ) : remaining_fuel t = 100 - 6 * t := by
  -- Proof is omitted, as per instructions
  sorry

end remaining_fuel_relation_l453_453620


namespace simplify_divide_expression_l453_453566

noncomputable def a : ℝ := Real.sqrt 2 + 1

theorem simplify_divide_expression : 
  (1 - (a / (a + 1))) / ((a^2 - 1) / (a^2 + 2 * a + 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_divide_expression_l453_453566


namespace calculate_AB_length_l453_453140

-- Definitions for the midpoints
variables {A B C D E F : Point} (hC : midpoint C A B) (hD : midpoint D A C) (hE : midpoint E A D) (hF : midpoint F A E)
-- Given condition
variables (hAF : dist A F = 5)
-- The theorem to prove
theorem calculate_AB_length : dist A B = 80 :=
by sorry

end calculate_AB_length_l453_453140


namespace range_of_m_proof_value_of_m_proof_l453_453038

variable (a b m : ℝ)

/-- Definition of the function f --/
def f (x : ℝ) : ℝ := Real.sqrt (|x + 1| + |x - m| - 5)

/-- Conditions for part (I) --/
def condition1 : Prop := ∀ x : ℝ, |x + 1| + |x - m| ≥ 5

/-- Range of m for part (I) --/
def range_of_m : Set ℝ := {m | m ≤ -6 ∨ m ≥ 4}

/-- Conditions for part (II) --/
def condition2 : Prop := a + b + m = 4 ∧ a^2 + b^2 + m^2 = 16

/-- Value of m for part (II) --/
def value_of_m := 4

theorem range_of_m_proof : (∀ x, f m x) → condition1 m → m ∈ range_of_m := sorry

theorem value_of_m_proof : condition2 a b m → condition1 m → m = value_of_m := sorry

end range_of_m_proof_value_of_m_proof_l453_453038


namespace cos_alpha_beta_l453_453398

noncomputable def f (x : Real) : Real := 2 * sin (2 * x - π / 3)
noncomputable def g (x : Real) : Real := (2 + Real.sqrt 3) * cos (2 * x)

theorem cos_alpha_beta (α β : Real) (hα : 0 ≤ α ∧ α < π) (hβ : 0 ≤ β ∧ β < π)
    (h : f α + g α = -2 ∧ f β + g β = -2 ∧ α ≠ β) :
    Real.cos (α - β) = (2 * Real.sqrt 5) / 5 :=
by
  sorry

end cos_alpha_beta_l453_453398


namespace total_fish_l453_453232

-- Definitions based on the problem conditions
def will_catch_catfish : ℕ := 16
def will_catch_eels : ℕ := 10
def henry_trout_per_catfish : ℕ := 3
def fraction_to_return : ℚ := 1/2

-- Calculation of required quantities
def will_total_fish : ℕ := will_catch_catfish + will_catch_eels
def henry_target_trout : ℕ := henry_trout_per_catfish * will_catch_catfish
def henry_return_trout : ℚ := fraction_to_return * henry_target_trout
def henry_kept_trout : ℤ := henry_target_trout -  henry_return_trout.to_nat

-- Goal statement to prove
theorem total_fish (will_catch_catfish = 16) (will_catch_eels = 10) 
  (henry_trout_per_catfish = 3) (fraction_to_return = 1/2) :
  will_total_fish + henry_kept_trout = 50 :=
by
  sorry

end total_fish_l453_453232


namespace radius_of_ball_l453_453664

theorem radius_of_ball (d h : ℝ) (hd : d = 24) (hh : h = 8) : 
  let R := sqrt ((d / 2)^2 + h^2) in R = 13 :=
by
  sorry

end radius_of_ball_l453_453664


namespace non_real_roots_interval_l453_453454

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l453_453454


namespace right_handed_players_total_l453_453549

def cricket_team : Type := 
  {
    total_players : ℕ, 
    throwers : ℕ, 
    non_throwers : ℕ, 
    left_handed_non_throwers : ℕ, 
    right_handed_non_throwers : ℕ, 
    total_right_handed : ℕ 
  }

def cricket_conditions := cricket_team {
  total_players := 120,
  throwers := 58,
  non_throwers := 120 - 58,
  left_handed_non_throwers := Nat.floor (0.4 * ↑(120 - 58)),
  right_handed_non_throwers := (120 - 58) - Nat.floor (0.4 * ↑(120 - 58)),
  total_right_handed := 58 + ((120 - 58) - Nat.floor (0.4 * ↑(120 - 58)))
}

theorem right_handed_players_total : cricket_conditions.total_right_handed = 96 :=
by
  sorry

end right_handed_players_total_l453_453549


namespace number_of_triplets_with_sum_6n_l453_453375

theorem number_of_triplets_with_sum_6n (n : ℕ) : 
  ∃ (count : ℕ), count = 3 * n^2 ∧ 
  (∀ (x y z : ℕ), x ≤ y → y ≤ z → x + y + z = 6 * n → count = 1) :=
sorry

end number_of_triplets_with_sum_6n_l453_453375


namespace telescoping_series_result_l453_453187

noncomputable def telescoping_series : ℤ :=
  let f : ℕ → ℤ := λ n => n! * (n + 2)
  let g : ℕ → ℤ := λ n => (n + 1)! * (n + 3)
  let series_sum := ∑ k in finset.range 2010, (if k % 2 = 0 then f k else -g (k - 1)) + (2011!)
  series_sum

theorem telescoping_series_result : telescoping_series = 1 := by
  sorry

end telescoping_series_result_l453_453187


namespace proposition_truth_value_l453_453011

-- Definitions of the propositions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (|x|) ≥ 1

-- The proof problem statement
theorem proposition_truth_value : (p ∧ q) ∧ ¬ (¬p ∧ q) ∧ ¬ (p ∧ ¬q) ∧ ¬ (¬ (p ∨ q)) :=
by
  sorry

end proposition_truth_value_l453_453011


namespace option_C_not_necessarily_true_l453_453791

variable {a b c : ℝ}

theorem option_C_not_necessarily_true (h1 : c < b) (h2 : b < a) (h3 : ac < 0) : ¬(∀ (c b a : ℝ), c < b → b < a → ac < 0 → cb^2 < ca^2) :=
sorry

end option_C_not_necessarily_true_l453_453791


namespace total_animals_l453_453997

theorem total_animals (total_legs : ℕ) (number_of_sheep : ℕ)
  (legs_per_chicken : ℕ) (legs_per_sheep : ℕ)
  (H1 : total_legs = 60) 
  (H2 : number_of_sheep = 10)
  (H3 : legs_per_chicken = 2)
  (H4 : legs_per_sheep = 4) : 
  number_of_sheep + (total_legs - number_of_sheep * legs_per_sheep) / legs_per_chicken = 20 :=
by {
  sorry
}

end total_animals_l453_453997


namespace difference_numbers_odd_even_l453_453635

def count_odd_digits (n : ℕ) : Bool :=
  let digits := String.toList (n.repr);
  List.all digits (λ c => c = '1' ∨ c = '3' ∨ c = '5' ∨ c = '7' ∨ c = '9')

def count_even_digits (n : ℕ) : Bool :=
  let digits := String.toList (n.repr);
  List.all digits (λ c => c = '0' ∨ c = '2' ∨ c = '4' ∨ c = '6' ∨ c = '8')

def count_numbers_with_odd_digits (m n : ℕ) : ℕ :=
  List.length (List.filter count_odd_digits (List.range (n + 1)).drop (m - 1))

def count_numbers_with_even_digits (m n : ℕ) : ℕ :=
  List.length (List.filter count_even_digits (List.range (n + 1)).drop (m - 1))

theorem difference_numbers_odd_even (a b : ℕ) (h₁ : a = 1) (h₂ : b = 80000) :
  count_numbers_with_odd_digits a b - count_numbers_with_even_digits a b = 780 :=
by
  sorry

end difference_numbers_odd_even_l453_453635


namespace even_integers_top_15_rows_l453_453991

theorem even_integers_top_15_rows :
  let even_in_top_10 := 22 in
  let evens_from_11_to_15 := 31 in
  even_in_top_10 + evens_from_11_to_15 = 53 :=
by
  let even_in_top_10 := 22
  let evens_from_11_to_15 := 31
  have h : even_in_top_10 + evens_from_11_to_15 = 53 := rfl
  exact h

end even_integers_top_15_rows_l453_453991


namespace quadratic_non_real_roots_l453_453433

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l453_453433


namespace contain_points_in_triangle_l453_453267

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem contain_points_in_triangle (k : ℕ) 
  (points : fin k → (ℝ × ℝ)) 
  (h : ∀ i j l : fin k, i ≠ j ∧ j ≠ l ∧ i ≠ l  → area_triangle (points i) (points j) (points l) ≤ 1) :
  ∃ A B C : ℝ × ℝ, area_triangle A B C = 4 ∧ ∀ i : fin k, ∃ a b c : ℝ, a + b + c = 1 ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ points i = (a * A.1 + b * B.1 + c * C.1, a * A.2 + b * B.2 + c * C.2) := 
sorry

end contain_points_in_triangle_l453_453267


namespace g_shift_identity_l453_453459

noncomputable def g (x : ℝ) : ℝ := 6 * x^2 + 3 * x - 4

theorem g_shift_identity (x h : ℝ) : g(x + h) - g(x) = h * (12 * x + 6 * h + 3) :=
by
  sorry

end g_shift_identity_l453_453459


namespace number_of_solutions_sine_quadratic_l453_453351

theorem number_of_solutions_sine_quadratic :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → 3 * (Real.sin x) ^ 2 - 5 * (Real.sin x) + 2 = 0 →
  ∃ a b c, x = a ∨ x = b ∨ x = c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c :=
sorry

end number_of_solutions_sine_quadratic_l453_453351


namespace tetrahedron_faces_congruent_l453_453929

theorem tetrahedron_faces_congruent (A B C D : Point) (hABCD : is_tetrahedron A B C D)
  (m : ℝ) (height_eq : ∀ (F : Face), height_from_opposite_vertex F = m) :
  ∀ F1 F2, is_face F1 A B C D → is_face F2 A B C D → face_congruent F1 F2 :=
begin
  sorry
end

end tetrahedron_faces_congruent_l453_453929


namespace fixed_point_l453_453179

noncomputable def fixed_point_function (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : (ℝ × ℝ) :=
  (1, a^(1 - (1 : ℝ)) + 5)

theorem fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : fixed_point_function a h₀ h₁ = (1, 6) :=
by 
  sorry

end fixed_point_l453_453179


namespace part1_part2_l453_453264

open Real

-- Part 1
theorem part1 {x : ℝ} (h1 : 0 < x) (h2 : x < 1) : (x - x^2 < sin x ∧ sin x < x) :=
sorry

-- Part 2
theorem part2 {a : ℝ} 
  (h1 : ∀ x, f x = cos (a * x) - log (1 - x ^ 2)) 
  (h2 : ∃ ε > 0, ∀ x, |x| < ε → f x ≤ f 0) 
  : a < -sqrt 2 ∨ a > sqrt 2 :=
sorry

end part1_part2_l453_453264


namespace coloring_red_cubes_count_l453_453064

theorem coloring_red_cubes_count : 
  let n := 4 in
  ∃ (latin_squares : Fin n → Fin n → Fin n) [latin_square latin_squares],
  (4! * 3!) * 4 = 576
  :=
begin
  sorry
end

end coloring_red_cubes_count_l453_453064


namespace find_a_range_l453_453005

variable {a : ℝ}

def p : Prop := ∃ x : ℝ, x^2 + 2*a*x + (2 - a) = 0
def q : Prop := ∀ x : ℝ, a*x^2 - √2*a*x + 2 > 0

theorem find_a_range (hna : ¬p) (hp_or_hq : p ∨ q) : 0 ≤ a ∧ a < 1 :=
by sorry

end find_a_range_l453_453005


namespace division_pairs_of_divisors_l453_453930

theorem division_pairs_of_divisors (n : ℕ) (h1 : 0 < n) (h2 : ∀ k : ℕ, k * k ≠ n) :
  ∃ (pairs : list (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ pairs → a ∣ b) ∧ 
                              (∀ d : ℕ, d ∣ n → ∃ (a b : ℕ), (a, b) ∈ pairs ∧ (d = a ∨ d = b)) :=
sorry

end division_pairs_of_divisors_l453_453930


namespace part_a_part_b_l453_453920

-- Part (a)
theorem part_a :
  ∃ (circles : List Circle) (O : Point), 
    List.length circles = 5 ∧ 
    ∀ ray : Ray,
      ∃ c1 c2,
        c1 ∈ circles ∧ c2 ∈ circles ∧ 
        c1 ≠ c2 ∧ 
        ray ∩ c1 ≠ ∅ ∧ ray ∩ c2 ≠ ∅ :=
sorry

-- Part (b)
theorem part_b :
  ¬ ∃ (circles : List Circle) (O : Point),
    List.length circles = 4 ∧ 
    (∀ circle ∈ circles, O ∉ circle) ∧ 
    ∀ ray : Ray,
      ∃ c1 c2,
        c1 ∈ circles ∧ c2 ∈ circles ∧ 
        c1 ≠ c2 ∧ 
        ray ∩ c1 ≠ ∅ ∧ ray ∩ c2 ≠ ∅ :=
sorry

end part_a_part_b_l453_453920


namespace cost_of_acai_berry_juice_l453_453977

theorem cost_of_acai_berry_juice (cost_per_litre_cocktail : ℝ)
                                 (cost_per_litre_fruit_juice : ℝ)
                                 (litres_fruit_juice : ℝ)
                                 (litres_acai_juice : ℝ)
                                 (total_cost_cocktail : ℝ)
                                 (cost_per_litre_acai : ℝ) :
  cost_per_litre_cocktail = 1399.45 →
  cost_per_litre_fruit_juice = 262.85 →
  litres_fruit_juice = 34 →
  litres_acai_juice = 22.666666666666668 →
  total_cost_cocktail = (34 + 22.666666666666668) * 1399.45 →
  (litres_fruit_juice * cost_per_litre_fruit_juice + litres_acai_juice * cost_per_litre_acai) = total_cost_cocktail →
  cost_per_litre_acai = 3106.66666666666666 :=
by
  intros
  sorry

end cost_of_acai_berry_juice_l453_453977


namespace PQ_parallel_AB_l453_453502

-- Definition of points and their properties on a circle.
variables {A B C D F P Q : Point}

-- Hypotheses based on the conditions.
hypothesis (h_circle : ∃ (circle : Circle), circle.contains A ∧ circle.contains B ∧ circle.contains C ∧ circle.contains D)
hypothesis (h_congruent_arcs : congruent_arc AF BF)
hypothesis (h_P_intersection : P = intersection (line_through D F) (line_through A C))
hypothesis (h_Q_intersection : Q = intersection (line_through C F) (line_through B D))

-- The theorem to prove.
theorem PQ_parallel_AB : PQ ∥ AB := by
  sorry

end PQ_parallel_AB_l453_453502


namespace pond_contains_total_money_correct_l453_453339

def value_of_dime := 10
def value_of_quarter := 25
def value_of_nickel := 5
def value_of_penny := 1

def cindy_dimes := 5
def eric_quarters := 3
def garrick_nickels := 8
def ivy_pennies := 60

def total_money : ℕ := 
  cindy_dimes * value_of_dime + 
  eric_quarters * value_of_quarter + 
  garrick_nickels * value_of_nickel + 
  ivy_pennies * value_of_penny

theorem pond_contains_total_money_correct:
  total_money = 225 := by
  sorry

end pond_contains_total_money_correct_l453_453339


namespace problem_D_l453_453389

-- Define the lines m and n, and planes α and β
variables (m n : Type) (α β : Type)

-- Define the parallel and perpendicular relations
variables (parallel : Type → Type → Prop) (perpendicular : Type → Type → Prop)

-- Assume the conditions of problem D
variables (h1 : perpendicular m α) (h2 : parallel n β) (h3 : parallel α β)

-- The proof problem statement: Prove that under these assumptions, m is perpendicular to n
theorem problem_D : perpendicular m n :=
sorry

end problem_D_l453_453389


namespace simplify_trig_expression_l453_453943

theorem simplify_trig_expression (x : ℝ) :
  (sin x + (3 * sin x - 4 * (sin x)^3)) / (1 + cos x + (4 * (cos x)^3 - 3 * cos x)) =
  4 * sin x * (cos x)^2 / (1 - 4 * (cos x)^2) :=
by
  sorry

end simplify_trig_expression_l453_453943


namespace card_sorting_problem_l453_453130

open List

-- Definition of the radix sort as described in the problem
noncomputable def radixSort (l : List ℕ) : List ℕ :=
  let l1 := l.groupBy (λ n, n % 10)
                 |> join |> List.reverse.groupBy (λ n, (n / 10) % 10)
  l1 |> join |> List.reverse.groupBy (λ n, n / 100) |> join

theorem card_sorting_problem :
  let cards := (List.range' 100 900)
  let sorted_cards := radixSort cards
  sorted_cards = cards.sort (≤) :=
by
  sorry

end card_sorting_problem_l453_453130


namespace quadratic_non_real_roots_l453_453445

theorem quadratic_non_real_roots (b : ℝ) : 
  let a : ℝ := 1 
  let c : ℝ := 16 in
  (b^2 - 4 * a * c < 0) ↔ (-8 < b ∧ b < 8) :=
sorry

end quadratic_non_real_roots_l453_453445


namespace expr_equals_five_thirds_l453_453711

noncomputable def expr : ℝ := 
  (3 / 2)^(-1 / 3) - (1 / 3) * (-7 / 6)^0 + 8^(1 / 4) * (2^(1 / 4)) - real.sqrt ((-2 / 3)^(2 / 3))

theorem expr_equals_five_thirds : expr = 5 / 3 :=
by
  sorry

end expr_equals_five_thirds_l453_453711


namespace LCM_other_factor_l453_453953

theorem LCM_other_factor (A B X : ℕ) (HCF : ℕ) (hHCF: HCF = 42) 
  (hA: A = 504) 
  (hLCM1: LCM A B = (A * B) / HCF) 
  (hLCM2: LCM A B = HCF * X) : 
  X = 12 := 
by 
  sorry

end LCM_other_factor_l453_453953


namespace diameter_of_other_base_l453_453962

theorem diameter_of_other_base (R r : ℝ) (h : ℝ) 
  (h_diameter_of_base : R = 50) 
  (h_diameter_increase : R * 1.21 = 60.5) 
  (h_volume_increase : ∀ V : ℝ, V' = 1.21 * V 
    → V' = (π * h / 3) * ((R * 1.21)^2 + R * 1.21 * r + r^2)) 
    (h_initial_volume : V = (π * h / 3) * (R^2 + R * r + r^2))
  : 2 * r = 110 := 
by 
  have h_final_volume : (π * h / 3) * ((R * 1.21)^2 + R * 1.21 * r + r^2) = 1.21 * V :=
    by apply h_volume_increase (π * h / 3) * (R^2 + R * r + r^2), sorry
  have h_equiv : (R * 1.21)^2 + R * 1.21 * r + r^2 = 1.21 * (R^2 + R * r + r^2) := sorry
  have h_solve : r = 55, by sorry
  have h_diameter_other_base : 2 * r = 110 := 
    by rw h_solve, sorry
  exact h_diameter_other_base

end diameter_of_other_base_l453_453962


namespace cnc_processing_time_l453_453457

theorem cnc_processing_time :
  (∃ (hours: ℕ), 3 * (960 / hours) = 960 / 3) → 1 * (400 / 5) = 400 / 1 :=
by
  sorry

end cnc_processing_time_l453_453457


namespace travel_within_three_roads_l453_453872

theorem travel_within_three_roads (n : ℕ) (m : ℕ) (cities : Finset ℕ) 
  (outgoingRoads : ℕ → Finset ℕ) (incomingRoads : ℕ → Finset ℕ) :
  n = 101 →
  (∀ c, c ∈ cities → (outgoingRoads c).card = 40) →
  (∀ c, c ∈ cities → (incomingRoads c).card = 40) →
  (∀ c₁ c₂, c₁ ∈ cities → c₂ ∈ cities → outgoingRoads c₁ ≠ outgoingRoads c₂) →
  (∀ c₁ c₂, c₁ ∈ cities → c₂ ∈ cities → incomingRoads c₁ ≠ incomingRoads c₂) →
  (∀ c₁ c₂, c₁ ∈ cities → c₂ ∈ cities → ∃ p, p.length ≤ 3 ∧ path p c₁ c₂) :=
begin
  sorry
end

end travel_within_three_roads_l453_453872


namespace min_a_for_decreasing_f_l453_453177

theorem min_a_for_decreasing_f {a : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1 - a / (2 * Real.sqrt x) ≤ 0) →
  a ≥ 4 :=
sorry

end min_a_for_decreasing_f_l453_453177


namespace cylinder_surface_area_l453_453072

theorem cylinder_surface_area
  (l : ℝ) (r : ℝ) (unfolded_square_side : ℝ) (base_circumference : ℝ)
  (hl : unfolded_square_side = 2 * π)
  (hl_gen : l = 2 * π)
  (hc : base_circumference = 2 * π)
  (hr : r = 1) :
  2 * π * r * (r + l) = 2 * π + 4 * π^2 :=
by
  sorry

end cylinder_surface_area_l453_453072


namespace quadratic_solution_1_quadratic_solution_2_l453_453568

theorem quadratic_solution_1 (x : ℝ) : x^2 - 8 * x + 12 = 0 ↔ x = 2 ∨ x = 6 := 
by
  sorry

theorem quadratic_solution_2 (x : ℝ) : (x - 3)^2 = 2 * x * (x - 3) ↔ x = 3 ∨ x = -3 := 
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l453_453568


namespace shopkeeper_discount_l453_453296

theorem shopkeeper_discount
  (CP LP SP : ℝ)
  (H_CP : CP = 100)
  (H_LP : LP = CP + 0.4 * CP)
  (H_SP : SP = CP + 0.33 * CP)
  (discount_percent : ℝ) :
  discount_percent = ((LP - SP) / LP) * 100 → discount_percent = 5 :=
by
  sorry

end shopkeeper_discount_l453_453296


namespace problem_statement_l453_453420

noncomputable def a_sequence (n : ℕ) : ℕ :=
if n = 1 then 1 else 2 * a_sequence (n - 1)

def b_sequence (n : ℕ) : ℕ :=
nat.log2 (a_sequence n) + 1

def sum_until (f : ℕ → ℕ) (n : ℕ) : ℕ :=
nat.sum (finset.range n.succ) f

def S_n (n : ℕ) : ℕ :=
sum_until (λ i, a_sequence i * b_sequence i) n

theorem problem_statement (n : ℕ) (h₀ : 1 ≤ n) :
  (a_sequence n = 2^(n - 1)) ∧ (S_n n = 1 + (n - 1) * 2^n) := by
  sorry

end problem_statement_l453_453420


namespace Nigella_earnings_l453_453543

def cost_A : ℕ := 60000
def cost_B : ℕ := 3 * cost_A
def cost_C : ℕ := 2 * cost_A - 110000

def total_sales : ℕ := cost_A + cost_B + cost_C

def commission (sales : ℕ) : ℕ := sales * 2 / 100

def base_salary : ℕ := 3000

def total_earnings : ℕ := base_salary + commission(total_sales)

theorem Nigella_earnings :
  total_earnings = 8000 := by
  sorry

end Nigella_earnings_l453_453543


namespace sqrt_area_inequality_l453_453611

theorem sqrt_area_inequality (ABCD S: ℤ × ℤ) (P P1: ℝ)
  (integer_coords: ∃ (A B C D S : ℤ × ℤ), is_lattice_point A ∧ is_lattice_point B ∧ is_lattice_point C ∧ is_lattice_point D ∧ is_lattice_point S) 
  (area_ABCD: is_area_of_quadrilateral ABCD P)
  (area_ABS: is_area_of_triangle_by_diagonals ABCD S P1):
  √P ≥ √P1 + (√2) / 2 
  := sorry

end sqrt_area_inequality_l453_453611


namespace angle_bisectors_intersect_l453_453145

theorem angle_bisectors_intersect 
  (A B C D : Type) 
  [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup D] 
  (quad : Quadrilateral A B C D)
  {M : A} {N : C}
  (h1 : M ∈ (angle_bisector quad.A quad.C (quad.diagonal BD))):
  (angle_bisectors_intersect quad.B quad.D quad.AC N) :=
sorry

end angle_bisectors_intersect_l453_453145


namespace squares_to_square_3x3_4x4_l453_453416

theorem squares_to_square_3x3_4x4 :
  ∃ parts_3x3 parts_4x4,
  (parts_3x3.piece1.size + parts_3x3.piece2.size + 
    parts_4x4.piece1.size + parts_4x4.piece2.size = (5 * 5)) :=
by
  sorry

end squares_to_square_3x3_4x4_l453_453416


namespace minimum_set_size_l453_453244

theorem minimum_set_size (n : ℕ) :
  (2 * n + 1) ≥ 11 :=
begin
  have h1 : 7 * (2 * n + 1) ≤ 15 * n + 2,
  sorry,
  have h2 : 14 * n + 7 ≤ 15 * n + 2,
  sorry,
  have h3 : n ≥ 5,
  sorry,
  show 2 * n + 1 ≥ 11,
  from calc
    2 * n + 1 = 2 * 5 + 1 : by linarith
          ... ≥ 11 : by linarith,
end

end minimum_set_size_l453_453244


namespace intersection_solution_l453_453737

theorem intersection_solution (x : ℝ) (y : ℝ) (h₁ : y = 12 / (x^2 + 6)) (h₂ : x + y = 4) : x = 2 :=
by
  sorry

end intersection_solution_l453_453737


namespace max_beauty_convex_board_l453_453506

theorem max_beauty_convex_board (n : ℕ) (hn : 0 < n) :
  ∃ beauty : ℕ, (beauty = (n^3 - n) / 3) ∧ 
  (∀ board : array (array ℕ n) n, -- Representing the n x n board
    ∀ i j : ℕ, i < n → j < n →
    (board[i][j] = 1 → (i = 0 ∨ board[i-1][j] = 1) ∧ (j = 0 ∨ board[i][j-1] = 1)) → -- Convex condition
    let
      a := λ r : ℕ, ∑ c in finset.range(n), if board[r][c] = 1 then 1 else 0,
      beauty_row := ∑ r in finset.range(n), a r * (n - a r),
      beauty_col := ∑ c in finset.range(n), 
                     ∑ i in finset.range(n), 
                     (if board[i][c] = 1 then (n - i) * (a i - a (i+1)) else 0),
      total_beauty := beauty_row + beauty_col
    in
    total_beauty ≤ beauty) := sorry

end max_beauty_convex_board_l453_453506


namespace sin_bounds_l453_453266

theorem sin_bounds {x : ℝ} (h : 0 < x ∧ x < 1) : x - x^2 < sin x ∧ sin x < x :=
sorry

end sin_bounds_l453_453266


namespace basketball_game_scores_l453_453856

theorem basketball_game_scores
  (a r b d : ℕ)
  (h_falcon_geometric_sequence : a > 0 ∧ r > 1)
  (h_eagle_arithmetic_sequence : b > 0 ∧ d > 0)
  (h_falcon_quarter_score : ∀ (n : ℕ), n < 4 → a * r^n ≤ 36)
  (h_eagle_quarter_score : ∀ (n : ℕ), n < 4 → b + n * d ≤ 36)
  (h_falcon_total_score : a * (1 + r + r^2 + r^3) ≤ 120)
  (h_eagle_total_score : 4 * b + 6 * d ≤ 120)
  (h_falcon_wins_by_2 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 2) :
  (a + a * r) + (b + (b + d)) = 52 :=
begin
  sorry
end

end basketball_game_scores_l453_453856


namespace order_of_ten_mod_thirteen_l453_453327

theorem order_of_ten_mod_thirteen : ∃ k : ℕ, k > 0 ∧ (10^k ≡ 1 [MOD 13]) ∧ ∀ m : ℕ, 0 < m < k → ¬ (10^m ≡ 1 [MOD 13]) := 
by
  sorry

end order_of_ten_mod_thirteen_l453_453327


namespace intersection_is_correct_l453_453050

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_is_correct : A ∩ B = {2, 4, 8} :=
by
  sorry

end intersection_is_correct_l453_453050


namespace proposition_truth_value_l453_453012

-- Definitions of the propositions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (|x|) ≥ 1

-- The proof problem statement
theorem proposition_truth_value : (p ∧ q) ∧ ¬ (¬p ∧ q) ∧ ¬ (p ∧ ¬q) ∧ ¬ (¬ (p ∨ q)) :=
by
  sorry

end proposition_truth_value_l453_453012


namespace gcd_lcm_sum_l453_453908

open Nat

theorem gcd_lcm_sum (A B : ℕ) (hA : A = gcd (gcd 6 18) 24) (hB : B = lcm (lcm 6 18) 24):
  A + B = 78 :=
by
  sorry

end gcd_lcm_sum_l453_453908


namespace remaining_cooking_time_l453_453319

-- Define the recommended cooking time in minutes and the time already cooked in seconds
def recommended_cooking_time_min := 5
def time_cooked_seconds := 45

-- Define the conversion from minutes to seconds
def minutes_to_seconds (min : Nat) : Nat := min * 60

-- Define the total recommended cooking time in seconds
def total_recommended_cooking_time_seconds := minutes_to_seconds recommended_cooking_time_min

-- State the theorem to prove the remaining cooking time
theorem remaining_cooking_time :
  (total_recommended_cooking_time_seconds - time_cooked_seconds) = 255 :=
by
  sorry

end remaining_cooking_time_l453_453319


namespace solve_real_eq_l453_453157

theorem solve_real_eq (x : ℝ) : 
  (∑ n in Finset.range 2017, x / ((n + 1) * (n + 2))) = 2017 → 
  x = 2018 := 
by
  intros h
  sorry

end solve_real_eq_l453_453157


namespace cherry_trees_leaves_l453_453687

-- Define the original number of trees
def original_num_trees : ℕ := 7

-- Define the number of trees actually planted
def actual_num_trees : ℕ := 2 * original_num_trees

-- Define the number of leaves each tree drops
def leaves_per_tree : ℕ := 100

-- Define the total number of leaves that fall
def total_leaves : ℕ := actual_num_trees * leaves_per_tree

-- Theorem statement for the problem
theorem cherry_trees_leaves : total_leaves = 1400 := by
  sorry

end cherry_trees_leaves_l453_453687


namespace a_2017_correct_l453_453035

theorem a_2017_correct (a : ℕ → ℤ) (b : ℕ → ℝ)
  (h1 : ∀ n, b n = real.exp (a n * real.log 2))
  (h2 : b 1 * b 2 * b 3 = 64)
  (h3 : b 1 + b 2 + b 3 = 14)
  (h4 : ∃ d < 0, ∀ n, a (n + 1) = a n + d) :
  a 2017 = -2013 :=
sorry

end a_2017_correct_l453_453035


namespace problem_find_m_find_range_of_a_l453_453405

theorem problem_find_m (m : ℝ) :
  (2 * m^2 - m = 1) ∧ (2 * m + 3).natAbs % 2 = 0 → m = -1 / 2 :=
sorry

theorem find_range_of_a (a : ℝ) (m : ℝ) :
  m = -1 / 2 ∧ (1 < a) ∧ (a < 2) ∧ (3 / 2 < a) →
  (a-1)^m < (2 * a - 3)^m :=
sorry

end problem_find_m_find_range_of_a_l453_453405


namespace general_formula_sum_formula_l453_453124

noncomputable def a : ℕ → ℕ
| 0     := 3
| (n+1) := 2 * a n + n * 2^(n+1) + 3^n

theorem general_formula (n : ℕ) : a (n+1) = 2^n * (n^2 - n) + 3^(n+1) := by
  sorry

noncomputable def S : ℕ → ℕ
| 0     := a 0
| (n+1) := S n + a (n+1)

theorem sum_formula (n : ℕ) : S n = -(n-2) * 2^(n+1) + (n-1) * n * 2^n - 4 := by
  sorry

end general_formula_sum_formula_l453_453124


namespace verify_extrema_l453_453100

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * x^4 - 2 * x^3 + (11 / 2) * x^2 - 6 * x + (9 / 4)

theorem verify_extrema :
  f 1 = 0 ∧ f 2 = 1 ∧ f 3 = 0 := by
  sorry

end verify_extrema_l453_453100


namespace number_of_sets_l453_453268

/-- Define the sets for the problem --/
def set1 : set ℕ := {2, 3}
def set2 : set ℕ := {1, 2, 3, 4, 5}

/-- The main theorem stating the number of possible sets M under the given conditions --/
theorem number_of_sets (M : set ℕ) (h1 : set1 ⊂ M) (h2 : M ⊂ set2) : 
  (∃ n, n = 6) :=
begin
  use 6,
  sorry
end

end number_of_sets_l453_453268


namespace non_real_roots_of_quadratic_l453_453443

theorem non_real_roots_of_quadratic (b : ℝ) : 
  (¬ ∃ x1 x2 : ℝ, x1^2 + bx1 + 16 = 0 ∧ x2^2 + bx2 + 16 = 0 ∧ x1 = x2) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
by {
  sorry
}

end non_real_roots_of_quadratic_l453_453443


namespace three_digit_powers_of_three_l453_453823

theorem three_digit_powers_of_three : 
  {n : ℤ | 100 ≤ 3^n ∧ 3^n ≤ 999}.finset.card = 2 :=
by
  sorry

end three_digit_powers_of_three_l453_453823


namespace non_real_roots_of_quadratic_l453_453444

theorem non_real_roots_of_quadratic (b : ℝ) : 
  (¬ ∃ x1 x2 : ℝ, x1^2 + bx1 + 16 = 0 ∧ x2^2 + bx2 + 16 = 0 ∧ x1 = x2) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
by {
  sorry
}

end non_real_roots_of_quadratic_l453_453444


namespace sector_area_l453_453464

theorem sector_area (arc_length radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 2) : 
  (1/2) * arc_length * radius = 2 :=
by
  -- sorry placeholder for proof
  sorry

end sector_area_l453_453464


namespace hot_dogs_remainder_l453_453456

theorem hot_dogs_remainder : 25197641 % 6 = 1 :=
by
  sorry

end hot_dogs_remainder_l453_453456


namespace min_value_l453_453771

theorem min_value (x : ℝ) (h : 0 < x) : x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 :=
sorry

end min_value_l453_453771


namespace decimal_to_base_five_correct_l453_453344

theorem decimal_to_base_five_correct : 
  ∃ (d0 d1 d2 d3 : ℕ), 256 = d3 * 5^3 + d2 * 5^2 + d1 * 5^1 + d0 * 5^0 ∧ 
                          d3 = 2 ∧ d2 = 0 ∧ d1 = 1 ∧ d0 = 1 :=
by sorry

end decimal_to_base_five_correct_l453_453344


namespace num_three_digit_powers_of_three_l453_453825

theorem num_three_digit_powers_of_three : 
  ∃ n1 n2 : ℕ, 100 ≤ 3^n1 ∧ 3^n1 ≤ 999 ∧ 100 ≤ 3^n2 ∧ 3^n2 ≤ 999 ∧ n1 ≠ n2 ∧ 
  (∀ n : ℕ, 100 ≤ 3^n ∧ 3^n ≤ 999 → n = n1 ∨ n = n2) :=
sorry

end num_three_digit_powers_of_three_l453_453825


namespace range_of_set_D_l453_453563

-- Define the set D containing all prime numbers between 10 and 25
def D : Set ℕ := { n | nat.prime n ∧ 10 < n ∧ n < 25 }

-- Prove that the range of set D is 12
theorem range_of_set_D : (set.range D).max' - (set.range D).min' = 12 := 
  sorry

end range_of_set_D_l453_453563


namespace inscribed_quadrilateral_exists_l453_453575

-- Definitions to set up the problem
variable (V : Point) (A B C D : Point)
variable (quadrilateral : ConvexQuadrilateral A B C D)
variable (pyramid : Pyramid V quadrilateral)

-- The condition is that A, B, C, and D form a convex quadrilateral, and V is the vertex of the pyramid
def is_convex_quadrilateral (A B C D : Point) : Prop := ConvexQuadrilateral A B C D

-- The proof goal: it is always possible to have a cross-section of the pyramid that does not intersect the base and is an inscribed quadrilateral.
theorem inscribed_quadrilateral_exists (cvx_quad : is_convex_quadrilateral A B C D) (pyr : pyramid V quadrilateral) : 
  ∃ E F G H, (InscribedQuadrilateral E F G H) ∧ (SectionDoesNotIntersectBase E F G H) := 
sorry

end inscribed_quadrilateral_exists_l453_453575


namespace dice_probability_l453_453891

theorem dice_probability :
  let outcomes := 1000 in
  let favorable := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) in
  let probability := favorable / outcomes in
  probability = 11 / 200 :=
by
  sorry

end dice_probability_l453_453891


namespace derivative_f_l453_453378

noncomputable def f (x : ℝ) : ℝ := x ^ (-5) + 3 * Real.sin x

theorem derivative_f (x : ℝ) : 
  (deriv f x) = -5 * x ^ (-6) + 3 * Real.cos x := 
by
  sorry

end derivative_f_l453_453378


namespace minimum_set_size_l453_453245

theorem minimum_set_size (n : ℕ) :
  (2 * n + 1) ≥ 11 :=
begin
  have h1 : 7 * (2 * n + 1) ≤ 15 * n + 2,
  sorry,
  have h2 : 14 * n + 7 ≤ 15 * n + 2,
  sorry,
  have h3 : n ≥ 5,
  sorry,
  show 2 * n + 1 ≥ 11,
  from calc
    2 * n + 1 = 2 * 5 + 1 : by linarith
          ... ≥ 11 : by linarith,
end

end minimum_set_size_l453_453245


namespace sector_area_proof_l453_453036

-- Defining the conditions and the formula for the area of a sector
def slant_height (r : ℝ) := r = 6
def central_angle (θ : ℝ) := θ = 120
def sector_area (θ r : ℝ) := (θ / 360) * π * (r ^ 2)

theorem sector_area_proof (r θ : ℝ) (h₁ : slant_height r) (h₂ : central_angle θ): 
  sector_area θ r = 12 * π :=
by
  rw [slant_height] at h₁
  rw [central_angle] at h₂
  rw [h₁, h₂]
  unfold sector_area
  norm_num
  ring
  done

end sector_area_proof_l453_453036


namespace integral_eq_k_l453_453047

theorem integral_eq_k (k : ℝ) (h : ∫ x in 0..1, k * x + 1 = k) : k = 2 :=
by
  sorry

end integral_eq_k_l453_453047


namespace sum_of_possible_values_of_x_l453_453774

def number_list := [10, 4, 7, 4, 6, 4]

def mean (x : ℝ) : ℝ := (35 + x) / 7

def mode : ℝ := 4

def median (x : ℝ) : ℝ :=
  if x ≤ 4 then 4
  else if 4 < x ∧ x < 6 then x
  else if 6 ≤ x ∧ x ≤ 7 then 6
  else 0 -- This case should not occur in the given problem

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem sum_of_possible_values_of_x :
  ∑ x in (set_of (λ x : ℝ, is_arithmetic_progression mode (median x) (mean x))), x = 63 / 13 :=
sorry

end sum_of_possible_values_of_x_l453_453774


namespace correct_tan_squared_sum_l453_453260

variable {α β γ : ℝ}
noncomputable def tetrahedron_dihedral_angles (P A B C : Prop) : Prop := 
  ∃ (α β γ : ℝ), 
  ∃ (T₁ T₂ T₃ T₄ : ℝ), 
  (T₁ = (∠ APB) = π / 2 ∧ 
   (T₂ = ∠ BPC) = π / 2 ∧ 
   (T₃ = ∠ CPA) = π / 2)

theorem correct_tan_squared_sum 
  (h : tetrahedron_dihedral_angles P A B C)
  : (tan α) ^ 2 + (tan β) ^ 2 + (tan γ) ^ 2 ≥ 0 := sorry

end correct_tan_squared_sum_l453_453260


namespace max_men_with_all_items_l453_453076

theorem max_men_with_all_items (total_men married men_with_TV men_with_radio men_with_AC men_with_car men_with_smartphone : ℕ) 
  (H_married : married = 2300) 
  (H_TV : men_with_TV = 2100) 
  (H_radio : men_with_radio = 2600) 
  (H_AC : men_with_AC = 1800) 
  (H_car : men_with_car = 2500) 
  (H_smartphone : men_with_smartphone = 2200) : 
  ∃ m, m ≤ married ∧ m ≤ men_with_TV ∧ m ≤ men_with_radio ∧ m ≤ men_with_AC ∧ m ≤ men_with_car ∧ m ≤ men_with_smartphone ∧ m = 1800 := 
  sorry

end max_men_with_all_items_l453_453076


namespace arrange_ascending_l453_453373

noncomputable def a : ℝ := 0.8 ^ 0.7
noncomputable def b : ℝ := 0.8 ^ 0.9
noncomputable def c : ℝ := 1.2 ^ 0.8

theorem arrange_ascending : b < a ∧ a < c :=
by
  sorry

end arrange_ascending_l453_453373


namespace find_a_6_l453_453602

-- Define the sequence {a_n} and partial sum {S_n}
def sequence_a : ℕ → ℕ
| 0     := 1
| (n+1) := 3 * (Finset.range (n+1)).sum sequence_a

noncomputable def partial_sum (n : ℕ) : ℕ :=
Finset.range (n+1).sum sequence_a

theorem find_a_6 : sequence_a 5 = 3 * 4^4 := by
  -- use sorry to skip the proof
  sorry

end find_a_6_l453_453602


namespace probability_point_in_triangle_l453_453675

theorem probability_point_in_triangle : 
  let area_rect := 6 * 2
  let area_triangle := (1.5 * 1.5) / 2
  let prob := area_triangle / area_rect
  prob = (3 / 32) := 
by {
  let area_rect := 6 * 2
  let area_triangle := (1.5 * 1.5) / 2
  let prob := area_triangle / area_rect
  have h : prob = 3 / 32, by sorry,
  exact h
}

end probability_point_in_triangle_l453_453675


namespace quadratics_roots_l453_453906

theorem quadratics_roots (m n : ℝ) (r₁ r₂ : ℝ) 
  (h₁ : r₁^2 - m * r₁ + n = 0) (h₂ : r₂^2 - m * r₂ + n = 0) 
  (p q : ℝ) (h₃ : (r₁^2 - r₂^2)^2 + p * (r₁^2 - r₂^2) + q = 0) :
  p = 0 ∧ q = -m^4 + 4 * m^2 * n := 
sorry

end quadratics_roots_l453_453906


namespace flower_bed_planting_methods_l453_453488

theorem flower_bed_planting_methods : 
  ∃ ways : ℕ, 
  (ways = 84) ∧ (ways = number_of_planting_methods 4 4 (λ i j, i ≠ j)) :=
by
  sorry

end flower_bed_planting_methods_l453_453488


namespace find_expression_intervals_of_monotonicity_l453_453043

noncomputable def f (x : ℝ) : ℝ := x^3 + 4 * x^2 - 9 * x + 1

theorem find_expression :
  (∀ (x : ℝ), f(0) = 1) ∧
  let f' := 3 * x ^ 2 + 8 * x - 9,
      a := (2 : ℝ), k := 2 - (f 1 : ℝ)  in 
  (∀ (x : ℝ), 2 * x - k = 0 ∧ (f' 1) = a) →
  (∀ (x : ℝ), f 1 = 4) ∧ f = λ x, x^3 + 4 * x^2 - 9 * x + 1 :=
by
  sorry

theorem intervals_of_monotonicity :
  let f' := 3 * x^2 + 8 * x - 9 in
  (∀ (x : ℝ),
    (f' x > 0 → x > (-4 + sqrt 41) / 3 ∨ x < (-4 - sqrt 41) / 3) ∧
    (f' x < 0 → (-4 - sqrt 41) / 3 < x ∧ x < (-4 + sqrt 41) / 3)) :=
by
  sorry

end find_expression_intervals_of_monotonicity_l453_453043


namespace value_of_f_minus_m_l453_453968

noncomputable def f (a b l x : ℝ) : ℝ := a * x ^ 3 + b * x + l

theorem value_of_f_minus_m {a b l m : ℝ} (h : f a b l m = 2) : f a b l (-m) = 0 :=
by
  let g := λ x: ℝ, a * x ^ 3 + b * x
  have hg : ∀ x, g (-x) = -g x, by
    intro x
    simp [g, pow_succ]
    ring
  have hgm : g m = 1, from eq_sub_of_add_eq h.symm
  calc
    f a b l (-m) = g (-m) + l : by simp [f, g]
    ... = -g m + l : by rw [hg m]
    ... = -1 + l : by rw [hgm]
    ... = 0 : by simp [h.symm]

end value_of_f_minus_m_l453_453968


namespace stratified_sampling_l453_453286

theorem stratified_sampling (total_students boys girls sample_size x y : ℕ)
  (h1 : total_students = 8)
  (h2 : boys = 6)
  (h3 : girls = 2)
  (h4 : sample_size = 4)
  (h5 : x + y = sample_size)
  (h6 : (x : ℚ) / boys = 3 / 4)
  (h7 : (y : ℚ) / girls = 1 / 4) :
  x = 3 ∧ y = 1 :=
by
  sorry

end stratified_sampling_l453_453286


namespace field_length_is_112_l453_453592

-- Define the conditions
def is_pond_side_length : ℕ := 8
def pond_area : ℕ := is_pond_side_length * is_pond_side_length
def pond_to_field_area_ratio : ℚ := 1 / 98

-- Define the field properties
def field_area (w l : ℕ) : ℕ := w * l

-- Expressing the condition given length is double the width
def length_double_width (w l : ℕ) : Prop := l = 2 * w

-- Equating the areas based on the ratio given
def area_condition (w l : ℕ) : Prop := pond_area = pond_to_field_area_ratio * field_area w l

-- The main theorem
theorem field_length_is_112 : ∃ w l, length_double_width w l ∧ area_condition w l ∧ l = 112 := by
  sorry

end field_length_is_112_l453_453592


namespace derek_joe_ratio_l453_453104

theorem derek_joe_ratio (D J T : ℝ) (h0 : J = 23) (h1 : T = 30) (h2 : T = (1/3 : ℝ) * D + 16) :
  D / J = 42 / 23 :=
by
  sorry

end derek_joe_ratio_l453_453104


namespace proof_problem_l453_453016

variables (R : Type*) [Real R]

def p : Prop := ∃ x : R, Real.sin x < 1
def q : Prop := ∀ x : R, Real.exp (abs x) ≥ 1

theorem proof_problem : p ∧ q := 
by 
  sorry

end proof_problem_l453_453016


namespace sum_limit_eq_one_limit_sum_is_one_l453_453368

theorem sum_limit_eq_one (n : ℕ) :
  (∑ k in Finset.range n, (k + 1) / (k + 2)!) ∈ Nat :=
by
  sorry

theorem limit_sum_is_one : 
  filter.Tendsto (λ n, ∑ k in Finset.range n, (k + 1) / (k + 2)!) filter.at_top _ :=
by
  sorry

end sum_limit_eq_one_limit_sum_is_one_l453_453368


namespace weight_of_mixture_is_correct_l453_453197

noncomputable def weight_mixture_kg (weight_per_liter_a weight_per_liter_b ratio_a ratio_b total_volume_liters : ℕ) : ℝ :=
  let volume_a := (ratio_a * total_volume_liters) / (ratio_a + ratio_b)
  let volume_b := (ratio_b * total_volume_liters) / (ratio_a + ratio_b)
  let weight_a := (volume_a * weight_per_liter_a) 
  let weight_b := (volume_b * weight_per_liter_b) 
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_mixture_kg 900 700 3 2 4 = 3.280 := 
sorry

end weight_of_mixture_is_correct_l453_453197


namespace laran_weekly_profit_l453_453499

-- Conditions as definitions
def posters_per_day := 5
def large_posters_per_day := 2
def small_posters_per_day := posters_per_day - large_posters_per_day

def large_poster_sale_price := 10
def large_poster_tax := 0.1 * large_poster_sale_price
def large_poster_total_price := large_poster_sale_price + large_poster_tax
def large_poster_cost := 5

def small_poster_sale_price := 6
def small_poster_tax := 0.15 * small_poster_sale_price
def small_poster_total_price := small_poster_sale_price + small_poster_tax
def small_poster_cost := 3

def fixed_weekly_expense := 20

-- Total revenue and cost calculations
def daily_revenue_large := large_posters_per_day * large_poster_total_price
def daily_cost_large := large_posters_per_day * large_poster_cost
def weekly_revenue_large := daily_revenue_large * 5
def weekly_cost_large := daily_cost_large * 5

def daily_revenue_small := small_posters_per_day * small_poster_total_price
def daily_cost_small := small_posters_per_day * small_poster_cost
def weekly_revenue_small := daily_revenue_small * 5
def weekly_cost_small := daily_cost_small * 5

def total_weekly_revenue := weekly_revenue_large + weekly_revenue_small
def total_weekly_cost := weekly_cost_large + weekly_cost_small
def gross_profit_before_expenses := total_weekly_revenue - total_weekly_cost
def net_profit := gross_profit_before_expenses - fixed_weekly_expense

-- Proof problem
theorem laran_weekly_profit : net_profit = 98.50 :=
by
  -- Proof goes here
  sorry

end laran_weekly_profit_l453_453499


namespace alpha_beta_sum_cos_alpha_minus_beta_l453_453028

noncomputable def tan_alpha_root1 := 2
noncomputable def tan_alpha_root2 := 3

theorem alpha_beta_sum (α β : ℝ) (h1 : tan α = 2 ∨ tan β = 2)
                        (h2 : tan α = 3 ∨ tan β = 3)
                        (h3 : 0 < α ∧ α < π ∧ 0 < β ∧ β < π) :
  α + β = 3 * π / 4 :=
sorry

theorem cos_alpha_minus_beta (α β : ℝ) (h1 : tan α = 2 ∨ tan β = 2)
                             (h2 : tan α = 3 ∨ tan β = 3)
                             (h3 : 0 < α ∧ α < π ∧ 0 < β ∧ β < π) :
  cos (α - β) = 7 * real.sqrt 2 / 10 :=
sorry

end alpha_beta_sum_cos_alpha_minus_beta_l453_453028


namespace ratio_female_to_male_l453_453954

theorem ratio_female_to_male
  (a b c : ℕ)
  (ha : a = 60)
  (hb : b = 80)
  (hc : c = 65) :
  f / m = 1 / 3 := 
by
  sorry

end ratio_female_to_male_l453_453954


namespace min_value_of_2x_plus_y_l453_453769

theorem min_value_of_2x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 8 / y = 2) : 2 * x + y ≥ 7 :=
sorry

end min_value_of_2x_plus_y_l453_453769


namespace measure_of_angle_ADC_l453_453978

theorem measure_of_angle_ADC
  (ABC : ∀ A B C : Type, triangle A B C)
  (m_ABC : ABC.angle A B C = 60)
  (AD_bisects_BAC : ∀ A B C D : Type, bisect A D B C .angle (A B C) = true)
  (DC_bisects_BCA : ∀ A B C D : Type, bisect D C A B .angle (C A B) = true)
  (angle_relation : ∀ A B C D : Type, angle(D, C, A) = 2 * angle(B, C, D))
  (x y : ℝ) : angle(D, C, A) = 100 := sorry

end measure_of_angle_ADC_l453_453978


namespace maximum_volume_smaller_pyramid_l453_453379

-- Define the conditions and the given data
variable {P A B C D O O' A' B' C' D' : Type} -- Types for points in space
variable [Inhabited P] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variable [Inhabited O] [Inhabited O'] [Inhabited A'] [Inhabited B'] [Inhabited C'] [Inhabited D']
variable (AB_length PO_height : ℝ)

-- Conditions of the problem
def conditions : Prop :=
  AB_length = 2 ∧ PO_height = 3 ∧
  ∃ O' : Type, O' ∈ PO ∧
  ∃⟨A', B', C', D'⟩ : Type,
    plane_parallel_to_base O' A' B' C' D' ∧
    intersects_PA_PB_PC_PD P A B C D A' B' C' D'

-- Prove maximum volume of the smaller pyramid
theorem maximum_volume_smaller_pyramid (h₁ : conditions AB_length PO_height) :
  ∃ (max_volume : ℝ), max_volume = 16 / 27 :=
sorry

end maximum_volume_smaller_pyramid_l453_453379


namespace decode_message_l453_453963

-- Defining the groups and their digit mappings
def group1 : List Nat := [0, 5, 6]
def group2 : List Nat := [1, 3, 4, 7, 8]
def singletonGroup : List Nat := [2, 9]  -- Singleton groups either [2] and [9] or combined

-- Define the letter mapping for simplicity
def russianAlphabet : List Char := (
  ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 
   'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 
   'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
)

-- Define the number mapping based on permutation and ordering logic
def letterToNumberMapping : List (Char × Int) := 
  [('Н', 8), ('А', 7), ('У', 3), ('К', 1), ('А', 6)]

-- Define the encoded message and expected decoded message
def encodedMessage : Nat := 873146507381
def decodedMessage : String := "НАУКА"

-- The equivalent proof problem in Lean 4
theorem decode_message : 
  ∀ (encoded : Nat) (decoded : String), 
  (encoded = 873146507381) → 
  (decoded = "НАУКА") → 
  decodedMessage = "НАУКА" := 
begin
  -- Establish the given encoded message
  intros,
  have h₁ : encoded = encodedMessage, from rfl,
  have h₂ : decoded = decodedMessage, from rfl,
  rw [h₁, h₂],
end

end decode_message_l453_453963


namespace probability_of_region_C_l453_453274

theorem probability_of_region_C (P_A P_B P_C : ℚ) (hA : P_A = 1/3) (hB : P_B = 1/2) (hTotal : P_A + P_B + P_C = 1) : P_C = 1/6 := 
by
  sorry

end probability_of_region_C_l453_453274


namespace ellipse_property_l453_453787

noncomputable section

def foci_of_ellipse (a b : ℝ) : ℝ := (sqrt (b^2 - a^2)).abs

theorem ellipse_property
  (A B F1 F2 : ℝ → ℝ)
  (ellipse : ∀ x y : ℝ, (x^2 / 25) + (y^2 / 169) = 1)
  (passes_through_f1 : ∀ points : ℝ → ℝ, points F1)
  (intersection_points : ℝ → ℝ)
  (F2A F2B : ℝ)
  (hyp1 : dist F2 A + dist F2 B = 30)
  (hyp2 : dist A F1 + dist A F2 = 26)
  (hyp3 : dist B F1 + dist B F2 = 26) :
  dist A B = 22 :=
sorry

end ellipse_property_l453_453787


namespace verify_minor_premise_l453_453877

-- Define the premises and conclusion
def major_premise : Prop := ∀ r : Type, (rectangle r) → (parallelogram r)
def minor_premise : Prop := ∀ t : Type, ¬ (parallelogram t)
def conclusion : Prop := ∀ t : Type, ¬ (rectangle t)

-- Theorem to prove the minor premise
theorem verify_minor_premise (h_major : major_premise) (h_minor : minor_premise) (h_conclusion: conclusion) :
  ¬ (parallelogram (triangle : Type)) := 
begin
  apply h_minor,
  sorry
end

end verify_minor_premise_l453_453877


namespace total_fish_l453_453233

-- Definitions based on the problem conditions
def will_catch_catfish : ℕ := 16
def will_catch_eels : ℕ := 10
def henry_trout_per_catfish : ℕ := 3
def fraction_to_return : ℚ := 1/2

-- Calculation of required quantities
def will_total_fish : ℕ := will_catch_catfish + will_catch_eels
def henry_target_trout : ℕ := henry_trout_per_catfish * will_catch_catfish
def henry_return_trout : ℚ := fraction_to_return * henry_target_trout
def henry_kept_trout : ℤ := henry_target_trout -  henry_return_trout.to_nat

-- Goal statement to prove
theorem total_fish (will_catch_catfish = 16) (will_catch_eels = 10) 
  (henry_trout_per_catfish = 3) (fraction_to_return = 1/2) :
  will_total_fish + henry_kept_trout = 50 :=
by
  sorry

end total_fish_l453_453233


namespace square_diagonals_equal_l453_453054

theorem square_diagonals_equal
  (h1 : ∀ (P : Type) [parallelogram P], diagonals_equal P)
  (h2 : ∀ (S : Type) [square S], parallelogram S) :
  ∀ (S : Type) [square S], diagonals_equal S :=
by
  sorry

end square_diagonals_equal_l453_453054


namespace cheyenne_earnings_l453_453337

def total_pots := 80
def cracked_fraction := (2 : ℕ) / 5
def price_per_pot := 40

def cracked_pots (total_pots : ℕ) (fraction : ℚ) : ℕ :=
  (fraction * total_pots).toNat

def remaining_pots (total_pots : ℕ) (cracked_pots : ℕ) : ℕ :=
  total_pots - cracked_pots

def total_earnings (remaining_pots : ℕ) (price_per_pot : ℕ) : ℕ :=
  remaining_pots * price_per_pot

theorem cheyenne_earnings :
  total_earnings (remaining_pots total_pots (cracked_pots total_pots cracked_fraction)) price_per_pot = 1920 :=
by
  sorry

end cheyenne_earnings_l453_453337


namespace proposition_p_and_q_l453_453020

-- Define the propositions as per given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (|x|) ≥ 1

-- The theorem to be proved
theorem proposition_p_and_q : p ∧ q :=
by
  sorry

end proposition_p_and_q_l453_453020


namespace f_96_value_l453_453763

noncomputable def f : ℕ → ℕ :=
sorry

axiom condition_1 (a b : ℕ) : 
  f (a * b) = f a + f b

axiom condition_2 (n : ℕ) (hp : Nat.Prime n) (hlt : 10 < n) : 
  f n = 0

axiom condition_3 : 
  f 1 < f 243 ∧ f 243 < f 2 ∧ f 2 < 11

axiom condition_4 : 
  f 2106 < 11

theorem f_96_value :
  f 96 = 31 :=
sorry

end f_96_value_l453_453763


namespace number_of_permutations_l453_453059

open Nat

theorem number_of_permutations (digits : Finset ℕ)
  (h_digits : digits = {3, 3, 3, 5, 5, 7, 7, 7, 2}) :
  (∃ n : ℕ, n = 1120 ∧ digits = {3, 3, 3, 5, 5, 7, 7, 7, 2} ∧ n = (factorial 8) / ((factorial 3) * (factorial 3))) :=
begin
  sorry
end

end number_of_permutations_l453_453059


namespace three_digit_numbers_form_3_pow_l453_453830

theorem three_digit_numbers_form_3_pow (n : ℤ) : 
  ∃! (n : ℤ), 100 ≤ 3^n ∧ 3^n ≤ 999 :=
by {
  use [5, 6],
  sorry
}

end three_digit_numbers_form_3_pow_l453_453830


namespace nat_nums_div2_not3_count_l453_453699

theorem nat_nums_div2_not3_count : 
  (finset.range 101).filter (λ n, n % 2 = 0 ∧ n % 3 ≠ 0).card = 34 := 
by sorry

end nat_nums_div2_not3_count_l453_453699


namespace largest_third_altitude_l453_453859

theorem largest_third_altitude 
  (A B C : Point) 
  (AB AC BC: ℝ) 
  (h1 : Altitude A B = 18) 
  (h2 : Altitude A C = 6) 
  (triangle_ABC : is_right_triangle A B C 90) 
  : Altitude B C < 6 := 
  sorry

end largest_third_altitude_l453_453859


namespace seating_arrangements_l453_453867

theorem seating_arrangements (n : ℕ) (fixed_group_size : ℕ) (total_people : ℕ) :
  n = 10 → fixed_group_size = 4 → total_people = 10 → 
  factorial total_people - factorial (total_people - fixed_group_size + 1) * factorial fixed_group_size = 3507840 :=
by {
  intros h_total h_group h_people,
  rw [h_total, h_group, h_people],
  simp only [factorial],
  sorry
}

end seating_arrangements_l453_453867


namespace non_real_roots_of_quadratic_l453_453441

theorem non_real_roots_of_quadratic (b : ℝ) : 
  (¬ ∃ x1 x2 : ℝ, x1^2 + bx1 + 16 = 0 ∧ x2^2 + bx2 + 16 = 0 ∧ x1 = x2) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
by {
  sorry
}

end non_real_roots_of_quadratic_l453_453441


namespace max_jogs_l453_453316

theorem max_jogs (x y z : ℕ) (h1 : 3 * x + 2 * y + 8 * z = 60) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  z ≤ 6 := 
sorry

end max_jogs_l453_453316


namespace jackson_sandwiches_l453_453886

noncomputable def total_sandwiches (weeks : ℕ) (miss_wed : ℕ) (miss_fri : ℕ) : ℕ :=
  let total_wednesdays := weeks - miss_wed
  let total_fridays := weeks - miss_fri
  total_wednesdays + total_fridays

theorem jackson_sandwiches : total_sandwiches 36 1 2 = 69 := by
  sorry

end jackson_sandwiches_l453_453886


namespace constant_term_of_expanded_expression_l453_453735

noncomputable def expanded_expression (x : ℝ) : ℝ :=
  (x + 4 / x - 4) ^ 3

theorem constant_term_of_expanded_expression (x : ℝ) (h : x ≠ 0) :
  constant_term (expanded_expression x) = -160 := 
by {
  sorry
}

end constant_term_of_expanded_expression_l453_453735


namespace number_of_female_officers_is_382_l453_453548

noncomputable def F : ℝ := 
  let total_on_duty := 210
  let ratio_male_female := 3 / 2
  let percent_female_on_duty := 22 / 100
  let female_on_duty := total_on_duty * (2 / (3 + 2))
  let total_females := female_on_duty / percent_female_on_duty
  total_females

theorem number_of_female_officers_is_382 : F = 382 := 
by
  sorry

end number_of_female_officers_is_382_l453_453548


namespace factorial_identity_l453_453342

theorem factorial_identity : (10.factorial / (7.factorial * 3.factorial)) * 2.factorial = 240 := by
  sorry

end factorial_identity_l453_453342


namespace boys_same_first_last_name_l453_453612

theorem boys_same_first_last_name :
  ∃ i j : ℕ, i ≠ j ∧ ∃ f l : Fin 14 → ℕ,
    ∀ k : Fin 14, f k ∈ {0, 1, 2, 3, 4, 5, 6} ∧ l k ∈ {0, 1, 2, 3, 4, 5, 6} ∧
    (f i = f j ∧ l i = l j) := sorry

end boys_same_first_last_name_l453_453612


namespace nine_point_circle_circumcircle_perpendiculars_concurrent_l453_453110

-- Define the excenters of the triangle
variables {α : Type*} [EuclideanSpace α]
variables {A B C I_A I_B I_C P Q R : α}

-- Given conditions
axioms 
(I_A_excenter : is_excenter A I_A)
(I_B_excenter : is_excenter B I_B)
(I_C_excenter : is_excenter C I_C)

-- First part: The nine-point circle of triangle I_A I_B I_C is the circumcircle of triangle ABC
theorem nine_point_circle_circumcircle (h₁: Π I_A I_B I_C A B C, nine_point_circle (triangle I_A I_B I_C) = circumcircle (triangle A B C)) :
  nine_point_circle (triangle I_A I_B I_C) = circumcircle (triangle A B C) := 
by sorry

-- Second part: Perpendiculars from I_A, I_B, I_C to the sides BC, CA, AB respectively are concurrent
theorem perpendiculars_concurrent (h₂ : Π I_A I_B I_C, concurrent (perpendicular I_A BC, perpendicular I_B CA, perpendicular I_C AB)) :
  concurrent (perpendicular I_A BC, perpendicular I_B CA, perpendicular I_C AB) := 
by sorry

end nine_point_circle_circumcircle_perpendiculars_concurrent_l453_453110


namespace lineup_and_selection_count_l453_453976

noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

def binomial_coefficient (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lineup_and_selection_count :
  let members := 9
  let captains := 2
  factorial members * binomial_coefficient members captains = 13063680 :=
by
  sorry

end lineup_and_selection_count_l453_453976


namespace ratio_of_tetrahedrons_volume_l453_453478

theorem ratio_of_tetrahedrons_volume (d R s s' V_ratio m n : ℕ) (h1 : d = 4)
  (h2 : R = 2)
  (h3 : s = 4 * R / Real.sqrt 6)
  (h4 : s' = s / Real.sqrt 8)
  (h5 : V_ratio = (s' / s) ^ 3)
  (hm : m = 1)
  (hn : n = 32)
  (h_ratio : V_ratio = m / n) :
  m + n = 33 :=
by
  sorry

end ratio_of_tetrahedrons_volume_l453_453478


namespace circle_radius_l453_453366

theorem circle_radius (x y : ℝ) : 
    let equation : ℝ := x^2 - 8 * x + y^2 + 6 * y + 1
    in equation = 0 → ∃ r : ℝ, r = 2 * Real.sqrt 6 :=
sorry

end circle_radius_l453_453366


namespace price_of_other_frisbees_l453_453679

theorem price_of_other_frisbees :
  ∃ F3 Fx Px : ℕ, F3 + Fx = 60 ∧ 3 * F3 + Px * Fx = 204 ∧ Fx ≥ 24 ∧ Px = 4 := 
by
  sorry

end price_of_other_frisbees_l453_453679


namespace accelerations_l453_453276

open Real

namespace Problem

variables (m M g : ℝ) (a1 a2 : ℝ)

theorem accelerations (mass_condition : 4 * m + M ≠ 0):
  (a1 = 2 * ((2 * m + M) * g) / (4 * m + M)) ∧
  (a2 = ((2 * m + M) * g) / (4 * m + M)) :=
sorry

end Problem

end accelerations_l453_453276


namespace find_angle_and_area_l453_453849

def triangle_data : Type :=
  {a b c A B C : ℝ}

theorem find_angle_and_area
  (a b c A B C : ℝ)
  (h1 : c = real.sqrt 3)
  (h2 : b = 1)
  (h3 : B = 30) :
  (C = 60 ∧ (1/2 * b * c = real.sqrt 3 / 2))
  ∨ (C = 120 ∧ (1/2 * b * c * (real.sin $ A) = real.sqrt 3 / 4)) :=
sorry

end find_angle_and_area_l453_453849


namespace simplify_correct_l453_453156

open Polynomial

noncomputable def simplify_expression (y : ℚ) : Polynomial ℚ :=
  (3 * (Polynomial.C y) + 2) * (2 * (Polynomial.C y)^12 + 3 * (Polynomial.C y)^11 - (Polynomial.C y)^9 - (Polynomial.C y)^8)

theorem simplify_correct (y : ℚ) : 
  simplify_expression y = 6 * (Polynomial.C y)^13 + 13 * (Polynomial.C y)^12 + 6 * (Polynomial.C y)^11 - 3 * (Polynomial.C y)^10 - 5 * (Polynomial.C y)^9 - 2 * (Polynomial.C y)^8 := 
by 
  simp [simplify_expression]
  sorry

end simplify_correct_l453_453156


namespace part1_part2_l453_453263

open Real

-- Part 1
theorem part1 {x : ℝ} (h1 : 0 < x) (h2 : x < 1) : (x - x^2 < sin x ∧ sin x < x) :=
sorry

-- Part 2
theorem part2 {a : ℝ} 
  (h1 : ∀ x, f x = cos (a * x) - log (1 - x ^ 2)) 
  (h2 : ∃ ε > 0, ∀ x, |x| < ε → f x ≤ f 0) 
  : a < -sqrt 2 ∨ a > sqrt 2 :=
sorry

end part1_part2_l453_453263


namespace max_subsets_intersection_at_most_two_l453_453535

open Finset

theorem max_subsets_intersection_at_most_two (A : Finset ℕ) (B : Finset (Finset ℕ)) :
  (A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) →
  (∀ B ∈ B, B ≠ ∅) →
  (∀ B1 B2 ∈ B, B1 ≠ B2 → (B1 ∩ B2).card ≤ 2) →
  B.card ≤ 175 :=
by
  intros hA hNonEmpty hIntersection
  -- Proof here
  sorry

end max_subsets_intersection_at_most_two_l453_453535


namespace ellipse_problem_l453_453534

theorem ellipse_problem {a b : ℝ} (ha : a > b) (hb : b > 0)
  (pass_through : ∀ x y, (x = 0 ∧ y = 4 → x^2 / a^2 + y^2 / b^2 = 1))
  (eccentricity : ∀ x, x = (3/5) → x = real.sqrt(1 - (b^2 / a^2))) :
  (a = 5 ∧ b = 4 ∧ 
  ∀ x y, (x^2 / 25 + y^2 / 16 = 1) ∧ 
  ∀ x y, (x = 3 ∧ y = 0 → y = (3/4) * (x - 3)) →
  ∃ x1 x2 y1 y2, 
  ((4*x1^2 / 25 + (3*x1 - 3)^2 / 16 = 1) ∧ (4*x2^2 / 25 + (3*x2 - 3)^2 / 16 = 1)) ∧ 
  (1 = (x1 + x2) / 2 ∧ -9/4 = (y1 + y2) / 2)) :=
by sorry

end ellipse_problem_l453_453534


namespace digit_415_of_13_div_19_is_6_l453_453215

theorem digit_415_of_13_div_19_is_6 :
  let d : ℚ := 13 / 19,
      cycle : List ℕ := [6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7, 3] in
  d.decimalPeriod = 18 ∧
  (415 % 18) = 1 →
  (cycle.nth (415 % 18 - 1)).getOrElse 0 = 6 :=
by
  intro d cycle h
  sorry

end digit_415_of_13_div_19_is_6_l453_453215


namespace correct_answer_l453_453386

def p1 : Prop := ∀ (x : ℝ), real.exp (real.ln 2 * x) - real.exp (-real.ln 2 * x) > 0
def p2 : Prop := ∀ (x : ℝ), real.exp (real.ln 2 * x) + real.exp (-real.ln 2 * x) < 0

def q1 : Prop := p1 ∨ p2
def q2 : Prop := p1 ∧ p2
def q3 : Prop := ¬p1 ∨ p2
def q4 : Prop := p1 ∨ ¬p2

theorem correct_answer : q1 ∧ q4 :=
by {
  sorry  -- We can skip the proof as per the requirements
}

end correct_answer_l453_453386


namespace distinct_prime_factors_of_2310_l453_453719

-- Define the number 2310
def n : ℕ := 2310

-- Define the statement that proves 2310 has exactly 5 distinct prime factors
theorem distinct_prime_factors_of_2310 : (nat.factors n).nodup ∧ (nat.factors n).length = 5 :=
by
  sorry

end distinct_prime_factors_of_2310_l453_453719


namespace probability_equivalence_l453_453803

-- Definitions for the conditions:
def total_products : ℕ := 7
def genuine_products : ℕ := 4
def defective_products : ℕ := 3

-- Function to return the probability of selecting a genuine product on the second draw, given first is defective
def probability_genuine_given_defective : ℚ := 
  (defective_products / total_products) * (genuine_products / (total_products - 1))

-- The theorem we need to prove:
theorem probability_equivalence :
  probability_genuine_given_defective = 2 / 3 :=
by
  sorry -- Proof placeholder

end probability_equivalence_l453_453803


namespace part1_part2_l453_453095

section 
variables {A B C D : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]

variables {AB AC BC AD : ℝ}
variable {angle_A : ℝ}
variable {sin_angle_ACB : ℝ}
variable {length_AD : ℝ}

-- Conditions
variable h1 : AB = 3
variable h2 : AC = 1
variable h3 : angle_A = real.pi / 3 -- 60 degrees in radians
variable h4 : D = midpoint B C -- Definition of midpoint

-- Proof statements
theorem part1 (h1 : AB = 3) (h2 : AC = 1) (h3 : angle_A = real.pi / 3) :
  sin_angle_ACB = 3 * (real.sqrt 21) / 14 :=
sorry

theorem part2 (h1 : AB = 3) (h2 : AC = 1) (h4 : D = midpoint B C) :
  length_AD = real.sqrt (13) / 2 :=
sorry
end

end part1_part2_l453_453095


namespace maximum_value_l453_453517

noncomputable def conditions (m n t : ℝ) : Prop :=
  -- m, n, t are positive real numbers
  (0 < m) ∧ (0 < n) ∧ (0 < t) ∧
  -- Equation condition
  (m^2 - 3 * m * n + 4 * n^2 - t = 0)

noncomputable def minimum_u (m n t : ℝ) : Prop :=
  -- Minimum value condition for t / mn
  (t / (m * n) = 1)

theorem maximum_value (m n t : ℝ) (h1 : conditions m n t) (h2 : minimum_u m n t) :
  -- Proving the maximum value of m + 2n - t
  (m + 2 * n - t) = 2 :=
sorry

end maximum_value_l453_453517


namespace functions_of_the_same_cluster_l453_453847

def f1 (x : ℝ) : ℝ := Real.sin x * Real.cos x
def f2 (x : ℝ) : ℝ := (Real.sqrt 2) * Real.sin (2 * x) + 2
def f3 (x : ℝ) : ℝ := 2 * Real.sin (x + (Real.pi / 4))
def f4 (x : ℝ) : ℝ := Real.sin x - (Real.sqrt 3) * Real.cos x

theorem functions_of_the_same_cluster : 
  ∃ d : ℝ, ∀ x : ℝ, f3 x = f4 (x + d) :=
sorry

end functions_of_the_same_cluster_l453_453847


namespace probability_two_tails_is_3_over_8_l453_453462

open ProbabilityTheory

def probability_two_tails_four_coins_toss (prob_head prob_tail : ℚ) (indep : ∀ (x y : ℕ), x ≠ y → Prob_event x * Prob_event y = Prob_event (x ∩ y)) : ℚ :=
  -- Tossing four coins independently
  let prob_sequence := (prob_head + prob_tail) ^ 4 in
  let specific_sequence_prob := (prob_tail ^ 2) * (prob_head ^ 2) in
  let total_ways := Nat.choose 4 2 in
  total_ways * specific_sequence_prob / prob_sequence

theorem probability_two_tails_is_3_over_8 :
  probability_two_tails_four_coins_toss (1 / 2) (1 / 2) (λ x y hxy, sorry) = 3 / 8 :=
by
  sorry

end probability_two_tails_is_3_over_8_l453_453462


namespace composition_of_rotations_l453_453557

variable {A B : Type} [MetricSpace A] (C : Type) [MetricSpace B]

def is_rotation (center : A) (angle : ℝ) := sorry
def is_translation (start : A) (end : A) (distance : ℝ) := sorry

theorem composition_of_rotations (α β : ℝ) (x y : A) :
  (α + β) % 360 ≠ 0 → is_rotation C (α + β) ∨ (α + β) % 360 = 0 → is_translation x y (α + β) :=
sorry

end composition_of_rotations_l453_453557


namespace find_amount_l453_453758

noncomputable def x : ℝ := 14 / 15

theorem find_amount :
  let amount := 14 - x in
  15 * x = 14 → amount = 14 - x :=
by
  sorry

end find_amount_l453_453758


namespace non_real_roots_b_range_l453_453439

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l453_453439


namespace option_a_option_d_l453_453772

variables {z : ℂ} {a b : ℝ}

/-- Definition of complex number and its conjugate -/
def conj (z : ℂ) : ℂ := complex.conj z

/-- Definition of absolute value squared for a complex number -/
def abs_sq (z : ℂ) : ℝ := complex.norm_sq z

/-- Theorem 1: The product of a complex number and its conjugate is equal to the square of its magnitude -/
theorem option_a (z : ℂ) : z * conj(z) = abs_sq(z) := by
  sorry

/-- Theorem 2: The sum of magnitudes of a complex number and its conjugate is greater than or equal to the magnitude of their sum -/
theorem option_d (z : ℂ) : complex.abs(z) + complex.abs(conj(z)) ≥ complex.abs(z + conj(z)) := by
  sorry

end option_a_option_d_l453_453772


namespace brand_z_percentage_final_l453_453315

theorem brand_z_percentage_final (initial_qty : ℝ) 
  (h0 : initial_qty = 1) 
  (filling_1 : ∀ (x : ℝ), x = initial_qty * (1 - 3/4) → initial_qty = x + (initial_qty * 3/4)) 
  (filling_2 : ∀ (x : ℝ), x = (initial_qty * (1 - 2/3)) * (1 - 3/4) → let new_qty := x + ( (initial_qty * (1 - (1/3))) * (1/12)) in initial_qty = new_qty + (initial_qty - new_qty))
  (filling_3 : ∀ (x : ℝ), x = initial_qty * (1 - 5/8) → initial_qty = (initial_qty * (3/8)) +  (initial_qty * (5/8))) 
  (filling_4 : ∀ (x : ℝ), x = (initial_qty * (3/40)) * (1 - 5/8) → let new_qty := x + ((initial_qty * (1 - (1/5))) * (3/40)) in initial_qty = new_qty + (initial_qty - new_qty))
  (filling_5 : ∀ (x : ℝ), x = initial_qty * (1 - 7/8) → initial_qty = (initial_qty * (1/8)) +  (initial_qty * (7/8))) : 
  (initial_qty * (1/8) / initial_qty) * 100 = 12.5 := by
  sorry

end brand_z_percentage_final_l453_453315


namespace constant_term_binomial_expansion_l453_453871

theorem constant_term_binomial_expansion :
  ∃ (r : ℕ), (8 - 2 * r = 0) ∧ Nat.choose 8 r = 70 := by
  sorry

end constant_term_binomial_expansion_l453_453871


namespace find_function_l453_453358

open Nat

theorem find_function (f : ℕ+ → ℕ+) (h_incr : ∀ n, f (n + 1) > f n) (h_f_f_eq : ∀ n, f (f n) = 3 * n) :
  ∀ n, f n = 3 * n - 2 :=
sorry

end find_function_l453_453358


namespace fraction_difference_eq_l453_453522

theorem fraction_difference_eq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end fraction_difference_eq_l453_453522


namespace sum_of_three_consecutive_odd_integers_l453_453222

-- Define the variables and conditions
variables (a : ℤ) (h1 : (a + (a + 4) = 100))

-- Define the statement that needs to be proved
theorem sum_of_three_consecutive_odd_integers (ha : a = 48) : a + (a + 2) + (a + 4) = 150 := by
  sorry

end sum_of_three_consecutive_odd_integers_l453_453222


namespace prob_factor_less_than_10_l453_453218

theorem prob_factor_less_than_10 (n : ℕ) 
  (prime_fact : n = 2 * 3^2 * 5) 
  (total_factors : 12) 
  (factors_less_than_10 : {1, 2, 3, 5, 6, 9}) : 
  (6 / 12 : ℚ) = (1 / 2 : ℚ) := 
sorry

end prob_factor_less_than_10_l453_453218


namespace sum_of_square_roots_of_subtriangles_l453_453306

variable (T T1 T2 T3 : ℝ)

/-- Prove that the sum of the square roots of the areas of the smaller triangles equals
    the square root of the area of the original triangle. -/
theorem sum_of_square_roots_of_subtriangles 
    (hT1 : T1 > 0) (hT2 : T2 > 0) (hT3 : T3 > 0) 
    (h_sum_areas : T1 + T2 + T3 = T) :
    sqrt T1 + sqrt T2 + sqrt T3 = sqrt T := 
sorry

end sum_of_square_roots_of_subtriangles_l453_453306


namespace consecutive_integers_average_and_product_l453_453999

theorem consecutive_integers_average_and_product (n m : ℤ) (hnm : n ≤ m) 
  (h1 : (n + m) / 2 = 20) 
  (h2 : n * m = 391) :  m - n + 1 = 7 :=
  sorry

end consecutive_integers_average_and_product_l453_453999


namespace cherry_tree_leaves_l453_453684

theorem cherry_tree_leaves (original_plan : ℕ) (multiplier : ℕ) (leaves_per_tree : ℕ) 
  (h1 : original_plan = 7) (h2 : multiplier = 2) (h3 : leaves_per_tree = 100) : 
  (original_plan * multiplier * leaves_per_tree = 1400) :=
by
  sorry

end cherry_tree_leaves_l453_453684


namespace distance_between_lines_l453_453810

/-
We want to prove that the distance between the lines l₁ and l₂ is equal to 3 * sqrt 5 / 10.
The conditions specify that l₁ is given by the equation 2x + y + 1 = 0 and l₂ by 4x + 2y - 1 = 0.
-/

theorem distance_between_lines (A B : ℝ) (C₁ C₂ : ℝ) 
  (hl₁ : 2 * A + B + 1 = 0) (hl₂ : 4 * A + 2 * B - 1 = 0) : 
  abs ((-1) - 2) / real.sqrt (4^2 + 2^2) = 3 * real.sqrt 5 / 10 := 
by
  sorry

# Exit the theorem environment without providing the actual proof.

end distance_between_lines_l453_453810


namespace probability_of_two_sunny_days_l453_453175

def prob_two_sunny_days (prob_sunny prob_rain : ℚ) (days : ℕ) : ℚ :=
  (days.choose 2) * (prob_sunny^2 * prob_rain^(days-2))

theorem probability_of_two_sunny_days :
  prob_two_sunny_days (2/5) (3/5) 3 = 36/125 :=
by 
  sorry

end probability_of_two_sunny_days_l453_453175


namespace powers_of_3_but_not_9_l453_453425

theorem powers_of_3_but_not_9 : 
  {n : ℕ | 0 < n ∧ n < 500000 ∧
    ∃ k : ℕ, n = 3^k ∧ ¬ ∃ m : ℕ, n = 9^m }.to_finset.card = 6 :=
by sorry

end powers_of_3_but_not_9_l453_453425


namespace sixteenth_answer_is_three_l453_453613

theorem sixteenth_answer_is_three (total_members : ℕ)
  (answers_1 answers_2 answers_3 : ℕ) 
  (h_total : total_members = 16) 
  (h_answers_1 : answers_1 = 6) 
  (h_answers_2 : answers_2 = 6) 
  (h_answers_3 : answers_3 = 3) :
  ∃ answer : ℕ, answer = 3 ∧ (answers_1 + answers_2 + answers_3 + 1 = total_members) :=
sorry

end sixteenth_answer_is_three_l453_453613


namespace right_triangle_area_and_semicircle_length_l453_453294

theorem right_triangle_area_and_semicircle_length 
(A B O : Point)
(hypotenuse_div : dist A O = 20 ∧ dist B O = 15)
(r : ℝ)
(hypotenuse_divides : dist A B = 35)
(radius_diameter_relation : 2 * r = 30) :
(∃ (area : ℝ) (arc_length : ℝ), 
 area = 294 ∧ arc_length = 12 * Real.pi) :=
sorry

end right_triangle_area_and_semicircle_length_l453_453294


namespace people_who_like_both_l453_453077

variable (U : Type) (A C : Finset U)
variable [Finite U]
variable (h1 : ∀ x, x ∈ A ∪ C) -- |A ∪ C| = 100
variable (h2 : A.card = 50) -- |A| = 50
variable (h3 : C.card = 70) -- |C| = 70

theorem people_who_like_both (h_total : (A ∪ C).card = 100) : (A ∩ C).card = 20 := by
  sorry

end people_who_like_both_l453_453077


namespace divides_iff_l453_453905

open Int

theorem divides_iff (n m : ℤ) : (9 ∣ (2 * n + 5 * m)) ↔ (9 ∣ (5 * n + 8 * m)) := 
sorry

end divides_iff_l453_453905


namespace find_m_l453_453838

theorem find_m (x y m : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x + m * y = 5) : m = 3 := 
by
  sorry

end find_m_l453_453838


namespace necessary_but_not_sufficient_condition_for_geometric_sequence_l453_453387

theorem necessary_but_not_sufficient_condition_for_geometric_sequence
  (a b c : ℝ) :
  (∃ (r : ℝ), a = r * b ∧ b = r * c) → (b^2 = a * c) ∧ ¬((b^2 = a * c) → (∃ (r : ℝ), a = r * b ∧ b = r * c)) := 
by
  sorry

end necessary_but_not_sufficient_condition_for_geometric_sequence_l453_453387


namespace non_real_roots_interval_l453_453451

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l453_453451


namespace triangle_side_length_l453_453491

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) (tan_B : ℝ) (hA : A = π / 3)
  (hTanB : tan_B = 1 / 2) (hAB : AB = 2 * sqrt 3 + 1) :
  BC = sqrt 15 :=
by
  sorry

end triangle_side_length_l453_453491


namespace sphere_views_identical_l453_453720

-- Define the geometric shape as a type
inductive GeometricShape
| sphere
| cube
| other (name : String)

-- Define a function to get the view of a sphere
def view (s : GeometricShape) (direction : String) : String :=
  match s with
  | GeometricShape.sphere => "circle"
  | GeometricShape.cube => "square"
  | GeometricShape.other _ => "unknown"

-- The theorem to prove that a sphere has identical front, top, and side views
theorem sphere_views_identical :
  ∀ (direction1 direction2 : String), view GeometricShape.sphere direction1 = view GeometricShape.sphere direction2 :=
by
  intros direction1 direction2
  sorry

end sphere_views_identical_l453_453720


namespace proof_example_l453_453007

open Real

theorem proof_example (p q : Prop) :
  (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  have p : ∃ x : ℝ, sin x < 1 := ⟨0, by norm_num⟩
  have q : ∀ x : ℝ, exp (abs x) ≥ 1 := by
    intro x
    have : abs x ≥ 0 := abs_nonneg x
    exact exp_pos (abs x)
  exact ⟨p, q⟩

end proof_example_l453_453007


namespace intersection_points_of_line_l453_453169

noncomputable def coordinates_of_intersections : Prop := 
  let line (x : ℝ) := x + 3 in
  let A := (-3, 0) in
  let B := (0, 3) in
  ∃ x_intersect y_intersect, 
    (line x_intersect = 0) ∧ (y_intersect = line 0) ∧ (A = (x_intersect, 0)) ∧ (B = (0, y_intersect))

theorem intersection_points_of_line : coordinates_of_intersections :=
by
  -- proof goes here
  sorry

end intersection_points_of_line_l453_453169


namespace harry_speed_on_friday_l453_453055

theorem harry_speed_on_friday :
  ∀ (speed_monday speed_tuesday_to_thursday speed_friday : ℝ)
  (ran_50_percent_faster ran_60_percent_faster: ℝ),
  speed_monday = 10 →
  ran_50_percent_faster = 0.50 →
  ran_60_percent_faster = 0.60 →
  speed_tuesday_to_thursday = speed_monday + (ran_50_percent_faster * speed_monday) →
  speed_friday = speed_tuesday_to_thursday + (ran_60_percent_faster * speed_tuesday_to_thursday) →
  speed_friday = 24 := by {
  intros speed_monday speed_tuesday_to_thursday speed_friday ran_50_percent_faster ran_60_percent_faster,
  intros h0 h1 h2 h3 h4,
  rw [h0, h1, h2, h3, h4],
  norm_num,
}

end harry_speed_on_friday_l453_453055


namespace x_varies_as_z_l453_453841

variable {x y z : ℝ}
variable (k j : ℝ)
variable (h1 : x = k * y^3)
variable (h2 : y = j * z^(1/3))

theorem x_varies_as_z (m : ℝ) (h3 : m = k * j^3) : x = m * z := by
  sorry

end x_varies_as_z_l453_453841


namespace arithmetic_sequence_tenth_term_l453_453610

theorem arithmetic_sequence_tenth_term (a d : ℤ) 
  (h1 : a + 2 * d = 23) 
  (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
by 
  -- proof goes here
  sorry

end arithmetic_sequence_tenth_term_l453_453610


namespace exists_index_with_inequality_l453_453214

-- Given 100 distinct real numbers, extended periodically.
noncomputable def sequence_extends_periodically (a: ℕ → ℝ) :=
  ∀ i,  
    (i > 100 → a(i) = a(i % 100)) ∧ -- periodic extension
    (i ≤ 100 → a(i) ≠ a(j) for distinct i and j, 1 ≤ i, j ≤ 100) -- distinctness

theorem exists_index_with_inequality (a : ℕ → ℝ) (h_distinct : sequence_extends_periodically a) :
  ∃ i, 1 ≤ i ∧ i ≤ 100 ∧ a i + a (i + 3) > a (i + 1) + a (i + 2) :=
sorry

end exists_index_with_inequality_l453_453214


namespace lcm_factor_of_hcf_and_larger_number_l453_453973

theorem lcm_factor_of_hcf_and_larger_number (A B : ℕ) (hcf : ℕ) (hlarger : A = 450) (hhcf : hcf = 30) (hwrel : A % hcf = 0) : ∃ x y, x = 15 ∧ (A * B = hcf * x * y) :=
by
  sorry

end lcm_factor_of_hcf_and_larger_number_l453_453973
