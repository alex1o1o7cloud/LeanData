import Mathlib

namespace function_decreasing_odd_function_m_zero_l1700_170006

-- First part: Prove that the function is decreasing
theorem function_decreasing (m : ℝ) (x1 x2 : ℝ) (h : x1 < x2) :
    let f := fun x => -2 * x + m
    f x1 > f x2 :=
by
    sorry

-- Second part: Find the value of m when the function is odd
theorem odd_function_m_zero (m : ℝ) :
    (∀ x : ℝ, let f := fun x => -2 * x + m
              f (-x) = -f x) → m = 0 :=
by
    sorry

end function_decreasing_odd_function_m_zero_l1700_170006


namespace mike_taller_than_mark_l1700_170071

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end mike_taller_than_mark_l1700_170071


namespace distance_from_home_to_school_l1700_170082

variable (t : ℕ) (D : ℕ)

-- conditions
def condition1 := 60 * (t - 10) = D
def condition2 := 50 * (t + 4) = D

-- the mathematical equivalent proof problem: proving the distance is 4200 given conditions
theorem distance_from_home_to_school :
  (∃ t, condition1 t 4200 ∧ condition2 t 4200) :=
  sorry

end distance_from_home_to_school_l1700_170082


namespace triangle_count_with_perimeter_11_l1700_170017

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l1700_170017


namespace part1_l1700_170074

def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - a^2 - 2*a < 0}
def setB (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x - 2*a ∧ x ≤ 2}

theorem part1 (a : ℝ) (h : a = 3) : setA 3 ∪ setB 3 = Set.Ioo (-6) 5 :=
by
  sorry

end part1_l1700_170074


namespace find_P_x_l1700_170041

noncomputable def P (x : ℝ) : ℝ :=
  (-17 / 3) * x^3 + (68 / 3) * x^2 - (31 / 3) * x - 18

variable (a b c : ℝ)

axiom h1 : a^3 - 4 * a^2 + 2 * a + 3 = 0
axiom h2 : b^3 - 4 * b^2 + 2 * b + 3 = 0
axiom h3 : c^3 - 4 * c^2 + 2 * c + 3 = 0

axiom h4 : P a = b + c
axiom h5 : P b = a + c
axiom h6 : P c = a + b
axiom h7 : a + b + c = 4
axiom h8 : P 4 = -20

theorem find_P_x :
  P x = (-17 / 3) * x^3 + (68 / 3) * x^2 - (31 / 3) * x - 18 := sorry

end find_P_x_l1700_170041


namespace circles_max_ab_l1700_170045

theorem circles_max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ (x y : ℝ), (x + a)^2 + (y - 2)^2 = 1 ∧ (x - b)^2 + (y - 2)^2 = 4) →
  a + b = 3 →
  ab ≤ 9 / 4 := 
  by
  sorry

end circles_max_ab_l1700_170045


namespace concentric_circles_radius_difference_l1700_170095

theorem concentric_circles_radius_difference (r R : ℝ)
  (h : R^2 = 4 * r^2) :
  R - r = r :=
by
  sorry

end concentric_circles_radius_difference_l1700_170095


namespace line_passing_quadrants_l1700_170032

theorem line_passing_quadrants (a k : ℝ) (a_nonzero : a ≠ 0)
  (x1 x2 y1 y2 : ℝ) (hx1 : y1 = a * x1^2 - a) (hx2 : y2 = a * x2^2 - a)
  (hx1_y1 : y1 = k * x1) (hx2_y2 : y2 = k * x2) 
  (sum_x : x1 + x2 < 0) : 
  ∃ (q1 q4 : (ℝ × ℝ)), 
  (q1.1 > 0 ∧ q1.2 > 0 ∧ q1.2 = a * q1.1 + k) ∧ (q4.1 > 0 ∧ q4.2 < 0 ∧ q4.2 = a * q4.1 + k) := 
sorry

end line_passing_quadrants_l1700_170032


namespace trigonometric_expression_value_l1700_170005

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l1700_170005


namespace B_work_rate_l1700_170001

theorem B_work_rate (A B C : ℕ) (combined_work_rate_A_B_C : ℕ)
  (A_work_days B_work_days C_work_days : ℕ)
  (combined_abc : combined_work_rate_A_B_C = 4)
  (a_work_rate : A_work_days = 6)
  (c_work_rate : C_work_days = 36) :
  B = 18 :=
by
  sorry

end B_work_rate_l1700_170001


namespace spring_extension_l1700_170049

theorem spring_extension (A1 A2 : ℝ) (x1 x2 : ℝ) (hA1 : A1 = 29.43) (hx1 : x1 = 0.05) (hA2 : A2 = 9.81) : x2 = 0.029 :=
by 
  sorry

end spring_extension_l1700_170049


namespace intersect_point_l1700_170013

noncomputable def f (x : ℤ) (b : ℤ) : ℤ := 5 * x + b
noncomputable def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 5

theorem intersect_point (a b : ℤ) (h_intersections : (f (-3) b = a ∧ f a b = -3)) : a = -3 :=
by
  sorry

end intersect_point_l1700_170013


namespace gcd_polynomial_eq_one_l1700_170076

theorem gcd_polynomial_eq_one (b : ℤ) (hb : Even b) (hmb : 431 ∣ b) : 
  Int.gcd (8 * b^2 + 63 * b + 143) (4 * b + 17) = 1 := by
  sorry

end gcd_polynomial_eq_one_l1700_170076


namespace symmetric_probability_l1700_170023

-- Definitions based on the problem conditions
def total_points : ℕ := 121
def central_point : ℕ × ℕ := (6, 6)
def remaining_points : ℕ := total_points - 1
def symmetric_points : ℕ := 40

-- Predicate for the probability that line PQ is a line of symmetry
def is_symmetrical_line (p q : (ℕ × ℕ)) : Prop := 
  (q.fst = 11 - p.fst ∧ q.snd = p.snd) ∨
  (q.fst = p.fst ∧ q.snd = 11 - p.snd) ∨
  (q.fst + q.snd = 12) ∨ 
  (q.fst - q.snd = 0)

-- The theorem stating the probability is 1/3
theorem symmetric_probability :
  ∃ (total_points : ℕ) (remaining_points : ℕ) (symmetric_points : ℕ),
    total_points = 121 ∧
    remaining_points = total_points - 1 ∧
    symmetric_points = 40 ∧
    (symmetric_points : ℚ) / (remaining_points : ℚ) = 1 / 3 :=
by
  sorry

end symmetric_probability_l1700_170023


namespace cuboid_height_l1700_170051

-- Define the base area and volume of the cuboid
def base_area : ℝ := 50
def volume : ℝ := 2000

-- Prove that the height is 40 cm given the base area and volume
theorem cuboid_height : volume / base_area = 40 := by
  sorry

end cuboid_height_l1700_170051


namespace minimum_quadratic_value_l1700_170088

theorem minimum_quadratic_value (h : ℝ) (x : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → (x - h)^2 + 1 ≥ 10) ∧ (∃ x, 1 ≤ x ∧ x ≤ 3 ∧ (x - h)^2 + 1 = 10) 
  ↔ h = -2 ∨ h = 6 :=
by
  sorry

end minimum_quadratic_value_l1700_170088


namespace inequality_l1700_170012

theorem inequality (a b c d e p q : ℝ) 
  (h0 : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (h1 : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * ((1 / a) + (1 / b) + (1 / c) + (1 / d) + (1 / e)) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
by
  sorry

end inequality_l1700_170012


namespace faster_runner_l1700_170048

-- Define the speeds of A and B
variables (v_A v_B : ℝ)
-- A's speed as a multiple of B's speed
variables (k : ℝ)

-- A's and B's distances in the race
variables (d_A d_B : ℝ)
-- Distance of the race
variables (distance : ℝ)
-- Head start given to B
variables (head_start : ℝ)

-- The theorem to prove that the factor k is 4 given the conditions
theorem faster_runner (k : ℝ) (v_A v_B : ℝ) (d_A d_B distance head_start : ℝ) :
  v_A = k * v_B ∧ d_B = distance - head_start ∧ d_A = distance ∧ (d_A / v_A) = (d_B / v_B) → k = 4 :=
by
  sorry

end faster_runner_l1700_170048


namespace solve_for_x_l1700_170084

theorem solve_for_x (x : ℝ) : (1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) → x = 10 :=
by
  sorry

end solve_for_x_l1700_170084


namespace hyperbola_eccentricity_l1700_170038

variable {a b : ℝ}
variable (h1 : a > 0) (h2 : b > 0)
variable (h3 : (a : ℝ) / (b : ℝ) = 3)

theorem hyperbola_eccentricity (h1 : a > 0) (h2 : b > 0) (h3 : b / a = 1 / 3) : 
  (Real.sqrt ((a ^ 2 + b ^ 2) / (a ^ 2))) = Real.sqrt 10 := by sorry

end hyperbola_eccentricity_l1700_170038


namespace current_swans_number_l1700_170098

noncomputable def swans_doubling (S : ℕ) : Prop :=
  let S_after_10_years := S * 2^5 -- Doubling every 2 years for 10 years results in multiplying by 2^5
  S_after_10_years = 480

theorem current_swans_number (S : ℕ) (h : swans_doubling S) : S = 15 := by
  sorry

end current_swans_number_l1700_170098


namespace eight_lines_no_parallel_no_concurrent_l1700_170086

-- Define the number of regions into which n lines divide the plane
def regions (n : ℕ) : ℕ :=
if n = 0 then 1
else if n = 1 then 2
else n * (n - 1) / 2 + n + 1

theorem eight_lines_no_parallel_no_concurrent :
  regions 8 = 37 :=
by
  sorry

end eight_lines_no_parallel_no_concurrent_l1700_170086


namespace binomial_12_10_l1700_170092

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l1700_170092


namespace smallest_number_is_51_l1700_170026

-- Definitions based on conditions
def conditions (x y : ℕ) : Prop :=
  (x + y = 2014) ∧ (∃ n a : ℕ, (x = 100 * n + a) ∧ (a < 100) ∧ (3 * n = y + 6))

-- The proof problem statement that needs to be proven
theorem smallest_number_is_51 :
  ∃ x y : ℕ, conditions x y ∧ min x y = 51 := 
sorry

end smallest_number_is_51_l1700_170026


namespace votes_cast_l1700_170000

theorem votes_cast (V : ℝ) (candidate_votes : ℝ) (rival_margin : ℝ)
  (h1 : candidate_votes = 0.30 * V)
  (h2 : rival_margin = 4000)
  (h3 : 0.30 * V + (0.30 * V + rival_margin) = V) :
  V = 10000 := 
by 
  sorry

end votes_cast_l1700_170000


namespace days_to_complete_l1700_170034

variable {m n : ℕ}

theorem days_to_complete (h : ∀ (m n : ℕ), (m + n) * m = 1) : 
  ∀ (n m : ℕ), (m * (m + n)) / n = m * (m + n) / n :=
by
  sorry

end days_to_complete_l1700_170034


namespace average_grade_of_male_students_l1700_170015

theorem average_grade_of_male_students (M : ℝ) (H1 : (90 : ℝ) = (8 + 32 : ℝ) / 40) 
(H2 : (92 : ℝ) = 32 / 40) :
  M = 82 := 
sorry

end average_grade_of_male_students_l1700_170015


namespace min_value_of_sum_of_squares_l1700_170035

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4.8 :=
sorry

end min_value_of_sum_of_squares_l1700_170035


namespace man_to_son_age_ratio_l1700_170042

-- Definitions based on conditions
variable (son_age : ℕ) (man_age : ℕ)
variable (h1 : man_age = son_age + 18) -- The man is 18 years older than his son
variable (h2 : 2 * (son_age + 2) = man_age + 2) -- In two years, the man's age will be a multiple of the son's age
variable (h3 : son_age = 16) -- The present age of the son is 16

-- Theorem statement to prove the desired ratio
theorem man_to_son_age_ratio (son_age man_age : ℕ) (h1 : man_age = son_age + 18) (h2 : 2 * (son_age + 2) = man_age + 2) (h3 : son_age = 16) :
  (man_age + 2) / (son_age + 2) = 2 :=
by
  sorry

end man_to_son_age_ratio_l1700_170042


namespace ratio_of_areas_l1700_170069

-- Definitions of the perimeters for each region
def perimeter_I : ℕ := 16
def perimeter_II : ℕ := 36
def perimeter_IV : ℕ := 48

-- Define the side lengths based on the given perimeters
def side_length (P : ℕ) : ℕ := P / 4

-- Calculate the areas from the side lengths
def area (s : ℕ) : ℕ := s * s

-- Now we state the theorem
theorem ratio_of_areas : 
  (area (side_length perimeter_II)) / (area (side_length perimeter_IV)) = 9 / 16 := 
by sorry

end ratio_of_areas_l1700_170069


namespace tan_105_eq_neg2_sub_sqrt3_l1700_170039

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l1700_170039


namespace valentino_farm_total_birds_l1700_170079

-- The definitions/conditions from the problem statement
def chickens := 200
def ducks := 2 * chickens
def turkeys := 3 * ducks

-- The theorem to prove the total number of birds
theorem valentino_farm_total_birds : 
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof is not required, so we use 'sorry'
  sorry

end valentino_farm_total_birds_l1700_170079


namespace angle_y_is_80_l1700_170083

def parallel (m n : ℝ) : Prop := sorry

def angle_at_base (θ : ℝ) := θ = 40
def right_angle (θ : ℝ) := θ = 90
def exterior_angle (θ1 θ2 : ℝ) := θ1 + θ2 = 180

theorem angle_y_is_80 (m n : ℝ) (θ1 θ2 θ3 θ_ext : ℝ) :
  parallel m n →
  angle_at_base θ1 →
  right_angle θ2 →
  angle_at_base θ3 →
  exterior_angle θ_ext θ3 →
  θ_ext = 80 := by
  sorry

end angle_y_is_80_l1700_170083


namespace units_digit_of_n_cubed_minus_n_squared_l1700_170021

-- Define n for the purpose of the problem
def n : ℕ := 9867

-- Prove that the units digit of n^3 - n^2 is 4
theorem units_digit_of_n_cubed_minus_n_squared : ∃ d : ℕ, d = (n^3 - n^2) % 10 ∧ d = 4 := by
  sorry

end units_digit_of_n_cubed_minus_n_squared_l1700_170021


namespace area_of_shaded_triangle_l1700_170024

-- Definitions of the conditions
def AC := 4
def BC := 3
def BD := 10
def CD := BD - BC

-- Statement of the proof problem
theorem area_of_shaded_triangle :
  (1 / 2 * CD * AC = 14) := by
  sorry

end area_of_shaded_triangle_l1700_170024


namespace convert_to_base5_l1700_170036

theorem convert_to_base5 : ∀ n : ℕ, n = 1729 → Nat.digits 5 n = [2, 3, 4, 0, 4] :=
by
  intros n hn
  rw [hn]
  -- proof steps can be filled in here
  sorry

end convert_to_base5_l1700_170036


namespace grasshoppers_cannot_return_to_initial_positions_l1700_170065

theorem grasshoppers_cannot_return_to_initial_positions :
  (∀ (a b c : ℕ), a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 → a + b + c ≠ 1985) :=
by
  sorry

end grasshoppers_cannot_return_to_initial_positions_l1700_170065


namespace ages_total_l1700_170055

variable (A B C : ℕ)

theorem ages_total (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : B = 10) : A + B + C = 27 :=
by
  sorry

end ages_total_l1700_170055


namespace polynomial_root_abs_sum_eq_80_l1700_170060

theorem polynomial_root_abs_sum_eq_80 (a b c : ℤ) (m : ℤ) 
  (h1 : a + b + c = 0) 
  (h2 : ab + bc + ac = -2023) 
  (h3 : ∃ m, ∀ x : ℤ, x^3 - 2023 * x + m = (x - a) * (x - b) * (x - c)) : 
  |a| + |b| + |c| = 80 := 
by {
  sorry
}

end polynomial_root_abs_sum_eq_80_l1700_170060


namespace simplify_expression_l1700_170004

theorem simplify_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_condition : a^3 + b^3 = 3 * (a + b)) : 
  (a / b + b / a + 1 / (a * b) = 4 / (a * b) + 1) :=
by
  sorry

end simplify_expression_l1700_170004


namespace center_of_circle_l1700_170063

theorem center_of_circle (x y : ℝ) : (x^2 + y^2 - 10 * x + 4 * y + 13 = 0) → (x - y = 7) :=
by
  -- Statement, proof omitted
  sorry

end center_of_circle_l1700_170063


namespace rice_mixture_ratio_l1700_170052

theorem rice_mixture_ratio (x y : ℝ) (h1 : 7 * x + 8.75 * y = 7.50 * (x + y)) : x / y = 2.5 :=
by
  sorry

end rice_mixture_ratio_l1700_170052


namespace triangle_largest_angle_l1700_170028

theorem triangle_largest_angle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180)
  (h2 : ∃ k, 3 * k + 4 * k + 5 * k = 180) :
  5 * k = 75 :=
sorry

end triangle_largest_angle_l1700_170028


namespace domain_of_f_l1700_170050

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 6 * x + 10⌋

theorem domain_of_f : {x : ℝ | ∀ y, f y ≠ 0 → x ≠ 3} = {x : ℝ | x < 3 ∨ x > 3} :=
by
  sorry

end domain_of_f_l1700_170050


namespace work_finished_earlier_due_to_additional_men_l1700_170011

-- Define the conditions as given facts in Lean
def original_men := 10
def original_days := 12
def additional_men := 10

-- State the theorem to be proved
theorem work_finished_earlier_due_to_additional_men :
  let total_men := original_men + additional_men
  let original_work := original_men * original_days
  let days_earlier := original_days - x
  original_work = total_men * days_earlier → x = 6 :=
by
  sorry

end work_finished_earlier_due_to_additional_men_l1700_170011


namespace parabola_intersections_l1700_170085

theorem parabola_intersections :
  ∃ y1 y2, (∀ x y, (y = 2 * x^2 + 5 * x + 1 ∧ y = - x^2 + 4 * x + 6) → 
     (x = ( -1 + Real.sqrt 61) / 6 ∧ y = y1) ∨ (x = ( -1 - Real.sqrt 61) / 6 ∧ y = y2)) := 
by
  sorry

end parabola_intersections_l1700_170085


namespace decreasing_interval_f_l1700_170022

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (4*x - x^2)

theorem decreasing_interval_f : ∀ x, (2 < x) ∧ (x < 4) → f x < f (2 : ℝ) :=
by
sorry

end decreasing_interval_f_l1700_170022


namespace simplify_expression_l1700_170014

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : (4 * m^2 - 2 * m) / (2 * m) = 2 * m - 1 := by
  sorry

end simplify_expression_l1700_170014


namespace high_probability_event_is_C_l1700_170067

-- Define the probabilities of events A, B, and C
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.1
def prob_C : ℝ := 0.9

-- Statement asserting Event C has the high possibility of occurring
theorem high_probability_event_is_C : prob_C > prob_A ∧ prob_C > prob_B :=
by
  sorry

end high_probability_event_is_C_l1700_170067


namespace popsicle_sticks_left_l1700_170064

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end popsicle_sticks_left_l1700_170064


namespace value_of_a_pow_sum_l1700_170027

variable {a : ℝ}
variable {m n : ℕ}

theorem value_of_a_pow_sum (h1 : a^m = 5) (h2 : a^n = 3) : a^(m + n) = 15 := by
  sorry

end value_of_a_pow_sum_l1700_170027


namespace marian_baked_cookies_l1700_170066

theorem marian_baked_cookies :
  let cookies_per_tray := 12
  let trays_used := 23
  trays_used * cookies_per_tray = 276 :=
by
  sorry

end marian_baked_cookies_l1700_170066


namespace find_quadruples_l1700_170094

def is_prime (n : ℕ) := ∀ m, m ∣ n → m = 1 ∨ m = n

 theorem find_quadruples (p q a b : ℕ) (hp : is_prime p) (hq : is_prime q) (ha : 1 < a)
  : (p^a = 1 + 5 * q^b ↔ ((p = 2 ∧ q = 3 ∧ a = 4 ∧ b = 1) ∨ (p = 3 ∧ q = 2 ∧ a = 4 ∧ b = 4))) :=
by {
  sorry
}

end find_quadruples_l1700_170094


namespace jane_wins_game_l1700_170072

noncomputable def jane_win_probability : ℚ :=
  1/3 / (1 - (2/3 * 1/3 * 2/3))

theorem jane_wins_game :
  jane_win_probability = 9/23 :=
by
  -- detailed proof steps would be filled in here
  sorry

end jane_wins_game_l1700_170072


namespace equation_contains_2020_l1700_170002

def first_term (n : Nat) : Nat :=
  2 * n^2

theorem equation_contains_2020 :
  ∃ n, first_term n = 2020 :=
by
  use 31
  sorry

end equation_contains_2020_l1700_170002


namespace equalities_implied_by_sum_of_squares_l1700_170030

variable {a b c d : ℝ}

theorem equalities_implied_by_sum_of_squares (h1 : a = b) (h2 : c = d) : 
  (a - b) ^ 2 + (c - d) ^ 2 = 0 :=
sorry

end equalities_implied_by_sum_of_squares_l1700_170030


namespace round_trip_in_first_trip_l1700_170091

def percentage_rt_trip_first_trip := 0.3 -- 30%
def percentage_2t_trip_second_trip := 0.6 -- 60%
def percentage_ow_trip_third_trip := 0.45 -- 45%

theorem round_trip_in_first_trip (P1 P2 P3: ℝ) (C1 C2 C3: ℝ) 
  (h1 : P1 = 0.3) 
  (h2 : 0 < P1 ∧ P1 < 1) 
  (h3 : P2 = 0.6) 
  (h4 : 0 < P2 ∧ P2 < 1) 
  (h5 : P3 = 0.45) 
  (h6 : 0 < P3 ∧ P3 < 1) 
  (h7 : C1 + C2 + C3 = 1) 
  (h8 : (C1 = (1 - P1) * 0.15)) 
  (h9 : C2 = 0.2 * P2) 
  (h10 : C3 = 0.1 * P3) :
  P1 = 0.3 := by
  sorry

end round_trip_in_first_trip_l1700_170091


namespace xy_value_l1700_170046

theorem xy_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 3 / x = y + 3 / y) (hxy : x ≠ y) : x * y = 3 :=
sorry

end xy_value_l1700_170046


namespace problem1_problem2_l1700_170054

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := { x | a - b < x ∧ x < a + b }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- First problem: prove the range of a
theorem problem1 (a : ℝ) (h : A a 1 ⊆ B) : a ≤ -2 ∨ a ≥ 6 := by
  sorry

-- Second problem: prove the range of b
theorem problem2 (b : ℝ) (h : A 1 b ∩ B = ∅) : b ≤ 2 := by
  sorry

end problem1_problem2_l1700_170054


namespace number_of_eggs_left_l1700_170068

theorem number_of_eggs_left (initial_eggs : ℕ) (eggs_eaten_morning : ℕ) (eggs_eaten_afternoon : ℕ) (eggs_left : ℕ) :
    initial_eggs = 20 → eggs_eaten_morning = 4 → eggs_eaten_afternoon = 3 → eggs_left = initial_eggs - (eggs_eaten_morning + eggs_eaten_afternoon) → eggs_left = 13 :=
by
  intros h_initial h_morning h_afternoon h_calc
  rw [h_initial, h_morning, h_afternoon] at h_calc
  norm_num at h_calc
  exact h_calc

end number_of_eggs_left_l1700_170068


namespace train_pass_time_l1700_170073

-- Definitions based on conditions
def train_length : Float := 250
def pole_time : Float := 10
def platform_length : Float := 1250
def incline_angle : Float := 5 -- degrees
def speed_reduction_factor : Float := 0.75

-- The statement to be proved
theorem train_pass_time :
  let original_speed := train_length / pole_time
  let incline_speed := original_speed * speed_reduction_factor
  let total_distance := train_length + platform_length
  let time_to_pass_platform := total_distance / incline_speed
  time_to_pass_platform = 80 := by
  simp [train_length, pole_time, platform_length, incline_angle, speed_reduction_factor]
  sorry

end train_pass_time_l1700_170073


namespace total_pieces_of_junk_mail_l1700_170078

-- Definition of the problem based on given conditions
def pieces_per_house : ℕ := 4
def number_of_blocks : ℕ := 16
def houses_per_block : ℕ := 17

-- Statement of the theorem to prove the total number of pieces of junk mail
theorem total_pieces_of_junk_mail :
  (houses_per_block * pieces_per_house * number_of_blocks) = 1088 :=
by
  sorry

end total_pieces_of_junk_mail_l1700_170078


namespace range_of_a_l1700_170007

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * a * x + a + 2 ≤ 0 → 1 ≤ x ∧ x ≤ 4) ↔ a ∈ Set.Ioo (-1 : ℝ) (18 / 7) ∨ a = 18 / 7 := 
by
  sorry

end range_of_a_l1700_170007


namespace no_coprime_xy_multiple_l1700_170062

theorem no_coprime_xy_multiple (n : ℕ) (hn : ∀ d : ℕ, d ∣ n → d^2 ∣ n → d = 1)
  (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h_coprime : Nat.gcd x y = 1) :
  ¬ ((x^n + y^n) % ((x + y)^3) = 0) :=
by
  sorry

end no_coprime_xy_multiple_l1700_170062


namespace inequality_x4_y4_z2_l1700_170053

theorem inequality_x4_y4_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^4 + y^4 + z^2 ≥  xyz * 8^(1/2) :=
  sorry

end inequality_x4_y4_z2_l1700_170053


namespace minimum_guests_l1700_170031

theorem minimum_guests (x : ℕ) : (120 + 18 * x > 250 + 15 * x) → (x ≥ 44) := by
  intro h
  sorry

end minimum_guests_l1700_170031


namespace one_divides_the_other_l1700_170040

theorem one_divides_the_other (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  ∃ m n : ℕ, (x = m * y) ∨ (y = n * x) :=
by 
  -- Proof goes here
  sorry

end one_divides_the_other_l1700_170040


namespace problem_statement_l1700_170096

theorem problem_statement :
  ((8^5 / 8^2) * 2^10 - 2^2) = 2^19 - 4 := 
by 
  sorry

end problem_statement_l1700_170096


namespace find_number_l1700_170089

variable (x : ℝ)

theorem find_number (h : 0.46 * x = 165.6) : x = 360 :=
sorry

end find_number_l1700_170089


namespace probability_neither_red_nor_purple_l1700_170058

section Probability

def total_balls : ℕ := 60
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def total_red_or_purple_balls : ℕ := red_balls + purple_balls
def non_red_or_purple_balls : ℕ := total_balls - total_red_or_purple_balls

theorem probability_neither_red_nor_purple :
  (non_red_or_purple_balls : ℚ) / (total_balls : ℚ) = 7 / 10 :=
by
  sorry

end Probability

end probability_neither_red_nor_purple_l1700_170058


namespace four_digit_number_conditions_l1700_170099

theorem four_digit_number_conditions :
  ∃ (a b c d : ℕ), 
    (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 10 * 23) ∧ 
    (a + b + c + d = 26) ∧ 
    ((b * d / 10) % 10 = a + c) ∧ 
    ∃ (n : ℕ), (b * d - c^2 = 2^n) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 1979) :=
sorry

end four_digit_number_conditions_l1700_170099


namespace dealership_sales_l1700_170043

theorem dealership_sales (sports_cars sedans suvs : ℕ) (h_sc : sports_cars = 35)
  (h_ratio_sedans : 5 * sedans = 8 * sports_cars) 
  (h_ratio_suvs : 5 * suvs = 3 * sports_cars) : 
  sedans = 56 ∧ suvs = 21 := by
  sorry

#print dealership_sales

end dealership_sales_l1700_170043


namespace rhombus_perimeter_l1700_170008

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  (4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))) = 52 := by
  sorry

end rhombus_perimeter_l1700_170008


namespace work_days_together_l1700_170037

theorem work_days_together (A_rate B_rate : ℚ) (h1 : A_rate = 1 / 12) (h2 : B_rate = 5 / 36) : 
  1 / (A_rate + B_rate) = 4.5 := by
  sorry

end work_days_together_l1700_170037


namespace basketball_team_avg_weight_l1700_170097

theorem basketball_team_avg_weight :
  let n_tallest := 5
  let w_tallest := 90
  let n_shortest := 4
  let w_shortest := 75
  let n_remaining := 3
  let w_remaining := 80
  let total_weight := (n_tallest * w_tallest) + (n_shortest * w_shortest) + (n_remaining * w_remaining)
  let total_players := n_tallest + n_shortest + n_remaining
  (total_weight / total_players) = 82.5 :=
by
  sorry

end basketball_team_avg_weight_l1700_170097


namespace ratio_mark_days_used_l1700_170003

-- Defining the conditions
def num_sick_days : ℕ := 10
def num_vacation_days : ℕ := 10
def total_hours_left : ℕ := 80
def hours_per_workday : ℕ := 8

-- Total days allotted
def total_days_allotted : ℕ :=
  num_sick_days + num_vacation_days

-- Days left for Mark
def days_left : ℕ :=
  total_hours_left / hours_per_workday

-- Days used by Mark
def days_used : ℕ :=
  total_days_allotted - days_left

-- The ratio of days used to total days allotted (expected to be 1:2)
def ratio_used_to_allotted : ℚ :=
  days_used / total_days_allotted

theorem ratio_mark_days_used :
  ratio_used_to_allotted = 1 / 2 :=
sorry

end ratio_mark_days_used_l1700_170003


namespace chickens_in_farm_l1700_170090

theorem chickens_in_farm (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 := by sorry

end chickens_in_farm_l1700_170090


namespace directrix_of_parabola_l1700_170056

-- Define the parabola x^2 = 16y
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

-- Define the directrix equation
def directrix (y : ℝ) : Prop := y = -4

-- Theorem stating that the directrix of the given parabola is y = -4
theorem directrix_of_parabola : ∀ x y: ℝ, parabola x y → ∃ y, directrix y :=
by
  sorry

end directrix_of_parabola_l1700_170056


namespace wire_length_after_two_bends_is_three_l1700_170059

-- Let's define the initial length and the property of bending the wire.
def initial_length : ℕ := 12

def half_length (length : ℕ) : ℕ :=
  length / 2

-- Define the final length after two bends.
def final_length_after_two_bends : ℕ :=
  half_length (half_length initial_length)

-- The theorem stating that the final length is 3 cm after two bends.
theorem wire_length_after_two_bends_is_three :
  final_length_after_two_bends = 3 :=
by
  -- The proof can be added later.
  sorry

end wire_length_after_two_bends_is_three_l1700_170059


namespace flowchart_output_correct_l1700_170044

-- Define the conditions of the problem
def program_flowchart (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 * 2
  let step3 := step2 * 2
  step3

-- State the proof problem
theorem flowchart_output_correct : program_flowchart 1 = 8 :=
by
  -- Sorry to skip the proof
  sorry

end flowchart_output_correct_l1700_170044


namespace mark_weekly_reading_time_l1700_170077

-- Define the conditions
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7
def additional_hours : ℕ := 4

-- State the main theorem to prove
theorem mark_weekly_reading_time : (hours_per_day * days_per_week) + additional_hours = 18 := 
by
  -- The proof steps are omitted as per instructions
  sorry

end mark_weekly_reading_time_l1700_170077


namespace first_year_with_sum_of_digits_10_after_2200_l1700_170025

/-- Prove that the first year after 2200 in which the sum of the digits equals 10 is 2224. -/
theorem first_year_with_sum_of_digits_10_after_2200 :
  ∃ y, y > 2200 ∧ (List.sum (y.digits 10) = 10) ∧ 
       ∀ z, (2200 < z ∧ z < y) → (List.sum (z.digits 10) ≠ 10) :=
sorry

end first_year_with_sum_of_digits_10_after_2200_l1700_170025


namespace infinite_solutions_abs_eq_ax_minus_2_l1700_170009

theorem infinite_solutions_abs_eq_ax_minus_2 (a : ℝ) :
  (∀ x : ℝ, |x - 2| = ax - 2) ↔ a = 1 :=
by {
  sorry
}

end infinite_solutions_abs_eq_ax_minus_2_l1700_170009


namespace elimination_method_equation_y_l1700_170019

theorem elimination_method_equation_y (x y : ℝ)
    (h1 : 5 * x - 3 * y = -5)
    (h2 : 5 * x + 4 * y = -1) :
    7 * y = 4 :=
by
  -- Adding the required conditions as hypotheses and skipping the proof.
  sorry

end elimination_method_equation_y_l1700_170019


namespace units_digit_sum_base8_l1700_170080

theorem units_digit_sum_base8 : 
  ∀ (x y : ℕ), (x = 64 ∧ y = 34 ∧ (x % 8 = 4) ∧ (y % 8 = 4) → (x + y) % 8 = 0) :=
by
  sorry

end units_digit_sum_base8_l1700_170080


namespace profit_ratio_a_to_b_l1700_170075

noncomputable def capital_a : ℕ := 3500
noncomputable def time_a : ℕ := 12
noncomputable def capital_b : ℕ := 10500
noncomputable def time_b : ℕ := 6

noncomputable def capital_months (capital : ℕ) (time : ℕ) : ℕ :=
  capital * time

noncomputable def capital_months_a : ℕ :=
  capital_months capital_a time_a

noncomputable def capital_months_b : ℕ :=
  capital_months capital_b time_b

theorem profit_ratio_a_to_b : (capital_months_a / Nat.gcd capital_months_a capital_months_b) =
                             2 ∧
                             (capital_months_b / Nat.gcd capital_months_a capital_months_b) =
                             3 := 
by
  sorry

end profit_ratio_a_to_b_l1700_170075


namespace Ravi_probability_l1700_170047

-- Conditions from the problem
def P_Ram : ℚ := 4 / 7
def P_BothSelected : ℚ := 0.11428571428571428

-- Statement to prove
theorem Ravi_probability :
  ∃ P_Ravi : ℚ, P_Rami = 0.2 ∧ P_Ram * P_Ravi = P_BothSelected := by
  sorry

end Ravi_probability_l1700_170047


namespace range_of_a_l1700_170070

def A := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + (a^2 -1) = 0}

theorem range_of_a (a : ℝ) :
  (A ∩ B a = B a) → (a = 1 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l1700_170070


namespace evaluate_expression_at_two_l1700_170010

theorem evaluate_expression_at_two : (2 * (2:ℝ)^2 - 3 * 2 + 4) = 6 := by
  sorry

end evaluate_expression_at_two_l1700_170010


namespace necessary_but_not_sufficient_l1700_170087

def quadratic_inequality (x : ℝ) : Prop :=
  x^2 - 3 * x + 2 < 0

def necessary_condition_A (x : ℝ) : Prop :=
  -1 < x ∧ x < 2

def necessary_condition_D (x : ℝ) : Prop :=
  -2 < x ∧ x < 2

theorem necessary_but_not_sufficient :
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_A x ∧ ¬(quadratic_inequality x ∧ necessary_condition_A x)) ∧ 
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_D x ∧ ¬(quadratic_inequality x ∧ necessary_condition_D x)) :=
sorry

end necessary_but_not_sufficient_l1700_170087


namespace given_problem_l1700_170016

theorem given_problem :
  3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end given_problem_l1700_170016


namespace three_pow_m_plus_2n_l1700_170029

theorem three_pow_m_plus_2n (m n : ℕ) (h1 : 3^m = 5) (h2 : 9^n = 10) : 3^(m + 2 * n) = 50 :=
by
  sorry

end three_pow_m_plus_2n_l1700_170029


namespace triangle_min_area_l1700_170020

theorem triangle_min_area :
  ∃ (p q : ℤ), (p, q).fst = 3 ∧ (p, q).snd = 3 ∧ 1/2 * |18 * p - 30 * q| = 3 := 
sorry

end triangle_min_area_l1700_170020


namespace sufficient_but_not_necessary_condition_l1700_170093

theorem sufficient_but_not_necessary_condition (x : ℝ) : 
  (x > 2 → (x-1)^2 > 1) ∧ (∃ (y : ℝ), y ≤ 2 ∧ (y-1)^2 > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1700_170093


namespace bicycles_wheels_l1700_170018

theorem bicycles_wheels (b : ℕ) (h1 : 3 * b + 4 * 3 + 7 * 1 = 25) : b = 2 :=
sorry

end bicycles_wheels_l1700_170018


namespace find_sides_of_rectangle_l1700_170033

-- Define the conditions
def isRectangle (w l : ℝ) : Prop :=
  l = 3 * w ∧ 2 * l + 2 * w = l * w

-- Main theorem statement
theorem find_sides_of_rectangle (w l : ℝ) :
  isRectangle w l → w = 8 / 3 ∧ l = 8 :=
by
  sorry

end find_sides_of_rectangle_l1700_170033


namespace probability_sum_9_is_correct_l1700_170057

def num_faces : ℕ := 6

def possible_outcomes : ℕ := num_faces * num_faces

def favorable_outcomes : ℕ := 4  -- (3,6), (6,3), (4,5), (5,4)

def probability_sum_9 : ℚ := favorable_outcomes / possible_outcomes

theorem probability_sum_9_is_correct :
  probability_sum_9 = 1/9 :=
sorry

end probability_sum_9_is_correct_l1700_170057


namespace sequence_general_formula_l1700_170081

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 3 / 2 * a n - 3) : 
  (∀ n, a n = 2 * 3 ^ n) :=
by 
  sorry

end sequence_general_formula_l1700_170081


namespace two_digit_numbers_tens_greater_ones_l1700_170061

theorem two_digit_numbers_tens_greater_ones : 
  ∃ (count : ℕ), count = 45 ∧ ∀ (n : ℕ), 10 ≤ n ∧ n < 100 → 
    let tens := n / 10;
    let ones := n % 10;
    tens > ones → count = 45 :=
by {
  sorry
}

end two_digit_numbers_tens_greater_ones_l1700_170061
