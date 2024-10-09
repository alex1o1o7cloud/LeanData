import Mathlib

namespace monotonicity_f_range_of_b_l969_96984

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

def p (a b : ℝ) (x : ℝ) : Prop := f a x ≤ 2 * b
def q (b : ℝ) : Prop := ∀ x, (x = -3 → (x^2 + (2*b + 1)*x - b - 1) > 0) ∧ 
                           (x = -2 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 0 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 1 → (x^2 + (2*b + 1)*x - b - 1) > 0)

theorem monotonicity_f (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : ∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 := by
  sorry

theorem range_of_b (b : ℝ) (hp_or : ∃ x, p a b x ∨ q b) (hp_and : ∀ x, ¬(p a b x ∧ q b)) :
    (1/5 < b ∧ b < 1/2) ∨ (b ≥ 5/7) := by
    sorry

end monotonicity_f_range_of_b_l969_96984


namespace pascal_triangle_21st_number_l969_96992

theorem pascal_triangle_21st_number 
: (Nat.choose 22 2) = 231 :=
by 
  sorry

end pascal_triangle_21st_number_l969_96992


namespace prime_pair_solution_l969_96981

-- Steps a) and b) are incorporated into this Lean statement
theorem prime_pair_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p * q ∣ 3^p + 3^q ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 3 ∧ q = 3) ∨ (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) :=
sorry

end prime_pair_solution_l969_96981


namespace range_of_a_l969_96983

variable (a : ℝ)
variable (x : ℝ)

noncomputable def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (h : ∀ x, otimes (x - a) (x + a) < 1) : - 1 / 2 < a ∧ a < 3 / 2 :=
sorry

end range_of_a_l969_96983


namespace painting_house_cost_l969_96987

theorem painting_house_cost 
  (judson_contrib : ℕ := 500)
  (kenny_contrib : ℕ := judson_contrib + (judson_contrib * 20) / 100)
  (camilo_contrib : ℕ := kenny_contrib + 200) :
  judson_contrib + kenny_contrib + camilo_contrib = 1900 :=
by
  sorry

end painting_house_cost_l969_96987


namespace base10_to_base7_of_804_l969_96974

def base7 (n : ℕ) : ℕ :=
  let d3 := n / 343
  let r3 := n % 343
  let d2 := r3 / 49
  let r2 := r3 % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

theorem base10_to_base7_of_804 :
  base7 804 = 2226 :=
by
  -- Proof to be filled in.
  sorry

end base10_to_base7_of_804_l969_96974


namespace total_population_increase_l969_96929
-- Import the required library

-- Define the conditions for Region A and Region B
def regionA_births_0_14 (time: ℕ) := time / 20
def regionA_births_15_64 (time: ℕ) := time / 30
def regionB_births_0_14 (time: ℕ) := time / 25
def regionB_births_15_64 (time: ℕ) := time / 35

-- Define the total number of people in each age group for both regions
def regionA_population_0_14 := 2000
def regionA_population_15_64 := 6000
def regionB_population_0_14 := 1500
def regionB_population_15_64 := 5000

-- Define the total time in seconds
def total_time := 25 * 60

-- Proof statement
theorem total_population_increase : 
  regionA_population_0_14 * regionA_births_0_14 total_time +
  regionA_population_15_64 * regionA_births_15_64 total_time +
  regionB_population_0_14 * regionB_births_0_14 total_time +
  regionB_population_15_64 * regionB_births_15_64 total_time = 227 := 
by sorry

end total_population_increase_l969_96929


namespace positive_divisors_of_x_l969_96955

theorem positive_divisors_of_x (x : ℕ) (h : ∀ d : ℕ, d ∣ x^3 → d = 1 ∨ d = x^3 ∨ d ∣ x^2) : (∀ d : ℕ, d ∣ x → d = 1 ∨ d = x ∨ d ∣ p) :=
by
  sorry

end positive_divisors_of_x_l969_96955


namespace sequence_problem_l969_96965

theorem sequence_problem
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : 1 + a1 + a1 = a1 + a1)
  (h2 : b1 * b1 = b2)
  (h3 : 4 = b2 * b2):
  (a1 + a2) / b2 = 2 :=
by
  -- The proof would go here
  sorry

end sequence_problem_l969_96965


namespace marcia_minutes_worked_l969_96923

/--
If Marcia worked for 5 hours on her science project,
then she worked for 300 minutes.
-/
theorem marcia_minutes_worked (hours : ℕ) (h : hours = 5) : (hours * 60) = 300 := by
  sorry

end marcia_minutes_worked_l969_96923


namespace find_y_plus_one_over_y_l969_96911

variable (y : ℝ)

theorem find_y_plus_one_over_y (h : y^3 + (1/y)^3 = 110) : y + 1/y = 5 :=
by
  sorry

end find_y_plus_one_over_y_l969_96911


namespace line_point_relation_l969_96951

theorem line_point_relation (x1 y1 x2 y2 a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * x1 + b1 * y1 = c1)
  (h2 : a2 * x2 + b2 * y2 = c2)
  (h3 : a1 + b1 = c1)
  (h4 : a2 + b2 = 2 * c2)
  (h5 : dist (x1, y1) (x2, y2) ≥ (Real.sqrt 2) / 2) :
  c1 / a1 + a2 / c2 = 3 := 
sorry

end line_point_relation_l969_96951


namespace multiply_add_square_l969_96900

theorem multiply_add_square : 15 * 28 + 42 * 15 + 15^2 = 1275 :=
by
  sorry

end multiply_add_square_l969_96900


namespace toothpicks_15_l969_96910

def toothpicks (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Not used, placeholder for 1-based indexing.
  | 1 => 3
  | k+1 => let p := toothpicks k
           2 + if k % 2 = 0 then 1 else 0 + p

theorem toothpicks_15 : toothpicks 15 = 38 :=
by
  sorry

end toothpicks_15_l969_96910


namespace A_beats_B_by_40_meters_l969_96969

-- Definitions based on conditions
def distance_A := 1000 -- Distance in meters
def time_A := 240      -- Time in seconds
def time_diff := 10      -- Time difference in seconds

-- Intermediate calculations
def velocity_A : ℚ := distance_A / time_A
def time_B := time_A + time_diff
def velocity_B : ℚ := distance_A / time_B

-- Distance B covers in 240 seconds
def distance_B_in_240 : ℚ := velocity_B * time_A

-- Proof goal
theorem A_beats_B_by_40_meters : (distance_A - distance_B_in_240 = 40) :=
by
  -- Insert actual steps to prove here
  sorry

end A_beats_B_by_40_meters_l969_96969


namespace red_apples_count_l969_96952

-- Definitions based on conditions
def green_apples : ℕ := 2
def yellow_apples : ℕ := 14
def total_apples : ℕ := 19

-- Definition of red apples as a theorem to be proven
theorem red_apples_count :
  green_apples + yellow_apples + red_apples = total_apples → red_apples = 3 :=
by
  -- You would need to prove this using Lean
  sorry

end red_apples_count_l969_96952


namespace two_digit_number_satisfies_conditions_l969_96990

theorem two_digit_number_satisfies_conditions :
  ∃ N : ℕ, (N > 0) ∧ (N < 100) ∧ (N % 2 = 1) ∧ (N % 13 = 0) ∧ (∃ a b : ℕ, N = 10 * a + b ∧ (a * b) = (k : ℕ) * k) ∧ (N = 91) :=
by
  sorry

end two_digit_number_satisfies_conditions_l969_96990


namespace probability_of_same_length_segments_l969_96913

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l969_96913


namespace monotonically_increasing_f_l969_96943

open Set Filter Topology

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem monotonically_increasing_f : MonotoneOn f (Ioi 0) :=
sorry

end monotonically_increasing_f_l969_96943


namespace paint_replacement_l969_96919

theorem paint_replacement :
  ∀ (original_paint new_paint : ℝ), 
  original_paint = 100 →
  new_paint = 0.10 * (original_paint - 0.5 * original_paint) + 0.20 * (0.5 * original_paint) →
  new_paint / original_paint = 0.15 :=
by
  intros original_paint new_paint h_orig h_new
  sorry

end paint_replacement_l969_96919


namespace positive_real_inequality_l969_96975

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end positive_real_inequality_l969_96975


namespace find_y_from_condition_l969_96976

variable (y : ℝ) (h : (3 * y) / 7 = 15)

theorem find_y_from_condition : y = 35 :=
by {
  sorry
}

end find_y_from_condition_l969_96976


namespace find_number_l969_96909

theorem find_number (x : ℝ) (h : 7 * x = 50.68) : x = 7.24 :=
sorry

end find_number_l969_96909


namespace simplified_equation_equivalent_l969_96922

theorem simplified_equation_equivalent  (x : ℝ) :
    (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) ↔ (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by sorry

end simplified_equation_equivalent_l969_96922


namespace tonya_payment_l969_96927

def original_balance : ℝ := 150.00
def new_balance : ℝ := 120.00

noncomputable def payment_amount : ℝ := original_balance - new_balance

theorem tonya_payment :
  payment_amount = 30.00 :=
by
  sorry

end tonya_payment_l969_96927


namespace number_of_paths_l969_96972

-- Define the coordinates and the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

def E := (0, 7)
def F := (4, 5)
def G := (9, 0)

-- Define the number of steps required for each path segment
def steps_to_F := 6
def steps_to_G := 10

-- Capture binomial coefficients for the calculated path segments
def paths_E_to_F := binomial steps_to_F 4
def paths_F_to_G := binomial steps_to_G 5

-- Prove the total number of paths from E to G through F
theorem number_of_paths : paths_E_to_F * paths_F_to_G = 3780 :=
by rw [paths_E_to_F, paths_F_to_G]; sorry

end number_of_paths_l969_96972


namespace A_share_of_annual_gain_l969_96941

-- Definitions based on the conditions
def investment_A (x : ℝ) : ℝ := 12 * x
def investment_B (x : ℝ) : ℝ := 12 * x
def investment_C (x : ℝ) : ℝ := 12 * x
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def annual_gain : ℝ := 15000

-- Theorem based on the question and correct answer
theorem A_share_of_annual_gain (x : ℝ) : (investment_A x / total_investment x) * annual_gain = 5000 :=
by
  sorry

end A_share_of_annual_gain_l969_96941


namespace correct_sentence_l969_96907

-- Define an enumeration for different sentences
inductive Sentence
| A : Sentence
| B : Sentence
| C : Sentence
| D : Sentence

-- Define a function stating properties of each sentence
def sentence_property (s : Sentence) : Bool :=
  match s with
  | Sentence.A => false  -- "The chromosomes from dad are more than from mom" is false
  | Sentence.B => false  -- "The chromosomes in my cells and my brother's cells are exactly the same" is false
  | Sentence.C => true   -- "Each pair of homologous chromosomes is provided by both parents" is true
  | Sentence.D => false  -- "Each pair of homologous chromosomes in my brother's cells are the same size" is false

-- The theorem to prove that Sentence.C is the correct one
theorem correct_sentence : sentence_property Sentence.C = true :=
by
  unfold sentence_property
  rfl

end correct_sentence_l969_96907


namespace perimeter_to_side_ratio_l969_96962

variable (a b c h_a r : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < h_a ∧ 0 < r ∧ a + b > c ∧ a + c > b ∧ b + c > a)

theorem perimeter_to_side_ratio (P : ℝ) (hP : P = a + b + c) :
  P / a = h_a / r := by
  sorry

end perimeter_to_side_ratio_l969_96962


namespace english_students_23_l969_96978

def survey_students_total : Nat := 35
def students_in_all_three : Nat := 2
def solely_english_three_times_than_french (x y : Nat) : Prop := y = 3 * x
def english_but_not_french_or_spanish (x y : Nat) : Prop := y + students_in_all_three = 35 ∧ y - students_in_all_three = 23

theorem english_students_23 :
  ∃ (x y : Nat), solely_english_three_times_than_french x y ∧ english_but_not_french_or_spanish x y :=
by
  sorry

end english_students_23_l969_96978


namespace james_total_spent_l969_96993

noncomputable def total_cost : ℝ :=
  let milk_price := 3.0
  let bananas_price := 2.0
  let bread_price := 1.5
  let cereal_price := 4.0
  let milk_tax := 0.20
  let bananas_tax := 0.15
  let bread_tax := 0.10
  let cereal_tax := 0.25
  let milk_total := milk_price * (1 + milk_tax)
  let bananas_total := bananas_price * (1 + bananas_tax)
  let bread_total := bread_price * (1 + bread_tax)
  let cereal_total := cereal_price * (1 + cereal_tax)
  milk_total + bananas_total + bread_total + cereal_total

theorem james_total_spent : total_cost = 12.55 :=
  sorry

end james_total_spent_l969_96993


namespace exists_integers_a_b_l969_96938

theorem exists_integers_a_b : 
  ∃ (a b : ℤ), 2003 < a + b * (Real.sqrt 2) ∧ a + b * (Real.sqrt 2) < 2003.01 :=
by
  sorry

end exists_integers_a_b_l969_96938


namespace g_at_10_l969_96963

noncomputable def g : ℕ → ℝ := sorry

axiom g_zero : g 0 = 2
axiom g_one : g 1 = 1
axiom g_func_eq (m n : ℕ) (h : m ≥ n) : 
  g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2 + 2

theorem g_at_10 : g 10 = 102 := sorry

end g_at_10_l969_96963


namespace equation_of_parallel_line_l969_96902

theorem equation_of_parallel_line 
  (l : ℝ → ℝ) 
  (passes_through : l 0 = 7) 
  (parallel_to : ∀ x : ℝ, l x = -4 * x + (l 0)) :
  ∀ x : ℝ, l x = -4 * x + 7 :=
by
  sorry

end equation_of_parallel_line_l969_96902


namespace f_even_function_l969_96932

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even_function : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  show f x = f (-x)
  sorry

end f_even_function_l969_96932


namespace ron_l969_96946

-- Definitions for the given problem conditions
def cost_of_chocolate_bar : ℝ := 1.5
def s'mores_per_chocolate_bar : ℕ := 3
def number_of_scouts : ℕ := 15
def s'mores_per_scout : ℕ := 2

-- Proof that Ron will spend $15.00 on chocolate bars
theorem ron's_chocolate_bar_cost :
  (number_of_scouts * s'mores_per_scout / s'mores_per_chocolate_bar) * cost_of_chocolate_bar = 15 :=
by
  sorry

end ron_l969_96946


namespace parabola_constant_l969_96908

theorem parabola_constant (b c : ℝ)
  (h₁ : -20 = 2 * (-2)^2 + b * (-2) + c)
  (h₂ : 24 = 2 * 2^2 + b * 2 + c) : 
  c = -6 := 
by 
  sorry

end parabola_constant_l969_96908


namespace find_c_l969_96933

theorem find_c (a b c : ℝ) (h1 : ∃ x y : ℝ, x = a * (y - 2)^2 + 3 ∧ (x,y) = (3,2))
  (h2 : (1 : ℝ) = a * ((4 : ℝ) - 2)^2 + 3) : c = 1 :=
sorry

end find_c_l969_96933


namespace ratio_equality_l969_96999

theorem ratio_equality (x y u v p q : ℝ) (h : (x / y) * (u / v) * (p / q) = 1) :
  (x / y) * (u / v) * (p / q) = 1 := 
by sorry

end ratio_equality_l969_96999


namespace geometric_sequence_problem_l969_96970

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

def condition (a : ℕ → ℝ) : Prop :=
a 4 + a 8 = -3

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : condition a) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 :=
sorry

end geometric_sequence_problem_l969_96970


namespace fraction_to_decimal_l969_96948

theorem fraction_to_decimal (numerator : ℚ) (denominator : ℚ) (h : numerator = 5 ∧ denominator = 40) : 
  (numerator / denominator) = 0.125 :=
sorry

end fraction_to_decimal_l969_96948


namespace zeros_of_f_l969_96996

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem zeros_of_f : (f (-1) = 0) ∧ (f 1 = 0) ∧ (f 2 = 0) :=
by 
  -- Placeholder for the proof
  sorry

end zeros_of_f_l969_96996


namespace negation_of_all_students_are_punctual_l969_96942

variable (Student : Type)
variable (student : Student → Prop)
variable (punctual : Student → Prop)

theorem negation_of_all_students_are_punctual :
  ¬ (∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) := by
  sorry

end negation_of_all_students_are_punctual_l969_96942


namespace max_ratio_of_mean_70_l969_96914

theorem max_ratio_of_mean_70 (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hmean : (x + y) / 2 = 70) : (x / y ≤ 99 / 41) :=
sorry

end max_ratio_of_mean_70_l969_96914


namespace slower_train_speed_l969_96950

-- Defining the conditions

def length_of_each_train := 80 -- in meters
def faster_train_speed := 52 -- in km/hr
def time_to_pass := 36 -- in seconds

-- Main statement: 
theorem slower_train_speed (v : ℝ) : 
    let relative_speed := (faster_train_speed - v) * (1000 / 3600) -- converting relative speed from km/hr to m/s
    let total_distance := 2 * length_of_each_train
    let speed_equals_distance_over_time := total_distance / time_to_pass 
    (relative_speed = speed_equals_distance_over_time) -> v = 36 :=
by
  intros
  sorry

end slower_train_speed_l969_96950


namespace minimum_value_m_l969_96973

theorem minimum_value_m (x0 : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 1| ≤ m) → m = 2 :=
by
  sorry

end minimum_value_m_l969_96973


namespace circle_equation_l969_96985

theorem circle_equation : 
  ∃ (b : ℝ), (∀ (x y : ℝ), (x^2 + (y - b)^2 = 1 ↔ (x = 1 ∧ y = 2) → b = 2)) :=
sorry

end circle_equation_l969_96985


namespace union_of_A_and_B_l969_96924

open Set

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > -1} :=
by sorry

end union_of_A_and_B_l969_96924


namespace coffee_consumption_l969_96935

variables (h w g : ℝ)

theorem coffee_consumption (k : ℝ) 
  (H1 : ∀ h w g, h * g = k * w)
  (H2 : h = 8 ∧ g = 4.5 ∧ w = 2)
  (H3 : h = 4 ∧ w = 3) : g = 13.5 :=
by {
  sorry
}

end coffee_consumption_l969_96935


namespace find_parallel_and_perpendicular_lines_through_A_l969_96954

def point_A : ℝ × ℝ := (2, 2)

def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 20 = 0

def parallel_line_l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 14 = 0

def perpendicular_line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y - 2 = 0

theorem find_parallel_and_perpendicular_lines_through_A :
  (∀ x y, line_l x y → parallel_line_l1 x y) ∧
  (∀ x y, line_l x y → perpendicular_line_l2 x y) :=
by
  sorry

end find_parallel_and_perpendicular_lines_through_A_l969_96954


namespace right_triangle_median_l969_96997

noncomputable def median_to_hypotenuse_length (a b : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (a^2 + b^2)
  hypotenuse / 2

theorem right_triangle_median
  (a b : ℝ) (h_a : a = 3) (h_b : b = 4) :
  median_to_hypotenuse_length a b = 2.5 :=
by
  sorry

end right_triangle_median_l969_96997


namespace silvia_shorter_route_l969_96998

theorem silvia_shorter_route :
  let jerry_distance := 3 + 4
  let silvia_distance := Real.sqrt (3^2 + 4^2)
  let percentage_reduction := ((jerry_distance - silvia_distance) / jerry_distance) * 100
  (28.5 ≤ percentage_reduction ∧ percentage_reduction < 30.5) →
  percentage_reduction = 30 := by
  intro h
  sorry

end silvia_shorter_route_l969_96998


namespace solve_for_n_l969_96926

theorem solve_for_n (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2) ^ 2 = 12 * 12 * (n - 2)) :
  n = 26 :=
by {
  sorry
}

end solve_for_n_l969_96926


namespace trip_first_part_distance_l969_96991

theorem trip_first_part_distance (x : ℝ) :
  let total_distance : ℝ := 60
  let speed_first : ℝ := 48
  let speed_remaining : ℝ := 24
  let avg_speed : ℝ := 32
  (x / speed_first + (total_distance - x) / speed_remaining = total_distance / avg_speed) ↔ (x = 30) :=
by sorry

end trip_first_part_distance_l969_96991


namespace students_basketball_cricket_l969_96917

theorem students_basketball_cricket (A B: ℕ) (AB: ℕ):
  A = 12 →
  B = 8 →
  AB = 3 →
  (A + B - AB) = 17 :=
by
  intros
  sorry

end students_basketball_cricket_l969_96917


namespace find_abc_l969_96971

theorem find_abc (a b c : ℝ) 
  (h1 : 2 * b = a + c)  -- a, b, c form an arithmetic sequence
  (h2 : a + b + c = 12) -- The sum of a, b, and c is 12
  (h3 : (b + 2)^2 = (a + 2) * (c + 5)) -- a+2, b+2, and c+5 form a geometric sequence
: (a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2) :=
sorry

end find_abc_l969_96971


namespace range_of_function_l969_96953

theorem range_of_function : 
  ∀ y : ℝ, (∃ x : ℝ, y = x / (1 + x^2)) ↔ (-1 / 2 ≤ y ∧ y ≤ 1 / 2) := 
by sorry

end range_of_function_l969_96953


namespace pradeep_passing_percentage_l969_96957

-- Define the constants based on the conditions
def totalMarks : ℕ := 550
def marksObtained : ℕ := 200
def marksFailedBy : ℕ := 20

-- Calculate the passing marks
def passingMarks : ℕ := marksObtained + marksFailedBy

-- Define the percentage calculation as a noncomputable function
noncomputable def requiredPercentageToPass : ℚ := (passingMarks / totalMarks) * 100

-- The theorem to prove
theorem pradeep_passing_percentage :
  requiredPercentageToPass = 40 := 
sorry

end pradeep_passing_percentage_l969_96957


namespace penny_dime_halfdollar_same_probability_l969_96903

def probability_same_penny_dime_halfdollar : ℚ :=
  let total_outcomes := 2 ^ 5
  let successful_outcomes := 2 * 2 * 2
  successful_outcomes / total_outcomes

theorem penny_dime_halfdollar_same_probability :
  probability_same_penny_dime_halfdollar = 1 / 4 :=
by 
  sorry

end penny_dime_halfdollar_same_probability_l969_96903


namespace onlyD_is_PythagoreanTriple_l969_96928

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def validTripleA := ¬ isPythagoreanTriple 12 15 18
def validTripleB := isPythagoreanTriple 3 4 5 ∧ (¬ (3 = 3 ∧ 4 = 4 ∧ 5 = 5)) -- Since 0.3, 0.4, 0.5 not integers
def validTripleC := ¬ isPythagoreanTriple 15 25 30 -- Conversion of 1.5, 2.5, 3 to integers
def validTripleD := isPythagoreanTriple 12 16 20

theorem onlyD_is_PythagoreanTriple : validTripleA ∧ validTripleB ∧ validTripleC ∧ validTripleD :=
by {
  sorry
}

end onlyD_is_PythagoreanTriple_l969_96928


namespace josie_animal_counts_l969_96956

/-- Josie counted 80 antelopes, 34 more rabbits than antelopes, 42 fewer hyenas than 
the total number of antelopes and rabbits combined, some more wild dogs than hyenas, 
and the number of leopards was half the number of rabbits. The total number of animals 
Josie counted was 605. Prove that the difference between the number of wild dogs 
and hyenas Josie counted is 50. -/
theorem josie_animal_counts :
  ∃ (antelopes rabbits hyenas wild_dogs leopards : ℕ),
    antelopes = 80 ∧
    rabbits = antelopes + 34 ∧
    hyenas = (antelopes + rabbits) - 42 ∧
    leopards = rabbits / 2 ∧
    (antelopes + rabbits + hyenas + wild_dogs + leopards = 605) ∧
    wild_dogs - hyenas = 50 := 
by
  sorry

end josie_animal_counts_l969_96956


namespace n_cubed_plus_two_not_divisible_by_nine_l969_96940

theorem n_cubed_plus_two_not_divisible_by_nine (n : ℕ) : ¬ (9 ∣ n^3 + 2) :=
sorry

end n_cubed_plus_two_not_divisible_by_nine_l969_96940


namespace total_students_correct_l969_96947

def students_in_general_hall : ℕ := 30
def students_in_biology_hall : ℕ := 2 * students_in_general_hall
def combined_students_general_biology : ℕ := students_in_general_hall + students_in_biology_hall
def students_in_math_hall : ℕ := (3 * combined_students_general_biology) / 5
def total_students_in_all_halls : ℕ := students_in_general_hall + students_in_biology_hall + students_in_math_hall

theorem total_students_correct : total_students_in_all_halls = 144 := by
  -- Proof omitted, it should be
  sorry

end total_students_correct_l969_96947


namespace locus_of_point_C_l969_96918

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_isosceles_triangle (A B C : Point) : Prop := 
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let AC := (A.x - C.x)^2 + (A.y - C.y)^2
  AB = AC

def circle_eqn (C : Point) : Prop :=
  C.x^2 + C.y^2 - 3 * C.x + C.y = 2

def not_points (C : Point) : Prop :=
  (C ≠ {x := 3, y := -2}) ∧ (C ≠ {x := 0, y := 1})

theorem locus_of_point_C :
  ∀ (A B C : Point),
    A = {x := 3, y := -2} →
    B = {x := 0, y := 1} →
    is_isosceles_triangle A B C →
    circle_eqn C ∧ not_points C :=
by
  intros A B C hA hB hIso
  sorry

end locus_of_point_C_l969_96918


namespace maximized_area_using_squares_l969_96916

theorem maximized_area_using_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
  by sorry

end maximized_area_using_squares_l969_96916


namespace find_x_l969_96949

variables (a b c d x y : ℚ)

noncomputable def modified_fraction (a b x y : ℚ) := (a + x) / (b + y)

theorem find_x (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : modified_fraction a b x y = c / d) :
  x = (b * c - a * d + y * c) / d :=
by
  sorry

end find_x_l969_96949


namespace geometric_sum_4_terms_l969_96961

theorem geometric_sum_4_terms 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 2 = 9) 
  (h2 : a 5 = 243) 
  (hq : ∀ n, a (n + 1) = a n * q) 
  : a 1 * (1 - q^4) / (1 - q) = 120 := 
sorry

end geometric_sum_4_terms_l969_96961


namespace remainder_when_divided_by_x_minus_4_l969_96939

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^5 - 8 * x^4 + 15 * x^3 + 20 * x^2 - 5 * x - 20

-- State the problem as a theorem
theorem remainder_when_divided_by_x_minus_4 : 
    (f 4 = 216) := 
by 
    -- Calculation goes here
    sorry

end remainder_when_divided_by_x_minus_4_l969_96939


namespace log_product_eq_one_sixth_log_y_x_l969_96936

variable (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

theorem log_product_eq_one_sixth_log_y_x :
  (Real.log x ^ 2 / Real.log (y ^ 5)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 4)) *
  (Real.log (x ^ 4) / Real.log (y ^ 3)) *
  (Real.log (y ^ 5) / Real.log (x ^ 3)) *
  (Real.log (x ^ 3) / Real.log (y ^ 4)) = 
  (1 / 6) * (Real.log x / Real.log y) := 
sorry

end log_product_eq_one_sixth_log_y_x_l969_96936


namespace product_of_roots_l969_96945

theorem product_of_roots :
  (∃ r s t : ℝ, (r + s + t) = 15 ∧ (r*s + s*t + r*t) = 50 ∧ (r*s*t) = -35) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 50*x + 35 = (x - r) * (x - s) * (x - t)) :=
sorry

end product_of_roots_l969_96945


namespace boat_speed_in_still_water_l969_96988

namespace BoatSpeed

variables (V_b V_s : ℝ)

def condition1 : Prop := V_b + V_s = 15
def condition2 : Prop := V_b - V_s = 5

theorem boat_speed_in_still_water (h1 : condition1 V_b V_s) (h2 : condition2 V_b V_s) : V_b = 10 :=
by
  sorry

end BoatSpeed

end boat_speed_in_still_water_l969_96988


namespace angus_caught_4_more_l969_96921

theorem angus_caught_4_more (
  angus ollie patrick: ℕ
) (
  h1: ollie = angus - 7
) (
  h2: ollie = 5
) (
  h3: patrick = 8
) : (angus - patrick) = 4 := 
sorry

end angus_caught_4_more_l969_96921


namespace colorings_10x10_board_l969_96989

def colorings_count (n : ℕ) : ℕ := 2^11 - 2

theorem colorings_10x10_board : colorings_count 10 = 2046 :=
by
  sorry

end colorings_10x10_board_l969_96989


namespace find_5_minus_a_l969_96958

-- Define the problem conditions as assumptions
variable (a b : ℤ)
variable (h1 : 5 + a = 6 - b)
variable (h2 : 3 + b = 8 + a)

-- State the theorem we want to prove
theorem find_5_minus_a : 5 - a = 7 :=
by
  sorry

end find_5_minus_a_l969_96958


namespace min_pos_solution_eqn_l969_96960

theorem min_pos_solution_eqn (x : ℝ) (h : (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 25) : x = 7 * Real.sqrt 3 :=
sorry

end min_pos_solution_eqn_l969_96960


namespace factorize_poly_l969_96906

theorem factorize_poly (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) := by
  sorry

end factorize_poly_l969_96906


namespace hourly_rate_is_7_l969_96912

-- Define the fixed fee, the total payment, and the number of hours
def fixed_fee : ℕ := 17
def total_payment : ℕ := 80
def num_hours : ℕ := 9

-- Define the function calculating the hourly rate based on the given conditions
def hourly_rate (fixed_fee total_payment num_hours : ℕ) : ℕ :=
  (total_payment - fixed_fee) / num_hours

-- Prove that the hourly rate is 7 dollars per hour
theorem hourly_rate_is_7 :
  hourly_rate fixed_fee total_payment num_hours = 7 := 
by 
  -- proof is skipped
  sorry

end hourly_rate_is_7_l969_96912


namespace Seokjin_paper_count_l969_96930

theorem Seokjin_paper_count (Jimin_paper : ℕ) (h1 : Jimin_paper = 41) (h2 : ∀ x : ℕ, Seokjin_paper = Jimin_paper - 1) : Seokjin_paper = 40 :=
by {
  sorry
}

end Seokjin_paper_count_l969_96930


namespace avg_displacement_per_man_l969_96915

-- Problem definition as per the given conditions
def num_men : ℕ := 50
def tank_length : ℝ := 40  -- 40 meters
def tank_width : ℝ := 20   -- 20 meters
def rise_in_water_level : ℝ := 0.25  -- 25 cm -> 0.25 meters

-- Given the conditions, we need to prove the average displacement per man
theorem avg_displacement_per_man :
  (tank_length * tank_width * rise_in_water_level) / num_men = 4 := by
  sorry

end avg_displacement_per_man_l969_96915


namespace simplify_polynomial_simplify_expression_l969_96980

-- Problem 1:
theorem simplify_polynomial (x : ℝ) : 
  2 * x^3 - 4 * x^2 - 3 * x - 2 * x^2 - x^3 + 5 * x - 7 = x^3 - 6 * x^2 + 2 * x - 7 := 
by
  sorry

-- Problem 2:
theorem simplify_expression (m n : ℝ) (A B : ℝ) (hA : A = 2 * m^2 - m * n) (hB : B = m^2 + 2 * m * n - 5) : 
  4 * A - 2 * B = 6 * m^2 - 8 * m * n + 10 := 
by
  sorry

end simplify_polynomial_simplify_expression_l969_96980


namespace harold_car_payment_l969_96901

variables (C : ℝ)

noncomputable def harold_income : ℝ := 2500
noncomputable def rent : ℝ := 700
noncomputable def groceries : ℝ := 50
noncomputable def remaining_after_retirement : ℝ := 1300

-- Harold's utility cost is half his car payment
noncomputable def utilities (C : ℝ) : ℝ := C / 2

-- Harold's total expenses.
noncomputable def total_expenses (C : ℝ) : ℝ := rent + C + utilities C + groceries

-- Proving that Harold’s car payment \(C\) can be calculated with the remaining money
theorem harold_car_payment : (2500 - total_expenses C = 1300) → (C = 300) :=
by 
  sorry

end harold_car_payment_l969_96901


namespace atomic_weight_of_nitrogen_l969_96944

-- Definitions from conditions
def molecular_weight := 53.0
def hydrogen_weight := 1.008
def chlorine_weight := 35.45
def hydrogen_atoms := 4
def chlorine_atoms := 1

-- The proof goal
theorem atomic_weight_of_nitrogen : 
  53.0 - (4.0 * 1.008) - 35.45 = 13.518 :=
by
  sorry

end atomic_weight_of_nitrogen_l969_96944


namespace factorize_difference_of_squares_l969_96959

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
sorry

end factorize_difference_of_squares_l969_96959


namespace difference_of_areas_l969_96925

-- Defining the side length of the square
def square_side_length : ℝ := 8

-- Defining the side lengths of the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 5

-- Defining the area functions
def area_of_square (side_length : ℝ) : ℝ := side_length * side_length
def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ := length * width

-- Stating the theorem
theorem difference_of_areas :
  area_of_square square_side_length - area_of_rectangle rectangle_length rectangle_width = 14 :=
by
  sorry

end difference_of_areas_l969_96925


namespace square_side_length_l969_96920

theorem square_side_length (A : ℝ) (π : ℝ) (s : ℝ) (area_circle_eq : A = 100)
  (area_circle_eq_perimeter_square : A = 4 * s) : s = 25 := by
  sorry

end square_side_length_l969_96920


namespace min_value_frac_sin_cos_l969_96931

open Real

theorem min_value_frac_sin_cos (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ m : ℝ, (∀ x : ℝ, x = (1 / (sin α)^2 + 3 / (cos α)^2) → x ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
by
  have h_sin_cos : sin α ≠ 0 ∧ cos α ≠ 0 := sorry -- This is an auxiliary lemma in the process, a proof is required.
  sorry

end min_value_frac_sin_cos_l969_96931


namespace tenth_day_is_monday_l969_96986

theorem tenth_day_is_monday (runs_20_mins : ∀ d ∈ [1, 7], d = 1 ∨ d = 6 ∨ d = 7 → True)
                            (total_minutes : 5 * 60 = 300)
                            (first_day_is_saturday : 1 = 6) :
   (10 % 7 = 3) :=
by
  sorry

end tenth_day_is_monday_l969_96986


namespace contradiction_to_at_least_one_not_greater_than_60_l969_96994

-- Define a condition for the interior angles of a triangle being > 60
def all_angles_greater_than_60 (α β γ : ℝ) : Prop :=
  α > 60 ∧ β > 60 ∧ γ > 60

-- Define the negation of the proposition "At least one of the interior angles is not greater than 60"
def at_least_one_not_greater_than_60 (α β γ : ℝ) : Prop :=
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The mathematically equivalent proof problem
theorem contradiction_to_at_least_one_not_greater_than_60 (α β γ : ℝ) :
  ¬ at_least_one_not_greater_than_60 α β γ ↔ all_angles_greater_than_60 α β γ := by
  sorry

end contradiction_to_at_least_one_not_greater_than_60_l969_96994


namespace kyle_money_l969_96905

theorem kyle_money (dave_money : ℕ) (kyle_initial : ℕ) (kyle_remaining : ℕ)
  (h1 : dave_money = 46)
  (h2 : kyle_initial = 3 * dave_money - 12)
  (h3 : kyle_remaining = kyle_initial - kyle_initial / 3) :
  kyle_remaining = 84 :=
by
  -- Define Dave's money and provide the assumption
  let dave_money := 46
  have h1 : dave_money = 46 := rfl

  -- Define Kyle's initial money based on Dave's money
  let kyle_initial := 3 * dave_money - 12
  have h2 : kyle_initial = 3 * dave_money - 12 := rfl

  -- Define Kyle's remaining money after spending one third on snowboarding
  let kyle_remaining := kyle_initial - kyle_initial / 3
  have h3 : kyle_remaining = kyle_initial - kyle_initial / 3 := rfl

  -- Now we prove that Kyle's remaining money is 84
  sorry -- Proof steps omitted

end kyle_money_l969_96905


namespace total_waiting_days_l969_96995

-- Definitions based on the conditions
def wait_for_first_appointment : ℕ := 4
def wait_for_second_appointment : ℕ := 20
def wait_for_effectiveness : ℕ := 2 * 7  -- 2 weeks converted to days

-- The main theorem statement
theorem total_waiting_days : wait_for_first_appointment + wait_for_second_appointment + wait_for_effectiveness = 38 :=
by
  sorry

end total_waiting_days_l969_96995


namespace total_worksheets_l969_96934

theorem total_worksheets (worksheets_graded : ℕ) (problems_per_worksheet : ℕ) (problems_remaining : ℕ)
  (h1 : worksheets_graded = 7)
  (h2 : problems_per_worksheet = 2)
  (h3 : problems_remaining = 14): 
  worksheets_graded + (problems_remaining / problems_per_worksheet) = 14 := 
by 
  sorry

end total_worksheets_l969_96934


namespace area_of_rectangle_is_270_l969_96937

noncomputable def side_of_square := Real.sqrt 2025

noncomputable def radius_of_circle := side_of_square

noncomputable def length_of_rectangle := (2/5 : ℝ) * radius_of_circle

noncomputable def initial_breadth_of_rectangle := (1/2 : ℝ) * length_of_rectangle + 5

noncomputable def breadth_of_rectangle := if (length_of_rectangle + initial_breadth_of_rectangle) % 3 = 0 
                                          then initial_breadth_of_rectangle 
                                          else initial_breadth_of_rectangle + 1

noncomputable def area_of_rectangle := length_of_rectangle * breadth_of_rectangle

theorem area_of_rectangle_is_270 :
  area_of_rectangle = 270 := by
  sorry

end area_of_rectangle_is_270_l969_96937


namespace dice_sum_impossible_l969_96967

theorem dice_sum_impossible (a b c d : ℕ) (h1 : a * b * c * d = 216)
  (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
  (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) : 
  a + b + c + d ≠ 18 :=
sorry

end dice_sum_impossible_l969_96967


namespace geometric_series_sum_l969_96982

theorem geometric_series_sum (a r : ℝ) 
  (h1 : a * (1 - r / (1 - r)) = 18) 
  (h2 : a * (r / (1 - r)) = 8) : r = 4 / 5 :=
by sorry

end geometric_series_sum_l969_96982


namespace simplify_fraction_l969_96964

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)

theorem simplify_fraction : (1 / a) + (1 / b) - (2 * a + b) / (2 * a * b) = 1 / (2 * a) :=
by
  sorry

end simplify_fraction_l969_96964


namespace particular_solution_exists_l969_96904

noncomputable def general_solution (C : ℝ) (x : ℝ) : ℝ := C * x + 1

def differential_equation (x y y' : ℝ) : Prop := x * y' = y - 1

def initial_condition (y : ℝ) : Prop := y = 5

theorem particular_solution_exists :
  (∀ C x y, y = general_solution C x → differential_equation x y (C : ℝ)) →
  (∃ C, initial_condition (general_solution C 1)) →
  (∀ x, ∃ y, y = general_solution 4 x) :=
by
  intros h1 h2
  sorry

end particular_solution_exists_l969_96904


namespace cos_value_l969_96968

theorem cos_value (A : ℝ) (h : Real.sin (π + A) = 1/2) : Real.cos (3*π/2 - A) = 1/2 :=
sorry

end cos_value_l969_96968


namespace olympiad_scores_l969_96977

theorem olympiad_scores (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j : Fin 20, i < j → a i < a j)
  (h_condition : ∀ i j k : Fin 20, i ≠ j ∧ i ≠ k ∧ j ≠ k → a i < a j + a k) : 
  ∀ i : Fin 20, a i > 18 :=
by
  sorry

end olympiad_scores_l969_96977


namespace emily_beads_l969_96979

-- Definitions of the conditions as per step a)
def beads_per_necklace : ℕ := 8
def necklaces : ℕ := 2

-- Theorem statement to prove the equivalent math problem
theorem emily_beads : beads_per_necklace * necklaces = 16 :=
by
  sorry

end emily_beads_l969_96979


namespace roots_square_sum_eq_l969_96966

theorem roots_square_sum_eq (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) 
  (h3 : r * s * t = r) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by
  sorry

end roots_square_sum_eq_l969_96966
