import Mathlib

namespace frosting_need_l1689_168953

theorem frosting_need : 
  (let layer_cake_frosting := 1
   let single_cake_frosting := 0.5
   let brownie_frosting := 0.5
   let dozen_cupcakes_frosting := 0.5
   let num_layer_cakes := 3
   let num_dozen_cupcakes := 6
   let num_single_cakes := 12
   let num_pans_brownies := 18
   
   let total_frosting := 
     (num_layer_cakes * layer_cake_frosting) + 
     (num_dozen_cupcakes * dozen_cupcakes_frosting) + 
     (num_single_cakes * single_cake_frosting) + 
     (num_pans_brownies * brownie_frosting)
   
   total_frosting = 21) :=
  by
    sorry

end frosting_need_l1689_168953


namespace mean_temperature_l1689_168955

theorem mean_temperature
  (temps : List ℤ) 
  (h_temps : temps = [-8, -3, -7, -6, 0, 4, 6, 5, -1, 2]) :
  (temps.sum: ℚ) / temps.length = -0.8 := 
by
  sorry

end mean_temperature_l1689_168955


namespace anatoliy_handshakes_l1689_168963

-- Define the total number of handshakes
def total_handshakes := 197

-- Define friends excluding Anatoliy
def handshake_func (n : Nat) : Nat :=
  n * (n - 1) / 2

-- Define the target problem stating that Anatoliy made 7 handshakes
theorem anatoliy_handshakes (n k : Nat) (h : handshake_func n + k = total_handshakes) : k = 7 :=
by sorry

end anatoliy_handshakes_l1689_168963


namespace proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l1689_168925

open Classical

variable (a b x y : ℝ)

theorem proposition_A_correct (h : a > 1) : (1 / a < 1) ∧ ¬((1 / a < 1) → (a > 1)) :=
sorry

theorem proposition_B_incorrect (h_neg : ¬(x < 1 → x^2 < 1)) : ¬(∃ x, x ≥ 1 ∧ x^2 ≥ 1) :=
sorry

theorem proposition_C_incorrect (h_xy : x ≥ 2 ∧ y ≥ 2) : ¬((x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4) :=
sorry

theorem proposition_D_correct (h_a : a ≠ 0) : (a * b ≠ 0) ∧ ¬((a * b ≠ 0) → (a ≠ 0)) :=
sorry

end proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l1689_168925


namespace vertex_of_parabola_l1689_168994

def f (x : ℝ) : ℝ := 2 - (2*x + 1)^2

theorem vertex_of_parabola :
  (∀ x : ℝ, f x ≤ 2) ∧ (f (-1/2) = 2) :=
by
  sorry

end vertex_of_parabola_l1689_168994


namespace heptagon_diagonals_l1689_168961

theorem heptagon_diagonals (n : ℕ) (h : n = 7) : (n * (n - 3)) / 2 = 14 := by
  sorry

end heptagon_diagonals_l1689_168961


namespace find_four_numbers_l1689_168933

theorem find_four_numbers 
    (a b c d : ℕ) 
    (h1 : b - a = c - b)  -- first three numbers form an arithmetic sequence
    (h2 : d / c = c / (b - a + b))  -- last three numbers form a geometric sequence
    (h3 : a + d = 16)  -- sum of first and last numbers is 16
    (h4 : b + (12 - b) = 12)  -- sum of the two middle numbers is 12
    : (a = 15 ∧ b = 9 ∧ c = 3 ∧ d = 1) ∨ (a = 0 ∧ b = 4 ∧ c = 8 ∧ d = 16) :=
by
  -- Proof will be provided here
  sorry

end find_four_numbers_l1689_168933


namespace omega_not_possible_l1689_168983

noncomputable def f (ω x φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_not_possible (ω φ : ℝ) (h1 : ∀ x y, -π/3 ≤ x → x < y → y ≤ π/6 → f ω x φ ≤ f ω y φ)
  (h2 : f ω (π / 6) φ = f ω (4 * π / 3) φ)
  (h3 : f ω (π / 6) φ = -f ω (-π / 3) φ) :
  ω ≠ 7 / 5 :=
sorry

end omega_not_possible_l1689_168983


namespace length_of_base_of_vessel_l1689_168931

noncomputable def volume_of_cube (edge : ℝ) := edge ^ 3

theorem length_of_base_of_vessel 
  (cube_edge : ℝ)
  (vessel_width : ℝ)
  (rise_in_water_level : ℝ)
  (volume_cube : ℝ)
  (h1 : cube_edge = 15)
  (h2 : vessel_width = 15)
  (h3 : rise_in_water_level = 11.25)
  (h4 : volume_cube = volume_of_cube cube_edge)
  : ∃ L : ℝ, L = volume_cube / (vessel_width * rise_in_water_level) ∧ L = 20 :=
by
  sorry

end length_of_base_of_vessel_l1689_168931


namespace units_produced_by_line_B_l1689_168990

-- State the problem with the given conditions and prove the question equals the answer.
theorem units_produced_by_line_B (total_units : ℕ) (B : ℕ) (A C : ℕ) 
    (h1 : total_units = 13200)
    (h2 : A + B + C = total_units)
    (h3 : ∃ d : ℕ, A = B - d ∧ C = B + d) :
    B = 4400 :=
by
  sorry

end units_produced_by_line_B_l1689_168990


namespace infinite_primes_solutions_l1689_168988

theorem infinite_primes_solutions :
  ∀ (P : Finset ℕ), (∀ p ∈ P, Prime p) →
  ∃ q, Prime q ∧ q ∉ P ∧ ∃ x y : ℤ, x^2 + x + 1 = q * y :=
by sorry

end infinite_primes_solutions_l1689_168988


namespace domain_of_rational_function_l1689_168968

theorem domain_of_rational_function 
  (c : ℝ) 
  (h : -7 * (6 ^ 2) + 28 * c < 0) : 
  c < -9 / 7 :=
by sorry

end domain_of_rational_function_l1689_168968


namespace sqrt_expression_eval_l1689_168907

theorem sqrt_expression_eval :
  (Real.sqrt 8) + (Real.sqrt (1 / 2)) + (Real.sqrt 3 - 1) ^ 2 + (Real.sqrt 6 / (1 / 2 * Real.sqrt 2)) = (5 / 2) * Real.sqrt 2 + 4 := 
by
  sorry

end sqrt_expression_eval_l1689_168907


namespace remainder_2457634_div_8_l1689_168916

theorem remainder_2457634_div_8 : 2457634 % 8 = 2 := by
  sorry

end remainder_2457634_div_8_l1689_168916


namespace gcd_values_count_l1689_168939

theorem gcd_values_count (a b : ℕ) (h : a * b = 3600) : ∃ n, n = 29 ∧ ∀ d, d ∣ a ∧ d ∣ b → d = gcd a b → n = 29 :=
by { sorry }

end gcd_values_count_l1689_168939


namespace trig_identity_l1689_168954

-- Define the angle alpha with the given condition tan(alpha) = 2
variables (α : ℝ) (h : Real.tan α = 2)

-- State the theorem
theorem trig_identity : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end trig_identity_l1689_168954


namespace correct_equation_l1689_168975

-- Define the daily paving distances for Team A and Team B
variables (x : ℝ) (h₀ : x > 10)

-- Assuming Team A takes the same number of days to pave 150m as Team B takes to pave 120m
def same_days_to_pave (h₁ : x - 10 > 0) : Prop :=
  (150 / x = 120 / (x - 10))

-- The theorem to be proven
theorem correct_equation (h₁ : x - 10 > 0) : 150 / x = 120 / (x - 10) :=
by
  sorry

end correct_equation_l1689_168975


namespace condition_for_a_l1689_168958

theorem condition_for_a (a : ℝ) :
  (∀ x : ℤ, (x < 0 → (x + a) / 2 ≥ 1) → (x = -1 ∨ x = -2)) ↔ 4 ≤ a ∧ a < 5 :=
by
  sorry

end condition_for_a_l1689_168958


namespace find_number_l1689_168969

theorem find_number (N p q : ℝ) 
  (h1 : N / p = 6) 
  (h2 : N / q = 18) 
  (h3 : p - q = 1 / 3) : 
  N = 3 := 
by 
  sorry

end find_number_l1689_168969


namespace parallelepiped_intersection_l1689_168914

/-- Given a parallelepiped A B C D A₁ B₁ C₁ D₁.
    Point X is chosen on edge A₁ D₁, and point Y is chosen on edge B C.
    It is known that A₁ X = 5, B Y = 3, and B₁ C₁ = 14.
    The plane C₁ X Y intersects ray D A at point Z.
    Prove that D Z = 20. -/
theorem parallelepiped_intersection
  (A B C D A₁ B₁ C₁ D₁ X Y Z : ℝ)
  (h₁: A₁ - X = 5)
  (h₂: B - Y = 3)
  (h₃: B₁ - C₁ = 14) :
  D - Z = 20 :=
sorry

end parallelepiped_intersection_l1689_168914


namespace algebraic_expression_value_l1689_168979

open Real

theorem algebraic_expression_value
  (θ : ℝ)
  (a := (cos θ, sin θ))
  (b := (1, -2))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (2 * sin θ - cos θ) / (sin θ + cos θ) = 5 :=
by
  sorry

end algebraic_expression_value_l1689_168979


namespace min_value_correct_l1689_168920

noncomputable def min_value (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) : ℝ :=
(1 / m) + (2 / n)

theorem min_value_correct :
  ∃ m n : ℝ, ∃ h₁ : m > 0, ∃ h₂ : n > 0, ∃ h₃ : m + n = 1,
  min_value m n h₁ h₂ h₃ = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_correct_l1689_168920


namespace value_of_a_plus_b_minus_c_l1689_168972

theorem value_of_a_plus_b_minus_c (a b c : ℝ) 
  (h1 : abs a = 1) 
  (h2 : abs b = 2) 
  (h3 : abs c = 3) 
  (h4 : a > b) 
  (h5 : b > c) : 
  a + b - c = 2 := 
sorry

end value_of_a_plus_b_minus_c_l1689_168972


namespace simplify_eq_l1689_168926

theorem simplify_eq {x y z : ℕ} (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  9 * (x : ℝ) - ((10 / (2 * y) / 3 + 7 * z) * Real.pi) =
  9 * (x : ℝ) - (5 * Real.pi / (3 * y) + 7 * z * Real.pi) := by
  sorry

end simplify_eq_l1689_168926


namespace people_per_table_l1689_168956

def total_people_invited : ℕ := 68
def people_who_didn't_show_up : ℕ := 50
def number_of_tables_needed : ℕ := 6

theorem people_per_table (total_people_invited people_who_didn't_show_up number_of_tables_needed : ℕ) : 
  total_people_invited - people_who_didn't_show_up = 18 ∧
  (total_people_invited - people_who_didn't_show_up) / number_of_tables_needed = 3 :=
by
  sorry

end people_per_table_l1689_168956


namespace length_ac_l1689_168995

theorem length_ac (a b c d e : ℝ) (h1 : bc = 3 * cd) (h2 : de = 7) (h3 : ab = 5) (h4 : ae = 20) :
    ac = 11 :=
by
  sorry

end length_ac_l1689_168995


namespace sufficient_but_not_necessary_l1689_168973

variable (x : ℝ)

def condition_p := -1 ≤ x ∧ x ≤ 1
def condition_q := x ≥ -2

theorem sufficient_but_not_necessary :
  (condition_p x → condition_q x) ∧ ¬(condition_q x → condition_p x) :=
by 
  sorry

end sufficient_but_not_necessary_l1689_168973


namespace perimeter_triangle_PQR_is_24_l1689_168903

noncomputable def perimeter_triangle_PQR (QR PR : ℝ) : ℝ :=
  let PQ := Real.sqrt (QR^2 + PR^2)
  PQ + QR + PR

theorem perimeter_triangle_PQR_is_24 :
  perimeter_triangle_PQR 8 6 = 24 := by
  sorry

end perimeter_triangle_PQR_is_24_l1689_168903


namespace coral_remaining_pages_l1689_168941

def pages_after_week1 (total_pages : ℕ) : ℕ :=
  total_pages / 2

def pages_after_week2 (remaining_pages_week1 : ℕ) : ℕ :=
  remaining_pages_week1 - (3 * remaining_pages_week1 / 10)

def pages_after_week3 (remaining_pages_week2 : ℕ) (reading_hours : ℕ) (reading_speed : ℕ) : ℕ :=
  remaining_pages_week2 - (reading_hours * reading_speed)

theorem coral_remaining_pages (total_pages remaining_pages_week1 remaining_pages_week2 remaining_pages_week3 : ℕ) 
  (reading_hours reading_speed unread_pages : ℕ)
  (h1 : total_pages = 600)
  (h2 : remaining_pages_week1 = pages_after_week1 total_pages)
  (h3 : remaining_pages_week2 = pages_after_week2 remaining_pages_week1)
  (h4 : reading_hours = 10)
  (h5 : reading_speed = 15)
  (h6 : remaining_pages_week3 = pages_after_week3 remaining_pages_week2 reading_hours reading_speed)
  (h7 : unread_pages = remaining_pages_week3) :
  unread_pages = 60 :=
by
  sorry

end coral_remaining_pages_l1689_168941


namespace find_tuesday_temperature_l1689_168952

variable (T W Th F : ℝ)

def average_temperature_1 : Prop := (T + W + Th) / 3 = 52
def average_temperature_2 : Prop := (W + Th + F) / 3 = 54
def friday_temperature : Prop := F = 53

theorem find_tuesday_temperature (h1 : average_temperature_1 T W Th) (h2 : average_temperature_2 W Th F) (h3 : friday_temperature F) :
  T = 47 :=
by
  sorry

end find_tuesday_temperature_l1689_168952


namespace grill_run_time_l1689_168944

-- Definitions of conditions
def coals_burned_per_minute : ℕ := 15
def minutes_per_coal_burned : ℕ := 20
def coals_per_bag : ℕ := 60
def bags_burned : ℕ := 3

-- Theorems to prove the question
theorem grill_run_time (coals_burned_per_minute: ℕ) (minutes_per_coal_burned: ℕ) (coals_per_bag: ℕ) (bags_burned: ℕ): (coals_burned_per_minute * (minutes_per_coal_burned * bags_burned * coals_per_bag / (coals_burned_per_minute * coals_per_bag))) / 60 = 4 := 
by 
  -- Lean statement skips detailed proof steps for conciseness
  sorry

end grill_run_time_l1689_168944


namespace probability_symmetric_line_l1689_168945

theorem probability_symmetric_line (P : (ℕ × ℕ) := (5, 5))
    (n : ℕ := 10) (total_points remaining_points symmetric_points : ℕ) 
    (probability : ℚ) :
  total_points = n * n →
  remaining_points = total_points - 1 →
  symmetric_points = 4 * (n - 1) →
  probability = (symmetric_points : ℚ) / (remaining_points : ℚ) →
  probability = 32 / 99 :=
by
  sorry

end probability_symmetric_line_l1689_168945


namespace faye_homework_problems_left_l1689_168908

-- Defining the problem conditions
def M : ℕ := 46
def S : ℕ := 9
def A : ℕ := 40

-- The statement to prove
theorem faye_homework_problems_left : M + S - A = 15 := by
  sorry

end faye_homework_problems_left_l1689_168908


namespace temperature_difference_in_fahrenheit_l1689_168949

-- Define the conversion formula from Celsius to Fahrenheit as a function
def celsius_to_fahrenheit (C : ℝ) : ℝ := 1.8 * C + 32

-- Define the temperatures in Boston and New York
variables (C_B C_N : ℝ)

-- Condition: New York is 10 degrees Celsius warmer than Boston
axiom temp_difference : C_N = C_B + 10

-- Goal: The temperature difference in Fahrenheit
theorem temperature_difference_in_fahrenheit : celsius_to_fahrenheit C_N - celsius_to_fahrenheit C_B = 18 :=
by sorry

end temperature_difference_in_fahrenheit_l1689_168949


namespace ab_fraction_inequality_l1689_168927

theorem ab_fraction_inequality (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) :
  (a * b * (1 - a) * (1 - b)) / ((1 - a * b) ^ 2) < 1 / 4 :=
by
  sorry

end ab_fraction_inequality_l1689_168927


namespace sam_initial_pennies_l1689_168951

def initial_pennies_spent (spent: Nat) (left: Nat) : Nat :=
  spent + left

theorem sam_initial_pennies (spent: Nat) (left: Nat) : spent = 93 ∧ left = 5 → initial_pennies_spent spent left = 98 :=
by
  sorry

end sam_initial_pennies_l1689_168951


namespace range_of_b_l1689_168940

noncomputable def set_A : Set ℝ := {x | -2 < x ∧ x < 1/3}
noncomputable def set_B (b : ℝ) : Set ℝ := {x | x^2 - 4*b*x + 3*b^2 < 0}

theorem range_of_b (b : ℝ) : 
  (set_A ∩ set_B b = ∅) ↔ (b = 0 ∨ b ≥ 1/3 ∨ b ≤ -2) :=
sorry

end range_of_b_l1689_168940


namespace monotonic_decreasing_interval_l1689_168922

noncomputable def f (x : ℝ) : ℝ :=
  x / 4 + 5 / (4 * x) - Real.log x

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), (a = 0) ∧ (b = 5) ∧ (∀ x, 0 < x ∧ x < 5 → (deriv f x < 0)) :=
by
  sorry

end monotonic_decreasing_interval_l1689_168922


namespace smaller_part_area_l1689_168905

theorem smaller_part_area (x y : ℝ) (h1 : x + y = 500) (h2 : y - x = (1 / 5) * ((x + y) / 2)) : x = 225 :=
by
  sorry

end smaller_part_area_l1689_168905


namespace math_scores_population_l1689_168957

/-- 
   Suppose there are 50,000 students who took the high school entrance exam.
   The education department randomly selected 2,000 students' math scores 
   for statistical analysis. Prove that the math scores of the 50,000 students 
   are the population.
-/
theorem math_scores_population (students : ℕ) (selected : ℕ) 
    (students_eq : students = 50000) (selected_eq : selected = 2000) : 
    true :=
by
  sorry

end math_scores_population_l1689_168957


namespace passenger_difference_l1689_168934

theorem passenger_difference {x : ℕ} :
  (30 + x = 3 * x + 14) →
  6 = 3 * x - x - 16 :=
by
  sorry

end passenger_difference_l1689_168934


namespace area_of_hall_l1689_168929

-- Define the conditions
def length := 25
def breadth := length - 5

-- Define the area calculation
def area := length * breadth

-- The statement to prove
theorem area_of_hall : area = 500 :=
by
  sorry

end area_of_hall_l1689_168929


namespace opposite_z_is_E_l1689_168950

noncomputable def cube_faces := ["A", "B", "C", "D", "E", "z"]

def opposite_face (net : List String) (face : String) : String :=
  if face = "z" then "E" else sorry  -- generalize this function as needed

theorem opposite_z_is_E :
  opposite_face cube_faces "z" = "E" :=
by
  sorry

end opposite_z_is_E_l1689_168950


namespace fractions_equivalent_under_scaling_l1689_168996

theorem fractions_equivalent_under_scaling (a b d k x : ℝ) (h₀ : d ≠ 0) (h₁ : k ≠ 0) :
  (a * (k * x) + b) / (a * (k * x) + d) = (b * (k * x)) / (d * (k * x)) ↔ b = d :=
by sorry

end fractions_equivalent_under_scaling_l1689_168996


namespace find_value_of_a_l1689_168978

theorem find_value_of_a (a : ℝ) 
  (h : (2 * a + 16 + 3 * a - 8) / 2 = 69) : a = 26 := 
by
  sorry

end find_value_of_a_l1689_168978


namespace class_size_is_44_l1689_168932

theorem class_size_is_44 (n : ℕ) : 
  (n - 1) % 2 = 1 ∧ (n - 1) % 7 = 1 → n = 44 := 
by 
  sorry

end class_size_is_44_l1689_168932


namespace calories_per_burger_l1689_168989

-- Conditions given in the problem
def burgers_per_day : Nat := 3
def days : Nat := 2
def total_calories : Nat := 120

-- Total burgers Dimitri will eat in the given period
def total_burgers := burgers_per_day * days

-- Prove that the number of calories per burger is 20
theorem calories_per_burger : total_calories / total_burgers = 20 := 
by 
  -- Skipping the proof with 'sorry' as instructed
  sorry

end calories_per_burger_l1689_168989


namespace expression_evaluation_l1689_168971

theorem expression_evaluation :
  (0.8 ^ 3) - ((0.5 ^ 3) / (0.8 ^ 2)) + 0.40 + (0.5 ^ 2) = 0.9666875 := 
by 
  sorry

end expression_evaluation_l1689_168971


namespace new_supervisor_salary_l1689_168919

namespace FactorySalaries

variables (W S2 : ℝ)

def old_supervisor_salary : ℝ := 870
def old_average_salary : ℝ := 430
def new_average_salary : ℝ := 440

theorem new_supervisor_salary :
  (W + old_supervisor_salary) / 9 = old_average_salary →
  (W + S2) / 9 = new_average_salary →
  S2 = 960 :=
by
  intros h1 h2
  -- Proof steps would go here
  sorry

end FactorySalaries

end new_supervisor_salary_l1689_168919


namespace find_k_l1689_168993

-- Define the sequence and its sum
def Sn (k : ℝ) (n : ℕ) : ℝ := k + 3^n
def an (k : ℝ) (n : ℕ) : ℝ := Sn k n - (if n = 0 then 0 else Sn k (n - 1))

-- Define the condition that a sequence is geometric
def is_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = r * a n

theorem find_k (k : ℝ) :
  is_geometric (an k) (an k 1 / an k 0) → k = -1 := 
by sorry

end find_k_l1689_168993


namespace num_pairs_mod_eq_l1689_168917

theorem num_pairs_mod_eq (k : ℕ) (h : k ≥ 7) :
  ∃ n : ℕ, n = 2^(k+5) ∧
  (∀ x y : ℕ, 0 ≤ x ∧ x < 2^k ∧ 0 ≤ y ∧ y < 2^k → (73^(73^x) ≡ 9^(9^y) [MOD 2^k]) → true) :=
sorry

end num_pairs_mod_eq_l1689_168917


namespace andrew_stamps_permits_l1689_168980

theorem andrew_stamps_permits (n a T r permits : ℕ)
  (h1 : n = 2)
  (h2 : a = 3)
  (h3 : T = 8)
  (h4 : r = 50)
  (h5 : permits = (T - n * a) * r) :
  permits = 100 :=
by
  rw [h1, h2, h3, h4] at h5
  norm_num at h5
  exact h5

end andrew_stamps_permits_l1689_168980


namespace min_distance_to_water_all_trees_l1689_168942

/-- Proof that the minimum distance Xiao Zhang must walk to water all 10 trees is 410 meters -/
def minimum_distance_to_water_trees (num_trees : ℕ) (distance_between_trees : ℕ) : ℕ := 
  (sorry) -- implementation to calculate the minimum distance

theorem min_distance_to_water_all_trees (num_trees distance_between_trees : ℕ) :
  num_trees = 10 → 
  distance_between_trees = 10 →
  minimum_distance_to_water_trees num_trees distance_between_trees = 410 :=
by
  intros h_num_trees h_distance_between_trees
  rw [h_num_trees, h_distance_between_trees]
  -- Add proof here that the distance is 410
  sorry

end min_distance_to_water_all_trees_l1689_168942


namespace range_of_a_l1689_168981

theorem range_of_a 
  (f : ℝ → ℝ)
  (h_even : ∀ x, -5 ≤ x ∧ x ≤ 5 → f x = f (-x))
  (h_decreasing : ∀ a b, 0 ≤ a ∧ a < b ∧ b ≤ 5 → f b < f a)
  (h_inequality : ∀ a, f (2 * a + 3) < f a) :
  ∀ a, -5 ≤ a ∧ a ≤ 5 → a ∈ (Set.Icc (-4) (-3) ∪ Set.Ioc (-1) 1) := 
by
  sorry

end range_of_a_l1689_168981


namespace percentage_not_speak_french_l1689_168962

open Nat

theorem percentage_not_speak_french (students_surveyed : ℕ)
  (speak_french_and_english : ℕ) (speak_only_french : ℕ) :
  students_surveyed = 200 →
  speak_french_and_english = 25 →
  speak_only_french = 65 →
  ((students_surveyed - (speak_french_and_english + speak_only_french)) * 100 / students_surveyed) = 55 :=
by
  intros h1 h2 h3
  sorry

end percentage_not_speak_french_l1689_168962


namespace total_pages_of_book_l1689_168997

theorem total_pages_of_book (P : ℝ) (h : 0.4 * P = 16) : P = 40 :=
sorry

end total_pages_of_book_l1689_168997


namespace transaction_result_l1689_168904

theorem transaction_result
  (house_selling_price store_selling_price : ℝ)
  (house_loss_perc : ℝ)
  (store_gain_perc : ℝ)
  (house_selling_price_eq : house_selling_price = 15000)
  (store_selling_price_eq : store_selling_price = 15000)
  (house_loss_perc_eq : house_loss_perc = 0.1)
  (store_gain_perc_eq : store_gain_perc = 0.3) :
  (store_selling_price + house_selling_price - ((house_selling_price / (1 - house_loss_perc)) + (store_selling_price / (1 + store_gain_perc)))) = 1795 :=
by
  sorry

end transaction_result_l1689_168904


namespace correct_statements_l1689_168966

open Classical

variables {α l m n p : Type*}
variables (is_perpendicular_to : α → α → Prop) (is_parallel_to : α → α → Prop)
variables (is_in_plane : α → α → Prop)

noncomputable def problem_statement (l : α) (α : α) : Prop :=
  (∀ m, is_perpendicular_to m l → is_parallel_to m α) ∧
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α)

theorem correct_statements (l : α) (α : α) (h_l_α : is_perpendicular_to l α) :
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α) :=
sorry

end correct_statements_l1689_168966


namespace problem_statement_l1689_168911

variables {AB CD BC DA : ℝ} (E : ℝ) (midpoint_E : E = BC / 2) (ins_ABC : circle_inscribable AB ED)
  (ins_AEC : circle_inscribable AE CD) (a b c d : ℝ) (h_AB : AB = a) (h_BC : BC = b) (h_CD : CD = c)
  (h_DA : DA = d)

theorem problem_statement :
  a + c = b / 3 + d ∧ (1 / a + 1 / c = 3 / b) :=
by
  sorry

end problem_statement_l1689_168911


namespace find_other_number_l1689_168930

theorem find_other_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 83) (h3 : A = 210) (h4 : LCM * HCF = A * B) : B = 913 :=
by
  sorry

end find_other_number_l1689_168930


namespace abs_neg_one_third_l1689_168906

theorem abs_neg_one_third : abs (-1/3) = 1/3 := by
  sorry

end abs_neg_one_third_l1689_168906


namespace emma_possible_lists_l1689_168965

-- Define the number of balls
def number_of_balls : ℕ := 24

-- Define the number of draws Emma repeats independently
def number_of_draws : ℕ := 4

-- Define the calculation for the total number of different lists
def total_number_of_lists : ℕ := number_of_balls ^ number_of_draws

theorem emma_possible_lists : total_number_of_lists = 331776 := by
  sorry

end emma_possible_lists_l1689_168965


namespace math_problem_proof_l1689_168977

variable (Zhang Li Wang Zhao Liu : Prop)
variable (n : ℕ)
variable (reviewed_truth : Zhang → n = 0 ∧ Li → n = 1 ∧ Wang → n = 2 ∧ Zhao → n = 3 ∧ Liu → n = 4)
variable (reviewed_lie : ¬Zhang → ¬(n = 0) ∧ ¬Li → ¬(n = 1) ∧ ¬Wang → ¬(n = 2) ∧ ¬Zhao → ¬(n = 3) ∧ ¬Liu → ¬(n = 4))
variable (some_reviewed : ∃ x, x ∧ ¬x)

theorem math_problem_proof: n = 1 :=
by
  -- Proof omitted, insert logic here
  sorry

end math_problem_proof_l1689_168977


namespace domain_log_function_l1689_168918

open Real

def quadratic_term (x : ℝ) : ℝ := 4 - 3 * x - x^2

def valid_argument (x : ℝ) : Prop := quadratic_term x > 0

theorem domain_log_function : { x : ℝ | valid_argument x } = Set.Ioo (-4 : ℝ) (1 : ℝ) :=
by
  sorry

end domain_log_function_l1689_168918


namespace harry_morning_routine_l1689_168946

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end harry_morning_routine_l1689_168946


namespace james_shirts_l1689_168938

theorem james_shirts (S P : ℕ) (h1 : P = S / 2) (h2 : 6 * S + 8 * P = 100) : S = 10 :=
sorry

end james_shirts_l1689_168938


namespace time_to_cover_escalator_l1689_168999

-- Definitions for the provided conditions.
def escalator_speed : ℝ := 7
def escalator_length : ℝ := 180
def person_speed : ℝ := 2

-- Goal to prove the time taken to cover the escalator length.
theorem time_to_cover_escalator : (escalator_length / (escalator_speed + person_speed)) = 20 := by
  sorry

end time_to_cover_escalator_l1689_168999


namespace find_central_angle_l1689_168901

variable (L : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions
def arc_length_condition : Prop := L = 200
def radius_condition : Prop := r = 2
def arc_length_formula : Prop := L = r * α

-- Theorem statement
theorem find_central_angle 
  (hL : arc_length_condition L) 
  (hr : radius_condition r) 
  (hf : arc_length_formula L r α) : 
  α = 100 := by
  -- Proof goes here
  sorry

end find_central_angle_l1689_168901


namespace solve_inequality_l1689_168948

theorem solve_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2)
  (h3 : (x^2 + 3*x - 1) / (4 - x^2) < 1)
  (h4 : (x^2 + 3*x - 1) / (4 - x^2) ≥ -1) :
  x < -5 / 2 ∨ (-1 ≤ x ∧ x < 1) :=
by sorry

end solve_inequality_l1689_168948


namespace isosceles_triangle_smallest_angle_l1689_168985

-- Given conditions:
-- 1. The triangle is isosceles
-- 2. One angle is 40% larger than the measure of a right angle

theorem isosceles_triangle_smallest_angle :
  ∃ (A B C : ℝ), 
  A + B + C = 180 ∧ 
  (A = B ∨ A = C ∨ B = C) ∧ 
  (∃ (large_angle : ℝ), large_angle = 90 + 0.4 * 90 ∧ (A = large_angle ∨ B = large_angle ∨ C = large_angle)) →
  (A = 27 ∨ B = 27 ∨ C = 27) := sorry

end isosceles_triangle_smallest_angle_l1689_168985


namespace cubic_roots_equal_l1689_168921

theorem cubic_roots_equal (k : ℚ) (h1 : k > 0)
  (h2 : ∃ a b : ℚ, a ≠ b ∧ (a + a + b = -3) ∧ (2 * a * b + a^2 = -54) ∧ (3 * x^3 + 9 * x^2 - 162 * x + k = 0)) : 
  k = 7983 / 125 :=
sorry

end cubic_roots_equal_l1689_168921


namespace A_can_give_C_start_l1689_168986

noncomputable def start_A_can_give_C : ℝ :=
  let start_AB := 50
  let start_BC := 157.89473684210532
  start_AB + start_BC

theorem A_can_give_C_start :
  start_A_can_give_C = 207.89473684210532 :=
by
  sorry

end A_can_give_C_start_l1689_168986


namespace largest_sequence_sum_45_l1689_168967

theorem largest_sequence_sum_45 
  (S: ℕ → ℕ)
  (h_S: ∀ n, S n = n * (n + 1) / 2)
  (h_sum: ∃ m: ℕ, S m = 45):
  (∃ k: ℕ, k ≤ 9 ∧ S k = 45) ∧ (∀ m: ℕ, S m ≤ 45 → m ≤ 9) :=
by
  sorry

end largest_sequence_sum_45_l1689_168967


namespace fourth_throw_probability_l1689_168924

-- Define a fair dice where each face has an equal probability.
def fair_dice (n : ℕ) : Prop := (n >= 1 ∧ n <= 6)

-- Define the probability of rolling a 6 on a fair dice.
noncomputable def probability_of_6 : ℝ := 1 / 6

/-- 
  Prove that the probability of getting a "6" on the 4th throw is 1/6 
  given that the dice is fair and the first three throws result in "6".
-/
theorem fourth_throw_probability : 
  (∀ (n1 n2 n3 : ℕ), fair_dice n1 ∧ fair_dice n2 ∧ fair_dice n3 ∧ n1 = 6 ∧ n2 = 6 ∧ n3 = 6) 
  → (probability_of_6 = 1 / 6) :=
by 
  sorry

end fourth_throw_probability_l1689_168924


namespace part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l1689_168959

-- Part 1: Prove existence of rectangle B with sides 2 + sqrt(2)/2 and 2 - sqrt(2)/2
theorem part1_exists_rectangle_B : 
  ∃ (x y : ℝ), (x + y = 4) ∧ (x * y = 7 / 2) :=
by
  sorry

-- Part 2: Prove non-existence of rectangle B for given sides of the known rectangle
theorem part2_no_rectangle_B : 
  ¬ ∃ (x y : ℝ), (x + y = 5 / 2) ∧ (x * y = 2) :=
by
  sorry

-- Part 3: General proof for any given sides of the known rectangle
theorem general_exists_rectangle_B (m n : ℝ) : 
  ∃ (x y : ℝ), (x + y = 3 * (m + n)) ∧ (x * y = 3 * m * n) :=
by
  sorry

end part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l1689_168959


namespace rhombus_side_length_l1689_168935

theorem rhombus_side_length (s : ℝ) (h : 4 * s = 32) : s = 8 :=
by
  sorry

end rhombus_side_length_l1689_168935


namespace probability_adjacent_vertices_in_decagon_l1689_168947

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l1689_168947


namespace bug_crawl_distance_l1689_168910

-- Define the positions visited by the bug
def start_position := -3
def first_stop := 0
def second_stop := -8
def final_stop := 10

-- Define the function to calculate the total distance crawled by the bug
def total_distance : ℤ :=
  abs (first_stop - start_position) + abs (second_stop - first_stop) + abs (final_stop - second_stop)

-- Prove that the total distance is 29 units
theorem bug_crawl_distance : total_distance = 29 :=
by
  -- Definitions are used here to validate the statement
  sorry

end bug_crawl_distance_l1689_168910


namespace expected_rolls_in_non_leap_year_l1689_168915

-- Define the conditions and the expected value
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def stops_rolling (n : ℕ) : Prop := is_prime n ∨ is_multiple_of_4 n

def expected_rolls_one_day : ℚ := 6 / 7

def non_leap_year_days : ℕ := 365

def expected_rolls_one_year := expected_rolls_one_day * non_leap_year_days

theorem expected_rolls_in_non_leap_year : expected_rolls_one_year = 314 :=
by
  -- Verification of the mathematical model
  sorry

end expected_rolls_in_non_leap_year_l1689_168915


namespace total_surface_area_of_three_face_painted_cubes_l1689_168923

def cube_side_length : ℕ := 9
def small_cube_side_length : ℕ := 1
def num_small_cubes_with_three_faces_painted : ℕ := 8
def surface_area_of_each_painted_face : ℕ := 6

theorem total_surface_area_of_three_face_painted_cubes :
  num_small_cubes_with_three_faces_painted * surface_area_of_each_painted_face = 48 := by
  sorry

end total_surface_area_of_three_face_painted_cubes_l1689_168923


namespace compute_sum_of_squares_roots_l1689_168936

-- p, q, and r are roots of 3*x^3 - 2*x^2 + 6*x + 15 = 0.
def P (x : ℝ) : Prop := 3*x^3 - 2*x^2 + 6*x + 15 = 0

theorem compute_sum_of_squares_roots :
  ∀ p q r : ℝ, P p ∧ P q ∧ P r → p^2 + q^2 + r^2 = -32 / 9 :=
by
  intros p q r h
  sorry

end compute_sum_of_squares_roots_l1689_168936


namespace avg_growth_rate_first_brand_eq_l1689_168909

noncomputable def avg_growth_rate_first_brand : ℝ :=
  let t := 5.647
  let first_brand_households_2001 := 4.9
  let second_brand_households_2001 := 2.5
  let second_brand_growth_rate := 0.7
  let equalization_time := t
  (second_brand_households_2001 + second_brand_growth_rate * equalization_time - first_brand_households_2001) / equalization_time

theorem avg_growth_rate_first_brand_eq :
  avg_growth_rate_first_brand = 0.275 := by
  sorry

end avg_growth_rate_first_brand_eq_l1689_168909


namespace age_difference_l1689_168991

theorem age_difference 
  (A B : ℤ) 
  (h1 : B = 39) 
  (h2 : A + 10 = 2 * (B - 10)) :
  A - B = 9 := 
by 
  sorry

end age_difference_l1689_168991


namespace non_monotonic_piecewise_l1689_168943

theorem non_monotonic_piecewise (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ (x t : ℝ),
    (f x = if x ≤ t then (4 * a - 3) * x + (2 * a - 4) else (2 * x^3 - 6 * x)))
  : a ≤ 3 / 4 := 
sorry

end non_monotonic_piecewise_l1689_168943


namespace min_frac_sum_pos_real_l1689_168912

variable {x y z w : ℝ}

theorem min_frac_sum_pos_real (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h_sum : x + y + z + w = 1) : 
  (x + y + z) / (x * y * z * w) ≥ 144 := 
sorry

end min_frac_sum_pos_real_l1689_168912


namespace log_eq_solution_l1689_168998

open Real

noncomputable def solve_log_eq : Real :=
  let x := 62.5^(1/3)
  x

theorem log_eq_solution (x : Real) (hx : 3 * log x - 4 * log 5 = -1) :
  x = solve_log_eq :=
by
  sorry

end log_eq_solution_l1689_168998


namespace central_angle_of_unfolded_side_surface_l1689_168976

theorem central_angle_of_unfolded_side_surface
  (radius : ℝ) (slant_height : ℝ) (arc_length : ℝ) (central_angle_deg : ℝ)
  (h_radius : radius = 1)
  (h_slant_height : slant_height = 3)
  (h_arc_length : arc_length = 2 * Real.pi) :
  central_angle_deg = 120 :=
by
  sorry

end central_angle_of_unfolded_side_surface_l1689_168976


namespace exists_odd_integers_l1689_168928

theorem exists_odd_integers (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, x % 2 = 1 ∧ y % 2 = 1 ∧ x^2 + 7 * y^2 = 2^n :=
sorry

end exists_odd_integers_l1689_168928


namespace remaining_numbers_l1689_168900

theorem remaining_numbers (S S3 S2 N : ℕ) (h1 : S / 5 = 8) (h2 : S3 / 3 = 4) (h3 : S2 / N = 14) 
(hS  : S = 5 * 8) (hS3 : S3 = 3 * 4) (hS2 : S2 = S - S3) : N = 2 := by
  sorry

end remaining_numbers_l1689_168900


namespace cyclists_cannot_reach_point_B_l1689_168992

def v1 := 35 -- Speed of the first cyclist in km/h
def v2 := 25 -- Speed of the second cyclist in km/h
def t := 2   -- Total time in hours
def d  := 30 -- Distance from A to B in km

-- Each cyclist does not rest simultaneously
-- Time equations based on their speed proportions

theorem cyclists_cannot_reach_point_B 
  (v1 := 35) (v2 := 25) (t := 2) (d := 30) 
  (h1 : t * (v1 * (5 / (5 + 7)) / 60) + t * (v2 * (7 / (5 + 7)) / 60) < d) : 
  False := 
sorry

end cyclists_cannot_reach_point_B_l1689_168992


namespace find_f_when_x_lt_0_l1689_168960

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_defined (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 2 * x

theorem find_f_when_x_lt_0 (f : ℝ → ℝ) (h_odd : odd_function f) (h_defined : f_defined f) :
  ∀ x < 0, f x = -x^2 - 2 * x :=
by
  sorry

end find_f_when_x_lt_0_l1689_168960


namespace imaginary_part_is_empty_l1689_168970

def imaginary_part_empty (z : ℂ) : Prop :=
  z.im = 0

theorem imaginary_part_is_empty (z : ℂ) (h : z.im = 0) : imaginary_part_empty z :=
by
  -- proof skipped
  sorry

end imaginary_part_is_empty_l1689_168970


namespace jogging_track_circumference_l1689_168937

def speed_Suresh_km_hr : ℝ := 4.5
def speed_wife_km_hr : ℝ := 3.75
def meet_time_min : ℝ := 5.28

theorem jogging_track_circumference : 
  let speed_Suresh_km_min := speed_Suresh_km_hr / 60
  let speed_wife_km_min := speed_wife_km_hr / 60
  let distance_Suresh_km := speed_Suresh_km_min * meet_time_min
  let distance_wife_km := speed_wife_km_min * meet_time_min
  let total_distance_km := distance_Suresh_km + distance_wife_km
  total_distance_km = 0.726 :=
by sorry

end jogging_track_circumference_l1689_168937


namespace solve_linear_eq_l1689_168984

theorem solve_linear_eq (x : ℝ) : (x + 1) / 3 = 0 → x = -1 := 
by 
  sorry

end solve_linear_eq_l1689_168984


namespace quadratic_roots_bounds_l1689_168902

theorem quadratic_roots_bounds (m x1 x2 : ℝ) (h : m < 0)
  (hx : x1 < x2) 
  (hr : ∀ x, x^2 - x - 6 = m → x = x1 ∨ x = x2) :
  -2 < x1 ∧ x2 < 3 :=
by
  sorry

end quadratic_roots_bounds_l1689_168902


namespace fundraiser_goal_l1689_168974

theorem fundraiser_goal (bronze_donation silver_donation gold_donation goal : ℕ)
  (bronze_families silver_families gold_family : ℕ)
  (H_bronze_amount : bronze_families * bronze_donation = 250)
  (H_silver_amount : silver_families * silver_donation = 350)
  (H_gold_amount : gold_family * gold_donation = 100)
  (H_goal : goal = 750) :
  goal - (bronze_families * bronze_donation + silver_families * silver_donation + gold_family * gold_donation) = 50 :=
by
  sorry

end fundraiser_goal_l1689_168974


namespace no_solution_inequality_l1689_168982

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  sorry

end no_solution_inequality_l1689_168982


namespace classify_triangle_l1689_168913

theorem classify_triangle (m : ℕ) (h₁ : m > 1) (h₂ : 3 * m + 3 = 180) :
  (m < 60) ∧ (m + 1 < 90) ∧ (m + 2 < 90) :=
by
  sorry

end classify_triangle_l1689_168913


namespace jiwoo_magnets_two_digit_count_l1689_168964

def num_magnets : List ℕ := [1, 2, 7]

theorem jiwoo_magnets_two_digit_count : 
  (∀ (x y : ℕ), x ≠ y → x ∈ num_magnets → y ∈ num_magnets → 2 * 3 = 6) := 
by {
  sorry
}

end jiwoo_magnets_two_digit_count_l1689_168964


namespace circumference_of_wheels_l1689_168987

-- Define the variables and conditions
variables (x y : ℝ)

def condition1 (x y : ℝ) : Prop := (120 / x) - (120 / y) = 6
def condition2 (x y : ℝ) : Prop := (4 / 5) * (120 / x) - (5 / 6) * (120 / y) = 4

-- The main theorem to prove
theorem circumference_of_wheels (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 4 ∧ y = 5 :=
  sorry  -- Proof is omitted

end circumference_of_wheels_l1689_168987
