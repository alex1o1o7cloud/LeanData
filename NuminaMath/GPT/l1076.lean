import Mathlib

namespace sum_of_first_nine_terms_l1076_107618

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a n = a 1 + d * (n - 1)

variables (a : ℕ → ℝ) (h_seq : arithmetic_sequence a)

-- Given condition: a₂ + a₃ + a₇ + a₈ = 20
def condition : Prop := a 2 + a 3 + a 7 + a 8 = 20

-- Statement: Prove that the sum of the first 9 terms is 45
theorem sum_of_first_nine_terms (h : condition a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 45 :=
by sorry

end sum_of_first_nine_terms_l1076_107618


namespace exponentiation_multiplication_l1076_107604

theorem exponentiation_multiplication (a : ℝ) : a^6 * a^2 = a^8 :=
by sorry

end exponentiation_multiplication_l1076_107604


namespace work_rate_ab_together_l1076_107676

-- Define A, B, and C as the work rates of individuals
variables (A B C : ℝ)

-- We are given the following conditions:
-- 1. a, b, and c together can finish the job in 11 days
-- 2. c alone can finish the job in 41.25 days

-- Given these conditions, we aim to prove that a and b together can finish the job in 15 days
theorem work_rate_ab_together
  (h1 : A + B + C = 1 / 11)
  (h2 : C = 1 / 41.25) :
  1 / (A + B) = 15 :=
by
  sorry

end work_rate_ab_together_l1076_107676


namespace total_cookies_sold_l1076_107680

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end total_cookies_sold_l1076_107680


namespace oppose_estimation_l1076_107607

-- Define the conditions
def survey_total : ℕ := 50
def favorable_attitude : ℕ := 15
def total_population : ℕ := 9600

-- Calculate the proportion opposed
def proportion_opposed : ℚ := (survey_total - favorable_attitude) / survey_total

-- Define the statement to be proved
theorem oppose_estimation : 
  proportion_opposed * total_population = 6720 := by
  sorry

end oppose_estimation_l1076_107607


namespace solution_set_inequality_l1076_107699

theorem solution_set_inequality (x : ℝ) : 
  (abs (x + 3) - abs (x - 2) ≥ 3) ↔ (x ≥ 1) := 
by {
  sorry
}

end solution_set_inequality_l1076_107699


namespace calculate_S_value_l1076_107634

def operation_S (a b : ℕ) : ℕ := 4 * a + 7 * b

theorem calculate_S_value : operation_S 8 3 = 53 :=
by
  -- proof goes here
  sorry

end calculate_S_value_l1076_107634


namespace om_4_2_eq_18_l1076_107641

def om (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem om_4_2_eq_18 : om 4 2 = 18 :=
by
  sorry

end om_4_2_eq_18_l1076_107641


namespace equation_represents_hyperbola_l1076_107639

theorem equation_represents_hyperbola (x y : ℝ) :
  x^2 - 4*y^2 - 2*x + 8*y - 8 = 0 → ∃ a b h k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a * (x - h)^2 - b * (y - k)^2 = 1) := 
sorry

end equation_represents_hyperbola_l1076_107639


namespace lucy_cleans_aquariums_l1076_107610

theorem lucy_cleans_aquariums :
  (∃ rate : ℕ, rate = 2 / 3) →
  (∃ hours : ℕ, hours = 24) →
  (∃ increments : ℕ, increments = 24 / 3) →
  (∃ aquariums : ℕ, aquariums = (2 * (24 / 3))) →
  aquariums = 16 :=
by
  sorry

end lucy_cleans_aquariums_l1076_107610


namespace probability_three_even_dice_l1076_107666

theorem probability_three_even_dice :
  let p_even := 1 / 2
  let combo := Nat.choose 5 3
  let probability := combo * (p_even ^ 3) * ((1 - p_even) ^ 2)
  probability = 5 / 16 := 
by
  sorry

end probability_three_even_dice_l1076_107666


namespace dragon_legs_l1076_107667

variable {x y n : ℤ}

theorem dragon_legs :
  (x = 40) ∧
  (y = 9) ∧
  (220 = 40 * x + n * y) →
  n = 4 :=
by
  sorry

end dragon_legs_l1076_107667


namespace container_capacity_l1076_107698

theorem container_capacity (C : ℝ) 
  (h1 : (0.30 * C : ℝ) + 27 = 0.75 * C) : C = 60 :=
sorry

end container_capacity_l1076_107698


namespace rectangle_length_width_l1076_107679

theorem rectangle_length_width (x y : ℝ) 
  (h1 : 2 * x + 2 * y = 16) 
  (h2 : x - y = 1) : 
  x = 4.5 ∧ y = 3.5 :=
by {
  sorry
}

end rectangle_length_width_l1076_107679


namespace largest_value_is_E_l1076_107649

-- Define the given values
def A := 1 - 0.1
def B := 1 - 0.01
def C := 1 - 0.001
def D := 1 - 0.0001
def E := 1 - 0.00001

-- Main theorem statement
theorem largest_value_is_E : E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end largest_value_is_E_l1076_107649


namespace complement_of_A_in_U_l1076_107624

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 2, 4, 5}

-- Proof statement
theorem complement_of_A_in_U : (U \ A) = {3, 6, 7} := by
  sorry

end complement_of_A_in_U_l1076_107624


namespace compute_product_sum_l1076_107606

theorem compute_product_sum (a b c : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  (a * b * c) * ((1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c) = 47 :=
by
  sorry

end compute_product_sum_l1076_107606


namespace VIP_ticket_price_l1076_107635

variable (total_savings : ℕ) 
variable (num_VIP_tickets : ℕ)
variable (num_regular_tickets : ℕ)
variable (price_per_regular_ticket : ℕ)
variable (remaining_savings : ℕ)

theorem VIP_ticket_price 
  (h1 : total_savings = 500)
  (h2 : num_VIP_tickets = 2)
  (h3 : num_regular_tickets = 3)
  (h4 : price_per_regular_ticket = 50)
  (h5 : remaining_savings = 150) :
  (total_savings - remaining_savings) - (num_regular_tickets * price_per_regular_ticket) = num_VIP_tickets * 100 := 
by
  sorry

end VIP_ticket_price_l1076_107635


namespace Sam_bought_cards_l1076_107611

theorem Sam_bought_cards (original_cards current_cards : ℕ) 
  (h1 : original_cards = 87) (h2 : current_cards = 74) : 
  original_cards - current_cards = 13 :=
by
  -- The 'sorry' here means the proof is omitted.
  sorry

end Sam_bought_cards_l1076_107611


namespace average_speed_l1076_107637

theorem average_speed (d d1 d2 s1 s2 : ℝ)
    (h1 : d = 100)
    (h2 : d1 = 50)
    (h3 : d2 = 50)
    (h4 : s1 = 20)
    (h5 : s2 = 50) :
    d / ((d1 / s1) + (d2 / s2)) = 28.57 :=
by
  sorry

end average_speed_l1076_107637


namespace trig_identity_l1076_107628

open Real

theorem trig_identity (α : ℝ) (h : 2 * sin α + cos α = 0) : 
  2 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = -12 / 5 :=
sorry

end trig_identity_l1076_107628


namespace unique_y_for_star_eq_9_l1076_107669

def star (x y : ℝ) : ℝ := 3 * x - 2 * y + x^2 * y

theorem unique_y_for_star_eq_9 : ∃! y : ℝ, star 2 y = 9 := by
  sorry

end unique_y_for_star_eq_9_l1076_107669


namespace distinctKeyArrangements_l1076_107654

-- Given conditions as definitions in Lean.
def houseNextToCar : Prop := sorry
def officeNextToBike : Prop := sorry
def noDifferenceByRotationOrReflection (arr1 arr2 : List ℕ) : Prop := sorry

-- Main statement to be proven
theorem distinctKeyArrangements : 
  houseNextToCar ∧ officeNextToBike ∧ (∀ (arr1 arr2 : List ℕ), noDifferenceByRotationOrReflection arr1 arr2 ↔ arr1 = arr2) 
  → ∃ n : ℕ, n = 16 :=
by sorry

end distinctKeyArrangements_l1076_107654


namespace kibble_left_l1076_107651

-- Define the initial amount of kibble
def initial_kibble := 3

-- Define the rate at which the cat eats kibble
def kibble_rate := 1 / 4

-- Define the time Kira was away
def time_away := 8

-- Define the amount of kibble eaten by the cat during the time away
def kibble_eaten := (time_away * kibble_rate)

-- Define the remaining kibble in the bowl
def remaining_kibble := initial_kibble - kibble_eaten

-- State and prove that the remaining amount of kibble is 1 pound
theorem kibble_left : remaining_kibble = 1 := by
  sorry

end kibble_left_l1076_107651


namespace number_in_scientific_notation_l1076_107617

/-- Condition: A constant corresponding to the number we are converting. -/
def number : ℕ := 9000000000

/-- Condition: The correct answer we want to prove. -/
def correct_answer : ℕ := 9 * 10^9

/-- Proof Problem: Prove that the number equals the correct_answer when expressed in scientific notation. -/
theorem number_in_scientific_notation : number = correct_answer := by
  sorry

end number_in_scientific_notation_l1076_107617


namespace min_y_value_l1076_107678

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 7*x + 10) / (x + 1)

theorem min_y_value : ∀ x > -1, f x ≥ 9 :=
by sorry

end min_y_value_l1076_107678


namespace option_A_correct_l1076_107619

variable (f g : ℝ → ℝ)

-- Given conditions
axiom cond1 : ∀ x : ℝ, f x - g (4 - x) = 2
axiom cond2 : ∀ x : ℝ, deriv g x = deriv f (x - 2)
axiom cond3 : ∀ x : ℝ, f (x + 2) = - f (- x - 2)

theorem option_A_correct : ∀ x : ℝ, f (4 + x) + f (- x) = 0 :=
by
  -- Proving the theorem
  sorry

end option_A_correct_l1076_107619


namespace least_common_duration_l1076_107623

theorem least_common_duration 
    (P Q R : ℝ) 
    (x : ℝ)
    (T : ℝ)
    (h1 : P / Q = 7 / 5)
    (h2 : Q / R = 5 / 3)
    (h3 : 8 * P / (6 * Q) = 7 / 10)
    (h4 : (6 * 10) * R / (30 * T) = 1)
    : T = 6 :=
by
  sorry

end least_common_duration_l1076_107623


namespace cauliflower_sales_l1076_107675

noncomputable def broccoli_sales : ℝ := 57
noncomputable def carrot_sales : ℝ := 2 * broccoli_sales
noncomputable def spinach_sales : ℝ := 16 + (1 / 2 * carrot_sales)
noncomputable def total_sales : ℝ := 380
noncomputable def other_sales : ℝ := broccoli_sales + carrot_sales + spinach_sales

theorem cauliflower_sales :
  total_sales - other_sales = 136 :=
by
  -- proof skipped
  sorry

end cauliflower_sales_l1076_107675


namespace milo_running_distance_l1076_107608

theorem milo_running_distance : 
  ∀ (cory_speed milo_skate_speed milo_run_speed time miles_run : ℕ),
  cory_speed = 12 →
  milo_skate_speed = cory_speed / 2 →
  milo_run_speed = milo_skate_speed / 2 →
  time = 2 →
  miles_run = milo_run_speed * time →
  miles_run = 6 :=
by 
  intros cory_speed milo_skate_speed milo_run_speed time miles_run hcory hmilo_skate hmilo_run htime hrun 
  -- Proof steps would go here
  sorry

end milo_running_distance_l1076_107608


namespace average_marks_all_students_proof_l1076_107683

-- Definitions based on the given conditions
def class1_student_count : ℕ := 35
def class2_student_count : ℕ := 45
def class1_average_marks : ℕ := 40
def class2_average_marks : ℕ := 60

-- Total marks calculations
def class1_total_marks : ℕ := class1_student_count * class1_average_marks
def class2_total_marks : ℕ := class2_student_count * class2_average_marks
def total_marks : ℕ := class1_total_marks + class2_total_marks

-- Total student count
def total_student_count : ℕ := class1_student_count + class2_student_count

-- Average marks of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_student_count

-- Lean statement to prove
theorem average_marks_all_students_proof
  (h1 : class1_student_count = 35)
  (h2 : class2_student_count = 45)
  (h3 : class1_average_marks = 40)
  (h4 : class2_average_marks = 60) :
  average_marks_all_students = 51.25 := by
  sorry

end average_marks_all_students_proof_l1076_107683


namespace final_temp_fahrenheit_correct_l1076_107645

noncomputable def initial_temp_celsius : ℝ := 50
noncomputable def conversion_c_to_f (c: ℝ) : ℝ := (c * 9 / 5) + 32
noncomputable def final_temp_celsius := initial_temp_celsius / 2

theorem final_temp_fahrenheit_correct : conversion_c_to_f final_temp_celsius = 77 :=
  by sorry

end final_temp_fahrenheit_correct_l1076_107645


namespace right_triangle_hypotenuse_length_l1076_107658

theorem right_triangle_hypotenuse_length (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 10 :=
by
  sorry

end right_triangle_hypotenuse_length_l1076_107658


namespace appropriate_sampling_method_l1076_107668

-- Defining the sizes of the boxes
def size_large : ℕ := 120
def size_medium : ℕ := 60
def size_small : ℕ := 20

-- Define a sample size
def sample_size : ℕ := 25

-- Define the concept of appropriate sampling method as being equivalent to stratified sampling in this context
theorem appropriate_sampling_method : 3 > 0 → sample_size > 0 → size_large = 120 ∧ size_medium = 60 ∧ size_small = 20 → 
("stratified sampling" = "stratified sampling") :=
by 
  sorry

end appropriate_sampling_method_l1076_107668


namespace magnitude_of_z_l1076_107659

open Complex

theorem magnitude_of_z (z : ℂ) (h : z + I = (2 + I) / I) : abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_z_l1076_107659


namespace fraction_of_products_inspected_jane_l1076_107682

theorem fraction_of_products_inspected_jane 
  (P : ℝ) 
  (J : ℝ) 
  (John_rejection_rate : ℝ) 
  (Jane_rejection_rate : ℝ)
  (Total_rejection_rate : ℝ) 
  (hJohn : John_rejection_rate = 0.005) 
  (hJane : Jane_rejection_rate = 0.008) 
  (hTotal : Total_rejection_rate = 0.0075) 
  : J = 5 / 6 := by
{
  sorry
}

end fraction_of_products_inspected_jane_l1076_107682


namespace math_problem_l1076_107643

noncomputable def x : ℝ := (Real.sqrt 5 + 1) / 2
noncomputable def y : ℝ := (Real.sqrt 5 - 1) / 2

theorem math_problem :
    x^3 * y + 2 * x^2 * y^2 + x * y^3 = 5 := 
by
  sorry

end math_problem_l1076_107643


namespace factors_and_multiple_of_20_l1076_107687

-- Define the relevant numbers
def a := 20
def b := 5
def c := 4

-- Given condition: the equation 20 / 5 = 4
def condition : Prop := a / b = c

-- Factors and multiples relationships to prove
def are_factors : Prop := a % b = 0 ∧ a % c = 0
def is_multiple : Prop := b * c = a

-- The main statement combining everything
theorem factors_and_multiple_of_20 (h : condition) : are_factors ∧ is_multiple :=
sorry

end factors_and_multiple_of_20_l1076_107687


namespace ratio_of_wire_lengths_l1076_107688

theorem ratio_of_wire_lengths 
  (bonnie_wire_length : ℕ := 80)
  (roark_wire_length : ℕ := 12000) :
  bonnie_wire_length / roark_wire_length = 1 / 150 :=
by
  sorry

end ratio_of_wire_lengths_l1076_107688


namespace shirt_original_price_l1076_107653

theorem shirt_original_price (original_price final_price : ℝ) (h1 : final_price = 0.5625 * original_price) 
  (h2 : final_price = 19) : original_price = 33.78 :=
by
  sorry

end shirt_original_price_l1076_107653


namespace find_x_l1076_107664

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉₊ * x = 198) : x = 13.2 :=
by
  sorry

end find_x_l1076_107664


namespace findC_coordinates_l1076_107602

-- Points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Defining points A, B, and stating that point C lies on the positive x-axis
def A : Point := {x := -4, y := -2}
def B : Point := {x := 0, y := -2}
def C (cx : ℝ) : Point := {x := cx, y := 0}

-- The condition that the triangle OBC is similar to triangle ABO
def isSimilar (A B O : Point) (C : Point) : Prop :=
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let OB := (B.x - O.x)^2 + (B.y - O.y)^2
  let OC := (C.x - O.x)^2 + (C.y - O.y)^2
  AB / OB = OB / OC

theorem findC_coordinates :
  ∃ (cx : ℝ), (C cx = {x := 1, y := 0} ∨ C cx = {x := 4, y := 0}) ∧
  isSimilar A B {x := 0, y := 0} (C cx) :=
by
  sorry

end findC_coordinates_l1076_107602


namespace tangent_line_ellipse_l1076_107600

theorem tangent_line_ellipse (x y : ℝ) (h : 2^2 / 8 + 1^2 / 2 = 1) :
    x / 4 + y / 2 = 1 := 
  sorry

end tangent_line_ellipse_l1076_107600


namespace probability_target_hit_l1076_107672

theorem probability_target_hit {P_A P_B : ℚ}
  (hA : P_A = 1 / 2) 
  (hB : P_B = 1 / 3) 
  : (1 - (1 - P_A) * (1 - P_B)) = 2 / 3 := 
by
  sorry

end probability_target_hit_l1076_107672


namespace sum_of_vars_l1076_107622

variables (a b c d k p : ℝ)

theorem sum_of_vars (h1 : a^2 + b^2 + c^2 + d^2 = 390)
                    (h2 : ab + bc + ca + ad + bd + cd = 5)
                    (h3 : ad + bd + cd = k)
                    (h4 : (a * b * c * d)^2 = p) :
                    a + b + c + d = 20 :=
by
  -- placeholder for the proof
  sorry

end sum_of_vars_l1076_107622


namespace domain_of_c_is_all_real_l1076_107615

theorem domain_of_c_is_all_real (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 3 * x + a ≠ 0) ↔ a < -3 / 4 :=
by
  sorry

end domain_of_c_is_all_real_l1076_107615


namespace circle_equation_correct_l1076_107621

-- Define the given elements: center and radius
def center : (ℝ × ℝ) := (1, -1)
def radius : ℝ := 2

-- Define the equation of the circle with the given center and radius
def circle_eqn (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = radius^2

-- Prove that the equation of the circle holds with the given center and radius
theorem circle_equation_correct : 
  ∀ x y : ℝ, circle_eqn x y ↔ (x - 1)^2 + (y + 1)^2 = 4 := 
by
  sorry

end circle_equation_correct_l1076_107621


namespace parabola_equation_l1076_107626

theorem parabola_equation (m : ℝ) (focus : ℝ × ℝ) (M : ℝ × ℝ) 
  (h_vertex : (0, 0) = (0, 0))
  (h_focus : focus = (p, 0))
  (h_point : M = (1, m))
  (h_distance : dist M focus = 2) 
  : (forall x y : ℝ, y^2 = 4*x) :=
sorry

end parabola_equation_l1076_107626


namespace expected_value_unfair_die_l1076_107633

theorem expected_value_unfair_die :
  let p8 := 3 / 8
  let p1_7 := (1 - p8) / 7
  let E := p1_7 * (1 + 2 + 3 + 4 + 5 + 6 + 7) + p8 * 8
  E = 5.5 := by
  sorry

end expected_value_unfair_die_l1076_107633


namespace salt_added_correct_l1076_107640

theorem salt_added_correct (x : ℝ)
  (hx : x = 119.99999999999996)
  (initial_salt : ℝ := 0.20 * x)
  (evaporation_volume : ℝ := x - (1/4) * x)
  (additional_water : ℝ := 8)
  (final_volume : ℝ := evaporation_volume + additional_water)
  (final_concentration : ℝ := 1 / 3)
  (final_salt : ℝ := final_concentration * final_volume)
  (salt_added : ℝ := final_salt - initial_salt) :
  salt_added = 8.67 :=
sorry

end salt_added_correct_l1076_107640


namespace problem_statement_l1076_107620

theorem problem_statement
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : ab + ac + bc ≠ 0) :
  (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
  sorry

end problem_statement_l1076_107620


namespace min_value_frac_function_l1076_107630

theorem min_value_frac_function (x : ℝ) (h : x > -1) : (x^2 / (x + 1)) ≥ 0 :=
sorry

end min_value_frac_function_l1076_107630


namespace minimum_value_of_expression_l1076_107631

noncomputable def min_value (a b : ℝ) : ℝ := 1 / a + 3 / b

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : min_value a b ≥ 16 := 
sorry

end minimum_value_of_expression_l1076_107631


namespace angle_B_is_60_l1076_107642

noncomputable def triangle_with_centroid (a b c : ℝ) (GA GB GC : ℝ) : Prop :=
  56 * a * GA + 40 * b * GB + 35 * c * GC = 0

theorem angle_B_is_60 {a b c GA GB GC : ℝ} (h : 56 * a * GA + 40 * b * GB + 35 * c * GC = 0) :
  ∃ B : ℝ, B = 60 :=
sorry

end angle_B_is_60_l1076_107642


namespace solution_of_system_l1076_107671

theorem solution_of_system 
  (k : ℝ) (x y : ℝ)
  (h1 : (1 : ℝ) = 2 * 1 - 1)
  (h2 : (1 : ℝ) = k * 1)
  (h3 : k ≠ 0)
  (h4 : 2 * x - y = 1)
  (h5 : k * x - y = 0) : 
  x = 1 ∧ y = 1 :=
by
  sorry

end solution_of_system_l1076_107671


namespace max_m_value_l1076_107605

theorem max_m_value {m : ℝ} : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0 → x < m)) ∧ ¬(∀ x : ℝ, (x^2 - 2 * x - 8 > 0 ↔ x < m)) → m ≤ -2 :=
sorry

end max_m_value_l1076_107605


namespace number_of_initial_cards_l1076_107695

theorem number_of_initial_cards (x : ℝ) (h1 : x + 276.0 = 580) : x = 304 :=
by
  sorry

end number_of_initial_cards_l1076_107695


namespace max_value_S_n_l1076_107612

theorem max_value_S_n 
  (a : ℕ → ℕ)
  (a1 : a 1 = 2)
  (S : ℕ → ℕ)
  (h : ∀ n, 6 * S n = 3 * a (n + 1) + 4 ^ n - 1) :
  ∃ n, S n = 10 := 
sorry

end max_value_S_n_l1076_107612


namespace find_x_unique_l1076_107665

def productOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of product of digits function
  sorry

def sumOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of sum of digits function
  sorry

theorem find_x_unique : ∀ x : ℕ, (productOfDigits x = 44 * x - 86868 ∧ ∃ n : ℕ, sumOfDigits x = n^3) -> x = 1989 :=
by
  intros x h
  sorry

end find_x_unique_l1076_107665


namespace washing_machine_capacity_l1076_107614

-- Define the problem conditions
def families : Nat := 3
def people_per_family : Nat := 4
def days : Nat := 7
def towels_per_person_per_day : Nat := 1
def loads : Nat := 6

-- Define the statement to prove
theorem washing_machine_capacity :
  (families * people_per_family * days * towels_per_person_per_day) / loads = 14 := by
  sorry

end washing_machine_capacity_l1076_107614


namespace petr_receives_1000000_l1076_107616

def initial_investment_vp := 200000
def initial_investment_pg := 350000
def third_share_value := 1100000
def total_company_value := 3 * third_share_value

theorem petr_receives_1000000 :
  initial_investment_vp = 200000 →
  initial_investment_pg = 350000 →
  third_share_value = 1100000 →
  total_company_value = 3300000 →
  ∃ (share_pg : ℕ), share_pg = 1000000 :=
by
  intros h_vp h_pg h_as h_total
  let x := initial_investment_vp * 1650000
  let y := initial_investment_pg * 1650000
  -- Skipping calculations
  sorry

end petr_receives_1000000_l1076_107616


namespace value_of_expression_l1076_107632

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem value_of_expression (h : a = Real.log 3 / Real.log 4) : 2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 :=
by
  sorry

end value_of_expression_l1076_107632


namespace cherries_used_l1076_107652

theorem cherries_used (initial remaining used : ℕ) (h_initial : initial = 77) (h_remaining : remaining = 17) (h_used : used = initial - remaining) : used = 60 :=
by
  rw [h_initial, h_remaining] at h_used
  simp at h_used
  exact h_used

end cherries_used_l1076_107652


namespace sum_of_roots_eq_14_l1076_107686

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) → 11 + 3 = 14 :=
by
  intro h
  have x1 : 11 = 11 := rfl
  have x2 : 3 = 3 := rfl
  exact rfl

end sum_of_roots_eq_14_l1076_107686


namespace combinatorial_solution_l1076_107691

theorem combinatorial_solution (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 14)
  (h3 : 0 ≤ 2 * x - 4) (h4 : 2 * x - 4 ≤ 14) : x = 4 ∨ x = 6 := by
  sorry

end combinatorial_solution_l1076_107691


namespace range_of_a_if_f_increasing_l1076_107692

theorem range_of_a_if_f_increasing (a : ℝ) :
  (∀ x : ℝ, 3*x^2 + 3*a ≥ 0) → (a ≥ 0) :=
sorry

end range_of_a_if_f_increasing_l1076_107692


namespace segments_do_not_intersect_l1076_107674

noncomputable def check_intersection (AP PB BQ QC CR RD DS SA : ℚ) : Bool :=
  (AP / PB) * (BQ / QC) * (CR / RD) * (DS / SA) = 1

theorem segments_do_not_intersect :
  let AP := (3 : ℚ)
  let PB := (6 : ℚ)
  let BQ := (2 : ℚ)
  let QC := (4 : ℚ)
  let CR := (1 : ℚ)
  let RD := (5 : ℚ)
  let DS := (4 : ℚ)
  let SA := (6 : ℚ)
  ¬ check_intersection AP PB BQ QC CR RD DS SA :=
by sorry

end segments_do_not_intersect_l1076_107674


namespace largest_even_number_l1076_107684

theorem largest_even_number (x : ℕ) (h : x + (x+2) + (x+4) = 1194) : x + 4 = 400 :=
by
  have : 3*x + 6 = 1194 := by linarith
  have : 3*x = 1188 := by linarith
  have : x = 396 := by linarith
  linarith

end largest_even_number_l1076_107684


namespace bargain_range_l1076_107625

theorem bargain_range (cost_price lowest_cp highest_cp : ℝ)
  (h_lowest : lowest_cp = 50)
  (h_highest : highest_cp = 200 / 3)
  (h_marked_at : cost_price = 100)
  (h_lowest_markup : lowest_cp * 2 = cost_price)
  (h_highest_markup : highest_cp * 1.5 = cost_price)
  (profit_margin : ∀ (cp : ℝ), (cp * 1.2 ≥ cp)) : 
  (60 ≤ cost_price * 1.2 ∧ cost_price * 1.2 ≤ 80) :=
by
  sorry

end bargain_range_l1076_107625


namespace satisfactory_fraction_is_28_over_31_l1076_107661

-- Define the number of students for each grade
def students_with_grade_A := 8
def students_with_grade_B := 7
def students_with_grade_C := 6
def students_with_grade_D := 4
def students_with_grade_E := 3
def students_with_grade_F := 3

-- Calculate the total number of students with satisfactory grades
def satisfactory_grades := students_with_grade_A + students_with_grade_B + students_with_grade_C + students_with_grade_D + students_with_grade_E

-- Calculate the total number of students
def total_students := satisfactory_grades + students_with_grade_F

-- Define the fraction of satisfactory grades
def satisfactory_fraction : ℚ := satisfactory_grades / total_students

-- The main proposition that the satisfactory fraction is 28/31
theorem satisfactory_fraction_is_28_over_31 : satisfactory_fraction = 28 / 31 := by {
  sorry
}

end satisfactory_fraction_is_28_over_31_l1076_107661


namespace inequality_ABC_l1076_107648

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l1076_107648


namespace train_speed_in_km_per_hour_l1076_107694

-- Definitions based on the conditions
def train_length : ℝ := 240  -- The length of the train in meters.
def time_to_pass_tree : ℝ := 8  -- The time to pass the tree in seconds.
def meters_per_second_to_kilometers_per_hour : ℝ := 3.6  -- Conversion factor from meters/second to kilometers/hour.

-- Statement based on the question and the correct answer
theorem train_speed_in_km_per_hour : (train_length / time_to_pass_tree) * meters_per_second_to_kilometers_per_hour = 108 :=
by
  sorry

end train_speed_in_km_per_hour_l1076_107694


namespace solve_for_a_l1076_107629

variable (x y a : ℤ)
variable (hx : x = 1)
variable (hy : y = -3)
variable (eq : a * x - y = 1)
 
theorem solve_for_a : a = -2 := by
  -- Placeholder to satisfy the lean prover, no actual proof steps
  sorry

end solve_for_a_l1076_107629


namespace this_year_sales_l1076_107650

def last_year_sales : ℝ := 320 -- in millions
def percent_increase : ℝ := 0.5 -- 50%

theorem this_year_sales : (last_year_sales * (1 + percent_increase)) = 480 := by
  sorry

end this_year_sales_l1076_107650


namespace distributor_income_proof_l1076_107627

noncomputable def income_2017 (a k x : ℝ) : ℝ :=
  (a + k / (x - 7)) * (x - 5)

theorem distributor_income_proof (a : ℝ) (x : ℝ) (h_range : 10 ≤ x ∧ x ≤ 14) (h_k : k = 3 * a):
  income_2017 a (3 * a) x = 12 * a ↔ x = 13 := by
  sorry

end distributor_income_proof_l1076_107627


namespace standard_deviation_distance_l1076_107656

-- Definitions and assumptions based on the identified conditions
def mean : ℝ := 12
def std_dev : ℝ := 1.2
def value : ℝ := 9.6

-- Statement to prove
theorem standard_deviation_distance : (value - mean) / std_dev = -2 :=
by sorry

end standard_deviation_distance_l1076_107656


namespace iceberg_submersion_l1076_107689

theorem iceberg_submersion (V_total V_immersed S_total S_submerged : ℝ) :
  convex_polyhedron ∧ floating_on_sea ∧
  V_total > 0 ∧ V_immersed > 0 ∧ S_total > 0 ∧ S_submerged > 0 ∧
  (V_immersed / V_total >= 0.90) ∧ ((S_total - S_submerged) / S_total >= 0.50) :=
sorry

end iceberg_submersion_l1076_107689


namespace simplify_and_rationalize_l1076_107681

noncomputable def expression := 
  (Real.sqrt 8 / Real.sqrt 3) * 
  (Real.sqrt 25 / Real.sqrt 30) * 
  (Real.sqrt 16 / Real.sqrt 21)

theorem simplify_and_rationalize :
  expression = 4 * Real.sqrt 14 / 63 :=
by
  sorry

end simplify_and_rationalize_l1076_107681


namespace sum_of_consecutive_integers_product_is_negative_336_l1076_107657

theorem sum_of_consecutive_integers_product_is_negative_336 :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) = -336 ∧ (n - 1) + n + (n + 1) = -21 :=
by
  sorry

end sum_of_consecutive_integers_product_is_negative_336_l1076_107657


namespace factorization_problem_l1076_107647

theorem factorization_problem :
  ∃ a b : ℤ, (∀ y : ℤ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b))
    ∧ (a - b = 11) := by
  sorry

end factorization_problem_l1076_107647


namespace kamal_marks_physics_correct_l1076_107663

-- Definition of the conditions
def kamal_marks_english : ℕ := 76
def kamal_marks_mathematics : ℕ := 60
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 74
def kamal_num_subjects : ℕ := 5

-- Definition of the total marks
def kamal_total_marks : ℕ := kamal_average_marks * kamal_num_subjects

-- Sum of known marks
def kamal_known_marks : ℕ := kamal_marks_english + kamal_marks_mathematics + kamal_marks_chemistry + kamal_marks_biology

-- The expected result for Physics
def kamal_marks_physics : ℕ := 82

-- Proof statement
theorem kamal_marks_physics_correct :
  kamal_total_marks - kamal_known_marks = kamal_marks_physics :=
by
  simp [kamal_total_marks, kamal_known_marks, kamal_marks_physics]
  sorry

end kamal_marks_physics_correct_l1076_107663


namespace correct_factorization_from_left_to_right_l1076_107685

theorem correct_factorization_from_left_to_right 
  (x a b c m n : ℝ) : 
  (2 * a * b - 2 * a * c = 2 * a * (b - c)) :=
sorry

end correct_factorization_from_left_to_right_l1076_107685


namespace conditions_iff_positive_l1076_107655

theorem conditions_iff_positive (a b : ℝ) (h₁ : a + b > 0) (h₂ : ab > 0) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ ab > 0) :=
sorry

end conditions_iff_positive_l1076_107655


namespace solve_system_l1076_107644

theorem solve_system :
  (∀ x y : ℝ, 
    (x^2 * y - x * y^2 - 3 * x + 3 * y + 1 = 0 ∧
     x^3 * y - x * y^3 - 3 * x^2 + 3 * y^2 + 3 = 0) → (x, y) = (2, 1)) :=
by simp [← solve_system]; sorry

end solve_system_l1076_107644


namespace remainders_identical_l1076_107609

theorem remainders_identical (a b : ℕ) (h1 : a > b) :
  ∃ r₁ r₂ q₁ q₂ : ℕ, 
  a = (a - b) * q₁ + r₁ ∧ 
  b = (a - b) * q₂ + r₂ ∧ 
  r₁ = r₂ := by 
sorry

end remainders_identical_l1076_107609


namespace range_of_k_for_distinct_roots_l1076_107690
-- Import necessary libraries

-- Define the quadratic equation and conditions
noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the property of having distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c > 0

-- Define the specific problem instance and range condition
theorem range_of_k_for_distinct_roots (k : ℝ) :
  has_two_distinct_real_roots 1 2 k ↔ k < 1 :=
by
  sorry

end range_of_k_for_distinct_roots_l1076_107690


namespace function_decreasing_on_interval_l1076_107696

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 3

theorem function_decreasing_on_interval : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → 1 ≤ x₂ → x₁ ≤ x₂ → f x₁ ≥ f x₂ := by
  sorry

end function_decreasing_on_interval_l1076_107696


namespace janet_income_difference_l1076_107603

def janet_current_job_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def janet_freelance_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def extra_fica_taxes (weekly_tax : ℝ) (weeks_per_month : ℕ) : ℝ :=
  weekly_tax * weeks_per_month

def healthcare_premiums (monthly_premium : ℝ) : ℝ :=
  monthly_premium

def janet_net_freelance_income (freelance_income : ℝ) (additional_costs : ℝ) : ℝ :=
  freelance_income - additional_costs

theorem janet_income_difference
  (hours_per_week : ℕ)
  (weeks_per_month : ℕ)
  (current_hourly_rate : ℝ)
  (freelance_hourly_rate : ℝ)
  (weekly_tax : ℝ)
  (monthly_premium : ℝ)
  (H_hours : hours_per_week = 40)
  (H_weeks : weeks_per_month = 4)
  (H_current_rate : current_hourly_rate = 30)
  (H_freelance_rate : freelance_hourly_rate = 40)
  (H_weekly_tax : weekly_tax = 25)
  (H_monthly_premium : monthly_premium = 400) :
  janet_net_freelance_income (janet_freelance_income 40 4 40) (extra_fica_taxes 25 4 + healthcare_premiums 400) 
  - janet_current_job_income 40 4 30 = 1100 := 
  by 
    sorry

end janet_income_difference_l1076_107603


namespace sin_pi_minus_a_l1076_107677

theorem sin_pi_minus_a (a : ℝ) (h_cos_a : Real.cos a = Real.sqrt 5 / 3) (h_range_a : a ∈ Set.Ioo (-Real.pi / 2) 0) : 
  Real.sin (Real.pi - a) = -2 / 3 :=
by sorry

end sin_pi_minus_a_l1076_107677


namespace jam_consumption_l1076_107660

theorem jam_consumption (x y t : ℝ) :
  x + y = 100 →
  t = 45 * x / y →
  t = 20 * y / x →
  x = 40 ∧ y = 60 ∧ 
  (y / 45 = 4 / 3) ∧ 
  (x / 20 = 2) := by
  sorry

end jam_consumption_l1076_107660


namespace parametric_line_segment_computation_l1076_107662

theorem parametric_line_segment_computation :
  ∃ (a b c d : ℝ), 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
   (-3, 10) = (a * t + b, c * t + d) ∧
   (4, 16) = (a * 1 + b, c * 1 + d)) ∧
  (b = -3) ∧ (d = 10) ∧ 
  (a + b = 4) ∧ (c + d = 16) ∧ 
  (a^2 + b^2 + c^2 + d^2 = 194) :=
sorry

end parametric_line_segment_computation_l1076_107662


namespace unique_solution_of_diophantine_l1076_107638

theorem unique_solution_of_diophantine (m n : ℕ) (hm_pos : m > 0) (hn_pos: n > 0) :
  m^2 = Int.sqrt n + Int.sqrt (2 * n + 1) → (m = 13 ∧ n = 4900) :=
by
  sorry

end unique_solution_of_diophantine_l1076_107638


namespace prime_factors_of_n_l1076_107601

def n : ℕ := 400000001

def is_prime (p: ℕ) : Prop := Nat.Prime p

theorem prime_factors_of_n (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : n = p * q) : 
  (p = 19801 ∧ q = 20201) ∨ (p = 20201 ∧ q = 19801) :=
by
  sorry

end prime_factors_of_n_l1076_107601


namespace expression_remainder_l1076_107673

theorem expression_remainder (n : ℤ) (h : n % 5 = 3) : (n + 1) % 5 = 4 :=
by
  sorry

end expression_remainder_l1076_107673


namespace false_proposition_l1076_107613

-- Definitions based on conditions
def opposite_angles (α β : ℝ) : Prop := α = β
def perpendicular (l m : ℝ → ℝ) : Prop := ∀ x, l x * m x = -1
def parallel (l m : ℝ → ℝ) : Prop := ∃ c, ∀ x, l x = m x + c
def corresponding_angles (α β : ℝ) : Prop := α = β

-- Propositions from the problem
def proposition1 : Prop := ∀ α β, opposite_angles α β → α = β
def proposition2 : Prop := ∀ l m n, perpendicular l n → perpendicular m n → parallel l m
def proposition3 : Prop := ∀ α β, α = β → opposite_angles α β
def proposition4 : Prop := ∀ α β, corresponding_angles α β → α = β

-- Statement to prove proposition 3 is false under given conditions
theorem false_proposition : ¬ proposition3 := by
  -- By our analysis, if proposition 3 is false, then it means the given definition for proposition 3 holds under all circumstances.
  sorry

end false_proposition_l1076_107613


namespace solve_for_F_l1076_107697

variable (S W F : ℝ)

def condition1 (S W : ℝ) : Prop := S = W / 3
def condition2 (W F : ℝ) : Prop := W = F + 60
def condition3 (S W F : ℝ) : Prop := S + W + F = 150

theorem solve_for_F (S W F : ℝ) (h1 : condition1 S W) (h2 : condition2 W F) (h3 : condition3 S W F) : F = 52.5 :=
sorry

end solve_for_F_l1076_107697


namespace cans_to_collect_l1076_107636

theorem cans_to_collect
  (martha_cans : ℕ)
  (diego_half_plus_ten : ℕ)
  (total_cans_required : ℕ)
  (martha_cans_collected : martha_cans = 90)
  (diego_collected : diego_half_plus_ten = (martha_cans / 2) + 10)
  (goal_cans : total_cans_required = 150) :
  total_cans_required - (martha_cans + diego_half_plus_ten) = 5 :=
by
  sorry

end cans_to_collect_l1076_107636


namespace inequality_solution_l1076_107670

theorem inequality_solution (x : ℝ) (h : 3 * x + 2 ≠ 0) : 
  3 - 2/(3 * x + 2) < 5 ↔ x > -2/3 := 
sorry

end inequality_solution_l1076_107670


namespace log_roots_equivalence_l1076_107646

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 5 / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 5

theorem log_roots_equivalence :
  (x : ℝ) → (x = a ∨ x = b ∨ x = c) ↔ (x^3 - (a + b + c)*x^2 + (a*b + b*c + c*a)*x - a*b*c = 0) := by
  sorry

end log_roots_equivalence_l1076_107646


namespace tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l1076_107693

-- Definition for Part (a)
theorem tetrahedron_parallelepiped_areas 
  (S1 S2 S3 S4 P1 P2 P3 : ℝ)
  (h1 : true)
  (h2 : true) :
  S1^2 + S2^2 + S3^2 + S4^2 = P1^2 + P2^2 + P3^2 := 
sorry

-- Definition for Part (b)
theorem tetrahedron_heights_distances 
  (h1 h2 h3 h4 d1 d2 d3 : ℝ)
  (h : true) :
  (1/(h1^2)) + (1/(h2^2)) + (1/(h3^2)) + (1/(h4^2)) = (1/(d1^2)) + (1/(d2^2)) + (1/(d3^2)) := 
sorry

end tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l1076_107693
