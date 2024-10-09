import Mathlib

namespace f_decreasing_on_negative_interval_and_min_value_l2228_222816

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

def minimum_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m ∧ ∃ x0, f x0 = m

-- Given the conditions
variables (condition1 : even_function f)
          (condition2 : increasing_on_interval f 3 7)
          (condition3 : minimum_value f 2)

-- Prove that f is decreasing on [-7,-3] and minimum value is 2
theorem f_decreasing_on_negative_interval_and_min_value :
  ∀ x y, -7 ≤ x → x ≤ y → y ≤ -3 → f y ≤ f x ∧ minimum_value f 2 :=
sorry

end f_decreasing_on_negative_interval_and_min_value_l2228_222816


namespace probability_of_specific_choice_l2228_222817

-- Define the sets of subjects
inductive Subject
| Chinese
| Mathematics
| ForeignLanguage
| Physics
| History
| PoliticalScience
| Geography
| Chemistry
| Biology

-- Define the conditions of the examination mode "3+1+2"
def threeSubjects := [Subject.Chinese, Subject.Mathematics, Subject.ForeignLanguage]
def oneSubject := [Subject.Physics, Subject.History]
def twoSubjects := [Subject.PoliticalScience, Subject.Geography, Subject.Chemistry, Subject.Biology]

-- Calculate the total number of ways to choose one subject from Physics or History and two subjects from PoliticalScience, Geography, Chemistry, and Biology
def totalWays : Nat := 2 * Nat.choose 4 2  -- 2 choices for "1" part, and C(4, 2) ways for "2" part

-- Calculate the probability that a candidate will choose Political Science, History, and Geography
def favorableOutcome := 1  -- Only one specific combination counts

theorem probability_of_specific_choice :
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  (specific_combination : ℚ) / total_ways = 1 / 12 :=
by
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  show (specific_combination : ℚ) / total_ways = 1 / 12
  sorry

end probability_of_specific_choice_l2228_222817


namespace pyramid_layers_total_l2228_222869

-- Since we are dealing with natural number calculations, noncomputable is generally not needed.

-- Definition of the pyramid layers and the number of balls in each layer
def number_of_balls (n : ℕ) : ℕ := n ^ 2

-- Given conditions for the layers
def third_layer_balls : ℕ := number_of_balls 3
def fifth_layer_balls : ℕ := number_of_balls 5

-- Statement of the problem proving that their sum is 34
theorem pyramid_layers_total : third_layer_balls + fifth_layer_balls = 34 := by
  sorry -- proof to be provided

end pyramid_layers_total_l2228_222869


namespace common_difference_of_arithmetic_sequence_l2228_222899

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

noncomputable def S_n (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (a1 d : ℕ) (h1 : a_n a1 d 3 = 8) (h2 : S_n a1 d 6 = 54) : d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l2228_222899


namespace staircase_toothpicks_l2228_222838

theorem staircase_toothpicks :
  ∀ (T : ℕ → ℕ), 
  (T 4 = 28) →
  (∀ n : ℕ, T (n + 1) = T n + (12 + 3 * (n - 3))) →
  T 6 - T 4 = 33 :=
by
  intros T T4_step H_increase
  -- proof goes here
  sorry

end staircase_toothpicks_l2228_222838


namespace function_machine_output_is_38_l2228_222811

def function_machine (input : ℕ) : ℕ :=
  let multiplied := input * 3
  if multiplied > 40 then
    multiplied - 7
  else
    multiplied + 10

theorem function_machine_output_is_38 :
  function_machine 15 = 38 :=
by
   sorry

end function_machine_output_is_38_l2228_222811


namespace arithmetic_sequence_general_formula_l2228_222883

theorem arithmetic_sequence_general_formula (a : ℤ) :
  ∀ n : ℕ, n ≥ 1 → (∃ a_1 a_2 a_3 : ℤ, a_1 = a - 1 ∧ a_2 = a + 1 ∧ a_3 = a + 3) →
  (a + 2 * n - 3 = a - 1 + (n - 1) * 2) :=
by
  intros n hn h_exists
  rcases h_exists with ⟨a_1, a_2, a_3, h1, h2, h3⟩
  sorry

end arithmetic_sequence_general_formula_l2228_222883


namespace geometric_mean_45_80_l2228_222847

theorem geometric_mean_45_80 : ∃ x : ℝ, x^2 = 45 * 80 ∧ (x = 60 ∨ x = -60) := 
by 
  sorry

end geometric_mean_45_80_l2228_222847


namespace remaining_apples_l2228_222853

def initial_apples : ℕ := 20
def shared_apples : ℕ := 7

theorem remaining_apples : initial_apples - shared_apples = 13 :=
by
  sorry

end remaining_apples_l2228_222853


namespace max_smaller_boxes_fit_l2228_222858

theorem max_smaller_boxes_fit (length_large width_large height_large : ℝ)
  (length_small width_small height_small : ℝ)
  (h1 : length_large = 6)
  (h2 : width_large = 5)
  (h3 : height_large = 4)
  (hs1 : length_small = 0.60)
  (hs2 : width_small = 0.50)
  (hs3 : height_small = 0.40) :
  length_large * width_large * height_large / (length_small * width_small * height_small) = 1000 := 
  by
  sorry

end max_smaller_boxes_fit_l2228_222858


namespace initial_children_on_bus_l2228_222837

-- Define the conditions
variables (x : ℕ)

-- Define the problem statement
theorem initial_children_on_bus (h : x + 7 = 25) : x = 18 :=
sorry

end initial_children_on_bus_l2228_222837


namespace dot_product_a_b_l2228_222844

-- Define the given vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

-- Define the dot product function
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- State the theorem with the correct answer
theorem dot_product_a_b : dot_product a b = 1 :=
by
  sorry

end dot_product_a_b_l2228_222844


namespace find_center_radius_sum_l2228_222819

theorem find_center_radius_sum :
    let x := x
    let y := y
    let a := 2
    let b := 3
    let r := 2 * Real.sqrt 6
    (x^2 - 4 * x + y^2 - 6 * y = 11) →
    (a + b + r = 5 + 2 * Real.sqrt 6) :=
by
  intros x y a b r
  sorry

end find_center_radius_sum_l2228_222819


namespace sin_eleven_pi_over_three_l2228_222805

theorem sin_eleven_pi_over_three : Real.sin (11 * Real.pi / 3) = -((Real.sqrt 3) / 2) :=
by
  -- Conversion factor between radians and degrees
  -- periodicity of sine function: sin theta = sin (theta + n * 360 degrees) for any integer n
  -- the sine function is odd: sin (-theta) = -sin theta
  -- sin 60 degrees = sqrt(3)/2
  sorry

end sin_eleven_pi_over_three_l2228_222805


namespace product_n_equals_7200_l2228_222859

theorem product_n_equals_7200 :
  (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 ^ 2 + 1) = 7200 := by
  sorry

end product_n_equals_7200_l2228_222859


namespace yard_length_l2228_222856

-- Definition of the problem conditions
def num_trees : Nat := 11
def distance_between_trees : Nat := 15

-- Length of the yard is given by the product of (num_trees - 1) and distance_between_trees
theorem yard_length :
  (num_trees - 1) * distance_between_trees = 150 :=
by
  sorry

end yard_length_l2228_222856


namespace fraction_of_girls_l2228_222889

variable {T G B : ℕ}
variable (ratio : ℚ)

theorem fraction_of_girls (X : ℚ) (h1 : ∀ (G : ℕ) (T : ℕ), X * G = (1/4) * T)
  (h2 : ratio = 5 / 3) (h3 : ∀ (G : ℕ) (B : ℕ), B / G = ratio) :
  X = 2 / 3 :=
by 
  sorry

end fraction_of_girls_l2228_222889


namespace average_income_of_other_40_customers_l2228_222818

theorem average_income_of_other_40_customers
    (avg_income_50 : ℝ)
    (num_50 : ℕ)
    (avg_income_10 : ℝ)
    (num_10 : ℕ)
    (total_num : ℕ)
    (remaining_num : ℕ)
    (total_income_50 : ℝ)
    (total_income_10 : ℝ)
    (total_income_40 : ℝ)
    (avg_income_40 : ℝ) 
    (hyp_avg_income_50 : avg_income_50 = 45000)
    (hyp_num_50 : num_50 = 50)
    (hyp_avg_income_10 : avg_income_10 = 55000)
    (hyp_num_10 : num_10 = 10)
    (hyp_total_num : total_num = 50)
    (hyp_remaining_num : remaining_num = 40)
    (hyp_total_income_50 : total_income_50 = 2250000)
    (hyp_total_income_10 : total_income_10 = 550000)
    (hyp_total_income_40 : total_income_40 = 1700000)
    (hyp_avg_income_40 : avg_income_40 = total_income_40 / remaining_num) :
  avg_income_40 = 42500 :=
  by
    sorry

end average_income_of_other_40_customers_l2228_222818


namespace greatest_divisor_of_arithmetic_sequence_sum_l2228_222823

theorem greatest_divisor_of_arithmetic_sequence_sum (x c : ℕ) (hx : x > 0) (hc : c > 0) :
  ∃ k, (∀ (S : ℕ), S = 6 * (2 * x + 11 * c) → k ∣ S) ∧ k = 6 :=
by
  sorry

end greatest_divisor_of_arithmetic_sequence_sum_l2228_222823


namespace find_functions_l2228_222833

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_functions
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_1 : ∀ x : ℝ, |x| ≤ 1 → |f a b c x| ≤ 1)
  (h_2 : ∃ x₀ : ℝ, |x₀| ≤ 1 ∧ ∀ x : ℝ, |x| ≤ 1 → |f' a b x₀| ≥ |f' a b x| )
  (K : ℝ)
  (h_3 : ∃ x₀ : ℝ, |x₀| ≤ 1 ∧ |f' a b x₀| = K) :
  (f a b c = fun x ↦ 2 * x^2 - 1) ∨ (f a b c = fun x ↦ -2 * x^2 + 1) ∧ K = 4 := 
sorry

end find_functions_l2228_222833


namespace sum_of_k_l2228_222828

theorem sum_of_k : ∃ (k_vals : List ℕ), 
  (∀ k ∈ k_vals, ∃ α β : ℤ, α + β = k ∧ α * β = -20) 
  ∧ k_vals.sum = 29 :=
by 
  sorry

end sum_of_k_l2228_222828


namespace sum_seven_consecutive_integers_l2228_222800

theorem sum_seven_consecutive_integers (m : ℕ) :
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) + (m + 6) = 7 * m + 21 :=
by
  -- Sorry to skip the actual proof steps.
  sorry

end sum_seven_consecutive_integers_l2228_222800


namespace mapping_image_l2228_222893

theorem mapping_image (f : ℕ → ℕ) (h : ∀ x, f x = x + 1) : f 3 = 4 :=
by {
  sorry
}

end mapping_image_l2228_222893


namespace sequence_a_n_perfect_square_l2228_222850

theorem sequence_a_n_perfect_square :
  (∃ a : ℕ → ℤ, ∃ b : ℕ → ℤ,
    a 0 = 1 ∧ b 0 = 0 ∧
    (∀ n : ℕ, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n : ℕ, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    (∀ n : ℕ, ∃ k : ℤ, a n = k^2)) :=
sorry

end sequence_a_n_perfect_square_l2228_222850


namespace probability_circle_l2228_222806

theorem probability_circle (total_figures triangles circles squares : ℕ)
  (h_total : total_figures = 10)
  (h_triangles : triangles = 4)
  (h_circles : circles = 3)
  (h_squares : squares = 3) :
  circles / total_figures = 3 / 10 :=
by
  sorry

end probability_circle_l2228_222806


namespace ice_cream_ordering_ways_l2228_222872

-- Define the possible choices for each category.
def cone_choices : Nat := 2
def scoop_choices : Nat := 1 + 10 + 20  -- Total choices for 1, 2, and 3 scoops.
def topping_choices : Nat := 1 + 4 + 6  -- Total choices for no topping, 1 topping, and 2 toppings.

-- Theorem to state the number of ways ice cream can be ordered.
theorem ice_cream_ordering_ways : cone_choices * scoop_choices * topping_choices = 748 := by
  let calc_cone := cone_choices  -- Number of cone choices.
  let calc_scoop := scoop_choices  -- Number of scoop combinations.
  let calc_topping := topping_choices  -- Number of topping combinations.
  have h1 : calc_cone * calc_scoop * calc_topping = 748 := sorry  -- Calculation hint.
  exact h1

end ice_cream_ordering_ways_l2228_222872


namespace zoo_ticket_problem_l2228_222842

theorem zoo_ticket_problem :
  ∀ (total_amount adult_ticket_cost children_ticket_cost : ℕ)
    (num_adult_tickets : ℕ),
  total_amount = 119 →
  adult_ticket_cost = 21 →
  children_ticket_cost = 14 →
  num_adult_tickets = 4 →
  6 = (num_adult_tickets + (total_amount - num_adult_tickets * adult_ticket_cost) / children_ticket_cost) :=
by 
  intros total_amount adult_ticket_cost children_ticket_cost num_adult_tickets 
         total_amt_eq adult_ticket_cost_eq children_ticket_cost_eq num_adult_tickets_eq
  sorry

end zoo_ticket_problem_l2228_222842


namespace arithmetic_sequence_term_number_l2228_222807

-- Given:
def first_term : ℕ := 1
def common_difference : ℕ := 3
def target_term : ℕ := 2011

-- To prove:
theorem arithmetic_sequence_term_number :
    ∃ n : ℕ, target_term = first_term + (n - 1) * common_difference ∧ n = 671 := 
by
  -- The proof is omitted
  sorry

end arithmetic_sequence_term_number_l2228_222807


namespace hyperbola_eccentricity_l2228_222804

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_angle : b / a = Real.sqrt 3 / 3) :
    let e := Real.sqrt (1 + (b / a)^2)
    e = 2 * Real.sqrt 3 / 3 := 
sorry

end hyperbola_eccentricity_l2228_222804


namespace prime_factors_of_product_l2228_222835

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : List ℕ :=
  -- Assume we have a function that returns a list of prime factors of n
  sorry

def num_distinct_primes (n : ℕ) : ℕ :=
  (prime_factors n).toFinset.card

theorem prime_factors_of_product :
  num_distinct_primes (85 * 87 * 91 * 94) = 8 :=
by
  have prod_factorizations : 85 = 5 * 17 ∧ 87 = 3 * 29 ∧ 91 = 7 * 13 ∧ 94 = 2 * 47 := 
    by sorry -- each factorization step
  sorry

end prime_factors_of_product_l2228_222835


namespace nth_equation_l2228_222851

theorem nth_equation (n : ℕ) : (n + 1)^2 - 1 = n * (n + 2) := 
by 
  sorry

end nth_equation_l2228_222851


namespace abs_inequality_example_l2228_222886

theorem abs_inequality_example (x : ℝ) : abs (5 - x) < 6 ↔ -1 < x ∧ x < 11 :=
by 
  sorry

end abs_inequality_example_l2228_222886


namespace A_eq_B_l2228_222803

def A : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 + 2 * y^2
}

def B : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 - 6 * x * y + 11 * y^2
}

theorem A_eq_B : A = B :=
by {
  sorry
}

end A_eq_B_l2228_222803


namespace sum_digits_in_possibilities_l2228_222808

noncomputable def sum_of_digits (a b c d : ℕ) : ℕ :=
  a + b + c + d

theorem sum_digits_in_possibilities :
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (sum_of_digits a b c d = 10 ∨ sum_of_digits a b c d = 18 ∨ sum_of_digits a b c d = 19) := sorry

end sum_digits_in_possibilities_l2228_222808


namespace smallest_piece_to_cut_l2228_222864

theorem smallest_piece_to_cut (x : ℕ) 
  (h1 : 9 - x > 0) 
  (h2 : 16 - x > 0) 
  (h3 : 18 - x > 0) :
  7 ≤ x ∧ 9 - x + 16 - x ≤ 18 - x :=
by {
  sorry
}

end smallest_piece_to_cut_l2228_222864


namespace find_angle_A_l2228_222841

noncomputable def angle_A (a b : ℝ) (B : ℝ) : ℝ :=
  Real.arcsin ((a * Real.sin B) / b)

theorem find_angle_A :
  ∀ (a b : ℝ) (angle_B : ℝ), 0 < a → 0 < b → 0 < angle_B → angle_B < 180 →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  angle_B = 60 →
  angle_A a b angle_B = 45 :=
by
  intros a b angle_B h1 h2 h3 h4 ha hb hB
  have ha' : a = Real.sqrt 2 := ha
  have hb' : b = Real.sqrt 3 := hb
  have hB' : angle_B = 60 := hB
  -- Proof omitted for demonstration
  sorry

end find_angle_A_l2228_222841


namespace arithmetic_sequence_sum_l2228_222882

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 3 = 6)
  (h2 : S 9 = 27) :
  S 6 = 15 :=
sorry

end arithmetic_sequence_sum_l2228_222882


namespace ancient_chinese_poem_l2228_222822

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) := by
  sorry

end ancient_chinese_poem_l2228_222822


namespace carol_additional_cupcakes_l2228_222821

-- Define the initial number of cupcakes Carol made
def initial_cupcakes : ℕ := 30

-- Define the number of cupcakes Carol sold
def sold_cupcakes : ℕ := 9

-- Define the total number of cupcakes Carol wanted to have
def total_cupcakes : ℕ := 49

-- Calculate the number of cupcakes Carol had left after selling
def remaining_cupcakes : ℕ := initial_cupcakes - sold_cupcakes

-- The number of additional cupcakes Carol made can be defined and proved as follows:
theorem carol_additional_cupcakes : initial_cupcakes - sold_cupcakes + 28 = total_cupcakes :=
by
  -- left side: initial_cupcakes (30) - sold_cupcakes (9) + additional_cupcakes (28) = total_cupcakes (49)
  sorry

end carol_additional_cupcakes_l2228_222821


namespace polynomial_square_l2228_222827

theorem polynomial_square (x : ℝ) : x^4 + 2*x^3 - 2*x^2 - 4*x - 5 = y^2 → x = 3 ∨ x = -3 := by
  sorry

end polynomial_square_l2228_222827


namespace cannot_form_1x1x2_blocks_l2228_222849

theorem cannot_form_1x1x2_blocks :
  let edge_length := 7
  let total_cubes := edge_length * edge_length * edge_length
  let central_cube := (3, 3, 3)
  let remaining_cubes := total_cubes - 1
  let checkerboard_color (x y z : Nat) : Bool := (x + y + z) % 2 = 0
  let num_white (k : Nat) := if k % 2 = 0 then 25 else 24
  let num_black (k : Nat) := if k % 2 = 0 then 24 else 25
  let total_white := 170
  let total_black := 171
  total_black > total_white →
  ¬(remaining_cubes % 2 = 0 ∧ total_white % 2 = 0 ∧ total_black % 2 = 0) → 
  ∀ (block: Nat × Nat × Nat → Bool) (x y z : Nat), block (x, y, z) = ((x*y*z) % 2 = 0) := sorry

end cannot_form_1x1x2_blocks_l2228_222849


namespace min_students_l2228_222873

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : ∃ k : ℕ, b + g = 10 * k) : b + g = 38 :=
sorry

end min_students_l2228_222873


namespace sin_alpha_plus_2beta_l2228_222826

theorem sin_alpha_plus_2beta
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcosalpha_plus_beta : Real.cos (α + β) = -5 / 13)
  (h sinbeta : Real.sin β = 3 / 5) :
  Real.sin (α + 2 * β) = 33 / 65 :=
  sorry

end sin_alpha_plus_2beta_l2228_222826


namespace eighty_first_number_in_set_l2228_222868

theorem eighty_first_number_in_set : ∃ n : ℕ, n = 81 ∧ ∀ k : ℕ, (k = 8 * (n - 1) + 5) → k = 645 := by
  sorry

end eighty_first_number_in_set_l2228_222868


namespace time_taken_by_C_l2228_222863

theorem time_taken_by_C (days_A B C : ℕ) (work_done_A work_done_B work_done_C : ℚ) 
  (h1 : days_A = 40) (h2 : work_done_A = 10 * (1/40)) 
  (h3 : days_B = 40) (h4 : work_done_B = 10 * (1/40)) 
  (h5 : work_done_C = 1/2)
  (h6 : 10 * work_done_C = 1/2) :
  (10 * 2) = 20 := 
sorry

end time_taken_by_C_l2228_222863


namespace cubic_coefficient_relationship_l2228_222839

theorem cubic_coefficient_relationship (a b c p q r : ℝ)
    (h1 : ∀ s1 s2 s3: ℝ, s1 + s2 + s3 = -a ∧ s1 * s2 + s2 * s3 + s3 * s1 = b ∧ s1 * s2 * s3 = -c)
    (h2 : ∀ s1 s2 s3: ℝ, s1^2 + s2^2 + s3^2 = -p ∧ s1^2 * s2^2 + s2^2 * s3^2 + s3^2 * s1^2 = q ∧ s1^2 * s2^2 * s3^2 = r) :
    p = a^2 - 2 * b ∧ q = b^2 + 2 * a * c ∧ r = c^2 :=
by
  sorry

end cubic_coefficient_relationship_l2228_222839


namespace triangle_right_angle_and_m_values_l2228_222898

open Real

-- Definitions and conditions
def line_AB (x y : ℝ) : Prop := 3 * x - 2 * y + 6 = 0
def line_AC (x y : ℝ) : Prop := 2 * x + 3 * y - 22 = 0
def line_BC (x y m : ℝ) : Prop := 3 * x + 4 * y - m = 0

-- Prove the shape and value of m when the height from BC is 1
theorem triangle_right_angle_and_m_values :
  (∃ (x y : ℝ), line_AB x y ∧ line_AC x y ∧ line_AB x y ∧ (-3/2) ≠ (2/3)) ∧
  (∀ x y, line_AB x y → line_AC x y → 3 * x + 4 * y - 25 = 0 ∨ 3 * x + 4 * y - 35 = 0) := 
sorry

end triangle_right_angle_and_m_values_l2228_222898


namespace modular_home_total_cost_l2228_222890

theorem modular_home_total_cost :
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  total_cost = 249000 := 
by
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  have h : total_cost = 249000 := sorry
  exact h

end modular_home_total_cost_l2228_222890


namespace mean_equals_l2228_222870

theorem mean_equals (z : ℝ) :
    (7 + 10 + 15 + 21) / 4 = (18 + z) / 2 → z = 8.5 := 
by
    intro h
    sorry

end mean_equals_l2228_222870


namespace find_number_l2228_222820

theorem find_number (x : ℝ) (h : 10 * x = 2 * x - 36) : x = -4.5 :=
sorry

end find_number_l2228_222820


namespace problem_solution_l2228_222814

noncomputable def problem_statement : Prop :=
  8 * (Real.cos (25 * Real.pi / 180)) ^ 2 - Real.tan (40 * Real.pi / 180) - 4 = Real.sqrt 3

theorem problem_solution : problem_statement :=
by
sorry

end problem_solution_l2228_222814


namespace find_N_l2228_222812

theorem find_N (N : ℕ) :
  ((5 + 6 + 7 + 8) / 4 = (2014 + 2015 + 2016 + 2017) / N) → N = 1240 :=
by
  sorry

end find_N_l2228_222812


namespace count_nine_in_1_to_1000_l2228_222888

def count_nine_units : ℕ := 100
def count_nine_tens : ℕ := 100
def count_nine_hundreds : ℕ := 100

theorem count_nine_in_1_to_1000 :
  count_nine_units + count_nine_tens + count_nine_hundreds = 300 :=
by
  sorry

end count_nine_in_1_to_1000_l2228_222888


namespace speed_of_A_is_7_l2228_222846

theorem speed_of_A_is_7
  (x : ℝ)
  (h1 : ∀ t : ℝ, t = 1)
  (h2 : ∀ y : ℝ, y = 3)
  (h3 : ∀ n : ℕ, n = 10)
  (h4 : x + 3 = 10) :
  x = 7 := by
  sorry

end speed_of_A_is_7_l2228_222846


namespace area_of_right_triangle_l2228_222885

theorem area_of_right_triangle (h : ℝ) 
  (a b : ℝ) 
  (h_a_triple : b = 3 * a)
  (h_hypotenuse : h ^ 2 = a ^ 2 + b ^ 2) : 
  (1 / 2) * a * b = (3 * h ^ 2) / 20 :=
by
  sorry

end area_of_right_triangle_l2228_222885


namespace notebook_distribution_l2228_222871

theorem notebook_distribution (x : ℕ) : 
  (∃ k₁ : ℕ, x = 3 * k₁ + 1) ∧ (∃ k₂ : ℕ, x = 4 * k₂ - 2) → (x - 1) / 3 = (x + 2) / 4 :=
by
  sorry

end notebook_distribution_l2228_222871


namespace John_spent_15_dollars_on_soap_l2228_222855

theorem John_spent_15_dollars_on_soap (number_of_bars : ℕ) (weight_per_bar : ℝ) (cost_per_pound : ℝ)
  (h1 : number_of_bars = 20) (h2 : weight_per_bar = 1.5) (h3 : cost_per_pound = 0.5) :
  (number_of_bars * weight_per_bar * cost_per_pound) = 15 :=
by
  sorry

end John_spent_15_dollars_on_soap_l2228_222855


namespace quadratic_roots_relation_l2228_222857

variable (a b c X1 X2 : ℝ)

theorem quadratic_roots_relation (h : a ≠ 0) : 
  (X1 + X2 = -b / a) ∧ (X1 * X2 = c / a) :=
sorry

end quadratic_roots_relation_l2228_222857


namespace carol_total_points_l2228_222824

/-- Conditions -/
def first_round_points : ℤ := 17
def second_round_points : ℤ := 6
def last_round_points : ℤ := -16

/-- Proof problem statement -/
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end carol_total_points_l2228_222824


namespace simplify_fraction_mul_l2228_222831

theorem simplify_fraction_mul (a b c d : ℕ) (h1 : 405 = 27 * a) (h2 : 1215 = 27 * b) (h3 : a / d = 1) (h4 : b / d = 3) : (a / d) * (27 : ℕ) = 9 :=
by
  sorry

end simplify_fraction_mul_l2228_222831


namespace integer_solutions_for_even_ratio_l2228_222865

theorem integer_solutions_for_even_ratio (a : ℤ) (h : ∃ k : ℤ, (a = 2 * k * (1011 - k))): 
  a = 1010 ∨ a = 1012 ∨ a = 1008 ∨ a = 1014 ∨ a = 674 ∨ a = 1348 ∨ a = 0 ∨ a = 2022 :=
sorry

end integer_solutions_for_even_ratio_l2228_222865


namespace f_x_minus_one_l2228_222867

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5

theorem f_x_minus_one (x : ℝ) : f (x - 1) = x^2 - 4 * x + 8 :=
by
  sorry

end f_x_minus_one_l2228_222867


namespace reciprocal_of_neg_two_l2228_222895

theorem reciprocal_of_neg_two : 1 / (-2) = -1 / 2 := by
  sorry

end reciprocal_of_neg_two_l2228_222895


namespace number_of_classes_l2228_222852

theorem number_of_classes (sheets_per_class_per_day : ℕ) 
                          (total_sheets_per_week : ℕ) 
                          (school_days_per_week : ℕ) 
                          (h1 : sheets_per_class_per_day = 200) 
                          (h2 : total_sheets_per_week = 9000) 
                          (h3 : school_days_per_week = 5) : 
                          total_sheets_per_week / school_days_per_week / sheets_per_class_per_day = 9 :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_classes_l2228_222852


namespace probability_of_C_l2228_222891

-- Definitions of probabilities for regions A, B, and D
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Sum of probabilities must be 1
def total_probability : ℚ := 1

-- The main proof statement
theorem probability_of_C : 
  P_A + P_B + P_D + (P_C : ℚ) = total_probability → P_C = 1 / 4 := sorry

end probability_of_C_l2228_222891


namespace largest_divisor_of_n5_minus_n_l2228_222894

theorem largest_divisor_of_n5_minus_n (n : ℤ) : 
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n^5 - n)) ∧ d = 30 :=
sorry

end largest_divisor_of_n5_minus_n_l2228_222894


namespace landscape_breadth_l2228_222881

theorem landscape_breadth (L B : ℕ) (h1 : B = 8 * L)
  (h2 : 3200 = 1 / 9 * (L * B))
  (h3 : B * B = 28800) :
  B = 480 := by
  sorry

end landscape_breadth_l2228_222881


namespace intersection_A_B_l2228_222840

def set_A (x : ℝ) : Prop := 2 * x^2 + 5 * x - 3 ≤ 0

def set_B (x : ℝ) : Prop := -2 < x

theorem intersection_A_B :
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -2 < x ∧ x ≤ 1/2} := 
by {
  sorry
}

end intersection_A_B_l2228_222840


namespace inequality_system_no_solution_l2228_222879

theorem inequality_system_no_solution (a : ℝ) : ¬ (∃ x : ℝ, x ≤ 5 ∧ x > a) ↔ a ≥ 5 :=
sorry

end inequality_system_no_solution_l2228_222879


namespace inverse_of_217_mod_397_l2228_222834

theorem inverse_of_217_mod_397 :
  ∃ a : ℤ, 0 ≤ a ∧ a < 397 ∧ 217 * a % 397 = 1 :=
sorry

end inverse_of_217_mod_397_l2228_222834


namespace correct_operation_l2228_222874

theorem correct_operation (a b x y m : Real) :
  (¬((a^2 * b)^2 = a^2 * b^2)) ∧
  (¬(a^6 / a^2 = a^3)) ∧
  (¬((x + y)^2 = x^2 + y^2)) ∧
  ((-m)^7 / (-m)^2 = -m^5) :=
by
  sorry

end correct_operation_l2228_222874


namespace cubes_even_sum_even_l2228_222843

theorem cubes_even_sum_even (p q : ℕ) (h : Even (p^3 - q^3)) : Even (p + q) := sorry

end cubes_even_sum_even_l2228_222843


namespace function_no_extrema_k_equals_one_l2228_222802

theorem function_no_extrema_k_equals_one (k : ℝ) (h : ∀ x : ℝ, ¬ ∃ m, (k - 1) * x^2 - 4 * x + 5 - k = m) : k = 1 :=
sorry

end function_no_extrema_k_equals_one_l2228_222802


namespace karsyn_total_payment_l2228_222875

def initial_price : ℝ := 600
def discount_rate : ℝ := 0.20
def phone_case_cost : ℝ := 25
def screen_protector_cost : ℝ := 15
def store_discount_rate : ℝ := 0.05
def sales_tax_rate : ℝ := 0.035

noncomputable def total_payment : ℝ :=
  let discounted_price := discount_rate * initial_price
  let total_cost := discounted_price + phone_case_cost + screen_protector_cost
  let store_discount := store_discount_rate * total_cost
  let discounted_total := total_cost - store_discount
  let tax := sales_tax_rate * discounted_total
  discounted_total + tax

theorem karsyn_total_payment : total_payment = 157.32 := by
  sorry

end karsyn_total_payment_l2228_222875


namespace bugs_diagonally_at_least_9_unoccupied_l2228_222876

theorem bugs_diagonally_at_least_9_unoccupied (bugs : ℕ × ℕ → Prop) :
  let board_size := 9
  let cells := (board_size * board_size)
  let black_cells := 45
  let white_cells := 36
  ∃ unoccupied_cells ≥ 9, true := 
sorry

end bugs_diagonally_at_least_9_unoccupied_l2228_222876


namespace next_unique_digits_date_l2228_222884

-- Define the conditions
def is_after (d1 d2 : String) : Prop := sorry -- Placeholder, needs a date comparison function
def has_8_unique_digits (date : String) : Prop := sorry -- Placeholder, needs a function to check unique digits

-- Specify the problem and assertion
theorem next_unique_digits_date :
  ∀ date : String, is_after date "11.08.1999" → has_8_unique_digits date → date = "17.06.2345" :=
by
  sorry

end next_unique_digits_date_l2228_222884


namespace ingrid_income_l2228_222877

theorem ingrid_income (I : ℝ) (h1 : 0.30 * 56000 = 16800) 
  (h2 : ∀ (I : ℝ), 0.40 * I = 0.4 * I) 
  (h3 : 0.35625 * (56000 + I) = 16800 + 0.4 * I) : 
  I = 49142.86 := 
by 
  sorry

end ingrid_income_l2228_222877


namespace multiple_of_27_l2228_222897

theorem multiple_of_27 (x y z : ℤ) 
  (h1 : (2 * x + 5 * y + 11 * z) = 4 * (x + y + z)) 
  (h2 : (2 * x + 20 * y + 110 * z) = 6 * (2 * x + 5 * y + 11 * z)) :
  ∃ k : ℤ, x + y + z = 27 * k :=
by
  sorry

end multiple_of_27_l2228_222897


namespace range_of_k_l2228_222845

theorem range_of_k (k : ℝ) :
  ∀ x : ℝ, ∃ a b c : ℝ, (a = k-1) → (b = -2) → (c = 1) → (a ≠ 0) → ((b^2 - 4 * a * c) ≥ 0) → k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end range_of_k_l2228_222845


namespace solve_hours_l2228_222848

variable (x y : ℝ)

-- Conditions
def Condition1 : x > 0 := sorry
def Condition2 : y > 0 := sorry
def Condition3 : (2:ℝ) / 3 * y / x + (3 * x * y - 2 * y^2) / (3 * x) = x * y / (x + y) + 2 := sorry
def Condition4 : 2 * y / (x + y) = (3 * x - 2 * y) / (3 * x) := sorry

-- Question: How many hours would it take for A and B to complete the task alone?
theorem solve_hours : x = 6 ∧ y = 3 := 
by
  -- Use assumed conditions and variables to define the context
  have h1 := Condition1
  have h2 := Condition2
  have h3 := Condition3
  have h4 := Condition4
  -- Combine analytical relationship and solve for x and y 
  sorry

end solve_hours_l2228_222848


namespace frank_spent_per_week_l2228_222809

theorem frank_spent_per_week (mowing_dollars : ℕ) (weed_eating_dollars : ℕ) (weeks : ℕ) 
    (total_dollars := mowing_dollars + weed_eating_dollars) 
    (spending_rate := total_dollars / weeks) :
    mowing_dollars = 5 → weed_eating_dollars = 58 → weeks = 9 → spending_rate = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end frank_spent_per_week_l2228_222809


namespace map_point_to_result_l2228_222810

def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem map_point_to_result :
  f 2 0 = (2, 2) :=
by
  unfold f
  simp

end map_point_to_result_l2228_222810


namespace emma_deposit_withdraw_ratio_l2228_222854

theorem emma_deposit_withdraw_ratio (initial_balance withdrawn new_balance : ℤ) 
  (h1 : initial_balance = 230) 
  (h2 : withdrawn = 60) 
  (h3 : new_balance = 290) 
  (deposited : ℤ) 
  (h_deposit : new_balance = initial_balance - withdrawn + deposited) :
  (deposited / withdrawn = 2) := 
sorry

end emma_deposit_withdraw_ratio_l2228_222854


namespace maximum_automobiles_on_ferry_l2228_222866

-- Define the conditions
def ferry_capacity_tons : ℕ := 50
def automobile_min_weight : ℕ := 1600
def automobile_max_weight : ℕ := 3200

-- Define the conversion factor from tons to pounds
def ton_to_pound : ℕ := 2000

-- Define the converted ferry capacity in pounds
def ferry_capacity_pounds := ferry_capacity_tons * ton_to_pound

-- Proof statement
theorem maximum_automobiles_on_ferry : 
  ferry_capacity_pounds / automobile_min_weight = 62 :=
by
  -- Given: ferry capacity is 50 tons and 1 ton = 2000 pounds
  -- Therefore, ferry capacity in pounds is 50 * 2000 = 100000 pounds
  -- The weight of the lightest automobile is 1600 pounds
  -- Maximum number of automobiles = 100000 / 1600 = 62.5
  -- Rounding down to the nearest whole number gives 62
  sorry  -- Proof steps would be filled here

end maximum_automobiles_on_ferry_l2228_222866


namespace min_value_f_l2228_222892

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^3 + (Real.cos x)^2

theorem min_value_f : ∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 26 / 27 :=
by
  sorry

end min_value_f_l2228_222892


namespace min_dist_on_circle_l2228_222887

theorem min_dist_on_circle :
  let P (θ : ℝ) := (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)
  let M := (0, 2)
  ∃ θ_min : ℝ, 
    (∀ θ : ℝ, 
      let dist (P : ℝ × ℝ) (M : ℝ × ℝ) := Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)
      dist (P θ) M ≥ dist (P θ_min) M) ∧ 
    dist (P θ_min) M = Real.sqrt 5 - 1 := sorry

end min_dist_on_circle_l2228_222887


namespace required_moles_H2SO4_l2228_222825

-- Definitions for the problem
def moles_NaCl := 2
def moles_H2SO4_needed := 2
def moles_HCl_produced := 2
def moles_NaHSO4_produced := 2

-- Condition representing stoichiometry of the reaction
axiom reaction_stoichiometry : ∀ (moles_NaCl moles_H2SO4 moles_HCl moles_NaHSO4 : ℕ), 
  moles_NaCl = moles_HCl ∧ moles_HCl = moles_NaHSO4 → moles_NaCl = moles_H2SO4

-- Proof statement we want to establish
theorem required_moles_H2SO4 : 
  ∃ (moles_H2SO4 : ℕ), moles_H2SO4 = 2 ∧ ∀ (moles_NaCl : ℕ), moles_NaCl = 2 → moles_H2SO4_needed = 2 := by
  sorry

end required_moles_H2SO4_l2228_222825


namespace minimum_number_of_kings_maximum_number_of_non_attacking_kings_l2228_222830

-- Definitions for the chessboard and king placement problem

-- Problem (a): Minimum number of kings covering the board
def minimum_kings_covering_board (board_size : Nat) : Nat :=
  sorry

theorem minimum_number_of_kings (h : 6 = board_size) :
  minimum_kings_covering_board 6 = 4 := 
  sorry

-- Problem (b): Maximum number of non-attacking kings
def maximum_non_attacking_kings (board_size : Nat) : Nat :=
  sorry

theorem maximum_number_of_non_attacking_kings (h : 6 = board_size) :
  maximum_non_attacking_kings 6 = 9 :=
  sorry

end minimum_number_of_kings_maximum_number_of_non_attacking_kings_l2228_222830


namespace difference_in_soda_bottles_l2228_222801

-- Define the given conditions
def regular_soda_bottles : ℕ := 81
def diet_soda_bottles : ℕ := 60

-- Define the difference in the number of bottles
def difference_bottles : ℕ := regular_soda_bottles - diet_soda_bottles

-- The theorem we want to prove
theorem difference_in_soda_bottles : difference_bottles = 21 := by
  sorry

end difference_in_soda_bottles_l2228_222801


namespace deductive_reasoning_example_is_A_l2228_222861

def isDeductive (statement : String) : Prop := sorry

-- Define conditions
def optionA : String := "Since y = 2^x is an exponential function, the function y = 2^x passes through the fixed point (0,1)"
def optionB : String := "Guessing the general formula for the sequence 1/(1×2), 1/(2×3), 1/(3×4), ... as a_n = 1/(n(n+1)) (n ∈ ℕ⁺)"
def optionC : String := "Drawing an analogy from 'In a plane, two lines perpendicular to the same line are parallel' to infer 'In space, two planes perpendicular to the same plane are parallel'"
def optionD : String := "From the circle's equation in the Cartesian coordinate plane (x-a)² + (y-b)² = r², predict that the equation of a sphere in three-dimensional Cartesian coordinates is (x-a)² + (y-b)² + (z-c)² = r²"

theorem deductive_reasoning_example_is_A : isDeductive optionA :=
by
  sorry

end deductive_reasoning_example_is_A_l2228_222861


namespace solution_set_of_inequality_l2228_222896

theorem solution_set_of_inequality (x : ℝ) : (|x - 3| < 1) → (2 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_inequality_l2228_222896


namespace find_x_l2228_222878

theorem find_x (x : ℕ) : 3 * 2^x + 5 * 2^x = 2048 → x = 8 := by
  sorry

end find_x_l2228_222878


namespace triangle_area_l2228_222813

-- Define the vertices of the triangle
def point_A : (ℝ × ℝ) := (0, 0)
def point_B : (ℝ × ℝ) := (8, -3)
def point_C : (ℝ × ℝ) := (4, 7)

-- Function to compute the area of a triangle given its vertices
def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Conjecture the area of triangle ABC is 30.0 square units
theorem triangle_area : area_of_triangle point_A point_B point_C = 30.0 := by
  sorry

end triangle_area_l2228_222813


namespace average_consecutive_from_c_l2228_222836

variable (a : ℕ) (c : ℕ)

-- Condition: c is the average of seven consecutive integers starting from a
axiom h1 : c = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7

-- Target statement: Prove the average of seven consecutive integers starting from c is a + 6
theorem average_consecutive_from_c : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 6 :=
by
  sorry

end average_consecutive_from_c_l2228_222836


namespace compute_difference_of_squares_l2228_222860

theorem compute_difference_of_squares : (303^2 - 297^2) = 3600 := by
  sorry

end compute_difference_of_squares_l2228_222860


namespace necessary_condition_not_sufficient_condition_l2228_222880

def P (x : ℝ) := x > 0
def Q (x : ℝ) := x > -2

theorem necessary_condition : ∀ x: ℝ, P x → Q x := 
by sorry

theorem not_sufficient_condition : ∃ x: ℝ, Q x ∧ ¬ P x := 
by sorry

end necessary_condition_not_sufficient_condition_l2228_222880


namespace factor_roots_l2228_222832

theorem factor_roots (t : ℝ) : (x - t) ∣ (8 * x^2 + 18 * x - 5) ↔ (t = 1 / 4 ∨ t = -5) :=
by
  sorry

end factor_roots_l2228_222832


namespace total_goals_correct_l2228_222829

-- Define the number of goals scored by each team in each period
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := (1 / 2) * kickers_first_period_goals
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- Define the total goals scored by both teams
def total_goals : ℕ :=
  kickers_first_period_goals + 
  kickers_second_period_goals + 
  spiders_first_period_goals + 
  spiders_second_period_goals

-- State the theorem to be proved
theorem total_goals_correct : total_goals = 15 := by
  sorry

end total_goals_correct_l2228_222829


namespace log_domain_l2228_222815

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.log 2

theorem log_domain :
  ∀ x : ℝ, (∃ y : ℝ, f y = Real.log (x - 1) / Real.log 2) ↔ x ∈ Set.Ioi 1 :=
by {
  sorry
}

end log_domain_l2228_222815


namespace Clea_Rides_Escalator_Alone_l2228_222862

-- Defining the conditions
variables (x y k : ℝ)
def Clea_Walking_Speed := x
def Total_Distance := y = 75 * x
def Time_with_Moving_Escalator := 30 * (x + k) = y
def Escalator_Speed := k = 1.5 * x

-- Stating the proof problem
theorem Clea_Rides_Escalator_Alone : 
  Total_Distance x y → 
  Time_with_Moving_Escalator x y k → 
  Escalator_Speed x k → 
  y / k = 50 :=
by
  intros
  sorry

end Clea_Rides_Escalator_Alone_l2228_222862
