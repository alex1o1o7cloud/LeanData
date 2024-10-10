import Mathlib

namespace allie_wildflowers_l1053_105340

/-- The number of wildflowers Allie picked -/
def total_flowers : ℕ := 44

/-- The number of yellow and white flowers -/
def yellow_white : ℕ := 13

/-- The number of red and yellow flowers -/
def red_yellow : ℕ := 17

/-- The number of red and white flowers -/
def red_white : ℕ := 14

/-- The difference between red and white flowers -/
def red_white_diff : ℕ := 4

theorem allie_wildflowers :
  total_flowers = yellow_white + red_yellow + red_white :=
by sorry

end allie_wildflowers_l1053_105340


namespace jess_height_l1053_105337

/-- Given the heights of Jana, Kelly, and Jess, prove that Jess is 72 inches tall. -/
theorem jess_height (jana kelly jess : ℕ) 
  (h1 : jana = kelly + 5)
  (h2 : kelly = jess - 3)
  (h3 : jana = 74) : 
  jess = 72 := by
  sorry

end jess_height_l1053_105337


namespace driving_distance_difference_l1053_105318

/-- Represents a driver's journey --/
structure Journey where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem statement --/
theorem driving_distance_difference 
  (liam : Journey) 
  (zoe : Journey) 
  (mia : Journey) 
  (h1 : zoe.time = liam.time + 2)
  (h2 : zoe.speed = liam.speed + 7)
  (h3 : zoe.distance = liam.distance + 80)
  (h4 : mia.time = liam.time + 3)
  (h5 : mia.speed = liam.speed + 15)
  (h6 : ∀ j : Journey, j.distance = j.speed * j.time) :
  mia.distance - liam.distance = 243 := by
  sorry

end driving_distance_difference_l1053_105318


namespace proof_by_contradiction_elements_l1053_105363

/-- Elements that can be used in a proof by contradiction -/
inductive ProofByContradictionElement : Type
  | NegationOfConclusion : ProofByContradictionElement
  | KnownConditions : ProofByContradictionElement
  | AxiomsTheoremsDefinitions : ProofByContradictionElement

/-- The set of elements used in proof by contradiction -/
def ProofByContradictionSet : Set ProofByContradictionElement :=
  {ProofByContradictionElement.NegationOfConclusion,
   ProofByContradictionElement.KnownConditions,
   ProofByContradictionElement.AxiomsTheoremsDefinitions}

/-- Theorem stating that the ProofByContradictionSet contains all necessary elements -/
theorem proof_by_contradiction_elements :
  ProofByContradictionElement.NegationOfConclusion ∈ ProofByContradictionSet ∧
  ProofByContradictionElement.KnownConditions ∈ ProofByContradictionSet ∧
  ProofByContradictionElement.AxiomsTheoremsDefinitions ∈ ProofByContradictionSet :=
by sorry

end proof_by_contradiction_elements_l1053_105363


namespace distance_from_point_to_x_axis_l1053_105319

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (x y : ℝ) : ℝ := |y|

/-- The theorem stating that the distance from (-2, -√5) to the x-axis is √5 -/
theorem distance_from_point_to_x_axis :
  distance_to_x_axis (-2 : ℝ) (-Real.sqrt 5) = Real.sqrt 5 := by
  sorry

end distance_from_point_to_x_axis_l1053_105319


namespace geometric_sequence_problem_l1053_105378

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → (∃ r : ℝ, 10 * r = b ∧ b * r = 2/3) → b = 2 * Real.sqrt 15 / 3 := by
  sorry

end geometric_sequence_problem_l1053_105378


namespace urn_probability_theorem_l1053_105320

/-- Represents the colors of balls in the urn -/
inductive Color
| Red
| Blue
| Green

/-- Represents the state of the urn -/
structure UrnState where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The operation of drawing a ball and adding a matching one -/
def draw_and_add (state : UrnState) : UrnState → Prop :=
  sorry

/-- Performs the draw_and_add operation n times -/
def perform_operations (n : ℕ) (initial : UrnState) : UrnState → Prop :=
  sorry

/-- The probability of a specific final state after n operations -/
noncomputable def probability_of_state (n : ℕ) (initial final : UrnState) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initial_state : UrnState := ⟨2, 1, 0⟩
  let final_state : UrnState := ⟨3, 3, 3⟩
  probability_of_state 6 initial_state final_state = 2/7 :=
by
  sorry

end urn_probability_theorem_l1053_105320


namespace duck_percentage_among_non_swans_l1053_105390

theorem duck_percentage_among_non_swans 
  (duck_percent : ℝ) 
  (swan_percent : ℝ) 
  (eagle_percent : ℝ) 
  (sparrow_percent : ℝ) 
  (h1 : duck_percent = 40)
  (h2 : swan_percent = 20)
  (h3 : eagle_percent = 15)
  (h4 : sparrow_percent = 25)
  (h5 : duck_percent + swan_percent + eagle_percent + sparrow_percent = 100) :
  (duck_percent / (100 - swan_percent)) * 100 = 50 := by
  sorry

end duck_percentage_among_non_swans_l1053_105390


namespace quadratic_inequalities_l1053_105379

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequalities (a b c : ℝ) :
  (∀ x, (1/2 : ℝ) ≤ x ∧ x ≤ 2 → f a b c x ≥ 0) ∧
  (∀ x, x < (1/2 : ℝ) ∨ x > 2 → f a b c x < 0) →
  b > 0 ∧ a + b + c > 0 := by
  sorry

end quadratic_inequalities_l1053_105379


namespace f_at_5_l1053_105372

/-- The polynomial function f(x) = 2x^5 - 5x^4 - 4x^3 + 3x^2 - 524 -/
def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 524

/-- Theorem: The value of f(5) is 2176 -/
theorem f_at_5 : f 5 = 2176 := by
  sorry

end f_at_5_l1053_105372


namespace abcdef_hex_to_binary_bits_l1053_105353

theorem abcdef_hex_to_binary_bits : ∃ (n : ℕ), n = 24 ∧ 
  (2^(n-1) : ℕ) ≤ (0xABCDEF : ℕ) ∧ (0xABCDEF : ℕ) < 2^n :=
by sorry

end abcdef_hex_to_binary_bits_l1053_105353


namespace range_of_m_for_p_or_q_l1053_105338

theorem range_of_m_for_p_or_q (m : ℝ) :
  (∃ x₀ : ℝ, m * x₀^2 + 1 ≤ 0) ∨ (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ m < 2 :=
by sorry

end range_of_m_for_p_or_q_l1053_105338


namespace sum_of_a_and_b_l1053_105333

theorem sum_of_a_and_b (a b : ℝ) : a^2 + b^2 + 2*a - 4*b + 5 = 0 → a + b = 1 := by
  sorry

end sum_of_a_and_b_l1053_105333


namespace divide_8900_by_6_and_4_l1053_105396

theorem divide_8900_by_6_and_4 : (8900 / 6) / 4 = 370.8333333333333 := by
  sorry

end divide_8900_by_6_and_4_l1053_105396


namespace sum_is_even_l1053_105328

theorem sum_is_even (a b p : ℕ) (ha : 4 ∣ a) (hb1 : 6 ∣ b) (hb2 : p ∣ b) (hp : Prime p) :
  Even (a + b) := by
  sorry

end sum_is_even_l1053_105328


namespace least_integer_absolute_value_inequality_l1053_105374

theorem least_integer_absolute_value_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |3*y - 4| ≤ 25 → x ≤ y) ∧ |3*x - 4| ≤ 25 :=
by
  -- The proof would go here
  sorry

end least_integer_absolute_value_inequality_l1053_105374


namespace complex_power_eight_l1053_105371

theorem complex_power_eight (z : ℂ) : 
  z = (1 - Complex.I * Real.sqrt 3) / 2 → 
  z^8 = -(1 + Complex.I * Real.sqrt 3) / 2 := by
sorry

end complex_power_eight_l1053_105371


namespace boat_speed_in_still_water_l1053_105361

/-- The speed of a boat in still water, given the speed of the current and the upstream speed. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : current_speed = 20) 
  (h2 : upstream_speed = 30) : 
  ∃ (still_water_speed : ℝ), still_water_speed = 50 ∧ 
    still_water_speed - current_speed = upstream_speed :=
by sorry

end boat_speed_in_still_water_l1053_105361


namespace absolute_value_inequality_solution_set_l1053_105301

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| - |x - 3| ≥ 0} = {x : ℝ | x ≥ 1} := by sorry

end absolute_value_inequality_solution_set_l1053_105301


namespace intersection_A_B_l1053_105356

def U : Set Int := {-1, 3, 5, 7, 9}
def complement_A : Set Int := {-1, 9}
def B : Set Int := {3, 7, 9}

theorem intersection_A_B :
  let A := U \ complement_A
  (A ∩ B) = {3, 7} := by sorry

end intersection_A_B_l1053_105356


namespace square_side_increase_l1053_105392

theorem square_side_increase (a : ℝ) (h : a > 0) : 
  let b := 2 * a
  let c := b * (1 + 60 / 100)
  c^2 = (a^2 + b^2) * (1 + 104.8 / 100) :=
by sorry

end square_side_increase_l1053_105392


namespace notebook_cost_l1053_105324

/-- Given the cost of items and the total spent, prove the cost of each notebook -/
theorem notebook_cost 
  (pen_cost : ℕ) 
  (folder_cost : ℕ) 
  (num_pens : ℕ) 
  (num_notebooks : ℕ) 
  (num_folders : ℕ) 
  (total_spent : ℕ) 
  (h1 : pen_cost = 1) 
  (h2 : folder_cost = 5) 
  (h3 : num_pens = 3) 
  (h4 : num_notebooks = 4) 
  (h5 : num_folders = 2) 
  (h6 : total_spent = 25) : 
  (total_spent - num_pens * pen_cost - num_folders * folder_cost) / num_notebooks = 3 := by
  sorry

end notebook_cost_l1053_105324


namespace custom_op_zero_l1053_105345

def custom_op (a b c : ℝ) : ℝ := 3 * (a - b - c)^2

theorem custom_op_zero (x y z : ℝ) : 
  custom_op ((x - y - z)^2) ((y - x - z)^2) ((z - x - y)^2) = 0 := by
  sorry

end custom_op_zero_l1053_105345


namespace final_wage_calculation_l1053_105386

/-- Calculates the final wage after a raise and a pay cut -/
theorem final_wage_calculation (initial_wage : ℝ) (raise_percentage : ℝ) (pay_cut_percentage : ℝ) :
  initial_wage = 10 →
  raise_percentage = 0.2 →
  pay_cut_percentage = 0.75 →
  initial_wage * (1 + raise_percentage) * pay_cut_percentage = 9 := by
  sorry

#check final_wage_calculation

end final_wage_calculation_l1053_105386


namespace mabels_daisy_problem_l1053_105397

/-- Given a number of daisies and petals per daisy, calculate the total number of petals --/
def total_petals (num_daisies : ℕ) (petals_per_daisy : ℕ) : ℕ :=
  num_daisies * petals_per_daisy

/-- Given an initial number of daisies and the number of daisies given away,
    calculate the remaining number of daisies --/
def remaining_daisies (initial_daisies : ℕ) (daisies_given : ℕ) : ℕ :=
  initial_daisies - daisies_given

theorem mabels_daisy_problem (initial_daisies : ℕ) (petals_per_daisy : ℕ) (daisies_given : ℕ)
    (h1 : initial_daisies = 5)
    (h2 : petals_per_daisy = 8)
    (h3 : daisies_given = 2) :
  total_petals (remaining_daisies initial_daisies daisies_given) petals_per_daisy = 24 := by
  sorry


end mabels_daisy_problem_l1053_105397


namespace range_of_expression_l1053_105399

theorem range_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  1 / a + a / b ≥ 1 + 2 * Real.sqrt 2 := by
  sorry

end range_of_expression_l1053_105399


namespace janabel_sales_sum_l1053_105303

theorem janabel_sales_sum (n : ℕ) (a₁ d : ℤ) (h1 : n = 12) (h2 : a₁ = 1) (h3 : d = 4) :
  (n : ℤ) * (2 * a₁ + (n - 1) * d) / 2 = 276 :=
by sorry

end janabel_sales_sum_l1053_105303


namespace increasing_sin_plus_linear_range_of_a_l1053_105341

/-- A function f : ℝ → ℝ is increasing if for all x₁ x₂, x₁ < x₂ implies f x₁ < f x₂ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- The main theorem: if y = sin x + ax is an increasing function on ℝ, then a ≥ 1 -/
theorem increasing_sin_plus_linear (a : ℝ) :
  IsIncreasing (fun x => Real.sin x + a * x) → a ≥ 1 := by
  sorry

/-- The range of a is [1, +∞) -/
theorem range_of_a (a : ℝ) :
  (IsIncreasing (fun x => Real.sin x + a * x) ↔ a ∈ Set.Ici 1) := by
  sorry

end increasing_sin_plus_linear_range_of_a_l1053_105341


namespace triangle_property_l1053_105367

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c / 2 = t.b - t.a * Real.cos t.C) : 
  (Real.cos t.A = 1 / 2) ∧ 
  (t.a = Real.sqrt 15 → t.b = 4 → t.c^2 - 4*t.c + 1 = 0) := by
  sorry

-- Note: The proof is omitted as per the instructions

end triangle_property_l1053_105367


namespace gcd_12347_30841_l1053_105352

theorem gcd_12347_30841 : Nat.gcd 12347 30841 = 1 := by
  sorry

end gcd_12347_30841_l1053_105352


namespace sequence_properties_l1053_105346

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ+) : ℚ := n^2 + 2*n

/-- The nth term of the sequence a_n -/
def a (n : ℕ+) : ℚ := 2*n + 1

/-- The nth term of the sequence b_n -/
def b (n : ℕ+) : ℚ := 1 / (a n * a (n + 1))

/-- The sum of the first n terms of the sequence b_n -/
def T (n : ℕ+) : ℚ := n / (3 * (2*n + 3))

theorem sequence_properties (n : ℕ+) :
  (∀ k : ℕ+, k ≤ n → S k = k^2 + 2*k) →
  (a n = 2*n + 1) ∧
  (T n = n / (3 * (2*n + 3))) :=
by sorry

end sequence_properties_l1053_105346


namespace zilla_savings_proof_l1053_105330

def monthly_savings (total_earnings rent other_expenses : ℝ) : ℝ :=
  total_earnings - rent - other_expenses

theorem zilla_savings_proof 
  (total_earnings : ℝ)
  (rent_percentage : ℝ)
  (rent : ℝ)
  (h1 : rent_percentage = 0.07)
  (h2 : rent = 133)
  (h3 : rent = total_earnings * rent_percentage)
  (h4 : let other_expenses := total_earnings / 2;
        monthly_savings total_earnings rent other_expenses = 817) : 
  ∃ (savings : ℝ), savings = 817 ∧ savings = monthly_savings total_earnings rent (total_earnings / 2) :=
sorry

end zilla_savings_proof_l1053_105330


namespace shampoo_comparison_l1053_105335

/-- Represents a bottle of shampoo with weight in grams and price in yuan -/
structure ShampooBottle where
  weight : ℚ
  price : ℚ

/-- Calculates the cost per gram of a shampoo bottle -/
def costPerGram (bottle : ShampooBottle) : ℚ :=
  bottle.price / bottle.weight

theorem shampoo_comparison (large small : ShampooBottle)
  (h_large_weight : large.weight = 450)
  (h_large_price : large.price = 36)
  (h_small_weight : small.weight = 150)
  (h_small_price : small.price = 25/2) :
  (∃ (a b : ℕ), a = 72 ∧ b = 25 ∧ large.price / small.price = a / b) ∧
  costPerGram large < costPerGram small :=
sorry

end shampoo_comparison_l1053_105335


namespace at_least_one_non_integer_distance_l1053_105376

/-- Given four points A, B, C, D on a plane with specified distances,
    prove that at least one of BD or CD is not an integer. -/
theorem at_least_one_non_integer_distance
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (h_AB : dist A B = 1)
  (h_BC : dist B C = 9)
  (h_CA : dist C A = 9)
  (h_AD : dist A D = 7) :
  ¬(∃ (bd cd : ℤ), dist B D = bd ∧ dist C D = cd) :=
sorry

end at_least_one_non_integer_distance_l1053_105376


namespace derivative_of_even_function_l1053_105398

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f as g
variable (g : ℝ → ℝ)

-- State the theorem
theorem derivative_of_even_function 
  (h1 : ∀ x, f (-x) = f x)  -- f is an even function
  (h2 : ∀ x, HasDerivAt f (g x) x)  -- g is the derivative of f
  : ∀ x, g (-x) = -g x := by
  sorry

end derivative_of_even_function_l1053_105398


namespace basketball_max_height_l1053_105331

/-- The height function of a basketball -/
def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 2

/-- The maximum height reached by the basketball -/
theorem basketball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 127 :=
sorry

end basketball_max_height_l1053_105331


namespace cubic_root_equation_solutions_l1053_105308

theorem cubic_root_equation_solutions :
  {x : ℝ | (15 * x - 1) ^ (1/3) + (13 * x + 1) ^ (1/3) = 4 * x ^ (1/3)} =
  {0, 1/14, -1/12} := by sorry

end cubic_root_equation_solutions_l1053_105308


namespace square_pyramid_frustum_volume_ratio_is_correct_l1053_105315

def square_pyramid_frustum_volume_ratio : ℚ :=
  let base_edge : ℚ := 24
  let altitude : ℚ := 10
  let small_pyramid_altitude_ratio : ℚ := 1/3
  
  let original_volume : ℚ := (1/3) * base_edge^2 * altitude
  let small_pyramid_base_edge : ℚ := base_edge * small_pyramid_altitude_ratio
  let small_pyramid_volume : ℚ := (1/3) * small_pyramid_base_edge^2 * (altitude * small_pyramid_altitude_ratio)
  let frustum_volume : ℚ := original_volume - small_pyramid_volume
  
  frustum_volume / original_volume

theorem square_pyramid_frustum_volume_ratio_is_correct :
  square_pyramid_frustum_volume_ratio = 924/960 := by
  sorry

end square_pyramid_frustum_volume_ratio_is_correct_l1053_105315


namespace square_of_difference_positive_l1053_105300

theorem square_of_difference_positive {a b : ℝ} (h : a ≠ b) : (a - b)^2 > 0 := by
  sorry

end square_of_difference_positive_l1053_105300


namespace expression_equality_l1053_105349

theorem expression_equality : -15 + 9 * (6 / 3) = 3 := by sorry

end expression_equality_l1053_105349


namespace cheries_sparklers_l1053_105389

/-- Represents the number of fireworks in a box -/
structure FireworksBox where
  sparklers : ℕ
  whistlers : ℕ

/-- The total number of fireworks in a box -/
def FireworksBox.total (box : FireworksBox) : ℕ := box.sparklers + box.whistlers

theorem cheries_sparklers (koby_box : FireworksBox) 
                          (cherie_box : FireworksBox) 
                          (h1 : koby_box.sparklers = 3)
                          (h2 : koby_box.whistlers = 5)
                          (h3 : cherie_box.whistlers = 9)
                          (h4 : 2 * koby_box.total + cherie_box.total = 33) :
  cherie_box.sparklers = 8 := by
  sorry

#check cheries_sparklers

end cheries_sparklers_l1053_105389


namespace triangle_half_angle_sine_inequality_l1053_105366

theorem triangle_half_angle_sine_inequality (A B C : Real) 
  (h : A + B + C = π) : 
  Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) < 1/4 := by
  sorry

end triangle_half_angle_sine_inequality_l1053_105366


namespace prob_each_class_one_student_prob_at_least_one_empty_class_prob_exactly_one_empty_class_l1053_105316

/-- The number of newly transferred students -/
def num_students : ℕ := 4

/-- The number of designated classes -/
def num_classes : ℕ := 4

/-- The total number of ways to distribute students into classes -/
def total_distributions : ℕ := num_classes ^ num_students

/-- The number of ways to distribute students such that each class receives one student -/
def each_class_one_student : ℕ := Nat.factorial num_classes

/-- The probability that each class receives one student -/
theorem prob_each_class_one_student :
  (each_class_one_student : ℚ) / total_distributions = 3 / 32 := by sorry

/-- The probability that at least one class does not receive any students -/
theorem prob_at_least_one_empty_class :
  1 - (each_class_one_student : ℚ) / total_distributions = 29 / 32 := by sorry

/-- The number of ways to distribute students such that exactly one class is empty -/
def exactly_one_empty_class : ℕ :=
  (num_classes.choose 1) * (num_classes.choose 2) * ((num_classes - 1).choose 1) * ((num_classes - 2).choose 1)

/-- The probability that exactly one class does not receive any students -/
theorem prob_exactly_one_empty_class :
  (exactly_one_empty_class : ℚ) / total_distributions = 9 / 16 := by sorry

end prob_each_class_one_student_prob_at_least_one_empty_class_prob_exactly_one_empty_class_l1053_105316


namespace time_to_paint_one_room_l1053_105327

theorem time_to_paint_one_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (time_for_remaining : ℕ) 
  (h1 : total_rooms = 9) 
  (h2 : painted_rooms = 5) 
  (h3 : time_for_remaining = 32) :
  (time_for_remaining / (total_rooms - painted_rooms) : ℚ) = 8 := by
  sorry

end time_to_paint_one_room_l1053_105327


namespace equation_equivalence_l1053_105355

theorem equation_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 9 = 0) ↔ ((x - 1)^2 = 10) :=
by sorry

end equation_equivalence_l1053_105355


namespace parallelogram_analogous_to_parallelepiped_l1053_105373

/-- A parallelepiped is a 3D shape with opposite faces parallel -/
structure Parallelepiped :=
  (opposite_faces_parallel : Bool)

/-- A parallelogram is a 2D shape with opposite sides parallel -/
structure Parallelogram :=
  (opposite_sides_parallel : Bool)

/-- An analogy between 3D and 2D shapes -/
def is_analogous (shape3D : Type) (shape2D : Type) : Prop :=
  ∃ (property3D : shape3D → Prop) (property2D : shape2D → Prop),
    ∀ (s3D : shape3D) (s2D : shape2D), property3D s3D ↔ property2D s2D

/-- Theorem: A parallelogram is the most analogous 2D shape to a parallelepiped -/
theorem parallelogram_analogous_to_parallelepiped :
  is_analogous Parallelepiped Parallelogram :=
sorry

end parallelogram_analogous_to_parallelepiped_l1053_105373


namespace num_purchasing_plans_eq_600_l1053_105384

/-- The number of different purchasing plans for souvenirs -/
def num_purchasing_plans : ℕ :=
  (Finset.filter (fun (x, y, z) => x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1)
    (Finset.filter (fun (x, y, z) => x + 2*y + 4*z = 101)
      (Finset.product (Finset.range 102)
        (Finset.product (Finset.range 51) (Finset.range 26))))).card

/-- Theorem stating that the number of purchasing plans is 600 -/
theorem num_purchasing_plans_eq_600 : num_purchasing_plans = 600 := by
  sorry

end num_purchasing_plans_eq_600_l1053_105384


namespace perpendicular_impossibility_l1053_105391

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)
variable (non_coincident : Line → Line → Prop)

-- State the theorem
theorem perpendicular_impossibility
  (a b : Line) (α : Plane) (P : Point)
  (h1 : non_coincident a b)
  (h2 : perpendicular a α)
  (h3 : intersect a b P) :
  ¬ (perpendicular b α) :=
sorry

end perpendicular_impossibility_l1053_105391


namespace dave_winfield_home_runs_l1053_105365

theorem dave_winfield_home_runs :
  let aaron_hr : ℕ := 755
  let winfield_hr : ℕ := 465
  aaron_hr = 2 * winfield_hr - 175 →
  winfield_hr = 465 :=
by sorry

end dave_winfield_home_runs_l1053_105365


namespace number_comparisons_l1053_105380

theorem number_comparisons :
  (-3.2 > -4.3) ∧ ((1/2 : ℚ) > -1/3) ∧ ((1/4 : ℚ) > 0) := by
  sorry

end number_comparisons_l1053_105380


namespace frog_prob_theorem_l1053_105343

/-- A triangular pond with 9 regions -/
structure TriangularPond :=
  (regions : Fin 9)

/-- The frog's position in the pond -/
inductive Position
  | A
  | Adjacent

/-- The probability of the frog being in a specific position after k jumps -/
def probability (k : ℕ) (pos : Position) : ℝ :=
  sorry

/-- The probability of the frog being in region A after 2022 jumps -/
def prob_in_A_after_2022 : ℝ :=
  probability 2022 Position.A

/-- The theorem stating the probability of the frog being in region A after 2022 jumps -/
theorem frog_prob_theorem :
  prob_in_A_after_2022 = 2/9 * (1/2)^1010 + 1/9 :=
sorry

end frog_prob_theorem_l1053_105343


namespace probability_two_slate_rocks_l1053_105383

/-- The probability of selecting two slate rocks from a field with 12 slate rocks, 
    17 pumice rocks, and 8 granite rocks, when choosing 2 rocks at random without replacement. -/
theorem probability_two_slate_rocks (slate : ℕ) (pumice : ℕ) (granite : ℕ) 
  (h_slate : slate = 12) (h_pumice : pumice = 17) (h_granite : granite = 8) :
  let total := slate + pumice + granite
  (slate / total) * ((slate - 1) / (total - 1)) = 132 / 1332 := by
  sorry

end probability_two_slate_rocks_l1053_105383


namespace min_value_expression_l1053_105325

theorem min_value_expression (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  (a + b^2) * (a^2 - b) / (a * b) = 14 ∧
  ∀ (p q : ℕ), p > 0 → q > 0 → p ≠ q →
    (p + q^2) * (p^2 - q) / (p * q) ≥ 14 := by
  sorry

end min_value_expression_l1053_105325


namespace arithmetic_calculation_l1053_105306

theorem arithmetic_calculation : 4 * (9 - 3)^2 - 8 = 136 := by
  sorry

end arithmetic_calculation_l1053_105306


namespace odot_solution_l1053_105309

-- Define the binary operation ⊙
noncomputable def odot (a b : ℝ) : ℝ :=
  a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt b^2)))

-- Theorem statement
theorem odot_solution (h : ℝ) :
  odot 9 h = 12 → h = Real.sqrt 6 := by
  sorry

end odot_solution_l1053_105309


namespace quadratic_real_root_condition_l1053_105347

theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end quadratic_real_root_condition_l1053_105347


namespace magnitude_of_complex_number_l1053_105313

theorem magnitude_of_complex_number (z : ℂ) : z = (5 * Complex.I) / (2 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end magnitude_of_complex_number_l1053_105313


namespace first_term_of_geometric_series_l1053_105336

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60. -/
theorem first_term_of_geometric_series : ∀ (a : ℝ),
  (∑' n, a * (1/4)^n) = 80 → a = 60 := by sorry

end first_term_of_geometric_series_l1053_105336


namespace probability_three_white_balls_l1053_105381

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 65 :=
by sorry

end probability_three_white_balls_l1053_105381


namespace rational_function_equality_l1053_105370

theorem rational_function_equality (x : ℝ) (h : x ≠ 1 ∧ x ≠ -2) : 
  (x^2 + 5) / (x^3 - 3*x + 2) = 1 / (x + 2) + 2 / (x - 1)^2 :=
by sorry

end rational_function_equality_l1053_105370


namespace intersection_implies_m_equals_one_l1053_105314

def A (m : ℝ) : Set ℝ := {3, 4, m^2 - 3*m - 1}
def B (m : ℝ) : Set ℝ := {2*m, -3}

theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, A m ∩ B m = {-3} → m = 1 := by
  sorry

end intersection_implies_m_equals_one_l1053_105314


namespace jerrys_remaining_debt_l1053_105354

/-- Given Jerry's debt payments over two months, calculate his remaining debt -/
theorem jerrys_remaining_debt (total_debt : ℕ) (first_payment : ℕ) (additional_payment : ℕ) :
  total_debt = 50 →
  first_payment = 12 →
  additional_payment = 3 →
  total_debt - (first_payment + (first_payment + additional_payment)) = 23 :=
by sorry

end jerrys_remaining_debt_l1053_105354


namespace infinite_complementary_sequences_with_arithmetic_l1053_105395

def is_strictly_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

def infinite_complementary_sequences (a b : ℕ → ℕ) : Prop :=
  (is_strictly_increasing a) ∧ 
  (is_strictly_increasing b) ∧
  (∀ n : ℕ, ∃ m : ℕ, n = a m ∨ n = b m) ∧
  (∀ n : ℕ, ¬(∃ m k : ℕ, n = a m ∧ n = b k))

def arithmetic_sequence (s : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, s (n + 1) = s n + d

theorem infinite_complementary_sequences_with_arithmetic (a b : ℕ → ℕ) :
  infinite_complementary_sequences a b →
  (∃ d : ℕ, arithmetic_sequence a d) →
  a 16 = 36 →
  (∀ n : ℕ, a n = 2 * n + 4) ∧
  (∀ n : ℕ, b n = if n ≤ 5 then n else 2 * n - 5) :=
sorry

end infinite_complementary_sequences_with_arithmetic_l1053_105395


namespace geometric_sequence_ninth_term_l1053_105358

theorem geometric_sequence_ninth_term :
  let a₁ : ℚ := 5
  let r : ℚ := 3/4
  let n : ℕ := 9
  let aₙ : ℕ → ℚ := λ k => a₁ * r^(k - 1)
  aₙ n = 32805/65536 := by
  sorry

end geometric_sequence_ninth_term_l1053_105358


namespace andrew_family_mask_duration_l1053_105351

/-- Calculates the number of days a package of masks will last for a family -/
def maskDuration (totalMasks : ℕ) (familySize : ℕ) (daysPerMask : ℕ) : ℕ :=
  (totalMasks / familySize) * daysPerMask

/-- Proves that for Andrew's family, 100 masks will last 80 days -/
theorem andrew_family_mask_duration :
  maskDuration 100 5 4 = 80 := by
  sorry

#eval maskDuration 100 5 4

end andrew_family_mask_duration_l1053_105351


namespace inequality_proof_l1053_105322

theorem inequality_proof (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
  2 * Real.sqrt ((1 - x^2) * (1 - y^2)) ≤ 2 * (1 - x) * (1 - y) + 1 := by
  sorry

end inequality_proof_l1053_105322


namespace pants_price_decrease_percentage_l1053_105312

theorem pants_price_decrease_percentage (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) : 
  purchase_price = 81 →
  markup_percentage = 0.25 →
  gross_profit = 5.40 →
  let original_price := purchase_price / (1 - markup_percentage)
  let decreased_price := original_price - gross_profit
  let decrease_amount := original_price - decreased_price
  (decrease_amount / original_price) * 100 = 5 := by
  sorry

end pants_price_decrease_percentage_l1053_105312


namespace sixth_term_constant_coefficient_x_squared_l1053_105375

/-- Expansion term of (x^(1/3) - 1/(2x^(1/3)))^n -/
def expansion_term (n : ℕ) (r : ℕ) : ℚ → ℚ :=
  λ x => (-1/2)^r * (n.choose r) * x^((n - 2*r : ℤ)/3)

/-- The 6th term (r = 5) is constant when n = 10 -/
theorem sixth_term_constant (n : ℕ) :
  (∀ x, expansion_term n 5 x = expansion_term n 5 1) → n = 10 :=
sorry

/-- When n = 10, the coefficient of x^2 is 45/4 -/
theorem coefficient_x_squared :
  expansion_term 10 2 = λ x => (45/4 : ℚ) * x^2 :=
sorry

end sixth_term_constant_coefficient_x_squared_l1053_105375


namespace billys_age_l1053_105339

theorem billys_age (my_age billy_age : ℕ) 
  (h1 : my_age = 4 * billy_age)
  (h2 : my_age - billy_age = 12) :
  billy_age = 4 := by
sorry

end billys_age_l1053_105339


namespace quadratic_inequality_theorem_l1053_105332

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - (a + 2) * x + b

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a b x ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2

-- Define the inequality function
def g (a b c x : ℝ) := (x - c) * (a * x - b)

-- Theorem statement
theorem quadratic_inequality_theorem (a b c : ℝ) (h : c ≠ 2) :
  solution_set a b →
  (a = 1 ∧ b = 2) ∧
  (∀ x, g a b c x > 0 ↔ 
    (c > 2 ∧ (x > c ∨ x < 2)) ∨
    (c < 2 ∧ (x > 2 ∨ x < c))) :=
by sorry

end quadratic_inequality_theorem_l1053_105332


namespace cube_root_equation_l1053_105342

theorem cube_root_equation (x : ℝ) : 
  x = 2 / (2 - Real.rpow 3 (1/3)) → 
  x = (2 * (2 + Real.rpow 3 (1/3))) / (4 - Real.rpow 9 (1/3)) :=
by sorry

end cube_root_equation_l1053_105342


namespace x_squared_plus_nine_y_squared_l1053_105377

theorem x_squared_plus_nine_y_squared (x y : ℝ) 
  (h1 : x + 3 * y = 5) (h2 : x * y = -8) : 
  x^2 + 9 * y^2 = 73 := by
  sorry

end x_squared_plus_nine_y_squared_l1053_105377


namespace remainder_of_polynomial_division_l1053_105329

theorem remainder_of_polynomial_division (x : ℤ) : 
  (x^2030 + 1) % (x^6 - x^4 + x^2 - 1) = x^2 - 1 := by sorry

end remainder_of_polynomial_division_l1053_105329


namespace polar_equation_C_max_area_OAB_l1053_105305

-- Define the curves C, C1, and C2
def C (x y : ℝ) : Prop := x^2 + y^2 = |x| + y ∧ y > 0

def C1 (x y t α : ℝ) : Prop := x = t * Real.cos α ∧ y = t * Real.sin α ∧ t > 0

def C2 (x y t α : ℝ) : Prop := x = -t * Real.sin α ∧ y = t * Real.cos α ∧ t > 0 ∧ 0 < α ∧ α < Real.pi / 2

-- Theorem for the polar coordinate equation of C
theorem polar_equation_C : 
  ∀ (ρ θ : ℝ), 0 < θ ∧ θ < Real.pi → 
  (C (ρ * Real.cos θ) (ρ * Real.sin θ) ↔ ρ = |Real.cos θ| + Real.sin θ) :=
sorry

-- Theorem for the maximum area of triangle OAB
theorem max_area_OAB :
  ∃ (x₁ y₁ x₂ y₂ t₁ t₂ α₁ α₂ : ℝ),
    C x₁ y₁ ∧ C x₂ y₂ ∧
    C1 x₁ y₁ t₁ α₁ ∧ C2 x₂ y₂ t₂ α₂ ∧
    (∀ (x₃ y₃ x₄ y₄ t₃ t₄ α₃ α₄ : ℝ),
      C x₃ y₃ ∧ C x₄ y₄ ∧ C1 x₃ y₃ t₃ α₃ ∧ C2 x₄ y₄ t₄ α₄ →
      (1 / 2 : ℝ) * |x₁ * y₂ - x₂ * y₁| ≥ (1 / 2 : ℝ) * |x₃ * y₄ - x₄ * y₃|) ∧
    (1 / 2 : ℝ) * |x₁ * y₂ - x₂ * y₁| = 1 :=
sorry

end polar_equation_C_max_area_OAB_l1053_105305


namespace projection_theorem_l1053_105302

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ := sorry

/-- The plane Q passing through the origin -/
def Q : Plane := sorry

theorem projection_theorem :
  project (6, 4, 6) Q = (4, 6, 2) →
  project (5, 2, 8) Q = (11/6, 31/6, 10/6) := by sorry

end projection_theorem_l1053_105302


namespace football_team_handedness_l1053_105310

theorem football_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 31)
  (h3 : right_handed = 57)
  (h4 : throwers ≤ right_handed) : 
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 :=
by sorry

end football_team_handedness_l1053_105310


namespace election_margin_l1053_105369

theorem election_margin (total_votes : ℕ) (vote_swing : ℕ) (final_margin_percent : ℚ) : 
  total_votes = 15000 →
  vote_swing = 3000 →
  final_margin_percent = 20 →
  let initial_winner_votes := (total_votes + vote_swing) / 2 + vote_swing / 2
  let initial_loser_votes := (total_votes - vote_swing) / 2 - vote_swing / 2
  let initial_margin := initial_winner_votes - initial_loser_votes
  initial_margin * 100 / total_votes = final_margin_percent :=
by sorry

end election_margin_l1053_105369


namespace circle_properties_l1053_105359

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + 45 = 0

-- Define the point P
def P : ℝ × ℝ := (9, 1)

-- Theorem statement
theorem circle_properties :
  -- 1. Common chord equation
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → 4*x + 3*y - 23 = 0) ∧
  -- 2. Length of common chord
  (∃ a b : ℝ, C₁ a b ∧ C₂ a b ∧
    ∃ c d : ℝ, C₁ c d ∧ C₂ c d ∧ (a ≠ c ∨ b ≠ d) ∧
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 2 * 7^(1/2 : ℝ)) ∧
  -- 3. Tangent lines
  (∀ x y : ℝ, (x = 9 ∨ 9*x + 40*y - 121 = 0) →
    ((x - P.1)^2 + (y - P.2)^2 = 0 ∨
     ∃ t : ℝ, C₂ (x + t) (y + t * (y - P.2) / (x - P.1)) ∧
              (∀ s : ℝ, s ≠ t → ¬C₂ (x + s) (y + s * (y - P.2) / (x - P.1))))) :=
sorry

end circle_properties_l1053_105359


namespace ellipse_equation_l1053_105393

/-- Given an ellipse centered at the origin with foci on the x-axis,
    focal length 4, and eccentricity √2/2, its equation is x²/8 + y²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let focal_length : ℝ := 4
  let eccentricity : ℝ := Real.sqrt 2 / 2
  x^2 / 8 + y^2 / 4 = 1 := by sorry

end ellipse_equation_l1053_105393


namespace B_power_101_l1053_105357

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 0, 1],
    ![0, 0, 0],
    ![-1, 0, 0]]

theorem B_power_101 : B^101 = B := by sorry

end B_power_101_l1053_105357


namespace total_fish_caught_l1053_105360

/-- Given 20 fishermen, where 19 caught 400 fish each and the 20th caught 2400,
    prove that the total number of fish caught is 10000. -/
theorem total_fish_caught (total_fishermen : Nat) (fish_per_fisherman : Nat) (fish_last_fisherman : Nat) :
  total_fishermen = 20 →
  fish_per_fisherman = 400 →
  fish_last_fisherman = 2400 →
  (total_fishermen - 1) * fish_per_fisherman + fish_last_fisherman = 10000 :=
by sorry

end total_fish_caught_l1053_105360


namespace optimal_arrangement_maximizes_sum_l1053_105334

/-- The type of arrangements of numbers from 1 to 1999 in a circle -/
def Arrangement := Fin 1999 → Fin 1999

/-- The sum of products of all sets of 10 consecutive numbers in an arrangement -/
def sumOfProducts (a : Arrangement) : ℕ :=
  sorry

/-- The optimal arrangement of numbers -/
def optimalArrangement : Arrangement :=
  fun i => if i.val % 2 = 0 then (1999 - i.val + 1) else (i.val + 1)

/-- Theorem stating that the optimal arrangement maximizes the sum of products -/
theorem optimal_arrangement_maximizes_sum :
  ∀ a : Arrangement, sumOfProducts a ≤ sumOfProducts optimalArrangement :=
sorry

end optimal_arrangement_maximizes_sum_l1053_105334


namespace number_of_tables_bought_l1053_105350

/-- Proves that the number of tables bought at the cost price is 15, given the conditions -/
theorem number_of_tables_bought (C S : ℝ) (N : ℕ) : 
  N * C = 20 * S → -- The cost price of N tables equals the selling price of 20 tables
  S = 0.75 * C →   -- The selling price is 75% of the cost price (due to 25% loss)
  N = 15 :=
by sorry

end number_of_tables_bought_l1053_105350


namespace amc_scoring_l1053_105344

theorem amc_scoring (total_problems : Nat) (correct_points : Int) (incorrect_points : Int) 
  (unanswered_points : Int) (attempted : Nat) (unanswered : Nat) (min_score : Int) : 
  let min_correct := ((min_score - unanswered * unanswered_points) - 
    (attempted * incorrect_points)) / (correct_points - incorrect_points)
  ⌈min_correct⌉ = 17 :=
by
  sorry

#check amc_scoring 30 7 (-1) 2 25 5 120

end amc_scoring_l1053_105344


namespace solve_equation_l1053_105385

theorem solve_equation : ∃ x : ℚ, 5 * x - 3 * x = 405 - 9 * (x + 4) → x = 369 / 11 := by
  sorry

end solve_equation_l1053_105385


namespace stock_price_change_l1053_105362

theorem stock_price_change (x : ℝ) : 
  (1 - x / 100) * 1.1 = 1 + 4.499999999999993 / 100 → x = 5 := by
  sorry

end stock_price_change_l1053_105362


namespace two_thirds_bucket_fill_time_l1053_105317

/-- Given a bucket that takes 3 minutes to fill completely, 
    prove that it takes 2 minutes to fill two-thirds of the bucket. -/
theorem two_thirds_bucket_fill_time :
  let total_time : ℝ := 3  -- Time to fill the entire bucket
  let fraction_to_fill : ℝ := 2/3  -- Fraction of the bucket we want to fill
  (fraction_to_fill * total_time) = 2 := by
  sorry

end two_thirds_bucket_fill_time_l1053_105317


namespace ballet_class_size_l1053_105388

/-- The number of large groups formed in the ballet class -/
def large_groups : ℕ := 12

/-- The number of members in each large group -/
def members_per_large_group : ℕ := 7

/-- The total number of members in the ballet class -/
def total_members : ℕ := large_groups * members_per_large_group

theorem ballet_class_size : total_members = 84 := by
  sorry

end ballet_class_size_l1053_105388


namespace correct_propositions_count_l1053_105307

/-- Represents a proposition about regression analysis -/
inductive RegressionProposition
  | residualSumOfSquares
  | correlationCoefficient
  | scatterPlotPoints
  | randomError

/-- Determines if a given proposition is correct -/
def is_correct (prop : RegressionProposition) : Bool :=
  match prop with
  | .residualSumOfSquares => true
  | .correlationCoefficient => false
  | .scatterPlotPoints => false
  | .randomError => true

/-- The set of all propositions -/
def all_propositions : List RegressionProposition :=
  [.residualSumOfSquares, .correlationCoefficient, .scatterPlotPoints, .randomError]

/-- Counts the number of correct propositions -/
def count_correct_propositions : Nat :=
  all_propositions.filter is_correct |>.length

/-- Theorem stating that the number of correct propositions is 2 -/
theorem correct_propositions_count :
  count_correct_propositions = 2 := by sorry

end correct_propositions_count_l1053_105307


namespace f_5_equals_18556_l1053_105348

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ :=
  horner_polynomial [5, 4, 3, 2, 1, 1] x

theorem f_5_equals_18556 : f 5 = 18556 := by
  sorry

end f_5_equals_18556_l1053_105348


namespace number_division_l1053_105304

theorem number_division (x : ℝ) : x - 17 = 55 → x / 9 = 8 := by
  sorry

end number_division_l1053_105304


namespace meghan_money_l1053_105387

/-- The total amount of money Meghan had -/
def total_money (hundred_bills : ℕ) (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  100 * hundred_bills + 50 * fifty_bills + 10 * ten_bills

/-- Proof that Meghan had $550 -/
theorem meghan_money : total_money 2 5 10 = 550 := by
  sorry

end meghan_money_l1053_105387


namespace specific_pyramid_volume_l1053_105326

/-- Represents a triangular pyramid with vertex P and base ABC -/
structure TriangularPyramid where
  BC : ℝ
  CA : ℝ
  AB : ℝ
  dihedral_angle : ℝ

/-- The volume of a triangular pyramid -/
def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The volume of the specific triangular pyramid is 2 -/
theorem specific_pyramid_volume :
  let p : TriangularPyramid := {
    BC := 3,
    CA := 4,
    AB := 5,
    dihedral_angle := π / 4  -- 45° in radians
  }
  volume p = 2 := by
  sorry

end specific_pyramid_volume_l1053_105326


namespace min_colors_for_pyramid_game_l1053_105394

/-- Represents a pyramid with a regular polygon base -/
structure Pyramid :=
  (base_vertices : ℕ)

/-- The total number of edges in a pyramid -/
def total_edges (p : Pyramid) : ℕ := 2 * p.base_vertices

/-- The maximum degree of any vertex in the pyramid -/
def max_vertex_degree (p : Pyramid) : ℕ := p.base_vertices

/-- The minimal number of colors needed for the coloring game on a pyramid -/
def min_colors_needed (p : Pyramid) : ℕ := p.base_vertices

theorem min_colors_for_pyramid_game (p : Pyramid) (h : p.base_vertices = 2016) :
  min_colors_needed p = 2016 :=
sorry

end min_colors_for_pyramid_game_l1053_105394


namespace fraction_leading_zeros_l1053_105382

-- Define the fraction
def fraction : ℚ := 1 / (2^4 * 5^7)

-- Define a function to count leading zeros in a decimal representation
def countLeadingZeros (q : ℚ) : ℕ :=
  sorry -- Implementation details omitted

-- Theorem statement
theorem fraction_leading_zeros :
  countLeadingZeros fraction = 6 := by
  sorry

end fraction_leading_zeros_l1053_105382


namespace circle_in_triangle_l1053_105323

/-- The distance traveled by the center of a circle rolling inside a right triangle -/
def distanceTraveled (a b c r : ℝ) : ℝ :=
  (a - 2*r) + (b - 2*r) + (c - 2*r)

theorem circle_in_triangle (a b c r : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 9) (h_b : b = 12) (h_c : c = 15) (h_r : r = 2) :
  distanceTraveled a b c r = 24 := by
sorry

end circle_in_triangle_l1053_105323


namespace negation_of_universal_positive_cubic_plus_exponential_negation_l1053_105311

theorem negation_of_universal_positive (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem cubic_plus_exponential_negation :
  (¬∀ x : ℝ, x^3 + 3^x > 0) ↔ (∃ x : ℝ, x^3 + 3^x ≤ 0) :=
by sorry

end negation_of_universal_positive_cubic_plus_exponential_negation_l1053_105311


namespace repeating_decimal_equals_fraction_l1053_105321

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 36 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end repeating_decimal_equals_fraction_l1053_105321


namespace prob_not_sunny_l1053_105368

/-- Given that the probability of a sunny day is 5/7, 
    prove that the probability of a not sunny day is 2/7 -/
theorem prob_not_sunny (prob_sunny : ℚ) (h : prob_sunny = 5 / 7) :
  1 - prob_sunny = 2 / 7 := by
  sorry

end prob_not_sunny_l1053_105368


namespace waiter_initial_customers_l1053_105364

/-- The number of customers who left the waiter's section -/
def customers_left : ℕ := 12

/-- The number of people at each table after some customers left -/
def people_per_table : ℕ := 3

/-- The number of tables in the waiter's section -/
def number_of_tables : ℕ := 3

/-- The initial number of customers in the waiter's section -/
def initial_customers : ℕ := customers_left + people_per_table * number_of_tables

theorem waiter_initial_customers :
  initial_customers = 21 :=
by sorry

end waiter_initial_customers_l1053_105364
