import Mathlib

namespace min_diff_composite_sum_105_l109_10914

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def sum_to_105 (a b : ℕ) : Prop := a + b = 105

theorem min_diff_composite_sum_105 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ sum_to_105 a b ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → sum_to_105 c d →
  (c : ℤ) - (d : ℤ) ≥ 3 ∨ (d : ℤ) - (c : ℤ) ≥ 3 :=
sorry

end min_diff_composite_sum_105_l109_10914


namespace quadratic_root_arithmetic_sequence_l109_10912

/-- Given real numbers a, b, c forming an arithmetic sequence with a ≥ b ≥ c ≥ 0,
    if the quadratic ax^2 + bx + c has exactly one root, then this root is -2 + √3 -/
theorem quadratic_root_arithmetic_sequence (a b c : ℝ) : 
  (∃ d : ℝ, b = a - d ∧ c = a - 2*d) →  -- arithmetic sequence
  a ≥ b ∧ b ≥ c ∧ c ≥ 0 →  -- ordering condition
  (∃! x : ℝ, a*x^2 + b*x + c = 0) →  -- exactly one root
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ x = -2 + Real.sqrt 3) :=
by sorry

end quadratic_root_arithmetic_sequence_l109_10912


namespace even_decreasing_inequality_l109_10916

-- Define an even function that is decreasing on (0,+∞)
def is_even_and_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x)

-- State the theorem
theorem even_decreasing_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h : is_even_and_decreasing f) : 
  f (-3/4) ≥ f (a^2 - a + 1) := by
  sorry

end even_decreasing_inequality_l109_10916


namespace train_speed_l109_10950

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : Real) (time : Real) (h1 : length = 200) (h2 : time = 12) :
  (length / 1000) / (time / 3600) = 60 := by
  sorry

end train_speed_l109_10950


namespace train_passing_time_l109_10913

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 20 → 
  train_speed_kmph = 36 → 
  (train_length / (train_speed_kmph * (1000 / 3600))) = 2 := by
  sorry

end train_passing_time_l109_10913


namespace symmetric_point_correct_l109_10998

/-- The point symmetric to A(3, 4) with respect to the x-axis -/
def symmetric_point : ℝ × ℝ := (3, -4)

/-- The original point A -/
def point_A : ℝ × ℝ := (3, 4)

/-- Theorem stating that symmetric_point is indeed symmetric to point_A with respect to the x-axis -/
theorem symmetric_point_correct :
  symmetric_point.1 = point_A.1 ∧
  symmetric_point.2 = -point_A.2 := by sorry

end symmetric_point_correct_l109_10998


namespace original_price_calculation_l109_10969

/-- Proves that given an article sold at a 30% profit with a selling price of 715, 
    the original price (cost price) of the article is 550. -/
theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 715)
    (h2 : profit_percentage = 30) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + profit_percentage / 100) ∧ 
    original_price = 550 := by
  sorry

end original_price_calculation_l109_10969


namespace functional_equation_solutions_l109_10985

/-- The functional equation that f must satisfy for all x and y -/
def functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

/-- The theorem stating the only solutions to the functional equation -/
theorem functional_equation_solutions :
  ∀ α : ℝ, ∀ f : ℝ → ℝ,
    functional_equation f α →
    ((α = 1 ∧ ∀ x, f x = -x) ∨ (α = -1 ∧ ∀ x, f x = x)) :=
by sorry

end functional_equation_solutions_l109_10985


namespace equation_solution_l109_10945

theorem equation_solution : ∃! x : ℝ, (x^2 - 6*x + 8)/(x^2 - 7*x + 12) = (x^2 - 3*x - 10)/(x^2 + x - 12) ∧ x = 0 := by
  sorry

end equation_solution_l109_10945


namespace coefficient_x_cubed_in_expansion_l109_10947

theorem coefficient_x_cubed_in_expansion : 
  let n : ℕ := 5
  let k : ℕ := 3
  let a : ℤ := 1
  let b : ℤ := -2
  (n.choose k) * (b ^ k) * (a ^ (n - k)) = -80 := by
  sorry

end coefficient_x_cubed_in_expansion_l109_10947


namespace resort_worker_tips_l109_10994

theorem resort_worker_tips (total_months : ℕ) (specific_month_multiplier : ℕ) :
  total_months = 7 ∧ specific_month_multiplier = 10 →
  (specific_month_multiplier : ℚ) / ((total_months - 1 : ℕ) + specific_month_multiplier : ℚ) = 5 / 8 := by
  sorry

end resort_worker_tips_l109_10994


namespace remainder_of_4n_mod_4_l109_10949

theorem remainder_of_4n_mod_4 (n : ℤ) (h : n % 4 = 3) : (4 * n) % 4 = 0 := by
  sorry

end remainder_of_4n_mod_4_l109_10949


namespace distance_BA_is_54_l109_10965

/-- Represents a circular path with three points -/
structure CircularPath where
  -- Distance from A to B
  dAB : ℝ
  -- Distance from B to C
  dBC : ℝ
  -- Distance from C to A
  dCA : ℝ
  -- Ensure all distances are positive
  all_positive : 0 < dAB ∧ 0 < dBC ∧ 0 < dCA

/-- The distance from B to A in the opposite direction on the circular path -/
def distance_BA (path : CircularPath) : ℝ :=
  path.dBC + path.dCA

/-- Theorem stating the distance from B to A in the opposite direction -/
theorem distance_BA_is_54 (path : CircularPath) 
  (h1 : path.dAB = 30) 
  (h2 : path.dBC = 28) 
  (h3 : path.dCA = 26) : 
  distance_BA path = 54 := by
  sorry

end distance_BA_is_54_l109_10965


namespace runner_time_difference_l109_10905

theorem runner_time_difference (total_distance : ℝ) (second_half_time : ℝ) : 
  total_distance = 40 ∧ second_half_time = 24 →
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    (total_distance / 2) / initial_speed + (total_distance / 2) / (initial_speed / 2) = second_half_time ∧
    (total_distance / 2) / (initial_speed / 2) - (total_distance / 2) / initial_speed = 12 := by
sorry

end runner_time_difference_l109_10905


namespace total_cost_calculation_l109_10967

def sandwich_cost : ℝ := 4
def soda_cost : ℝ := 3
def tax_rate : ℝ := 0.1
def num_sandwiches : ℕ := 4
def num_sodas : ℕ := 6

theorem total_cost_calculation :
  let subtotal := sandwich_cost * num_sandwiches + soda_cost * num_sodas
  let tax := subtotal * tax_rate
  let total_cost := subtotal + tax
  total_cost = 37.4 := by sorry

end total_cost_calculation_l109_10967


namespace custom_mul_property_l109_10987

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := (a + b + 1)^2

/-- Theorem stating that (x-1) * (1-x) = 1 for all real x -/
theorem custom_mul_property (x : ℝ) : custom_mul (x - 1) (1 - x) = 1 := by
  sorry

end custom_mul_property_l109_10987


namespace simplify_and_evaluate_l109_10970

theorem simplify_and_evaluate (m : ℝ) (h : m = 2) : 
  (2 * m - 6) / (m^2 - 9) / ((2 * m + 2) / (m + 3)) - m / (m + 1) = -1/3 := by
  sorry

end simplify_and_evaluate_l109_10970


namespace triangle_problem_l109_10962

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 2 * Real.sqrt 3 →
  C = π / 3 →
  Real.tan A = 3 / 4 →
  (Real.sin A = 3 / 5 ∧ b = 4 + Real.sqrt 3) := by
  sorry

end triangle_problem_l109_10962


namespace no_solutions_for_prime_power_equation_l109_10943

theorem no_solutions_for_prime_power_equation (p m n k : ℕ) : 
  Nat.Prime p → 
  p % 2 = 1 → 
  0 < n → 
  n ≤ m → 
  m ≤ 3 * n → 
  p^m + p^n + 1 = k^2 → 
  False :=
by sorry

end no_solutions_for_prime_power_equation_l109_10943


namespace two_car_speeds_l109_10957

/-- Represents the speed of two cars traveling in opposite directions -/
structure TwoCarSpeeds where
  slower : ℝ
  faster : ℝ
  speed_difference : faster = slower + 10
  total_distance : 5 * slower + 5 * faster = 500

/-- Theorem stating the speeds of the two cars -/
theorem two_car_speeds : ∃ (s : TwoCarSpeeds), s.slower = 45 ∧ s.faster = 55 := by
  sorry

end two_car_speeds_l109_10957


namespace log_product_equals_four_implies_y_equals_81_l109_10988

theorem log_product_equals_four_implies_y_equals_81 (m y : ℝ) 
  (h : m > 0) (k : y > 0) (eq : Real.log y / Real.log m * Real.log m / Real.log 3 = 4) : 
  y = 81 := by
  sorry

end log_product_equals_four_implies_y_equals_81_l109_10988


namespace larger_number_problem_l109_10973

theorem larger_number_problem (x y : ℤ) 
  (h1 : y = x + 10) 
  (h2 : x + y = 34) : 
  y = 22 := by
  sorry

end larger_number_problem_l109_10973


namespace inequality_proof_l109_10900

theorem inequality_proof (a₁ a₂ a₃ S : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1)
  (hS : S = a₁ + a₂ + a₃)
  (hₐ₁ : a₁^2 / (a₁ - 1) > S)
  (hₐ₂ : a₂^2 / (a₂ - 1) > S)
  (hₐ₃ : a₃^2 / (a₃ - 1) > S) :
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end inequality_proof_l109_10900


namespace intersection_of_A_and_B_l109_10956

def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l109_10956


namespace geometric_sum_remainder_l109_10980

theorem geometric_sum_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^(n+1) - 1)) / (r - 1) % m = 91 :=
by
  sorry

#check geometric_sum_remainder 2002 1 9 500

end geometric_sum_remainder_l109_10980


namespace triangle_tangent_inequality_l109_10989

/-- Given a triangle ABC with sides a, b, c, and points A₁, A₂, B₁, B₂, C₁, C₂ defined by lines
    parallel to the opposite sides and tangent to the incircle, prove the inequality. -/
theorem triangle_tangent_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (AA₁ AA₂ BB₁ BB₂ CC₁ CC₂ : ℝ)
  (hAA₁ : AA₁ = b * (b + c - a) / (a + b + c))
  (hAA₂ : AA₂ = c * (b + c - a) / (a + b + c))
  (hBB₁ : BB₁ = c * (c + a - b) / (a + b + c))
  (hBB₂ : BB₂ = a * (c + a - b) / (a + b + c))
  (hCC₁ : CC₁ = a * (a + b - c) / (a + b + c))
  (hCC₂ : CC₂ = b * (a + b - c) / (a + b + c)) :
  AA₁ * AA₂ + BB₁ * BB₂ + CC₁ * CC₂ ≥ (1 / 9) * (a^2 + b^2 + c^2) :=
sorry

end triangle_tangent_inequality_l109_10989


namespace president_and_committee_10_people_l109_10911

/-- The number of ways to choose a president and a 3-person committee from a group of n people,
    where the president cannot be part of the committee -/
def choose_president_and_committee (n : ℕ) : ℕ :=
  n * (Nat.choose (n - 1) 3)

/-- Theorem stating that choosing a president and a 3-person committee from 10 people,
    where the president cannot be part of the committee, can be done in 840 ways -/
theorem president_and_committee_10_people :
  choose_president_and_committee 10 = 840 := by
  sorry

end president_and_committee_10_people_l109_10911


namespace quadratic_shift_l109_10951

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

/-- Shift a function to the left -/
def shiftLeft (g : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x ↦ g (x + d)

/-- Shift a function down -/
def shiftDown (g : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x ↦ g x - d

/-- The resulting function after shifts -/
def g (x : ℝ) : ℝ := (x + 1)^2 - 2

theorem quadratic_shift :
  shiftDown (shiftLeft f 2) 3 = g := by sorry

end quadratic_shift_l109_10951


namespace team_scoring_problem_l109_10908

theorem team_scoring_problem (player1_score : ℕ) (player2_score : ℕ) (player3_score : ℕ) 
  (h1 : player1_score = 20)
  (h2 : player2_score = player1_score / 2)
  (h3 : ∃ X : ℕ, player3_score = X * player2_score)
  (h4 : (player1_score + player2_score + player3_score) / 3 = 30) :
  ∃ X : ℕ, player3_score = X * player2_score ∧ X = 6 := by
  sorry

end team_scoring_problem_l109_10908


namespace larger_number_proof_l109_10958

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 23) 
  (h2 : Nat.lcm a b = 23 * 13 * 15) (h3 : a > b) : a = 345 := by
  sorry

end larger_number_proof_l109_10958


namespace fermat_like_congruence_l109_10941

theorem fermat_like_congruence (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  let n : ℕ := (2^(2*p) - 1) / 3
  2^n - 2 ≡ 0 [MOD n] := by
  sorry

end fermat_like_congruence_l109_10941


namespace business_trip_bus_distance_l109_10978

theorem business_trip_bus_distance (total_distance : ℝ) 
  (h_total : total_distance = 1800) 
  (h_plane : total_distance / 4 = 450) 
  (h_train : total_distance / 6 = 300) 
  (h_taxi : total_distance / 8 = 225) 
  (h_bus_rental : ∃ (bus rental : ℝ), 
    bus + rental = total_distance - (450 + 300 + 225) ∧ 
    bus = rental / 2) : 
  ∃ (bus : ℝ), bus = 275 := by
  sorry

end business_trip_bus_distance_l109_10978


namespace usual_bus_time_l109_10910

/-- The usual time to catch the bus, given that walking at 4/5 of the usual speed results in missing the bus by 3 minutes, is 12 minutes. -/
theorem usual_bus_time (usual_speed : ℝ) (usual_time : ℝ) : 
  (4 / 5 * usual_speed * (usual_time + 3) = usual_speed * usual_time) → 
  usual_time = 12 := by
sorry

end usual_bus_time_l109_10910


namespace jack_cell_phone_cost_l109_10955

/- Define the cell phone plan parameters -/
def base_cost : ℝ := 25
def text_cost : ℝ := 0.08
def extra_minute_cost : ℝ := 0.10
def free_hours : ℝ := 25

/- Define Jack's usage -/
def texts_sent : ℕ := 150
def hours_talked : ℝ := 26

/- Calculate the total cost -/
def total_cost : ℝ :=
  base_cost +
  (↑texts_sent * text_cost) +
  ((hours_talked - free_hours) * 60 * extra_minute_cost)

/- Theorem to prove -/
theorem jack_cell_phone_cost : total_cost = 43 := by
  sorry

end jack_cell_phone_cost_l109_10955


namespace perimeter_sum_equals_original_l109_10959

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithIncircle where
  perimeter : ℝ
  incircle : Set ℝ × ℝ

/-- Represents a triangle cut off from the original triangle -/
structure CutOffTriangle where
  perimeter : ℝ
  touchesIncircle : Bool

/-- The theorem stating that the perimeter of the original triangle
    is equal to the sum of the perimeters of the cut-off triangles -/
theorem perimeter_sum_equals_original
  (original : TriangleWithIncircle)
  (cutoff1 cutoff2 cutoff3 : CutOffTriangle)
  (h1 : cutoff1.touchesIncircle = true)
  (h2 : cutoff2.touchesIncircle = true)
  (h3 : cutoff3.touchesIncircle = true) :
  original.perimeter = cutoff1.perimeter + cutoff2.perimeter + cutoff3.perimeter :=
sorry

end perimeter_sum_equals_original_l109_10959


namespace fourth_root_equality_l109_10968

theorem fourth_root_equality (M : ℝ) (h : M > 1) :
  (M^2 * (M * M^(1/4))^(1/3))^(1/4) = M^(29/48) := by
  sorry

end fourth_root_equality_l109_10968


namespace geometric_progressions_terms_l109_10922

theorem geometric_progressions_terms (a₁ b₁ q₁ q₂ : ℚ) (sum : ℚ) :
  a₁ = 20 →
  q₁ = 3/4 →
  b₁ = 4 →
  q₂ = 2/3 →
  sum = 158.75 →
  (∃ n : ℕ, sum = (a₁ * b₁) * (1 - (q₁ * q₂)^n) / (1 - q₁ * q₂)) →
  (∃ n : ℕ, n = 7 ∧
    sum = (a₁ * b₁) * (1 - (q₁ * q₂)^n) / (1 - q₁ * q₂)) :=
by sorry

end geometric_progressions_terms_l109_10922


namespace function_values_l109_10929

noncomputable def f (A : ℝ) (x : ℝ) : ℝ := A * Real.sin x - Real.sqrt 3 * Real.cos x

theorem function_values (A : ℝ) :
  f A (π / 3) = 0 →
  A = 1 ∧ f A (π / 12) = -Real.sqrt 2 := by
  sorry

end function_values_l109_10929


namespace quadratic_form_h_l109_10942

theorem quadratic_form_h (a k : ℝ) (h : ℝ) :
  (∀ x, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) →
  h = -3/2 := by
sorry

end quadratic_form_h_l109_10942


namespace range_of_a_l109_10938

-- Define the open interval (1, 2)
def open_interval := {x : ℝ | 1 < x ∧ x < 2}

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ open_interval, (x - 1)^2 < Real.log x / Real.log a

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, inequality_holds a ↔ a ∈ {a : ℝ | 1 < a ∧ a ≤ 2} :=
by sorry

end range_of_a_l109_10938


namespace polynomial_divisibility_theorem_l109_10926

theorem polynomial_divisibility_theorem : 
  ∃ (r : ℝ), 
    (∀ (x : ℝ), ∃ (q : ℝ), 8*x^3 - 4*x^2 - 42*x + 45 = (x - r)^2 * q) ∧ 
    (abs (r - 1.5) < 0.1) := by
  sorry

end polynomial_divisibility_theorem_l109_10926


namespace sum_of_erased_numbers_l109_10948

/-- Represents a sequence of odd numbers -/
def OddSequence (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * i + 1)

/-- The sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n ^ 2

/-- Theorem: Sum of erased numbers in the sequence -/
theorem sum_of_erased_numbers
  (n : ℕ) -- Length of the first part
  (h1 : sumOddNumbers (n + 2) = 4147) -- Sum of third part is 4147
  (h2 : n > 0) -- Ensure non-empty sequence
  : ∃ (a b : ℕ), a ∈ OddSequence (4 * n + 6) ∧ 
                 b ∈ OddSequence (4 * n + 6) ∧ 
                 a + b = 168 :=
sorry

end sum_of_erased_numbers_l109_10948


namespace math_interest_group_size_l109_10995

theorem math_interest_group_size (total_cards : ℕ) : 
  (total_cards = 182) → 
  (∃ n : ℕ, n * (n - 1) = total_cards ∧ n > 0) → 
  (∃ n : ℕ, n * (n - 1) = total_cards ∧ n = 14) :=
by
  sorry

end math_interest_group_size_l109_10995


namespace missing_number_proof_l109_10984

theorem missing_number_proof (x : ℝ) (y : ℝ) : 
  (12 + x + y + 78 + 104) / 5 = 62 ∧ 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 → 
  y = 42 :=
by
  sorry

end missing_number_proof_l109_10984


namespace perfect_square_addition_subtraction_l109_10972

theorem perfect_square_addition_subtraction : ∃! n : ℤ, 
  (∃ u : ℤ, n + 5 = u^2) ∧ (∃ v : ℤ, n - 11 = v^2) :=
by
  sorry

end perfect_square_addition_subtraction_l109_10972


namespace smallest_gcd_bc_l109_10936

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 168) (hac : Nat.gcd a c = 693) :
  ∃ (d : ℕ), d = Nat.gcd b c ∧ d ≥ 21 ∧ ∀ (e : ℕ), e = Nat.gcd b c → e ≥ d :=
sorry

end smallest_gcd_bc_l109_10936


namespace abc_inequality_l109_10919

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a^(1/7) + b^(1/7) + c^(1/7)) :
  a^a * b^b * c^c ≥ 1 := by
  sorry

end abc_inequality_l109_10919


namespace rosie_pies_theorem_l109_10997

def apples_per_pie (total_apples : ℕ) (pies : ℕ) : ℕ := total_apples / pies

def pies_from_apples (available_apples : ℕ) (apples_per_pie : ℕ) : ℕ := available_apples / apples_per_pie

def leftover_apples (available_apples : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ := available_apples - pies * apples_per_pie

theorem rosie_pies_theorem (available_apples : ℕ) (base_apples : ℕ) (base_pies : ℕ) :
  available_apples = 55 →
  base_apples = 15 →
  base_pies = 3 →
  let apples_per_pie := apples_per_pie base_apples base_pies
  let pies := pies_from_apples available_apples apples_per_pie
  let leftovers := leftover_apples available_apples pies apples_per_pie
  pies = 11 ∧ leftovers = 0 := by sorry

end rosie_pies_theorem_l109_10997


namespace unique_solution_l109_10903

/-- Prove that 7 is the only positive integer solution to the equation -/
theorem unique_solution : ∃! (x : ℕ), x > 0 ∧ (1/4 : ℚ) * (10*x + 7 - x^2) - x = 0 := by
  sorry

end unique_solution_l109_10903


namespace university_groups_l109_10964

theorem university_groups (total_students : ℕ) (group_reduction : ℕ) 
  (h1 : total_students = 2808)
  (h2 : group_reduction = 4)
  (h3 : ∃ (n : ℕ), n > 0 ∧ total_students % n = 0 ∧ total_students % (n + group_reduction) = 0)
  (h4 : ∀ (n : ℕ), n > 0 → total_students % n = 0 → (total_students / n < 30)) :
  ∃ (new_groups : ℕ), new_groups = 104 ∧ 
    total_students % new_groups = 0 ∧
    total_students % (new_groups + group_reduction) = 0 ∧
    total_students / new_groups < 30 :=
by sorry

end university_groups_l109_10964


namespace inverse_proportional_m_range_l109_10983

/-- Given an inverse proportional function y = (1 - 2m) / x with two points
    A(x₁, y₁) and B(x₂, y₂) on its graph, where x₁ < 0 < x₂ and y₁ < y₂,
    prove that the range of m is m < 1/2. -/
theorem inverse_proportional_m_range (m x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = (1 - 2*m) / x₁)
  (h2 : y₂ = (1 - 2*m) / x₂)
  (h3 : x₁ < 0)
  (h4 : 0 < x₂)
  (h5 : y₁ < y₂) :
  m < 1/2 :=
sorry

end inverse_proportional_m_range_l109_10983


namespace complex_equation_solution_l109_10953

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = Complex.I → z = (1/2 : ℂ) + (1/2 : ℂ) * Complex.I := by
  sorry

end complex_equation_solution_l109_10953


namespace preimage_of_2_neg2_l109_10933

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x^2 - y)

-- Define the theorem
theorem preimage_of_2_neg2 :
  ∃ (x y : ℝ), x ≥ 0 ∧ f x y = (2, -2) ∧ (x, y) = (0, 2) := by
  sorry

end preimage_of_2_neg2_l109_10933


namespace vertical_tangent_iff_negative_a_l109_10924

/-- A function f(x) = ax^2 + ln(x) has a vertical tangent line if and only if a < 0 -/
theorem vertical_tangent_iff_negative_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ¬ ∃ y : ℝ, HasDerivAt (fun x => a * x^2 + Real.log x) y x) ↔ a < 0 :=
sorry

end vertical_tangent_iff_negative_a_l109_10924


namespace alternating_sequence_property_l109_10944

def alternatingSequence (n : ℕ) : ℤ := (-1) ^ (n + 1)

theorem alternating_sequence_property : ∀ n : ℕ, 
  (alternatingSequence n = 1 ∧ alternatingSequence (n + 1) = -1) ∨
  (alternatingSequence n = -1 ∧ alternatingSequence (n + 1) = 1) :=
by
  sorry

end alternating_sequence_property_l109_10944


namespace reciprocal_sum_of_difference_product_relation_l109_10928

theorem reciprocal_sum_of_difference_product_relation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x - y = 3 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end reciprocal_sum_of_difference_product_relation_l109_10928


namespace fraction_equivalence_l109_10979

theorem fraction_equivalence :
  (14 / 12 : ℚ) = 7 / 6 ∧
  (1 + 1 / 6 : ℚ) = 7 / 6 ∧
  (1 + 5 / 30 : ℚ) = 7 / 6 ∧
  (1 + 2 / 6 : ℚ) ≠ 7 / 6 ∧
  (1 + 14 / 42 : ℚ) = 7 / 6 :=
by sorry

end fraction_equivalence_l109_10979


namespace ribbon_lengths_after_cutting_l109_10966

def initial_lengths : List ℝ := [15, 20, 24, 26, 30]

def median (l : List ℝ) : ℝ := sorry
def range (l : List ℝ) : ℝ := sorry
def average (l : List ℝ) : ℝ := sorry

theorem ribbon_lengths_after_cutting (new_lengths : List ℝ) :
  (average new_lengths = average initial_lengths - 5) →
  (median new_lengths = median initial_lengths) →
  (range new_lengths = range initial_lengths) →
  new_lengths.length = initial_lengths.length →
  (∀ x ∈ new_lengths, x > 0) →
  new_lengths = [9, 9, 24, 24, 24] :=
by sorry

end ribbon_lengths_after_cutting_l109_10966


namespace brian_trip_distance_l109_10906

/-- Calculates the distance traveled given car efficiency, initial tank capacity, and fuel used --/
def distanceTraveled (efficiency : ℝ) (initialTank : ℝ) (fuelUsed : ℝ) : ℝ :=
  efficiency * fuelUsed

/-- Represents Brian's trip --/
structure BrianTrip where
  efficiency : ℝ
  initialTank : ℝ
  remainingFuelFraction : ℝ
  drivingTime : ℝ

/-- Theorem stating the distance Brian traveled --/
theorem brian_trip_distance (trip : BrianTrip) 
  (h1 : trip.efficiency = 20)
  (h2 : trip.initialTank = 15)
  (h3 : trip.remainingFuelFraction = 3/7)
  (h4 : trip.drivingTime = 2) :
  ∃ (distance : ℝ), abs (distance - distanceTraveled trip.efficiency trip.initialTank (trip.initialTank * (1 - trip.remainingFuelFraction))) < 0.1 :=
by sorry

#check brian_trip_distance

end brian_trip_distance_l109_10906


namespace ball_placement_theorem_l109_10907

/-- The number of ways to place 5 numbered balls into 5 numbered boxes -/
def ball_placement_count : ℕ := 20

/-- A function that returns the number of ways to place n numbered balls into n numbered boxes
    such that exactly k balls match their box numbers -/
def place_balls (n k : ℕ) : ℕ := sorry

/-- The theorem stating that the number of ways to place 5 numbered balls into 5 numbered boxes,
    where each box contains one ball and exactly two balls match their box numbers, is 20 -/
theorem ball_placement_theorem : place_balls 5 2 = ball_placement_count := by sorry

end ball_placement_theorem_l109_10907


namespace max_cfriendly_diff_l109_10909

-- Define c-friendly function
def CFriendly (c : ℝ) (f : ℝ → ℝ) : Prop :=
  c > 1 ∧
  f 0 = 0 ∧
  f 1 = 1 ∧
  ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → |f x - f y| ≤ c * |x - y|

-- State the theorem
theorem max_cfriendly_diff (c : ℝ) (f : ℝ → ℝ) (x y : ℝ) :
  CFriendly c f →
  x ∈ Set.Icc 0 1 →
  y ∈ Set.Icc 0 1 →
  |f x - f y| ≤ (c + 1) / 2 := by
  sorry

end max_cfriendly_diff_l109_10909


namespace heptagon_diagonals_l109_10993

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l109_10993


namespace sum_of_distinct_prime_factors_l109_10981

theorem sum_of_distinct_prime_factors : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p ∧ p ∣ (25^3 - 27^2)) ∧ 
  (∀ p, Nat.Prime p → p ∣ (25^3 - 27^2) → p ∈ s) ∧
  (s.sum id = 28) := by
  sorry

end sum_of_distinct_prime_factors_l109_10981


namespace imaginary_part_of_z_l109_10954

def i : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) : z = (2 + i) / i → z.im = -2 := by
  sorry

end imaginary_part_of_z_l109_10954


namespace trigonometric_inequality_l109_10937

theorem trigonometric_inequality : 
  let a := Real.sin (2 * Real.pi / 5)
  let b := Real.cos (5 * Real.pi / 6)
  let c := Real.tan (7 * Real.pi / 5)
  c > a ∧ a > b := by sorry

end trigonometric_inequality_l109_10937


namespace problem_statement_l109_10952

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3) :
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → y/x + 3/y ≥ 4) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x*y ≤ 9/8) ∧
  (∀ ε > 0, ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 ∧ Real.sqrt a + Real.sqrt (2*b) > 2 - ε) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → a^2 + 4*b^2 ≥ 9/2) := by
sorry

end problem_statement_l109_10952


namespace march_greatest_drop_l109_10976

/-- Represents the months of the year --/
inductive Month
| January
| February
| March
| April
| May
| June

/-- The price change for each month --/
def price_change : Month → ℝ
| Month.January => -0.5
| Month.February => 1.5
| Month.March => -3.0
| Month.April => 2.0
| Month.May => -1.0
| Month.June => -2.5

/-- The fixed transaction fee --/
def transaction_fee : ℝ := 1.0

/-- The adjusted price change after applying the transaction fee --/
def adjusted_price_change (m : Month) : ℝ :=
  price_change m - transaction_fee

/-- Theorem stating that March has the greatest monthly drop --/
theorem march_greatest_drop :
  ∀ m : Month, m ≠ Month.March →
  adjusted_price_change Month.March ≤ adjusted_price_change m :=
by sorry

end march_greatest_drop_l109_10976


namespace lillian_mushroom_foraging_l109_10918

/-- Calculates the number of uncertain mushrooms given the total, safe, and poisonous counts. -/
def uncertain_mushrooms (total safe : ℕ) : ℕ :=
  total - (safe + 2 * safe)

/-- Proves that the number of uncertain mushrooms is 5 given the problem conditions. -/
theorem lillian_mushroom_foraging :
  uncertain_mushrooms 32 9 = 5 := by
  sorry

end lillian_mushroom_foraging_l109_10918


namespace cube_greater_than_one_iff_l109_10939

theorem cube_greater_than_one_iff (x : ℝ) : x > 1 ↔ x^3 > 1 := by
  sorry

end cube_greater_than_one_iff_l109_10939


namespace sufficient_not_necessary_l109_10932

theorem sufficient_not_necessary (a : ℝ) :
  (a > 9 → (1 / a) < (1 / 9)) ∧
  ∃ b : ℝ, (1 / b) < (1 / 9) ∧ b ≤ 9 :=
by sorry

end sufficient_not_necessary_l109_10932


namespace ned_second_table_trays_l109_10974

/-- The number of trays Ned can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Ned made -/
def total_trips : ℕ := 4

/-- The number of trays Ned picked up from the first table -/
def trays_first_table : ℕ := 27

/-- The number of trays Ned picked up from the second table -/
def trays_second_table : ℕ := total_trips * trays_per_trip - trays_first_table

theorem ned_second_table_trays : trays_second_table = 5 := by
  sorry

end ned_second_table_trays_l109_10974


namespace point_location_l109_10946

theorem point_location (x y : ℝ) : 
  (4 * x + 7 * y = 28) →  -- Line equation
  (abs x = abs y) →       -- Equidistant from axes
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=  -- In quadrant I or II
by sorry

end point_location_l109_10946


namespace remainder_444_power_444_mod_13_l109_10927

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l109_10927


namespace parabola_axis_equation_l109_10963

/-- Given a parabola with equation x = (1/4)y^2, its axis equation is x = -1 -/
theorem parabola_axis_equation (y : ℝ) :
  let x := (1/4) * y^2
  (∃ p : ℝ, p/2 = 1) → (x = -1) := by
  sorry

end parabola_axis_equation_l109_10963


namespace kongming_total_score_l109_10917

/-- Represents a recruitment exam with a written test and an interview -/
structure RecruitmentExam where
  writtenTestWeight : Real
  interviewWeight : Real
  writtenTestScore : Real
  interviewScore : Real

/-- Calculates the total score for a recruitment exam -/
def totalScore (exam : RecruitmentExam) : Real :=
  exam.writtenTestScore * exam.writtenTestWeight + exam.interviewScore * exam.interviewWeight

theorem kongming_total_score :
  let exam : RecruitmentExam := {
    writtenTestWeight := 0.6,
    interviewWeight := 0.4,
    writtenTestScore := 90,
    interviewScore := 85
  }
  totalScore exam = 88 := by sorry

end kongming_total_score_l109_10917


namespace conference_room_capacity_l109_10940

theorem conference_room_capacity 
  (num_rooms : ℕ) 
  (current_occupancy : ℕ) 
  (occupancy_ratio : ℚ) :
  num_rooms = 6 →
  current_occupancy = 320 →
  occupancy_ratio = 2/3 →
  (current_occupancy : ℚ) / occupancy_ratio / num_rooms = 80 := by
  sorry

end conference_room_capacity_l109_10940


namespace sine_product_less_than_quarter_l109_10925

-- Define a structure for a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Theorem statement
theorem sine_product_less_than_quarter (t : Triangle) :
  Real.sin (t.A / 2) * Real.sin (t.B / 2) * Real.sin (t.C / 2) < 1 / 4 := by
  sorry

end sine_product_less_than_quarter_l109_10925


namespace twenty_five_percent_less_than_80_l109_10930

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/3) * x = 60 → x = 45 := by
  sorry

end twenty_five_percent_less_than_80_l109_10930


namespace triangle_angle_measure_l109_10921

theorem triangle_angle_measure (D E F : ℝ) : 
  0 < D ∧ 0 < E ∧ 0 < F →  -- Angles are positive
  D + E + F = 180 →        -- Sum of angles in a triangle
  E = 3 * F →              -- Angle E is three times angle F
  F = 18 →                 -- Angle F is 18 degrees
  D = 108 :=               -- Conclusion: Angle D is 108 degrees
by sorry

end triangle_angle_measure_l109_10921


namespace pilot_weeks_flown_l109_10992

def miles_tuesday : ℕ := 1134
def miles_thursday : ℕ := 1475
def total_miles : ℕ := 7827

theorem pilot_weeks_flown : 
  (total_miles : ℚ) / (miles_tuesday + miles_thursday : ℚ) = 3 := by
  sorry

end pilot_weeks_flown_l109_10992


namespace square_puzzle_l109_10902

/-- Given a square with side length n satisfying the equation n^2 + 20 = (n + 1)^2 - 9,
    prove that the total number of small squares is 216. -/
theorem square_puzzle (n : ℕ) (h : n^2 + 20 = (n + 1)^2 - 9) : n^2 + 20 = 216 := by
  sorry

end square_puzzle_l109_10902


namespace neg_p_and_q_implies_not_p_and_q_l109_10977

theorem neg_p_and_q_implies_not_p_and_q (p q : Prop) :
  (¬p ∧ q) → (¬p ∧ q) :=
by
  sorry

end neg_p_and_q_implies_not_p_and_q_l109_10977


namespace quadratic_properties_l109_10904

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 - 3

-- State the theorem
theorem quadratic_properties :
  (∀ x y : ℝ, f x < f y → x < y ∨ x > y) ∧  -- Opens downwards
  (∀ x : ℝ, f (x + (-2)) = f ((-2) - x)) ∧  -- Axis of symmetry is x = -2
  (∀ x : ℝ, f x < 0) ∧                      -- Does not intersect x-axis
  (∀ x y : ℝ, x > -1 → y > x → f y < f x)   -- Decreases for x > -1
  :=
by sorry

end quadratic_properties_l109_10904


namespace imaginary_part_of_z_l109_10982

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  Complex.im (2 * i / (1 - i)) = 1 := by
  sorry

end imaginary_part_of_z_l109_10982


namespace three_fourths_of_48_plus_5_l109_10991

theorem three_fourths_of_48_plus_5 : (3 / 4 : ℚ) * 48 + 5 = 41 := by
  sorry

end three_fourths_of_48_plus_5_l109_10991


namespace geometric_sequence_third_term_l109_10986

theorem geometric_sequence_third_term
  (a : ℕ → ℕ)  -- The sequence
  (h1 : a 1 = 5)  -- First term is 5
  (h2 : a 4 = 320)  -- Fourth term is 320
  (h_geom : ∀ n : ℕ, n > 0 → a (n + 1) = a n * (a 2 / a 1))  -- Geometric sequence property
  : a 3 = 80 :=
by sorry

end geometric_sequence_third_term_l109_10986


namespace tailor_cut_l109_10996

theorem tailor_cut (skirt_cut pants_cut : ℝ) : 
  skirt_cut = 0.75 → 
  skirt_cut = pants_cut + 0.25 → 
  pants_cut = 0.50 := by
sorry

end tailor_cut_l109_10996


namespace min_value_of_expression_l109_10915

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 9) :
  2/y + 1/x ≥ 1 ∧ (2/y + 1/x = 1 ↔ x = 3 ∧ y = 3) :=
sorry

end min_value_of_expression_l109_10915


namespace factorization_equality_l109_10960

theorem factorization_equality (x y : ℝ) : 3*x^2 + 6*x*y + 3*y^2 = 3*(x+y)^2 := by
  sorry

end factorization_equality_l109_10960


namespace binomial_coefficient_seven_two_l109_10999

theorem binomial_coefficient_seven_two : Nat.choose 7 2 = 21 := by
  sorry

end binomial_coefficient_seven_two_l109_10999


namespace value_added_to_numbers_l109_10931

theorem value_added_to_numbers (n : ℕ) (initial_avg final_avg x : ℚ) : 
  n = 15 → initial_avg = 40 → final_avg = 52 → 
  n * final_avg = n * initial_avg + n * x → x = 12 := by
  sorry

end value_added_to_numbers_l109_10931


namespace quadratic_inequality_solution_l109_10971

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (a * x^2 + b * x + 2 > 0) ↔ (-1/2 < x ∧ x < 1/3)) →
  a - b = -10 := by
sorry

end quadratic_inequality_solution_l109_10971


namespace subset_of_A_l109_10961

def A : Set ℝ := {x | x > -1}

theorem subset_of_A : {0} ⊆ A := by sorry

end subset_of_A_l109_10961


namespace equation_equivalence_l109_10975

theorem equation_equivalence (x y z : ℝ) :
  (x - z)^2 - 4*(x - y)*(y - z) = 0 → z + x - 2*y = 0 := by
  sorry

end equation_equivalence_l109_10975


namespace parabola_shift_theorem_l109_10935

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- Shift a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

/-- Shift a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The resulting parabola after shifts -/
def resulting_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) 4

theorem parabola_shift_theorem :
  resulting_parabola = { a := 1, b := -6, c := 13 } := by
  sorry

end parabola_shift_theorem_l109_10935


namespace max_absolute_value_of_Z_l109_10934

theorem max_absolute_value_of_Z (Z : ℂ) (h : Complex.abs (Z - (3 + 4*I)) = 1) :
  ∃ (M : ℝ), M = 6 ∧ ∀ (W : ℂ), Complex.abs (W - (3 + 4*I)) = 1 → Complex.abs W ≤ M :=
sorry

end max_absolute_value_of_Z_l109_10934


namespace school_seminar_cost_l109_10920

/-- Calculates the total amount spent by a school for a seminar with discounts and food allowance -/
theorem school_seminar_cost
  (regular_fee : ℝ)
  (discount_percent : ℝ)
  (num_teachers : ℕ)
  (food_allowance : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_percent = 5)
  (h3 : num_teachers = 10)
  (h4 : food_allowance = 10)
  : (1 - discount_percent / 100) * regular_fee * num_teachers + food_allowance * num_teachers = 1525 := by
  sorry

#check school_seminar_cost

end school_seminar_cost_l109_10920


namespace graph_composition_l109_10923

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 * (x + y + 1) = y^3 * (x + y + 1)

-- Define the components of the graph
def parabola_component (x y : ℝ) : Prop := x^2 = y^3 ∧ x + y + 1 ≠ 0
def line_component (x y : ℝ) : Prop := y = -x - 1

-- Theorem stating that the graph consists of a parabola and a line
theorem graph_composition :
  ∀ x y : ℝ, equation x y ↔ parabola_component x y ∨ line_component x y :=
sorry

end graph_composition_l109_10923


namespace rosemary_pots_correct_l109_10990

/-- The number of pots of rosemary Annie planted -/
def rosemary_pots : ℕ := 
  let basil_pots : ℕ := 3
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let thyme_leaves_per_pot : ℕ := 30
  let total_leaves : ℕ := 354
  9

theorem rosemary_pots_correct : 
  let basil_pots : ℕ := 3
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let thyme_leaves_per_pot : ℕ := 30
  let total_leaves : ℕ := 354
  rosemary_pots * rosemary_leaves_per_pot + 
  basil_pots * basil_leaves_per_pot + 
  thyme_pots * thyme_leaves_per_pot = total_leaves :=
by sorry

end rosemary_pots_correct_l109_10990


namespace emily_toys_left_l109_10901

/-- The number of toys Emily started with -/
def initial_toys : ℕ := 7

/-- The number of toys Emily sold -/
def sold_toys : ℕ := 3

/-- The number of toys Emily has left -/
def remaining_toys : ℕ := initial_toys - sold_toys

/-- Theorem stating that Emily has 4 toys left -/
theorem emily_toys_left : remaining_toys = 4 := by
  sorry

end emily_toys_left_l109_10901
