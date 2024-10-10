import Mathlib

namespace charles_whistle_count_l3739_373988

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end charles_whistle_count_l3739_373988


namespace tray_height_proof_l3739_373946

/-- Given a square with side length 150 and cuts starting 8 units from each corner
    meeting at a 45° angle on the diagonal, the height of the resulting tray when folded
    is equal to the fourth root of 4096. -/
theorem tray_height_proof (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 150 →
  cut_distance = 8 →
  cut_angle = 45 →
  ∃ (h : ℝ), h = (8 * Real.sqrt 2 - 8) ∧ h^4 = 4096 :=
by sorry

end tray_height_proof_l3739_373946


namespace lighthouse_distance_l3739_373925

/-- Proves that in a triangle ABS with given side length and angles, BS = 72 km -/
theorem lighthouse_distance (AB : ℝ) (angle_A angle_B : ℝ) :
  AB = 36 * Real.sqrt 6 →
  angle_A = 45 * π / 180 →
  angle_B = 75 * π / 180 →
  let angle_S := π - (angle_A + angle_B)
  let BS := AB * Real.sin angle_A / Real.sin angle_S
  BS = 72 := by sorry

end lighthouse_distance_l3739_373925


namespace certain_number_proof_l3739_373956

theorem certain_number_proof : ∃ x : ℕ, 865 * 48 = 173 * x ∧ x = 240 := by
  sorry

end certain_number_proof_l3739_373956


namespace simplify_expression_l3739_373923

theorem simplify_expression : 5 * (18 / 6) * (21 / -63) = -5 := by
  sorry

end simplify_expression_l3739_373923


namespace abs_z_equals_sqrt_5_l3739_373918

-- Define the complex number z
def z : ℂ := -Complex.I * (1 + 2 * Complex.I)

-- Theorem stating that the absolute value of z is √5
theorem abs_z_equals_sqrt_5 : Complex.abs z = Real.sqrt 5 := by
  sorry

end abs_z_equals_sqrt_5_l3739_373918


namespace completing_square_form_l3739_373975

theorem completing_square_form (x : ℝ) : 
  (x^2 - 6*x - 3 = 0) ↔ ((x - 3)^2 = 12) := by sorry

end completing_square_form_l3739_373975


namespace unique_solution_for_radical_equation_l3739_373924

theorem unique_solution_for_radical_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (10 * x) = 10) : 
  x = 1 / 6 := by
sorry

end unique_solution_for_radical_equation_l3739_373924


namespace solar_project_analysis_l3739_373985

/-- Represents the net profit of a solar power generation project over n years -/
def net_profit (n : ℕ+) : ℚ :=
  -4 * n^2 + 80 * n - 144

/-- Represents the average annual profit of the project over n years -/
def avg_annual_profit (n : ℕ+) : ℚ :=
  net_profit n / n

theorem solar_project_analysis :
  ∀ n : ℕ+,
  -- 1. Net profit function
  net_profit n = -4 * n^2 + 80 * n - 144 ∧
  -- 2. Project starts making profit from the 3rd year
  (∀ k : ℕ+, k ≥ 3 → net_profit k > 0) ∧
  (∀ k : ℕ+, k < 3 → net_profit k ≤ 0) ∧
  -- 3. Maximum average annual profit occurs when n = 6
  (∀ k : ℕ+, avg_annual_profit k ≤ avg_annual_profit 6) ∧
  -- 4. Maximum net profit occurs when n = 10
  (∀ k : ℕ+, net_profit k ≤ net_profit 10) ∧
  -- 5. Both options result in the same total profit
  net_profit 6 + 72 = net_profit 10 + 8 ∧
  net_profit 6 + 72 = 264 :=
by sorry


end solar_project_analysis_l3739_373985


namespace power_of_two_geq_n_plus_one_l3739_373971

theorem power_of_two_geq_n_plus_one (n : ℕ) (h : n ≥ 1) : 2^n ≥ n + 1 := by
  sorry

end power_of_two_geq_n_plus_one_l3739_373971


namespace equation_solution_l3739_373940

theorem equation_solution : ∃ x : ℝ, (4 / 7) * (1 / 8) * x = 12 ∧ x = 168 := by
  sorry

end equation_solution_l3739_373940


namespace hexagon_diagonals_from_vertex_l3739_373936

/-- The number of diagonals that can be drawn from one vertex of a hexagon -/
def diagonals_from_vertex_hexagon : ℕ := 3

/-- Theorem stating that the number of diagonals from one vertex of a hexagon is 3 -/
theorem hexagon_diagonals_from_vertex :
  diagonals_from_vertex_hexagon = 3 := by
  sorry

end hexagon_diagonals_from_vertex_l3739_373936


namespace eighth_term_equals_general_term_l3739_373948

/-- The general term of the sequence -/
def generalTerm (n : ℕ) (a : ℝ) : ℝ := (-1)^n * n^2 * a^(n+1)

/-- The 8th term of the sequence -/
def eighthTerm (a : ℝ) : ℝ := 64 * a^9

theorem eighth_term_equals_general_term : 
  ∀ a : ℝ, generalTerm 8 a = eighthTerm a := by sorry

end eighth_term_equals_general_term_l3739_373948


namespace odd_function_property_l3739_373905

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) :
  ∀ x, f x * f (-x) ≤ 0 := by
  sorry

end odd_function_property_l3739_373905


namespace trapezoid_area_l3739_373997

/-- A trapezoid with the given properties -/
structure Trapezoid where
  /-- Length of one diagonal -/
  diagonal1 : ℝ
  /-- Length of the other diagonal -/
  diagonal2 : ℝ
  /-- Length of the segment connecting the midpoints of the bases -/
  midpoint_segment : ℝ
  /-- The first diagonal is 3 -/
  h1 : diagonal1 = 3
  /-- The second diagonal is 5 -/
  h2 : diagonal2 = 5
  /-- The segment connecting the midpoints of the bases is 2 -/
  h3 : midpoint_segment = 2

/-- The area of the trapezoid -/
def area (t : Trapezoid) : ℝ := 6

/-- Theorem stating that the area of the trapezoid with the given properties is 6 -/
theorem trapezoid_area (t : Trapezoid) : area t = 6 := by
  sorry

end trapezoid_area_l3739_373997


namespace a_in_M_sufficient_not_necessary_for_a_in_N_l3739_373986

def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x < 2}

theorem a_in_M_sufficient_not_necessary_for_a_in_N :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by
  sorry

end a_in_M_sufficient_not_necessary_for_a_in_N_l3739_373986


namespace bakery_puzzle_l3739_373992

/-- Represents the cost of items in a bakery -/
structure BakeryCosts where
  pastry : ℚ
  cupcake : ℚ
  bagel : ℚ

/-- Represents a purchase at the bakery -/
structure Purchase where
  pastries : ℕ
  cupcakes : ℕ
  bagels : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (costs : BakeryCosts) (purchase : Purchase) : ℚ :=
  costs.pastry * purchase.pastries + costs.cupcake * purchase.cupcakes + costs.bagel * purchase.bagels

theorem bakery_puzzle (costs : BakeryCosts) : 
  let petya := Purchase.mk 1 2 3
  let anya := Purchase.mk 3 0 1
  let kolya := Purchase.mk 0 6 0
  let lena := Purchase.mk 2 0 2
  totalCost costs petya = totalCost costs anya ∧ 
  totalCost costs anya = totalCost costs kolya → 
  totalCost costs lena = totalCost costs (Purchase.mk 0 5 0) := by
  sorry


end bakery_puzzle_l3739_373992


namespace alternate_seating_four_boys_three_girls_l3739_373998

/-- The number of ways to seat 4 boys and 3 girls in a row alternately -/
def alternate_seating (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  if num_boys = 4 ∧ num_girls = 3 then
    2 * (Nat.factorial num_boys * Nat.factorial num_girls)
  else
    0

theorem alternate_seating_four_boys_three_girls :
  alternate_seating 4 3 = 288 := by
  sorry

end alternate_seating_four_boys_three_girls_l3739_373998


namespace inequality_problem_l3739_373953

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c < b * d := by
  sorry

end inequality_problem_l3739_373953


namespace simple_interest_principal_l3739_373931

/-- Simple interest calculation --/
theorem simple_interest_principal (amount : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  amount = 1456 ∧ rate = 0.05 ∧ time = 2.4 →
  principal = 1300 ∧ amount = principal * (1 + rate * time) := by
  sorry

end simple_interest_principal_l3739_373931


namespace rocket_momentum_l3739_373934

/-- Given two rockets with masses m and 9m, subjected to the same constant force F 
    for the same distance d, if the rocket with mass m acquires momentum p, 
    then the rocket with mass 9m acquires momentum 3p. -/
theorem rocket_momentum 
  (m : ℝ) 
  (F : ℝ) 
  (d : ℝ) 
  (p : ℝ) 
  (h1 : m > 0) 
  (h2 : F > 0) 
  (h3 : d > 0) 
  (h4 : p = Real.sqrt (2 * d * m * F)) : 
  9 * m * Real.sqrt ((2 * F * d) / (9 * m)) = 3 * p := by
  sorry

end rocket_momentum_l3739_373934


namespace pair_sequence_existence_l3739_373903

theorem pair_sequence_existence (n q : ℕ) (h : n > 0) (h2 : q > 0) :
  ∃ (m : ℕ) (seq : List (Fin n × Fin n)),
    m = ⌈(2 * q : ℚ) / n⌉ ∧
    seq.length = m ∧
    seq.Nodup ∧
    (∀ i < m - 1, ∃ x, (seq.get ⟨i, by sorry⟩).1 = x ∨ (seq.get ⟨i, by sorry⟩).2 = x) ∧
    (∀ i < m - 1, (seq.get ⟨i, by sorry⟩).1.val < (seq.get ⟨i + 1, by sorry⟩).1.val) :=
by sorry

end pair_sequence_existence_l3739_373903


namespace sat_score_improvement_l3739_373928

theorem sat_score_improvement (first_score second_score : ℝ) : 
  (second_score = first_score * 1.1) → 
  (second_score = 1100) → 
  (first_score = 1000) := by
sorry

end sat_score_improvement_l3739_373928


namespace sqrt_sum_difference_equals_four_sqrt_three_plus_one_l3739_373974

theorem sqrt_sum_difference_equals_four_sqrt_three_plus_one :
  Real.sqrt 12 + Real.sqrt 27 - |1 - Real.sqrt 3| = 4 * Real.sqrt 3 + 1 := by
  sorry

end sqrt_sum_difference_equals_four_sqrt_three_plus_one_l3739_373974


namespace part_one_calculation_part_two_calculation_l3739_373958

-- Part I
theorem part_one_calculation : -(-1)^1000 - 2.45 * 8 + 2.55 * (-8) = -41 := by
  sorry

-- Part II
theorem part_two_calculation : (1/6 - 1/3 + 0.25) / (-1/12) = -1 := by
  sorry

end part_one_calculation_part_two_calculation_l3739_373958


namespace sum_of_two_numbers_l3739_373944

theorem sum_of_two_numbers (x y : ℝ) : 
  (0.45 * x = 2700) → (y = 2 * x) → (x + y = 18000) := by
  sorry

end sum_of_two_numbers_l3739_373944


namespace cauchy_schwarz_inequality_l3739_373926

theorem cauchy_schwarz_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a * c + b * d ≤ Real.sqrt ((a^2 + b^2) * (c^2 + d^2)) := by
  sorry

end cauchy_schwarz_inequality_l3739_373926


namespace sum_of_roots_squared_equation_l3739_373963

theorem sum_of_roots_squared_equation (x : ℝ) :
  (∀ x, (x - 4)^2 = 16 ↔ x = 8 ∨ x = 0) →
  (∃ a b : ℝ, (a - 4)^2 = 16 ∧ (b - 4)^2 = 16 ∧ a + b = 8) :=
by sorry

end sum_of_roots_squared_equation_l3739_373963


namespace rotated_angle_measure_l3739_373942

/-- Given an initial angle of 30 degrees rotated 450 degrees clockwise,
    the resulting new acute angle measures 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 30 →
  rotation = 450 →
  let effective_rotation := rotation % 360
  let new_angle := (initial_angle - effective_rotation) % 360
  let acute_angle := min new_angle (360 - new_angle)
  acute_angle = 60 := by
  sorry

end rotated_angle_measure_l3739_373942


namespace tetrahedron_volume_in_cube_l3739_373912

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume : ℝ := cube_side_length ^ 3
  let clear_tetrahedron_volume : ℝ := (1 / 6) * cube_side_length ^ 3
  let colored_tetrahedron_volume : ℝ := cube_volume - 4 * clear_tetrahedron_volume
  colored_tetrahedron_volume = 172 := by
  sorry

end tetrahedron_volume_in_cube_l3739_373912


namespace simplify_and_rationalize_l3739_373987

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) = Real.sqrt 70 / 16 := by
  sorry

end simplify_and_rationalize_l3739_373987


namespace coaches_next_meeting_l3739_373917

theorem coaches_next_meeting (ella_schedule : Nat) (felix_schedule : Nat) (greta_schedule : Nat) (harry_schedule : Nat)
  (h_ella : ella_schedule = 5)
  (h_felix : felix_schedule = 9)
  (h_greta : greta_schedule = 8)
  (h_harry : harry_schedule = 11) :
  Nat.lcm (Nat.lcm (Nat.lcm ella_schedule felix_schedule) greta_schedule) harry_schedule = 3960 := by
sorry

end coaches_next_meeting_l3739_373917


namespace stratified_sampling_theorem_l3739_373964

/-- Represents the number of volunteers in each grade --/
structure GradeVolunteers where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the number of volunteers selected in the sample from each grade --/
structure SampleVolunteers where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the probability of selecting two volunteers from the same grade --/
def probability_same_grade (sample : SampleVolunteers) : ℚ :=
  let total_pairs := sample.first.choose 2 + sample.second.choose 2
  let all_pairs := (sample.first + sample.second).choose 2
  total_pairs / all_pairs

theorem stratified_sampling_theorem (volunteers : GradeVolunteers) (sample : SampleVolunteers) :
  volunteers.first = 36 →
  volunteers.second = 72 →
  volunteers.third = 54 →
  sample.third = 3 →
  sample.first = 2 ∧
  sample.second = 4 ∧
  probability_same_grade sample = 7/15 := by
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l3739_373964


namespace percentage_problem_l3739_373904

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 4800) = 108) → P = 0.3 := by
  sorry

end percentage_problem_l3739_373904


namespace predicted_holiday_shoppers_l3739_373978

theorem predicted_holiday_shoppers 
  (packages_per_box : ℕ) 
  (boxes_ordered : ℕ) 
  (shopper_ratio : ℕ) 
  (h1 : packages_per_box = 25)
  (h2 : boxes_ordered = 5)
  (h3 : shopper_ratio = 3) :
  boxes_ordered * packages_per_box * shopper_ratio = 375 :=
by sorry

end predicted_holiday_shoppers_l3739_373978


namespace correct_adult_ticket_cost_l3739_373994

/-- The cost of an adult ticket in dollars -/
def adult_ticket_cost : ℕ := 19

/-- The cost of a child ticket in dollars -/
def child_ticket_cost : ℕ := adult_ticket_cost - 6

/-- The number of adults in the family -/
def num_adults : ℕ := 2

/-- The number of children in the family -/
def num_children : ℕ := 3

/-- The total cost of tickets for the family -/
def total_cost : ℕ := 77

theorem correct_adult_ticket_cost :
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_cost :=
by sorry

end correct_adult_ticket_cost_l3739_373994


namespace base5_98_to_base9_l3739_373970

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base-9 --/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- Theorem: The base-9 representation of 98₍₅₎ is 58₍₉₎ --/
theorem base5_98_to_base9 :
  decimalToBase9 (base5ToDecimal [8, 9]) = [5, 8] :=
sorry

end base5_98_to_base9_l3739_373970


namespace divisibility_implies_equality_l3739_373907

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end divisibility_implies_equality_l3739_373907


namespace product_of_sum_and_cube_sum_l3739_373993

theorem product_of_sum_and_cube_sum (p q : ℝ) 
  (h1 : p + q = 10) 
  (h2 : p^3 + q^3 = 370) : 
  p * q = 21 := by
sorry

end product_of_sum_and_cube_sum_l3739_373993


namespace opposite_property_opposite_of_neg_two_l3739_373922

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_property (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -2 is 2 -/
theorem opposite_of_neg_two :
  opposite (-2 : ℝ) = 2 := by sorry

end opposite_property_opposite_of_neg_two_l3739_373922


namespace power_calculation_l3739_373900

theorem power_calculation : 3^2022 * (1/3)^2023 = 1/3 := by
  sorry

end power_calculation_l3739_373900


namespace polygon_diagonals_l3739_373911

theorem polygon_diagonals (n : ℕ) (h : (n - 2) * 180 + 360 = 2160) : 
  n * (n - 3) / 2 = 54 := by sorry

end polygon_diagonals_l3739_373911


namespace no_perfect_square_in_range_l3739_373976

theorem no_perfect_square_in_range : 
  ∀ m : ℤ, 4 ≤ m ∧ m ≤ 12 → ¬∃ k : ℤ, 2 * m^2 + 3 * m + 2 = k^2 := by
  sorry

#check no_perfect_square_in_range

end no_perfect_square_in_range_l3739_373976


namespace square_perimeter_l3739_373968

theorem square_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) 
  (h1 : rectangle_length = 125)
  (h2 : rectangle_width = 64)
  (h3 : ∃ square_side : ℝ, square_side^2 = 5 * rectangle_length * rectangle_width) :
  ∃ square_perimeter : ℝ, square_perimeter = 800 := by
sorry

end square_perimeter_l3739_373968


namespace not_sufficient_not_necessary_l3739_373991

-- Define sets A and B
def A : Set ℝ := {x | 1 / x ≥ 1}
def B : Set ℝ := {x | Real.log (1 - x) ≤ 0}

-- Theorem statement
theorem not_sufficient_not_necessary : 
  ¬(∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) := by sorry

end not_sufficient_not_necessary_l3739_373991


namespace rectangular_prism_dimensions_l3739_373927

theorem rectangular_prism_dimensions :
  ∀ (l b h : ℝ),
    l = 3 * b →
    l = 2 * h →
    l * b * h = 12168 →
    l = 42 ∧ b = 14 ∧ h = 21 := by
  sorry

end rectangular_prism_dimensions_l3739_373927


namespace distinct_roots_sum_l3739_373999

theorem distinct_roots_sum (a b c : ℂ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * (a - 6) = 7 →
  b * (b - 6) = 7 →
  c * (c - 6) = 7 →
  a + b + c = 5 := by
sorry

end distinct_roots_sum_l3739_373999


namespace al_mass_percentage_l3739_373943

theorem al_mass_percentage (mass_percentage : ℝ) (h : mass_percentage = 20.45) :
  mass_percentage = 20.45 := by
sorry

end al_mass_percentage_l3739_373943


namespace smaller_number_proof_l3739_373954

theorem smaller_number_proof (x y : ℝ) : 
  x + y = 70 ∧ y = 3 * x + 10 → x = 15 ∧ x ≤ y := by
  sorry

end smaller_number_proof_l3739_373954


namespace common_value_proof_l3739_373979

theorem common_value_proof (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : 40 * a * b = 1800) :
  4 * a = 60 ∧ 5 * b = 60 := by
sorry

end common_value_proof_l3739_373979


namespace solution_approximation_l3739_373913

def equation (x : ℝ) : Prop :=
  (0.66^3 - x^3) = 0.5599999999999999 * ((0.66^2) + 0.066 + x^2)

theorem solution_approximation : ∃ x : ℝ, equation x ∧ abs (x - 0.1) < 1e-6 := by
  sorry

end solution_approximation_l3739_373913


namespace triangle_inequality_squared_l3739_373983

/-- Triangle side condition -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem -/
theorem triangle_inequality_squared (a b c : ℝ) 
  (h : is_triangle a b c) : 
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 :=
sorry

end triangle_inequality_squared_l3739_373983


namespace boys_in_row_l3739_373930

theorem boys_in_row (left_position right_position between : ℕ) : 
  left_position = 6 →
  right_position = 10 →
  between = 8 →
  left_position - 1 + between + right_position = 24 :=
by sorry

end boys_in_row_l3739_373930


namespace intersection_line_through_origin_l3739_373938

/-- Given two lines in the plane, this theorem proves that a specific line
    passes through their intersection point and the origin. -/
theorem intersection_line_through_origin :
  let line1 : ℝ → ℝ → Prop := λ x y => 2023 * x - 2022 * y - 1 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 2022 * x + 2023 * y + 1 = 0
  let intersection_line : ℝ → ℝ → Prop := λ x y => 4045 * x + y = 0
  (∃ x y, line1 x y ∧ line2 x y) →  -- Assumption: The two lines intersect
  (∀ x y, line1 x y ∧ line2 x y → intersection_line x y) ∧  -- The line passes through the intersection
  intersection_line 0 0  -- The line passes through the origin
  := by sorry

end intersection_line_through_origin_l3739_373938


namespace normal_distribution_two_std_below_mean_l3739_373952

theorem normal_distribution_two_std_below_mean :
  let μ : ℝ := 16.2  -- mean
  let σ : ℝ := 2.3   -- standard deviation
  let x : ℝ := μ - 2 * σ  -- value 2 standard deviations below mean
  x = 11.6 := by
  sorry

end normal_distribution_two_std_below_mean_l3739_373952


namespace theater_pricing_l3739_373977

/-- The price of orchestra seats in dollars -/
def orchestra_price : ℝ := 12

/-- The total number of tickets sold -/
def total_tickets : ℕ := 380

/-- The total revenue in dollars -/
def total_revenue : ℝ := 3320

/-- The difference between balcony and orchestra tickets sold -/
def ticket_difference : ℕ := 240

/-- The price of balcony seats in dollars -/
def balcony_price : ℝ := 8

theorem theater_pricing :
  ∃ (orchestra_tickets : ℕ) (balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = total_tickets ∧
    balcony_tickets = orchestra_tickets + ticket_difference ∧
    orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_revenue :=
by sorry

end theater_pricing_l3739_373977


namespace square_circle_union_area_l3739_373949

/-- The area of the union of a square with side length 12 and a circle with radius 12
    centered at one of the square's vertices is equal to 144 + 108π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let quarter_circle_area : ℝ := circle_area / 4
  square_area + circle_area - quarter_circle_area = 144 + 108 * π := by
  sorry

end square_circle_union_area_l3739_373949


namespace zeros_in_concatenated_number_l3739_373972

/-- Counts the number of zeros in a given positive integer -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Counts the total number of zeros in all integers from 1 to n -/
def totalZeros (n : ℕ) : ℕ := sorry

/-- The concatenated number formed by all integers from 1 to 2007 -/
def concatenatedNumber : ℕ := sorry

theorem zeros_in_concatenated_number :
  countZeros concatenatedNumber = 506 := by sorry

end zeros_in_concatenated_number_l3739_373972


namespace integral_equals_six_implies_b_equals_e_to_four_l3739_373984

theorem integral_equals_six_implies_b_equals_e_to_four (b : ℝ) :
  (∫ (x : ℝ) in Set.Icc (Real.exp 1) b, 2 / x) = 6 →
  b = Real.exp 4 := by
  sorry

end integral_equals_six_implies_b_equals_e_to_four_l3739_373984


namespace triangle_decomposition_l3739_373980

theorem triangle_decomposition (a b c : ℝ) 
  (h1 : b + c > a) (h2 : a + c > b) (h3 : a + b > c) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    a = y + z ∧ b = x + z ∧ c = x + y :=
by sorry

end triangle_decomposition_l3739_373980


namespace double_quarter_four_percent_l3739_373921

theorem double_quarter_four_percent : 
  (4 / 100 / 4 * 2 : ℝ) = 0.02 := by sorry

end double_quarter_four_percent_l3739_373921


namespace sum_of_prime_factors_2550_l3739_373957

def sum_of_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_prime_factors_2550 : sum_of_prime_factors 2550 = 27 := by sorry

end sum_of_prime_factors_2550_l3739_373957


namespace tooth_fairy_total_l3739_373969

/-- The total number of baby teeth a child has. -/
def totalTeeth : ℕ := 20

/-- The number of teeth lost or swallowed. -/
def lostTeeth : ℕ := 2

/-- The amount received for the first tooth. -/
def firstToothAmount : ℕ := 20

/-- The amount received for each subsequent tooth. -/
def regularToothAmount : ℕ := 2

/-- The total amount received from the tooth fairy. -/
def totalAmount : ℕ := firstToothAmount + regularToothAmount * (totalTeeth - lostTeeth - 1)

theorem tooth_fairy_total : totalAmount = 54 := by
  sorry

end tooth_fairy_total_l3739_373969


namespace f_decreasing_iff_a_in_range_l3739_373955

/-- The function f(x) defined as 2ax² + 4(a-3)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 + 4*(a-3)*x + 5

/-- The property of f(x) being decreasing on the interval (-∞, 3) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 3 → y < 3 → f a x > f a y

/-- The theorem stating the range of a for which f(x) is decreasing on (-∞, 3) -/
theorem f_decreasing_iff_a_in_range :
  ∀ a, is_decreasing_on_interval a ↔ a ∈ Set.Icc 0 (3/4) :=
sorry

end f_decreasing_iff_a_in_range_l3739_373955


namespace cara_seating_arrangements_l3739_373906

theorem cara_seating_arrangements (n : ℕ) (h : n = 8) :
  (n - 2 : ℕ) = 6 := by
  sorry

end cara_seating_arrangements_l3739_373906


namespace compound_interest_rate_l3739_373951

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r)^2 = 2420)
  (h2 : P * (1 + r)^3 = 3025) : 
  r = 0.25 := by
  sorry

end compound_interest_rate_l3739_373951


namespace a_4k_plus_2_div_3_l3739_373941

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => a (n + 2) + a (n + 1)

theorem a_4k_plus_2_div_3 (k : ℕ) : ∃ m : ℕ, a (4 * k + 2) = 3 * m := by
  sorry

end a_4k_plus_2_div_3_l3739_373941


namespace worker_completion_time_proof_l3739_373990

/-- Represents a worker with their working days and payment --/
structure Worker where
  days : ℕ
  payment : ℕ

/-- Calculates the time it would take a worker to complete the entire job --/
def timeToCompleteJob (w : Worker) (totalPayment : ℕ) : ℕ :=
  w.days * (totalPayment / w.payment)

theorem worker_completion_time_proof (w1 w2 w3 : Worker) 
  (h1 : w1.days = 6 ∧ w1.payment = 36)
  (h2 : w2.days = 3 ∧ w2.payment = 12)
  (h3 : w3.days = 8 ∧ w3.payment = 24) :
  let totalPayment := w1.payment + w2.payment + w3.payment
  (timeToCompleteJob w1 totalPayment = 12) ∧
  (timeToCompleteJob w2 totalPayment = 18) ∧
  (timeToCompleteJob w3 totalPayment = 24) := by
  sorry

#check worker_completion_time_proof

end worker_completion_time_proof_l3739_373990


namespace smallest_factor_for_perfect_cube_l3739_373945

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_factor_for_perfect_cube (x y : ℕ) : 
  x = 5 * 30 * 60 →
  y > 0 →
  is_perfect_cube (x * y) →
  (∀ z : ℕ, z > 0 → z < y → ¬ is_perfect_cube (x * z)) →
  y = 3 := by
  sorry

end smallest_factor_for_perfect_cube_l3739_373945


namespace sam_exchange_probability_l3739_373909

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 10

/-- The price of the first toy in cents -/
def first_toy_price : ℕ := 50

/-- The price increment between consecutive toys in cents -/
def price_increment : ℕ := 25

/-- The number of quarters Sam has -/
def sam_quarters : ℕ := 10

/-- The price of Sam's favorite toy in cents -/
def favorite_toy_price : ℕ := 225

/-- The total number of possible toy arrangements -/
def total_arrangements : ℕ := Nat.factorial num_toys

/-- The number of favorable arrangements where Sam can buy his favorite toy without exchanging his bill -/
def favorable_arrangements : ℕ := Nat.factorial 9 + Nat.factorial 8 + Nat.factorial 7 + Nat.factorial 6 + Nat.factorial 5

/-- The probability that Sam needs to exchange his bill -/
def exchange_probability : ℚ := 1 - (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

theorem sam_exchange_probability :
  exchange_probability = 8 / 9 := by sorry

end sam_exchange_probability_l3739_373909


namespace smallest_label_on_final_position_l3739_373935

/-- The number of points on the circle -/
def n : ℕ := 70

/-- The function that calculates the position of a label -/
def position (k : ℕ) : ℕ := (k * (k + 1) / 2) % n

/-- The final label we're interested in -/
def final_label : ℕ := 2014

/-- The smallest label we claim to be on the same point as the final label -/
def smallest_label : ℕ := 5

theorem smallest_label_on_final_position :
  position final_label = position smallest_label ∧
  ∀ m : ℕ, m < smallest_label → position final_label ≠ position m :=
sorry

end smallest_label_on_final_position_l3739_373935


namespace sin_two_x_value_l3739_373929

theorem sin_two_x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 3) : 
  Real.sin (2 * x) = 7 / 9 := by
sorry

end sin_two_x_value_l3739_373929


namespace inequality_solution_set_l3739_373982

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is increasing on a set S if f(x) ≤ f(y) for all x, y in S with x ≤ y -/
def IncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_odd : OddFunction f)
  (h_incr : IncreasingOn f (Set.Iic 0))
  (h_f2 : f 2 = 4) :
  {x : ℝ | 4 + f (x^2 - x) > 0} = Set.univ :=
by sorry

end inequality_solution_set_l3739_373982


namespace alpha_value_l3739_373910

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (2 * (α - 3 * β)).re > 0)
  (h3 : β = 5 + 4 * Complex.I) :
  α = 16 - 4 * Complex.I := by
sorry

end alpha_value_l3739_373910


namespace inscribed_circle_radius_l3739_373901

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 18) :
  let r := (1 / a + 1 / b + 1 / c + 4 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 9 / (5 + 6 * Real.sqrt 3) := by sorry

end inscribed_circle_radius_l3739_373901


namespace f_of_10_l3739_373961

/-- Given a function f(x) = 2x^2 + y where f(2) = 30, prove that f(10) = 222 -/
theorem f_of_10 (f : ℝ → ℝ) (y : ℝ) 
    (h1 : ∀ x, f x = 2 * x^2 + y) 
    (h2 : f 2 = 30) : 
  f 10 = 222 := by
sorry

end f_of_10_l3739_373961


namespace hash_equals_100_l3739_373915

def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_equals_100 (a b : ℕ) (h : a + b + 6 = 11) : hash a b = 100 := by
  sorry

end hash_equals_100_l3739_373915


namespace cube_face_diagonal_edge_angle_l3739_373914

/-- Represents a cube in 3D space -/
structure Cube where
  -- Define necessary properties of a cube

/-- Represents a line segment in 3D space -/
structure LineSegment where
  -- Define necessary properties of a line segment

/-- Represents an angle between two line segments -/
def angle (l1 l2 : LineSegment) : ℝ := sorry

/-- Predicate to check if a line segment is an edge of the cube -/
def is_edge (c : Cube) (l : LineSegment) : Prop := sorry

/-- Predicate to check if a line segment is a face diagonal of the cube -/
def is_face_diagonal (c : Cube) (l : LineSegment) : Prop := sorry

/-- Predicate to check if two line segments are incident to the same vertex -/
def incident_to_same_vertex (l1 l2 : LineSegment) : Prop := sorry

/-- Theorem: In a cube, the angle between a face diagonal and an edge 
    incident to the same vertex is 60 degrees -/
theorem cube_face_diagonal_edge_angle (c : Cube) (d e : LineSegment) :
  is_face_diagonal c d → is_edge c e → incident_to_same_vertex d e →
  angle d e = 60 := by sorry

end cube_face_diagonal_edge_angle_l3739_373914


namespace londolozi_lions_growth_l3739_373996

/-- The number of lion cubs born per month in Londolozi -/
def cubs_per_month : ℕ := sorry

/-- The initial number of lions in Londolozi -/
def initial_lions : ℕ := 100

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of lions that die per month -/
def lions_die_per_month : ℕ := 1

/-- The number of lions after one year -/
def lions_after_year : ℕ := 148

theorem londolozi_lions_growth :
  cubs_per_month * months_in_year - lions_die_per_month * months_in_year + initial_lions = lions_after_year ∧
  cubs_per_month = 5 := by sorry

end londolozi_lions_growth_l3739_373996


namespace balloon_count_l3739_373916

/-- The number of violet balloons Dan has -/
def dans_balloons : ℕ := 29

/-- The number of times more balloons Tim has compared to Dan -/
def tims_multiplier : ℕ := 7

/-- The number of times more balloons Molly has compared to Dan -/
def mollys_multiplier : ℕ := 5

/-- The total number of violet balloons Dan, Tim, and Molly have -/
def total_balloons : ℕ := dans_balloons + tims_multiplier * dans_balloons + mollys_multiplier * dans_balloons

theorem balloon_count : total_balloons = 377 := by sorry

end balloon_count_l3739_373916


namespace percentage_relationship_l3739_373989

theorem percentage_relationship (a b : ℝ) (h : a = 1.5 * b) :
  3 * b = 2 * a := by sorry

end percentage_relationship_l3739_373989


namespace last_term_is_1344_l3739_373902

/-- Defines the nth term of the sequence -/
def sequenceTerm (n : ℕ) : ℕ :=
  if n % 3 = 1 then (n + 2) / 3 else (n + 1) / 3

/-- The last term of the sequence with 2015 elements -/
def lastTerm : ℕ := sequenceTerm 2015

theorem last_term_is_1344 : lastTerm = 1344 := by
  sorry

end last_term_is_1344_l3739_373902


namespace moving_circle_trajectory_l3739_373939

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def externally_tangent (M : MovingCircle) : Prop :=
  C₁ (M.center.1 + M.radius) M.center.2

def internally_tangent (M : MovingCircle) : Prop :=
  C₂ (M.center.1 - M.radius) M.center.2

-- State the theorem
theorem moving_circle_trajectory
  (M : MovingCircle)
  (h1 : externally_tangent M)
  (h2 : internally_tangent M) :
  ∃ x y : ℝ, x > 0 ∧ x^2 / 16 - y^2 / 9 = 1 ∧ M.center = (x, y) :=
sorry

end moving_circle_trajectory_l3739_373939


namespace four_points_reciprocal_sum_l3739_373937

theorem four_points_reciprocal_sum (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 
    1 / |x - a| + 1 / |x - b| + 1 / |x - c| + 1 / |x - d| ≤ 40 := by
  sorry

end four_points_reciprocal_sum_l3739_373937


namespace randy_biscuits_l3739_373947

/-- The number of biscuits Randy has after receiving and losing some -/
def final_biscuits (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (eaten_by_brother : ℕ) : ℕ :=
  initial + from_father + from_mother - eaten_by_brother

/-- Theorem stating that Randy ends up with 40 biscuits -/
theorem randy_biscuits : final_biscuits 32 13 15 20 = 40 := by
  sorry

end randy_biscuits_l3739_373947


namespace x_bijective_l3739_373960

def x : ℕ → ℤ
  | 0 => 0
  | n + 1 => 
    let r := (n + 1).log 3 + 1
    let k := (n + 1) / (3^(r-1)) - 1
    if (n + 1) = 3^(r-1) * (3*k + 1) then
      x n + (3^r - 1) / 2
    else if (n + 1) = 3^(r-1) * (3*k + 2) then
      x n - (3^r + 1) / 2
    else
      x n

theorem x_bijective : Function.Bijective x := by sorry

end x_bijective_l3739_373960


namespace three_digit_distinct_sum_remainder_l3739_373959

def S : ℕ := sorry

theorem three_digit_distinct_sum_remainder : S % 1000 = 680 := by sorry

end three_digit_distinct_sum_remainder_l3739_373959


namespace raffle_ticket_sales_l3739_373933

theorem raffle_ticket_sales (total_money : ℕ) (ticket_cost : ℕ) (num_tickets : ℕ) :
  total_money = 620 →
  ticket_cost = 4 →
  total_money = ticket_cost * num_tickets →
  num_tickets = 155 := by
sorry

end raffle_ticket_sales_l3739_373933


namespace equal_roots_quadratic_l3739_373995

theorem equal_roots_quadratic (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - a*y + 1 = 0 → y = x)) →
  a = 2 ∨ a = -2 := by
sorry

end equal_roots_quadratic_l3739_373995


namespace bees_12_feet_apart_l3739_373981

/-- Represents the position of a bee in 3D space -/
structure Position where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the movement cycle of a bee -/
structure MovementCycle where
  steps : List Position

/-- Calculates the position of a bee after a given number of steps -/
def beePosition (start : Position) (cycle : MovementCycle) (steps : ℕ) : Position :=
  sorry

/-- Calculates the distance between two positions -/
def distance (p1 p2 : Position) : ℝ :=
  sorry

/-- Determines the direction of movement for a bee given its current and next position -/
def movementDirection (current next : Position) : String :=
  sorry

/-- The theorem to be proved -/
theorem bees_12_feet_apart :
  ∀ (steps : ℕ),
  let start := Position.mk 0 0 0
  let cycleA := MovementCycle.mk [Position.mk 2 0 0, Position.mk 0 2 0]
  let cycleB := MovementCycle.mk [Position.mk 0 (-2) 1, Position.mk (-1) 0 0]
  let posA := beePosition start cycleA steps
  let posB := beePosition start cycleB steps
  let nextA := beePosition start cycleA (steps + 1)
  let nextB := beePosition start cycleB (steps + 1)
  distance posA posB = 12 →
  (∀ (s : ℕ), s < steps → distance (beePosition start cycleA s) (beePosition start cycleB s) < 12) →
  movementDirection posA nextA = "east" ∧ movementDirection posB nextB = "upwards" :=
sorry

end bees_12_feet_apart_l3739_373981


namespace price_decrease_unit_increase_ratio_l3739_373920

theorem price_decrease_unit_increase_ratio (P U V : ℝ) 
  (h1 : P > 0) 
  (h2 : U > 0) 
  (h3 : V > U) 
  (h4 : P * U = 0.25 * P * V) : 
  ((V - U) / U) / 0.75 = 4 := by
sorry

end price_decrease_unit_increase_ratio_l3739_373920


namespace favorite_movies_sum_l3739_373965

/-- Given the movie lengths of Joyce, Michael, Nikki, and Ryn, prove their sum is 76 hours -/
theorem favorite_movies_sum (michael nikki joyce ryn : ℝ) : 
  nikki = 30 ∧ 
  michael = nikki / 3 ∧ 
  joyce = michael + 2 ∧ 
  ryn = 4 / 5 * nikki → 
  joyce + michael + nikki + ryn = 76 := by
  sorry

end favorite_movies_sum_l3739_373965


namespace expression_evaluation_l3739_373966

theorem expression_evaluation :
  let a : ℤ := -3
  let b : ℤ := -2
  (3 * a^2 * b + 2 * a * b^2) - (2 * (a^2 * b - 1) + 3 * a * b^2 + 2) = -6 := by
  sorry

end expression_evaluation_l3739_373966


namespace arithmetic_sequence_sum_property_l3739_373908

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (d : ℝ) (a : ℕ → ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The property that the sum of any two distinct terms is a term in the sequence -/
def SumPropertyHolds (a : ℕ → ℝ) : Prop :=
  ∀ s t, s ≠ t → ∃ k, a s + a t = a k

/-- The theorem stating the equivalence of the sum property and the existence of m -/
theorem arithmetic_sequence_sum_property (d : ℝ) (a : ℕ → ℝ) :
  ArithmeticSequence d a →
  (SumPropertyHolds a ↔ ∃ m : ℤ, m ≥ -1 ∧ a 1 = m * d) := by
  sorry

end arithmetic_sequence_sum_property_l3739_373908


namespace min_value_theorem_l3739_373950

theorem min_value_theorem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) :
  (2/x + 3/y) ≥ 8 + 4*Real.sqrt 3 := by
sorry

end min_value_theorem_l3739_373950


namespace percent_calculation_l3739_373962

theorem percent_calculation (x : ℝ) (h : 0.2 * x = 60) : 0.8 * x = 240 := by
  sorry

end percent_calculation_l3739_373962


namespace jung_min_wire_purchase_l3739_373973

/-- The length of wire needed to make a regular pentagon with given side length -/
def pentagonWireLength (sideLength : ℝ) : ℝ := 5 * sideLength

/-- The total length of wire bought given the side length of the pentagon and the leftover wire -/
def totalWireBought (sideLength leftover : ℝ) : ℝ := pentagonWireLength sideLength + leftover

theorem jung_min_wire_purchase :
  totalWireBought 13 8 = 73 := by
  sorry

end jung_min_wire_purchase_l3739_373973


namespace distance_between_points_l3739_373919

def point_A : ℝ × ℝ := (2, 3)
def point_B : ℝ × ℝ := (5, 10)

theorem distance_between_points :
  Real.sqrt ((point_B.1 - point_A.1)^2 + (point_B.2 - point_A.2)^2) = Real.sqrt 58 := by
  sorry

end distance_between_points_l3739_373919


namespace stratified_sample_probability_l3739_373932

/-- Represents the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ := sorry

theorem stratified_sample_probability : 
  let total_sample_size := 6
  let elementary_teachers_in_sample := 3
  let further_selection_size := 2
  probability (choose elementary_teachers_in_sample further_selection_size) 
              (choose total_sample_size further_selection_size) = 1/5 := by
  sorry

end stratified_sample_probability_l3739_373932


namespace line_through_point_with_equal_intercepts_l3739_373967

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    pointOnLine ⟨1, 2⟩ l1 ∧
    pointOnLine ⟨1, 2⟩ l2 ∧
    equalIntercepts l1 ∧
    equalIntercepts l2 ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
by sorry

end line_through_point_with_equal_intercepts_l3739_373967
