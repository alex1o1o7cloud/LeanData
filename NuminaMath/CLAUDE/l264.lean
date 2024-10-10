import Mathlib

namespace divisibility_by_eight_and_nine_l264_26468

theorem divisibility_by_eight_and_nine (x y : Nat) : 
  x < 10 ∧ y < 10 →
  (1234 * 10 * x + 1234 * y) % 8 = 0 ∧ 
  (1234 * 10 * x + 1234 * y) % 9 = 0 ↔ 
  (x = 8 ∧ y = 0) ∨ (x = 0 ∧ y = 8) := by
sorry

end divisibility_by_eight_and_nine_l264_26468


namespace hyperbola_equation_with_eccentricity_hyperbola_equation_with_asymptote_l264_26447

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  right_focus : ℝ × ℝ

-- Define the standard form of a hyperbola equation
def standard_form (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 / a^2 - y^2 / b^2 = 1

-- Theorem for the first part of the problem
theorem hyperbola_equation_with_eccentricity 
  (C : Hyperbola) 
  (h_center : C.center = (0, 0))
  (h_focus : C.right_focus = (Real.sqrt 3, 0))
  (h_eccentricity : ∃ e, e = Real.sqrt 3) :
  ∃ a b, standard_form a b = λ x y => x^2 - y^2 / 2 = 1 :=
sorry

-- Theorem for the second part of the problem
theorem hyperbola_equation_with_asymptote
  (C : Hyperbola)
  (h_center : C.center = (0, 0))
  (h_focus : C.right_focus = (Real.sqrt 3, 0))
  (h_asymptote : ∃ X Y, X + Real.sqrt 2 * Y = 0) :
  ∃ a b, standard_form a b = λ x y => x^2 / 2 - y^2 = 1 :=
sorry

end hyperbola_equation_with_eccentricity_hyperbola_equation_with_asymptote_l264_26447


namespace dollar_three_neg_one_l264_26409

-- Define the $ operation
def dollar (a b : ℤ) : ℤ := a * (b + 2) + a * (b + 1)

-- Theorem to prove
theorem dollar_three_neg_one : dollar 3 (-1) = 3 := by
  sorry

end dollar_three_neg_one_l264_26409


namespace triangle_angle_max_value_l264_26448

theorem triangle_angle_max_value (A B C : ℝ) : 
  A + B + C = π →
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (Real.sqrt 3 * Real.cos A + Real.sin A) / (Real.sqrt 3 * Real.sin A - Real.cos A) = Real.tan (-7 * π / 12) →
  ∃ (x : ℝ), x = 2 * Real.cos B + Real.sin (2 * C) ∧ x ≤ 3 / 2 ∧ 
  ∀ (y : ℝ), y = 2 * Real.cos B + Real.sin (2 * C) → y ≤ x :=
by sorry

end triangle_angle_max_value_l264_26448


namespace deck_width_l264_26424

/-- Given a rectangular pool of dimensions 10 feet by 12 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 360 square feet, then the width of the deck is 4 feet. -/
theorem deck_width (w : ℝ) : 
  (10 + 2*w) * (12 + 2*w) = 360 → w = 4 := by sorry

end deck_width_l264_26424


namespace arrangement_speeches_not_adjacent_l264_26427

theorem arrangement_speeches_not_adjacent (n : ℕ) (m : ℕ) :
  n = 5 ∧ m = 3 →
  (n.factorial * (n + 1).factorial / ((n + 1 - m).factorial)) = 14400 :=
sorry

end arrangement_speeches_not_adjacent_l264_26427


namespace brendans_tax_payment_is_correct_l264_26403

/-- Calculates Brendan's weekly tax payment based on his work schedule and income reporting --/
def brendans_weekly_tax_payment (
  waiter_hourly_wage : ℚ)
  (barista_hourly_wage : ℚ)
  (waiter_shift_hours : List ℚ)
  (barista_shift_hours : List ℚ)
  (waiter_hourly_tips : ℚ)
  (barista_hourly_tips : ℚ)
  (waiter_tax_rate : ℚ)
  (barista_tax_rate : ℚ)
  (waiter_reported_tips_ratio : ℚ)
  (barista_reported_tips_ratio : ℚ) : ℚ :=
  let waiter_total_hours := waiter_shift_hours.sum
  let barista_total_hours := barista_shift_hours.sum
  let waiter_wage_income := waiter_total_hours * waiter_hourly_wage
  let barista_wage_income := barista_total_hours * barista_hourly_wage
  let waiter_total_tips := waiter_total_hours * waiter_hourly_tips
  let barista_total_tips := barista_total_hours * barista_hourly_tips
  let waiter_reported_tips := waiter_total_tips * waiter_reported_tips_ratio
  let barista_reported_tips := barista_total_tips * barista_reported_tips_ratio
  let waiter_reported_income := waiter_wage_income + waiter_reported_tips
  let barista_reported_income := barista_wage_income + barista_reported_tips
  let waiter_tax := waiter_reported_income * waiter_tax_rate
  let barista_tax := barista_reported_income * barista_tax_rate
  waiter_tax + barista_tax

theorem brendans_tax_payment_is_correct :
  brendans_weekly_tax_payment 6 8 [8, 8, 12] [6] 12 5 (1/5) (1/4) (1/3) (1/2) = 71.75 := by
  sorry

end brendans_tax_payment_is_correct_l264_26403


namespace brads_running_speed_l264_26456

/-- Prove that Brad's running speed is 6 km/h given the conditions of the problem -/
theorem brads_running_speed (maxwell_speed : ℝ) (total_distance : ℝ) (brad_delay : ℝ) (total_time : ℝ)
  (h1 : maxwell_speed = 4)
  (h2 : total_distance = 74)
  (h3 : brad_delay = 1)
  (h4 : total_time = 8) :
  (total_distance - maxwell_speed * total_time) / (total_time - brad_delay) = 6 := by
  sorry

end brads_running_speed_l264_26456


namespace fifth_score_proof_l264_26495

theorem fifth_score_proof (s1 s2 s3 s4 s5 : ℕ) : 
  s1 = 90 → s2 = 93 → s3 = 85 → s4 = 97 → 
  (s1 + s2 + s3 + s4 + s5) / 5 = 92 → 
  s5 = 95 := by
sorry

end fifth_score_proof_l264_26495


namespace arithmetic_sequence_n_equals_5_l264_26453

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_pos : ∀ n, a n > 0
  a_1 : a 1 = 3
  S_3 : (a 1) + (a 2) + (a 3) = 21
  a_n : ∃ n, a n = 48

/-- The theorem stating that n = 5 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_5 (seq : ArithmeticSequence) :
  ∃ n, seq.a n = 48 ∧ n = 5 := by
  sorry

end arithmetic_sequence_n_equals_5_l264_26453


namespace bakers_cake_inventory_l264_26464

/-- Baker's cake inventory problem -/
theorem bakers_cake_inventory (cakes_made cakes_bought cakes_sold : ℕ) :
  cakes_made = 8 →
  cakes_bought = 139 →
  cakes_sold = 145 →
  cakes_sold - cakes_bought = 6 := by
  sorry

end bakers_cake_inventory_l264_26464


namespace combined_boys_avg_is_correct_l264_26496

/-- Represents a high school with exam scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools -/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_girls_avg : ℝ

/-- Calculates the combined average score for boys given two schools' data -/
def combined_boys_avg (schools : CombinedSchools) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that the combined boys' average is approximately 48.57 -/
theorem combined_boys_avg_is_correct (schools : CombinedSchools) 
  (h1 : schools.school1 = ⟨68, 72, 70⟩)
  (h2 : schools.school2 = ⟨74, 88, 82⟩)
  (h3 : schools.combined_girls_avg = 83) :
  abs (combined_boys_avg schools - 48.57) < 0.01 := by
  sorry

end combined_boys_avg_is_correct_l264_26496


namespace debbie_tape_usage_l264_26460

/-- The amount of tape needed to pack boxes of different sizes --/
def total_tape_used (large_boxes medium_boxes small_boxes : ℕ) : ℕ :=
  let large_tape := 5 * large_boxes  -- 4 feet for sealing + 1 foot for label
  let medium_tape := 3 * medium_boxes  -- 2 feet for sealing + 1 foot for label
  let small_tape := 2 * small_boxes  -- 1 foot for sealing + 1 foot for label
  large_tape + medium_tape + small_tape

/-- Theorem stating that Debbie used 44 feet of tape --/
theorem debbie_tape_usage : total_tape_used 2 8 5 = 44 := by
  sorry

end debbie_tape_usage_l264_26460


namespace bee_hive_population_l264_26450

/-- The population growth function for bees in a hive -/
def bee_population (initial : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial * growth_factor ^ days

/-- Theorem stating the population of bees after 20 days -/
theorem bee_hive_population :
  bee_population 1 5 20 = 5^20 := by
  sorry

end bee_hive_population_l264_26450


namespace triangle_ABC_properties_l264_26402

open Real

theorem triangle_ABC_properties (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  2 * sin A * sin C * (1 / (tan A * tan C) - 1) = -1 ∧
  a + c = 3 * sqrt 3 / 2 ∧
  b = sqrt 3 →
  B = π / 3 ∧ 
  (1 / 2) * a * c * sin B = 5 * sqrt 3 / 16 :=
by sorry

end triangle_ABC_properties_l264_26402


namespace pyramid_edges_cannot_form_closed_polygon_l264_26423

/-- Represents a line segment in 3D space -/
structure Segment3D where
  parallel_to_plane : Bool

/-- Represents a collection of line segments in 3D space -/
structure SegmentCollection where
  segments : List Segment3D
  parallel_count : Nat
  non_parallel_count : Nat

/-- Checks if a collection of segments can form a closed polygon -/
def can_form_closed_polygon (collection : SegmentCollection) : Prop :=
  collection.parallel_count = collection.non_parallel_count ∧
  collection.parallel_count + collection.non_parallel_count = collection.segments.length

theorem pyramid_edges_cannot_form_closed_polygon :
  ¬ ∃ (collection : SegmentCollection),
    collection.parallel_count = 171 ∧
    collection.non_parallel_count = 171 ∧
    can_form_closed_polygon collection :=
by sorry

end pyramid_edges_cannot_form_closed_polygon_l264_26423


namespace rectangle_width_equal_square_side_l264_26406

theorem rectangle_width_equal_square_side 
  (square_side : ℝ) 
  (rect_length : ℝ) 
  (h1 : square_side = 3)
  (h2 : rect_length = 3)
  (h3 : square_side * square_side = rect_length * (square_side)) :
  square_side = rect_length :=
by sorry

end rectangle_width_equal_square_side_l264_26406


namespace lineup_organization_l264_26428

/-- The number of ways to organize a football lineup -/
def organize_lineup (total_members : ℕ) (defensive_linemen : ℕ) : ℕ :=
  defensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3)

/-- Theorem: The number of ways to organize a lineup for a team with 7 members,
    of which 4 can play defensive lineman, is 480 -/
theorem lineup_organization :
  organize_lineup 7 4 = 480 := by
  sorry

end lineup_organization_l264_26428


namespace inequalities_hold_l264_26474

theorem inequalities_hold (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  (a^2 * b < b^2 * c) ∧ (a^2 * c < b^2 * c) ∧ (a^2 * b < a^2 * c) := by
  sorry

end inequalities_hold_l264_26474


namespace abs_diff_opposite_l264_26467

theorem abs_diff_opposite (x : ℝ) (h : x < 0) : |x - (-x)| = -2*x := by
  sorry

end abs_diff_opposite_l264_26467


namespace andrews_age_l264_26417

theorem andrews_age :
  ∀ (a g : ℚ),
  g = 15 * a →
  g - a = 55 →
  a = 55 / 14 := by
sorry

end andrews_age_l264_26417


namespace prime_condition_characterization_l264_26408

def is_prime_for_all (a : ℕ+) : Prop :=
  ∀ n : ℕ, n < a → Nat.Prime (4 * n^2 + a)

theorem prime_condition_characterization :
  ∀ a : ℕ+, is_prime_for_all a ↔ (a = 3 ∨ a = 7) :=
sorry

end prime_condition_characterization_l264_26408


namespace square_and_cube_sum_l264_26412

theorem square_and_cube_sum (p q : ℝ) (h1 : p * q = 8) (h2 : p + q = 7) :
  p^2 + q^2 = 33 ∧ p^3 + q^3 = 175 := by
  sorry

end square_and_cube_sum_l264_26412


namespace beach_trip_time_difference_l264_26482

theorem beach_trip_time_difference (bus_time car_round_trip : ℕ) : 
  bus_time = 40 → car_round_trip = 70 → bus_time - car_round_trip / 2 = 5 := by
  sorry

end beach_trip_time_difference_l264_26482


namespace integral_x_squared_plus_sin_x_l264_26473

theorem integral_x_squared_plus_sin_x : ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by sorry

end integral_x_squared_plus_sin_x_l264_26473


namespace geometric_sequence_a6_l264_26476

/-- A geometric sequence with a_2 = 2 and a_4 = 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 2 = 2 ∧ a 4 = 4

/-- In a geometric sequence with a_2 = 2 and a_4 = 4, a_6 = 8 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h : geometric_sequence a) : a 6 = 8 := by
  sorry

end geometric_sequence_a6_l264_26476


namespace range_of_a_l264_26480

def p (x : ℝ) : Prop := x^2 - 5*x - 6 ≤ 0

def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - 4*a^2 ≤ 0

theorem range_of_a :
  (∀ x a : ℝ, a ≥ 0 →
    (∀ x, ¬(p x) → ¬(q x a)) ∧
    (∃ x, ¬(p x) ∧ q x a)) →
  {a : ℝ | a ≥ 5/2} = {a : ℝ | ∀ x, ¬(p x) → ¬(q x a)} := by sorry

end range_of_a_l264_26480


namespace alice_bushes_l264_26489

/-- The number of bushes needed to cover three sides of a yard -/
def bushes_needed (side_length : ℕ) (sides : ℕ) (bush_coverage : ℕ) : ℕ :=
  (side_length * sides) / bush_coverage

/-- Theorem: Alice needs 12 bushes for her yard -/
theorem alice_bushes :
  bushes_needed 16 3 4 = 12 := by
  sorry

end alice_bushes_l264_26489


namespace rectangle_area_properties_l264_26442

-- Define the rectangle's dimensions and measurement errors
def expected_length : Real := 2
def expected_width : Real := 1
def length_std_dev : Real := 0.003
def width_std_dev : Real := 0.002

-- Define the theorem
theorem rectangle_area_properties :
  let expected_area := expected_length * expected_width
  let area_variance := (expected_length^2 * width_std_dev^2) + (expected_width^2 * length_std_dev^2) + (length_std_dev^2 * width_std_dev^2)
  let area_std_dev := Real.sqrt area_variance
  (expected_area = 2) ∧ (area_std_dev * 100 = 5) := by
  sorry

end rectangle_area_properties_l264_26442


namespace evaluate_expression_l264_26404

theorem evaluate_expression (S : ℝ) : 
  S = 1 / (4 - Real.sqrt 10) - 1 / (Real.sqrt 10 - Real.sqrt 9) + 
      1 / (Real.sqrt 9 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
      1 / (Real.sqrt 7 - 3) → 
  S = 7 := by
sorry

end evaluate_expression_l264_26404


namespace factorization_equality_l264_26407

theorem factorization_equality (x : ℝ) : (x^2 - 1)^2 - 6*(x^2 - 1) + 9 = (x - 2)^2 * (x + 2)^2 := by
  sorry

end factorization_equality_l264_26407


namespace regular_polygon_interior_angle_sum_l264_26438

/-- A regular polygon with interior angles of 144° has a sum of interior angles equal to 1440°. -/
theorem regular_polygon_interior_angle_sum (n : ℕ) (h : n ≥ 3) :
  let interior_angle : ℝ := 144
  n * interior_angle = (n - 2) * 180 ∧ n * interior_angle = 1440 :=
by sorry

end regular_polygon_interior_angle_sum_l264_26438


namespace work_completion_time_l264_26462

/-- Represents the time it takes for a worker to complete a task alone -/
structure WorkTime where
  days : ℝ
  work_rate : ℝ
  inv_days_eq_work_rate : work_rate = 1 / days

/-- Represents a work scenario with two workers -/
structure WorkScenario where
  x : WorkTime
  y : WorkTime
  total_days : ℝ
  x_solo_days : ℝ
  both_days : ℝ
  total_days_eq_sum : total_days = x_solo_days + both_days

/-- The theorem to be proved -/
theorem work_completion_time (w : WorkScenario) (h1 : w.y.days = 12) 
  (h2 : w.x_solo_days = 4) (h3 : w.total_days = 10) : w.x.days = 20 := by
  sorry


end work_completion_time_l264_26462


namespace max_value_of_expression_l264_26497

theorem max_value_of_expression (a b : ℕ+) (ha : a < 6) (hb : b < 10) :
  (∀ x y : ℕ+, x < 6 → y < 10 → 2 * x - x * y ≤ 2 * a - a * b) →
  2 * a - a * b = 5 :=
sorry

end max_value_of_expression_l264_26497


namespace max_non_managers_l264_26430

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 36 :=
by sorry

end max_non_managers_l264_26430


namespace completing_square_equivalence_l264_26433

theorem completing_square_equivalence (x : ℝ) : 
  x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 := by sorry

end completing_square_equivalence_l264_26433


namespace quadratic_function_properties_l264_26477

-- Define the "graph number" type
def GraphNumber := ℝ × ℝ × ℝ

-- Define a function to get the graph number of a quadratic function
def getGraphNumber (a b c : ℝ) : GraphNumber :=
  (a, b, c)

-- Define a predicate for when a quadratic function intersects x-axis at one point
def intersectsXAxisOnce (a b c : ℝ) : Prop :=
  b ^ 2 - 4 * a * c = 0

theorem quadratic_function_properties :
  -- Part 1
  getGraphNumber (1/3) (-1) (-1) = (1/3, -1, -1) ∧
  -- Part 2
  ∀ m : ℝ, intersectsXAxisOnce m (m+1) (m+1) → (m = -1 ∨ m = 1/3) :=
by sorry

end quadratic_function_properties_l264_26477


namespace stability_comparison_A_more_stable_than_B_l264_26485

structure Student where
  name : String
  variance : ℝ

def more_stable (a b : Student) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : Student) 
  (h_mean : a.variance ≠ b.variance) :
  more_stable a b ∨ more_stable b a := by
  sorry

-- Define the specific students from the problem
def student_A : Student := ⟨"A", 1.4⟩
def student_B : Student := ⟨"B", 2.5⟩

-- Theorem for the specific case in the problem
theorem A_more_stable_than_B : 
  more_stable student_A student_B := by
  sorry

end stability_comparison_A_more_stable_than_B_l264_26485


namespace expand_product_l264_26475

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l264_26475


namespace quadratic_equation_factor_l264_26465

theorem quadratic_equation_factor (a : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + a = 0 ↔ 2 * (x - 2)^2 = 4) → a = 4 := by
  sorry

end quadratic_equation_factor_l264_26465


namespace system_solution_l264_26441

theorem system_solution (x y z : ℝ) 
  (eq1 : x + 3*y = 20)
  (eq2 : x + y + z = 25)
  (eq3 : x - z = 5) :
  x = 14 ∧ y = 2 ∧ z = 9 := by
sorry

end system_solution_l264_26441


namespace marks_lost_is_one_l264_26415

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  marks_per_correct : ℕ
  total_score : ℕ
  correct_answers : ℕ

/-- Calculates the marks lost per wrong answer -/
def marks_lost_per_wrong (exam : Examination) : ℚ :=
  (exam.marks_per_correct * exam.correct_answers - exam.total_score) / (exam.total_questions - exam.correct_answers)

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
  (h1 : exam.total_questions = 60)
  (h2 : exam.marks_per_correct = 4)
  (h3 : exam.total_score = 140)
  (h4 : exam.correct_answers = 40) : 
  marks_lost_per_wrong exam = 1 := by
  sorry

#eval marks_lost_per_wrong ⟨60, 4, 140, 40⟩

end marks_lost_is_one_l264_26415


namespace first_channel_ends_earlier_l264_26461

/-- Represents the runtime of a film on a TV channel with commercials -/
structure ChannelRuntime where
  segment_length : ℕ
  commercial_length : ℕ
  num_segments : ℕ

/-- Calculates the total runtime for a channel -/
def total_runtime (c : ChannelRuntime) : ℕ :=
  c.segment_length * c.num_segments + c.commercial_length * (c.num_segments - 1)

/-- The theorem to be proved -/
theorem first_channel_ends_earlier (film_length : ℕ) :
  ∃ (n : ℕ), 
    let channel1 := ChannelRuntime.mk 20 2 n
    let channel2 := ChannelRuntime.mk 10 1 (2 * n)
    film_length = 20 * n ∧ 
    film_length = 10 * (2 * n) ∧
    total_runtime channel1 < total_runtime channel2 := by
  sorry

end first_channel_ends_earlier_l264_26461


namespace ali_wallet_final_amount_l264_26411

def initial_wallet_value : ℕ := 7 * 5 + 1 * 10 + 3 * 20 + 1 * 50 + 8 * 1

def grocery_spending : ℕ := 65

def change_received : ℕ := 1 * 5 + 5 * 1

def friend_payment : ℕ := 2 * 20 + 2 * 1

theorem ali_wallet_final_amount :
  initial_wallet_value - grocery_spending + change_received + friend_payment = 150 := by
  sorry

end ali_wallet_final_amount_l264_26411


namespace farm_fencing_cost_l264_26418

/-- Calculates the cost of fencing a rectangular farm -/
theorem farm_fencing_cost 
  (area : ℝ) 
  (short_side : ℝ) 
  (cost_per_meter : ℝ) 
  (h_area : area = 1200) 
  (h_short : short_side = 30) 
  (h_cost : cost_per_meter = 14) : 
  let long_side := area / short_side
  let diagonal := Real.sqrt (long_side^2 + short_side^2)
  let total_length := long_side + short_side + diagonal
  cost_per_meter * total_length = 1680 :=
by sorry

end farm_fencing_cost_l264_26418


namespace sum_after_transformation_l264_26449

theorem sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * (a + 4) + 3 * (b + 4) = 3 * S + 24 := by
  sorry

end sum_after_transformation_l264_26449


namespace ellipse_foci_distance_l264_26493

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (3, 0) and tangent to the y-axis at (0, 2) -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ
  tangent_x : center.1 - semi_major_axis = 3
  tangent_y : center.2 - semi_minor_axis = 2

/-- The distance between the foci of the ellipse is 2√5 -/
theorem ellipse_foci_distance (e : Ellipse) : 
  Real.sqrt (4 * (e.semi_major_axis^2 - e.semi_minor_axis^2)) = 2 * Real.sqrt 5 := by
  sorry

end ellipse_foci_distance_l264_26493


namespace water_and_milk_amounts_l264_26471

/-- Sarah's special bread recipe -/
def special_bread_recipe (flour water milk : ℚ) : Prop :=
  water / flour = 75 / 300 ∧ milk / flour = 60 / 300

/-- The amount of flour Sarah uses -/
def flour_amount : ℚ := 900

/-- The theorem stating the required amounts of water and milk -/
theorem water_and_milk_amounts :
  ∀ water milk : ℚ,
  special_bread_recipe flour_amount water milk →
  water = 225 ∧ milk = 180 := by sorry

end water_and_milk_amounts_l264_26471


namespace art_fair_customers_l264_26426

theorem art_fair_customers (group1 group2 group3 : ℕ) 
  (paintings_per_customer1 paintings_per_customer2 paintings_per_customer3 : ℕ) 
  (total_paintings : ℕ) : 
  group1 = 4 → 
  group2 = 12 → 
  group3 = 4 → 
  paintings_per_customer1 = 2 → 
  paintings_per_customer2 = 1 → 
  paintings_per_customer3 = 4 → 
  total_paintings = 36 → 
  group1 * paintings_per_customer1 + 
  group2 * paintings_per_customer2 + 
  group3 * paintings_per_customer3 = total_paintings → 
  group1 + group2 + group3 = 20 := by
sorry

end art_fair_customers_l264_26426


namespace quadratic_intersection_points_specific_quadratic_roots_l264_26421

theorem quadratic_intersection_points (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔
  b^2 - 4*a*c > 0 :=
by sorry

theorem specific_quadratic_roots :
  ∃ x y : ℝ, x ≠ y ∧ 2 * x^2 + 3 * x - 2 = 0 ∧ 2 * y^2 + 3 * y - 2 = 0 :=
by sorry

end quadratic_intersection_points_specific_quadratic_roots_l264_26421


namespace cubic_sum_of_roots_l264_26410

theorem cubic_sum_of_roots (r s : ℝ) : 
  r^2 - 5*r + 3 = 0 → 
  s^2 - 5*s + 3 = 0 → 
  r^3 + s^3 = 80 := by
sorry

end cubic_sum_of_roots_l264_26410


namespace factor_t_squared_minus_64_l264_26413

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_t_squared_minus_64_l264_26413


namespace solve_system_l264_26431

theorem solve_system (x y : ℝ) : 
  (5 * x - 3 = 2 * x + 9) → 
  (x + y = 10) → 
  (x = 4 ∧ y = 6) := by
sorry


end solve_system_l264_26431


namespace rockets_win_in_7_l264_26466

/-- Probability of Warriors winning a single game -/
def p_warriors : ℚ := 3/4

/-- Probability of Rockets winning a single game -/
def p_rockets : ℚ := 1 - p_warriors

/-- Number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- Maximum number of games in the series -/
def max_games : ℕ := 7

/-- Probability of Rockets winning the series in exactly 7 games -/
def p_rockets_win_in_7 : ℚ := 135/4096

theorem rockets_win_in_7 :
  p_rockets_win_in_7 = (Nat.choose 6 3 : ℚ) * p_rockets^3 * p_warriors^3 * p_rockets :=
by sorry

end rockets_win_in_7_l264_26466


namespace ln_squared_plus_ln_inequality_l264_26486

theorem ln_squared_plus_ln_inequality (x : ℝ) :
  x > 0 → (Real.log x ^ 2 + Real.log x < 0 ↔ Real.exp (-1) < x ∧ x < 1) := by
  sorry

end ln_squared_plus_ln_inequality_l264_26486


namespace wilson_pays_twelve_l264_26452

/-- The total cost of Wilson's purchase at a fast-food restaurant --/
def total_cost (hamburger_price cola_price hamburger_quantity cola_quantity discount : ℕ) : ℕ :=
  hamburger_price * hamburger_quantity + cola_price * cola_quantity - discount

/-- Theorem stating that Wilson pays $12 in total --/
theorem wilson_pays_twelve :
  ∀ (hamburger_price cola_price hamburger_quantity cola_quantity discount : ℕ),
    hamburger_price = 5 →
    cola_price = 2 →
    hamburger_quantity = 2 →
    cola_quantity = 3 →
    discount = 4 →
    total_cost hamburger_price cola_price hamburger_quantity cola_quantity discount = 12 :=
by
  sorry


end wilson_pays_twelve_l264_26452


namespace complex_fraction_equality_l264_26469

theorem complex_fraction_equality : (1 - Complex.I * Real.sqrt 3) / ((Real.sqrt 3 + Complex.I) ^ 2) = -1/4 - (Complex.I * Real.sqrt 3) / 4 := by
  sorry

end complex_fraction_equality_l264_26469


namespace no_solution_equation_l264_26451

theorem no_solution_equation :
  ∀ x : ℝ, x ≠ 4 → x - 9 / (x - 4) ≠ 4 - 9 / (x - 4) := by
  sorry

end no_solution_equation_l264_26451


namespace original_square_area_l264_26419

theorem original_square_area : ∃ s : ℝ, s > 0 ∧ s^2 = 400 ∧ (s + 5)^2 = s^2 + 225 := by
  sorry

end original_square_area_l264_26419


namespace time_to_find_artifacts_is_120_months_l264_26458

/-- The time taken to find two artifacts given research and expedition times for the first artifact,
    and a multiplier for the second artifact's time. -/
def time_to_find_artifacts (research_time_1 : ℕ) (expedition_time_1 : ℕ) (multiplier : ℕ) : ℕ :=
  let first_artifact_time := research_time_1 + expedition_time_1
  let second_artifact_time := multiplier * first_artifact_time
  first_artifact_time + second_artifact_time

/-- Theorem stating that the time to find both artifacts is 120 months. -/
theorem time_to_find_artifacts_is_120_months :
  time_to_find_artifacts 6 24 3 = 120 := by
  sorry

end time_to_find_artifacts_is_120_months_l264_26458


namespace fourth_root_equation_solutions_l264_26439

theorem fourth_root_equation_solutions :
  let f (x : ℝ) := (Real.sqrt (Real.sqrt (43 - 2*x))) + (Real.sqrt (Real.sqrt (39 + 2*x)))
  ∃ (S : Set ℝ), S = {x | f x = 4} ∧ S = {21, -13.5} := by
  sorry

end fourth_root_equation_solutions_l264_26439


namespace funfair_visitors_l264_26479

theorem funfair_visitors (a : ℕ) : 
  a > 0 ∧ 
  (50 * a - 40 : ℤ) > 0 ∧ 
  (90 - 20 * a : ℤ) > 0 ∧ 
  (50 * a - 40 : ℤ) > (90 - 20 * a : ℤ) →
  (50 * a - 40 : ℤ) = 60 ∨ (50 * a - 40 : ℤ) = 110 ∨ (50 * a - 40 : ℤ) = 160 :=
by sorry

end funfair_visitors_l264_26479


namespace smallest_perfect_square_sum_of_24_consecutive_integers_l264_26488

theorem smallest_perfect_square_sum_of_24_consecutive_integers :
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (∃ (k : ℕ), k * k = 12 * (2 * n + 23)) ∧
    (∀ (m : ℕ), m > 0 → m < n → 
      ¬∃ (j : ℕ), j * j = 12 * (2 * m + 23)) :=
by
  sorry

end smallest_perfect_square_sum_of_24_consecutive_integers_l264_26488


namespace b_age_is_four_l264_26420

-- Define the ages as natural numbers
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- State the theorem
theorem b_age_is_four :
  (a = b + 2) →  -- a is two years older than b
  (b = 2 * c) →  -- b is twice as old as c
  (a + b + c = 12) →  -- The total of the ages is 12
  b = 4 := by
  sorry

end b_age_is_four_l264_26420


namespace lcm_of_incremented_numbers_l264_26498

theorem lcm_of_incremented_numbers : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 13 19)) = 8645 := by
  sorry

end lcm_of_incremented_numbers_l264_26498


namespace teacher_student_relationship_l264_26463

/-- In a school system, prove the relationship between teachers and students -/
theorem teacher_student_relationship (m n k l : ℕ) 
  (h1 : m > 0) -- Ensure there's at least one teacher
  (h2 : n > 0) -- Ensure there's at least one student
  (h3 : k > 0) -- Each teacher has at least one student
  (h4 : l > 0) -- Each student has at least one teacher
  (h5 : ∀ t, t ≤ m → (∃ s, s = k)) -- Each teacher has exactly k students
  (h6 : ∀ s, s ≤ n → (∃ t, t = l)) -- Each student has exactly l teachers
  : m * k = n * l := by
  sorry

end teacher_student_relationship_l264_26463


namespace range_of_a_l264_26425

open Set
open Real

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + a = 0) →
  (∀ x : ℝ, x > 1 → x + 1/(x-1) ≥ a) →
  a ∈ Ioo 0 3 := by
sorry

end range_of_a_l264_26425


namespace geometric_sequence_sum_l264_26440

theorem geometric_sequence_sum (a : ℚ) (r : ℚ) (n : ℕ) (h1 : a = 1/4) (h2 : r = 1/4) (h3 : n = 6) :
  a * (1 - r^n) / (1 - r) = 1365/4096 := by
  sorry

end geometric_sequence_sum_l264_26440


namespace distance_is_sqrt_206_l264_26490

def point : ℝ × ℝ × ℝ := (2, 3, 1)

def line_point : ℝ × ℝ × ℝ := (8, 10, 12)

def line_direction : ℝ × ℝ × ℝ := (2, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_206 : 
  distance_to_line point line_point line_direction = Real.sqrt 206 := by
  sorry

end distance_is_sqrt_206_l264_26490


namespace cubic_minus_linear_l264_26487

theorem cubic_minus_linear (n : ℕ) : ∃ n : ℕ, n^3 - n = 5814 :=
by
  -- We need to prove that there exists a natural number n such that n^3 - n = 5814
  -- given that n^3 - n is even and is the product of three consecutive natural numbers
  sorry

end cubic_minus_linear_l264_26487


namespace cone_volume_l264_26436

theorem cone_volume (d h : ℝ) (h1 : d = 16) (h2 : h = 12) :
  (1 / 3 : ℝ) * π * (d / 2) ^ 2 * h = 256 * π := by
  sorry

end cone_volume_l264_26436


namespace complex_division_equality_l264_26422

theorem complex_division_equality : ∀ (i : ℂ), i^2 = -1 →
  (3 - 2*i) / (2 + i) = 4/5 - 7/5*i :=
by sorry

end complex_division_equality_l264_26422


namespace sand_pile_base_area_l264_26435

/-- Given a rectangular compartment of sand and a conical pile, this theorem proves
    that the base area of the pile is 81/2 square meters. -/
theorem sand_pile_base_area
  (length width height : ℝ)
  (pile_height : ℝ)
  (h_length : length = 6)
  (h_width : width = 1.5)
  (h_height : height = 3)
  (h_pile_height : pile_height = 2)
  (h_volume_conservation : length * width * height = (1/3) * Real.pi * (pile_base_area / Real.pi) * pile_height)
  : pile_base_area = 81/2 := by
  sorry

end sand_pile_base_area_l264_26435


namespace roundness_of_1764000_l264_26459

/-- The roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- Theorem: The roundness of 1,764,000 is 11. -/
theorem roundness_of_1764000 : roundness 1764000 = 11 := by sorry

end roundness_of_1764000_l264_26459


namespace train_speed_calculation_l264_26401

/-- Proves that given a train journey with an original time of 50 minutes and a reduced time of 40 minutes
    at a speed of 60 km/h, the original average speed of the train is 48 km/h. -/
theorem train_speed_calculation (distance : ℝ) (original_time : ℝ) (reduced_time : ℝ) (new_speed : ℝ) :
  original_time = 50 / 60 →
  reduced_time = 40 / 60 →
  new_speed = 60 →
  distance = new_speed * reduced_time →
  distance / original_time = 48 :=
by sorry

end train_speed_calculation_l264_26401


namespace shoes_to_belts_ratio_l264_26478

def number_of_hats : ℕ := 5
def number_of_shoes : ℕ := 14
def belt_hat_difference : ℕ := 2

def number_of_belts : ℕ := number_of_hats + belt_hat_difference

theorem shoes_to_belts_ratio :
  (number_of_shoes : ℚ) / (number_of_belts : ℚ) = 2 / 1 := by
  sorry

end shoes_to_belts_ratio_l264_26478


namespace min_slope_tangent_line_l264_26432

/-- The function f(x) = x^3 + 3x^2 + 6x - 10 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem min_slope_tangent_line :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f' x ≥ m :=
sorry

end min_slope_tangent_line_l264_26432


namespace arithmetic_calculations_l264_26499

theorem arithmetic_calculations :
  (72 * 54 + 28 * 54 = 5400) ∧
  (60 * 25 * 8 = 12000) ∧
  (2790 / (250 * 12 - 2910) = 31) ∧
  ((100 - 1456 / 26) * 78 = 3432) := by
  sorry

end arithmetic_calculations_l264_26499


namespace tissue_length_l264_26457

/-- The total length of overlapped tissue pieces. -/
def totalLength (n : ℕ) (pieceLength : ℝ) (overlap : ℝ) : ℝ :=
  pieceLength + (n - 1 : ℝ) * (pieceLength - overlap)

/-- Theorem stating the total length of 30 pieces of tissue, each 25 cm long,
    overlapped by 6 cm, is 576 cm. -/
theorem tissue_length :
  totalLength 30 25 6 = 576 := by
  sorry

end tissue_length_l264_26457


namespace cube_cut_surface_area_l264_26434

/-- Represents a piece of the cube -/
structure Piece where
  height : ℝ

/-- Represents the solid formed by rearranging the cube pieces -/
structure Solid where
  pieces : List Piece

/-- Calculates the surface area of the solid -/
def surfaceArea (s : Solid) : ℝ :=
  sorry

theorem cube_cut_surface_area :
  let cube_volume : ℝ := 1
  let cut1 : ℝ := 1/2
  let cut2 : ℝ := 1/3
  let cut3 : ℝ := 1/17
  let piece_A : Piece := ⟨cut1⟩
  let piece_B : Piece := ⟨cut2⟩
  let piece_C : Piece := ⟨cut3⟩
  let piece_D : Piece := ⟨1 - (cut1 + cut2 + cut3)⟩
  let solid : Solid := ⟨[piece_A, piece_B, piece_C, piece_D]⟩
  surfaceArea solid = 11 :=
sorry

end cube_cut_surface_area_l264_26434


namespace unique_quadratic_with_real_roots_l264_26446

/-- A geometric progression of length 2016 -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i ∈ Finset.range 2015, a (i + 1) = r * a i

/-- An arithmetic progression of length 2016 -/
def arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i ∈ Finset.range 2015, b (i + 1) = b i + d

/-- The quadratic trinomial P_i(x) = x^2 + a_i * x + b_i -/
def P (a b : ℕ → ℝ) (i : ℕ) (x : ℝ) : ℝ :=
  x^2 + a i * x + b i

/-- P_k(x) has real roots iff its discriminant is non-negative -/
def has_real_roots (a b : ℕ → ℝ) (k : ℕ) : Prop :=
  (a k)^2 - 4 * b k ≥ 0

theorem unique_quadratic_with_real_roots
  (a b : ℕ → ℝ)
  (h_geom : geometric_progression a)
  (h_arith : arithmetic_progression b)
  (h_unique : ∃! k : ℕ, k ∈ Finset.range 2016 ∧ has_real_roots a b k) :
  ∃ k : ℕ, (k = 1 ∨ k = 2016) ∧ k ∈ Finset.range 2016 ∧ has_real_roots a b k :=
sorry

end unique_quadratic_with_real_roots_l264_26446


namespace zoo_rhinos_count_zoo_rhinos_count_is_three_l264_26444

/-- Calculates the number of endangered rhinos taken in by a zoo --/
theorem zoo_rhinos_count (initial_animals : ℕ) (gorilla_family : ℕ) (hippo : ℕ) 
  (lion_cubs : ℕ) (final_animals : ℕ) : ℕ :=
  let animals_after_gorillas := initial_animals - gorilla_family
  let animals_after_hippo := animals_after_gorillas + hippo
  let animals_after_cubs := animals_after_hippo + lion_cubs
  let meerkats := 2 * lion_cubs
  let animals_before_rhinos := animals_after_cubs + meerkats
  final_animals - animals_before_rhinos

/-- Proves that the number of endangered rhinos taken in is 3 --/
theorem zoo_rhinos_count_is_three : 
  zoo_rhinos_count 68 6 1 8 90 = 3 := by
  sorry

end zoo_rhinos_count_zoo_rhinos_count_is_three_l264_26444


namespace perfect_match_production_l264_26483

theorem perfect_match_production (total_workers : ℕ) 
  (tables_per_worker : ℕ) (chairs_per_worker : ℕ) 
  (table_workers : ℕ) (chair_workers : ℕ) : 
  total_workers = 36 → 
  tables_per_worker = 20 → 
  chairs_per_worker = 50 → 
  table_workers = 20 → 
  chair_workers = 16 → 
  table_workers + chair_workers = total_workers → 
  2 * (table_workers * tables_per_worker) = chair_workers * chairs_per_worker :=
by
  sorry

#check perfect_match_production

end perfect_match_production_l264_26483


namespace circle_equation_l264_26455

/-- The standard equation of a circle with center (-3, 4) and radius 2 is (x+3)^2 + (y-4)^2 = 4 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (-3, 4)
  let radius : ℝ := 2
  (x + 3)^2 + (y - 4)^2 = 4 ↔ 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_equation_l264_26455


namespace total_cost_calculation_l264_26481

def beef_amount : ℕ := 1000
def beef_price : ℕ := 8
def chicken_amount : ℕ := 2 * beef_amount
def chicken_price : ℕ := 3

theorem total_cost_calculation :
  beef_amount * beef_price + chicken_amount * chicken_price = 14000 := by
  sorry

end total_cost_calculation_l264_26481


namespace reservoir_fullness_after_storm_l264_26400

theorem reservoir_fullness_after_storm 
  (original_content : ℝ) 
  (original_percentage : ℝ) 
  (storm_deposit : ℝ) 
  (h1 : original_content = 245)
  (h2 : original_percentage = 54.44444444444444)
  (h3 : storm_deposit = 115) :
  let total_capacity := original_content / (original_percentage / 100)
  let new_content := original_content + storm_deposit
  (new_content / total_capacity) * 100 = 80 := by
sorry

end reservoir_fullness_after_storm_l264_26400


namespace board_officer_selection_ways_l264_26454

def board_size : ℕ := 30
def num_officers : ℕ := 4

def ways_without_special_members : ℕ := 26 * 25 * 24 * 23
def ways_with_one_pair : ℕ := 4 * 3 * 26 * 25
def ways_with_both_pairs : ℕ := 4 * 3 * 2 * 1

theorem board_officer_selection_ways :
  ways_without_special_members + 2 * ways_with_one_pair + ways_with_both_pairs = 374424 :=
sorry

end board_officer_selection_ways_l264_26454


namespace fixed_point_implies_stable_point_exists_stable_point_not_fixed_point_l264_26414

/-- A monotonically decreasing function -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

/-- Definition of a fixed point -/
def IsFixedPoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = x

/-- Definition of a stable point -/
def IsStablePoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = Function.invFun f x

theorem fixed_point_implies_stable_point
    (f : ℝ → ℝ) (hf : MonoDecreasing f) (x : ℝ) :
    IsFixedPoint f x → IsStablePoint f x :=
  sorry

theorem exists_stable_point_not_fixed_point
    (f : ℝ → ℝ) (hf : MonoDecreasing f) :
    ∃ x, IsStablePoint f x ∧ ¬IsFixedPoint f x :=
  sorry

end fixed_point_implies_stable_point_exists_stable_point_not_fixed_point_l264_26414


namespace quadratic_root_difference_l264_26445

theorem quadratic_root_difference (a b c : ℝ) (h : a > 0) :
  let equation := fun x => (5 + 2 * Real.sqrt 5) * x^2 - (3 + Real.sqrt 5) * x + 1
  let larger_root := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let smaller_root := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  equation larger_root = 0 ∧ equation smaller_root = 0 →
  larger_root - smaller_root = Real.sqrt (-3 + (2 * Real.sqrt 5) / 5) :=
by sorry

end quadratic_root_difference_l264_26445


namespace triangle_configuration_l264_26443

/-- Represents a triangle with side lengths x, y, and z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  hx : x > 0
  hy : y > 0
  hz : z > 0
  hxy : x < y + z
  hyz : y < x + z
  hzx : z < x + y

/-- Theorem about a specific triangle configuration -/
theorem triangle_configuration (a : ℝ) : 
  ∃ (t : Triangle), 
    t.x + t.y = 3 * t.z ∧ 
    t.z + t.y = t.x + a ∧ 
    t.x + t.z = 60 → 
    (0 < a ∧ a < 60) ∧
    (a = 30 → t.x = 42 ∧ t.y = 48 ∧ t.z = 30) := by
  sorry

#check triangle_configuration

end triangle_configuration_l264_26443


namespace construction_team_distance_l264_26492

/-- Calculates the total distance built by a construction team -/
def total_distance_built (days : ℕ) (rate : ℕ) : ℕ :=
  days * rate

/-- Proves that a construction team working for 5 days at 120 meters per day builds 600 meters -/
theorem construction_team_distance : total_distance_built 5 120 = 600 := by
  sorry

end construction_team_distance_l264_26492


namespace quadrilateral_problem_l264_26429

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Check if a quadrilateral is convex -/
def isConvex (quad : Quadrilateral) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Find the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_problem (PQRS : Quadrilateral) (T : Point) :
  isConvex PQRS →
  isPerpendicular PQRS.R PQRS.S PQRS.P PQRS.Q →
  isPerpendicular PQRS.P PQRS.Q PQRS.R PQRS.S →
  distance PQRS.R PQRS.S = 52 →
  distance PQRS.P PQRS.Q = 39 →
  isPerpendicular PQRS.Q T PQRS.P PQRS.S →
  T = lineIntersection PQRS.P PQRS.Q PQRS.Q T →
  distance PQRS.P T = 25 →
  distance PQRS.Q T = 14 := by
sorry

end quadrilateral_problem_l264_26429


namespace initial_shoe_pairs_l264_26470

/-- 
Given that a person loses 9 individual shoes and is left with a maximum of 20 matching pairs,
prove that the initial number of pairs of shoes was 25.
-/
theorem initial_shoe_pairs (lost_shoes : ℕ) (max_pairs_left : ℕ) : 
  lost_shoes = 9 →
  max_pairs_left = 20 →
  ∃ (initial_pairs : ℕ), initial_pairs = 25 ∧ 
    initial_pairs * 2 = max_pairs_left * 2 + lost_shoes :=
by sorry


end initial_shoe_pairs_l264_26470


namespace square_2023_position_l264_26494

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DABC
  | CBAD
  | DCBA

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DABC
  | SquarePosition.DABC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

-- Define the function to get the nth square position
def nthSquarePosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.DABC
  | 2 => SquarePosition.CBAD
  | 3 => SquarePosition.DCBA
  | _ => SquarePosition.ABCD -- This case is not actually possible

theorem square_2023_position : nthSquarePosition 2023 = SquarePosition.DABC := by
  sorry

end square_2023_position_l264_26494


namespace large_monkey_doll_cost_l264_26484

theorem large_monkey_doll_cost (total_spent : ℚ) (price_difference : ℚ) (extra_dolls : ℕ) 
  (h1 : total_spent = 320)
  (h2 : price_difference = 4)
  (h3 : extra_dolls = 40)
  (h4 : total_spent / (large_cost - price_difference) = total_spent / large_cost + extra_dolls) :
  large_cost = 8 :=
by
  sorry

end large_monkey_doll_cost_l264_26484


namespace quadratic_factorization_l264_26416

theorem quadratic_factorization (x : ℝ) :
  16 * x^2 + 8 * x - 24 = 8 * (2 * x + 3) * (x - 1) := by
  sorry

end quadratic_factorization_l264_26416


namespace temperature_rise_l264_26472

theorem temperature_rise (initial_temp final_temp rise : ℤ) : 
  initial_temp = -2 → rise = 3 → final_temp = initial_temp + rise → final_temp = 1 :=
by sorry

end temperature_rise_l264_26472


namespace jims_initial_reading_speed_l264_26437

/-- Represents Jim's reading habits and speeds -/
structure ReadingHabits where
  initial_speed : ℝ  -- Initial reading speed in pages per hour
  initial_hours : ℝ  -- Initial hours read per week
  new_speed : ℝ      -- New reading speed in pages per hour
  new_hours : ℝ      -- New hours read per week

/-- Theorem stating Jim's initial reading speed -/
theorem jims_initial_reading_speed 
  (h : ReadingHabits) 
  (initial_pages : h.initial_speed * h.initial_hours = 600) 
  (speed_increase : h.new_speed = 1.5 * h.initial_speed)
  (time_decrease : h.new_hours = h.initial_hours - 4)
  (new_pages : h.new_speed * h.new_hours = 660) : 
  h.initial_speed = 40 := by
  sorry


end jims_initial_reading_speed_l264_26437


namespace family_gathering_handshakes_l264_26405

/-- The number of handshakes at a family gathering -/
def total_handshakes (twin_sets quadruplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let quadruplets := quadruplet_sets * 4
  let twin_handshakes := twins * (twins - 2) / 2
  let quadruplet_handshakes := quadruplets * (quadruplets - 4) / 2
  let cross_handshakes := twins * (quadruplets / 3) + quadruplets * (twins / 4)
  twin_handshakes + quadruplet_handshakes + cross_handshakes

/-- Theorem stating the number of handshakes at the family gathering -/
theorem family_gathering_handshakes :
  total_handshakes 12 8 = 1168 := by
  sorry

end family_gathering_handshakes_l264_26405


namespace candy_division_l264_26491

theorem candy_division (total_candy : ℕ) (non_chocolate_candy : ℕ) 
  (chocolate_heart_bags : ℕ) (chocolate_kiss_bags : ℕ) :
  total_candy = 63 →
  non_chocolate_candy = 28 →
  chocolate_heart_bags = 2 →
  chocolate_kiss_bags = 3 →
  ∃ (pieces_per_bag : ℕ),
    pieces_per_bag > 0 ∧
    (total_candy - non_chocolate_candy) = 
      (chocolate_heart_bags + chocolate_kiss_bags) * pieces_per_bag ∧
    non_chocolate_candy % pieces_per_bag = 0 ∧
    chocolate_heart_bags + chocolate_kiss_bags + (non_chocolate_candy / pieces_per_bag) = 9 :=
by
  sorry

end candy_division_l264_26491
