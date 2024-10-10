import Mathlib

namespace ceiling_of_negative_fraction_squared_l1844_184443

theorem ceiling_of_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end ceiling_of_negative_fraction_squared_l1844_184443


namespace certain_value_calculation_l1844_184433

theorem certain_value_calculation (x : ℝ) (v : ℝ) (h1 : x = 100) (h2 : 0.8 * x + v = x) : v = 20 := by
  sorry

end certain_value_calculation_l1844_184433


namespace dinner_payment_difference_l1844_184439

/-- The problem of calculating the difference in payment between John and Jane --/
theorem dinner_payment_difference :
  let original_price : ℝ := 36.000000000000036
  let discount_rate : ℝ := 0.10
  let tip_rate : ℝ := 0.15
  let discounted_price := original_price * (1 - discount_rate)
  let john_tip := original_price * tip_rate
  let jane_tip := discounted_price * tip_rate
  let john_total := discounted_price + john_tip
  let jane_total := discounted_price + jane_tip
  john_total - jane_total = 0.5400000000000023 :=
by sorry

end dinner_payment_difference_l1844_184439


namespace remainder_sum_modulo_l1844_184464

theorem remainder_sum_modulo (p q : ℤ) 
  (hp : p % 98 = 84) 
  (hq : q % 126 = 117) : 
  (p + q) % 42 = 33 := by
  sorry

end remainder_sum_modulo_l1844_184464


namespace complex_equation_solution_l1844_184484

theorem complex_equation_solution (z : ℂ) :
  (3 + Complex.I) * z = 4 - 2 * Complex.I →
  z = 1 - Complex.I := by
sorry

end complex_equation_solution_l1844_184484


namespace complex_subtraction_l1844_184434

theorem complex_subtraction : (5 * Complex.I) - (2 + 2 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end complex_subtraction_l1844_184434


namespace eleven_day_rental_cost_l1844_184472

/-- Calculates the cost of a car rental given the number of days, daily rate, and weekly rate. -/
def rentalCost (days : ℕ) (dailyRate : ℕ) (weeklyRate : ℕ) : ℕ :=
  if days ≥ 7 then
    weeklyRate + (days - 7) * dailyRate
  else
    days * dailyRate

/-- Proves that the rental cost for 11 days is $310 given the specified rates. -/
theorem eleven_day_rental_cost :
  rentalCost 11 30 190 = 310 := by
  sorry

end eleven_day_rental_cost_l1844_184472


namespace car_tank_capacity_l1844_184409

/-- Calculates the capacity of a car's gas tank given initial and final mileage, efficiency, and number of fill-ups -/
def tank_capacity (initial_mileage final_mileage : ℕ) (efficiency : ℚ) (fill_ups : ℕ) : ℚ :=
  (final_mileage - initial_mileage : ℚ) / (efficiency * fill_ups)

/-- Proves that the car's tank capacity is 20 gallons given the problem conditions -/
theorem car_tank_capacity :
  tank_capacity 1728 2928 30 2 = 20 := by
  sorry

end car_tank_capacity_l1844_184409


namespace mark_born_in_1978_l1844_184423

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1985

/-- The year Mark took the ninth AMC 8 -/
def ninth_amc8_year : ℕ := first_amc8_year + 8

/-- Mark's age when he took the ninth AMC 8 -/
def marks_age : ℕ := 15

/-- Mark's birth year -/
def marks_birth_year : ℕ := ninth_amc8_year - marks_age

theorem mark_born_in_1978 : marks_birth_year = 1978 := by sorry

end mark_born_in_1978_l1844_184423


namespace positive_root_range_log_function_range_l1844_184456

-- Part 1
theorem positive_root_range (a : ℝ) :
  (∃ x > 0, 4^x + 2^x = a^2 + a) ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-2) :=
sorry

-- Part 2
theorem log_function_range (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, Real.log (x^2 + a*x + 1) = y) ↔ a ∈ Set.Ici 2 ∪ Set.Iic (-2) :=
sorry

end positive_root_range_log_function_range_l1844_184456


namespace parabola_vertex_coordinates_l1844_184404

/-- The vertex coordinates of the parabola y = -2x^2 + 8x - 3 are (2, 5) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := -2 * x^2 + 8 * x - 3
  ∃ (x y : ℝ), (x, y) = (2, 5) ∧ 
    (∀ t : ℝ, f t ≤ f x) ∧
    y = f x :=
by sorry

end parabola_vertex_coordinates_l1844_184404


namespace alcohol_mixture_percentage_l1844_184490

theorem alcohol_mixture_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_pure_alcohol : ℝ) : 
  initial_volume = 6 →
  initial_percentage = 35 / 100 →
  added_pure_alcohol = 1.8 →
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_pure_alcohol
  let final_volume := initial_volume + added_pure_alcohol
  final_alcohol / final_volume = 1 / 2 := by
sorry

end alcohol_mixture_percentage_l1844_184490


namespace red_pencils_count_l1844_184478

/-- Given a box of pencils with blue, red, and green colors, prove that the number of red pencils is 6 --/
theorem red_pencils_count (B R G : ℕ) : 
  B + R + G = 20 →  -- Total number of pencils
  B = 6 * G →       -- Blue pencils are 6 times green pencils
  R < B →           -- Fewer red pencils than blue ones
  R = 6 :=
by sorry

end red_pencils_count_l1844_184478


namespace f_difference_l1844_184435

-- Define the function f
def f (x : ℝ) : ℝ := x^6 + 3*x^4 - 4*x^3 + x^2 + 2*x

-- State the theorem
theorem f_difference : f 3 - f (-3) = -204 := by
  sorry

end f_difference_l1844_184435


namespace right_triangle_max_ratio_right_triangle_max_ratio_equality_l1844_184422

theorem right_triangle_max_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  (a^2 + b^2 + a*b) / c^2 ≤ 1.5 := by
sorry

theorem right_triangle_max_ratio_equality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ (a^2 + b^2 + a*b) / c^2 = 1.5 := by
sorry

end right_triangle_max_ratio_right_triangle_max_ratio_equality_l1844_184422


namespace ratio_consequent_l1844_184457

theorem ratio_consequent (antecedent : ℚ) (consequent : ℚ) : 
  antecedent = 30 → (4 : ℚ) / 6 = antecedent / consequent → consequent = 45 := by
  sorry

end ratio_consequent_l1844_184457


namespace cupcakes_theorem_l1844_184449

def cupcakes_problem (initial_cupcakes : ℕ) 
                     (delmont_class : ℕ) 
                     (donnelly_class : ℕ) 
                     (teachers : ℕ) 
                     (staff : ℕ) : Prop :=
  let given_away := delmont_class + donnelly_class + teachers + staff
  initial_cupcakes - given_away = 2

theorem cupcakes_theorem : 
  cupcakes_problem 40 18 16 2 2 := by
  sorry

end cupcakes_theorem_l1844_184449


namespace smallest_n_satisfying_conditions_l1844_184452

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

def is_perfect_square (x : ℚ) : Prop := ∃ y : ℚ, x = y^2

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 1 ∧
    is_perfect_cube (2002 * n) ∧
    is_perfect_square (n / 2002 : ℚ) ∧
    (∀ m : ℕ, m ≥ 1 →
      is_perfect_cube (2002 * m) →
      is_perfect_square (m / 2002 : ℚ) →
      n ≤ m) ∧
    n = 2002^5 :=
by sorry

end smallest_n_satisfying_conditions_l1844_184452


namespace fish_weight_is_eight_l1844_184487

/-- Represents the weight of a fish with three parts: tail, head, and body. -/
structure FishWeight where
  tail : ℝ
  head : ℝ
  body : ℝ

/-- The conditions given in the problem -/
def fish_conditions (f : FishWeight) : Prop :=
  f.tail = 1 ∧
  f.head = f.tail + f.body / 2 ∧
  f.body = f.head + f.tail

/-- The theorem stating that a fish satisfying the given conditions weighs 8 kg -/
theorem fish_weight_is_eight (f : FishWeight) 
  (h : fish_conditions f) : f.tail + f.head + f.body = 8 := by
  sorry


end fish_weight_is_eight_l1844_184487


namespace intersection_of_A_and_B_l1844_184426

def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l1844_184426


namespace swim_meet_car_occupancy_l1844_184403

theorem swim_meet_car_occupancy :
  let num_cars : ℕ := 2
  let num_vans : ℕ := 3
  let people_per_van : ℕ := 3
  let max_car_capacity : ℕ := 6
  let max_van_capacity : ℕ := 8
  let additional_capacity : ℕ := 17
  
  let total_van_occupancy : ℕ := num_vans * people_per_van
  let total_max_capacity : ℕ := num_cars * max_car_capacity + num_vans * max_van_capacity
  let actual_total_occupancy : ℕ := total_max_capacity - additional_capacity
  let car_occupancy : ℕ := actual_total_occupancy - total_van_occupancy
  
  car_occupancy / num_cars = 5 :=
by sorry

end swim_meet_car_occupancy_l1844_184403


namespace line_contains_both_points_l1844_184470

/-- The line equation is 2 - kx = -4y -/
def line_equation (k x y : ℝ) : Prop := 2 - k * x = -4 * y

/-- The first point (2, -1) -/
def point1 : ℝ × ℝ := (2, -1)

/-- The second point (3, -1.5) -/
def point2 : ℝ × ℝ := (3, -1.5)

/-- The line contains both points when k = -1 -/
theorem line_contains_both_points :
  ∃! k : ℝ, line_equation k point1.1 point1.2 ∧ line_equation k point2.1 point2.2 ∧ k = -1 := by
  sorry

end line_contains_both_points_l1844_184470


namespace cricketer_average_score_l1844_184408

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_set_matches : ℕ) 
  (second_set_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (h1 : total_matches = first_set_matches + second_set_matches)
  (h2 : total_matches = 5)
  (h3 : first_set_matches = 2)
  (h4 : second_set_matches = 3)
  (h5 : first_set_average = 40)
  (h6 : second_set_average = 10) :
  (first_set_matches * first_set_average + second_set_matches * second_set_average) / total_matches = 22 := by
  sorry

end cricketer_average_score_l1844_184408


namespace xy_max_and_x_plus_y_min_l1844_184483

theorem xy_max_and_x_plus_y_min (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + x + 2 * y = 6) :
  (∀ a b : ℝ, a > 0 → b > 0 → a * b + a + 2 * b = 6 → x * y ≥ a * b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a * b + a + 2 * b = 6 → x + y ≤ a + b) ∧
  (x * y = 2 ∨ x + y = 4 * Real.sqrt 2 - 3) :=
sorry

end xy_max_and_x_plus_y_min_l1844_184483


namespace stuffed_animals_difference_l1844_184498

theorem stuffed_animals_difference (thor jake quincy : ℕ) : 
  quincy = 100 * (thor + jake) →
  jake = 2 * thor + 15 →
  quincy = 4000 →
  quincy - jake = 3969 := by
  sorry

end stuffed_animals_difference_l1844_184498


namespace triangle_area_l1844_184475

/-- Given a triangle ABC with the following properties:
  1. The side opposite to angle B has length 1
  2. Angle B measures π/6 radians
  3. 1/tan(A) + 1/tan(C) = 2
Prove that the area of the triangle is 1/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b = 1 → 
  B = π / 6 → 
  1 / Real.tan A + 1 / Real.tan C = 2 → 
  (1 / 2) * a * b * Real.sin C = 1 / 4 := by
  sorry

end triangle_area_l1844_184475


namespace average_of_numbers_l1844_184412

theorem average_of_numbers (x : ℝ) : 
  ((x + 5) + 14 + x + 5) / 4 = 9 → x = 6 := by
sorry

end average_of_numbers_l1844_184412


namespace fraction_decomposition_l1844_184444

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 ≠ -1) :
  (-x^3 + 4*x^2 - 5*x + 3) / (x^4 + x^2) = 3/x^2 + (4*x + 1)/(x^2 + 1) - 5/x :=
by sorry

end fraction_decomposition_l1844_184444


namespace simons_school_students_l1844_184429

def total_students : ℕ := 2500

theorem simons_school_students (linas_students : ℕ) 
  (h1 : linas_students * 5 = total_students) : 
  linas_students * 4 = 2000 := by
  sorry

#check simons_school_students

end simons_school_students_l1844_184429


namespace candy_distribution_l1844_184477

theorem candy_distribution (x : ℝ) (x_pos : x > 0) : 
  let al_share := (4/9 : ℝ) * x
  let bert_share := (1/3 : ℝ) * (x - al_share)
  let carl_share := (2/9 : ℝ) * (x - al_share - bert_share)
  al_share + bert_share + carl_share = x :=
by sorry

end candy_distribution_l1844_184477


namespace log_expression_simplification_l1844_184407

theorem log_expression_simplification 
  (a b c d x y z w : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) : 
  Real.log (a / b) + Real.log (b / c) + Real.log (c / d) - Real.log (a * y * z / (d * x * w)) = Real.log (x * w / (y * z)) := by
  sorry

end log_expression_simplification_l1844_184407


namespace test_question_points_l1844_184479

theorem test_question_points :
  ∀ (total_points total_questions two_point_questions : ℕ) 
    (other_question_points : ℚ),
  total_points = 100 →
  total_questions = 40 →
  two_point_questions = 30 →
  total_points = 2 * two_point_questions + (total_questions - two_point_questions) * other_question_points →
  other_question_points = 4 := by
sorry

end test_question_points_l1844_184479


namespace prism_volume_l1844_184496

/-- Given a right rectangular prism with dimensions a, b, and c,
    if the areas of three faces are 30, 45, and 54 square centimeters,
    then the volume of the prism is 270 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 45) 
  (h3 : b * c = 54) : 
  a * b * c = 270 := by
sorry

end prism_volume_l1844_184496


namespace parallel_vectors_x_value_l1844_184427

def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b : ℝ × ℝ := (2, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (vector_a x) vector_b → x = -2/3 := by
  sorry

end parallel_vectors_x_value_l1844_184427


namespace ten_times_average_sum_positions_elida_length_adrianna_length_l1844_184428

/-- Represents the alphabetical position of a letter (A=1, B=2, ..., Z=26) -/
def alphabeticalPosition (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

/-- The name Elida -/
def elida : String := "ELIDA"

/-- The name Adrianna -/
def adrianna : String := "ADRIANNA"

/-- Sum of alphabetical positions of letters in a name -/
def sumAlphabeticalPositions (name : String) : ℕ :=
  name.toList.map alphabeticalPosition |>.sum

/-- Theorem stating that 10 times the average of the sum of alphabetical positions
    in both names is 465 -/
theorem ten_times_average_sum_positions : 
  (10 : ℚ) * ((sumAlphabeticalPositions elida + sumAlphabeticalPositions adrianna) / 2) = 465 := by
  sorry

/-- Elida has 5 letters -/
theorem elida_length : elida.length = 5 := by sorry

/-- Adrianna has 2 less than twice the number of letters Elida has -/
theorem adrianna_length : adrianna.length = 2 * elida.length - 2 := by sorry

end ten_times_average_sum_positions_elida_length_adrianna_length_l1844_184428


namespace equilateral_not_unique_from_angle_and_median_l1844_184410

/-- Represents a triangle -/
structure Triangle where
  a : ℝ  -- side length a
  b : ℝ  -- side length b
  c : ℝ  -- side length c
  α : ℝ  -- angle opposite to side a
  β : ℝ  -- angle opposite to side b
  γ : ℝ  -- angle opposite to side c

/-- Represents a median of a triangle -/
def Median (t : Triangle) (side : ℕ) : ℝ := sorry

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Theorem stating that one angle and the median to the opposite side
    do not uniquely determine an equilateral triangle -/
theorem equilateral_not_unique_from_angle_and_median :
  ∃ (t1 t2 : Triangle) (side : ℕ),
    t1.α = t2.α ∧
    Median t1 side = Median t2 side ∧
    IsEquilateral t1 ∧
    IsEquilateral t2 ∧
    t1 ≠ t2 :=
  sorry

end equilateral_not_unique_from_angle_and_median_l1844_184410


namespace periodic_function_value_l1844_184406

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2008 = -1 → f 2009 = 1 := by sorry

end periodic_function_value_l1844_184406


namespace problem_solution_l1844_184497

theorem problem_solution (x : ℝ) (h : Real.sqrt x = 6000 * (1/1000)) :
  (600 - Real.sqrt x)^2 + x = 352872 := by
  sorry

end problem_solution_l1844_184497


namespace no_real_roots_l1844_184451

theorem no_real_roots : ∀ x : ℝ, x^2 + x + 5 ≠ 0 := by
  sorry

end no_real_roots_l1844_184451


namespace survey_results_l1844_184468

/-- Represents the preferences of students in a class --/
structure ClassPreferences where
  total_students : Nat
  men_like_math : Nat
  men_like_lit : Nat
  men_dislike_both : Nat
  women_dislike_both : Nat
  total_men : Nat
  like_both : Nat
  like_only_math : Nat

/-- Theorem stating the results of the survey --/
theorem survey_results (prefs : ClassPreferences)
  (h1 : prefs.total_students = 35)
  (h2 : prefs.men_like_math = 7)
  (h3 : prefs.men_like_lit = 6)
  (h4 : prefs.men_dislike_both = 5)
  (h5 : prefs.women_dislike_both = 8)
  (h6 : prefs.total_men = 16)
  (h7 : prefs.like_both = 5)
  (h8 : prefs.like_only_math = 11) :
  (∃ (men_like_both women_like_only_lit : Nat),
    men_like_both = 2 ∧
    women_like_only_lit = 6) := by
  sorry


end survey_results_l1844_184468


namespace valley_of_five_lakes_streams_l1844_184469

structure Lake :=
  (name : String)

structure Valley :=
  (lakes : Finset Lake)
  (streams : Finset (Lake × Lake))
  (start : Lake)

def Valley.valid (v : Valley) : Prop :=
  v.lakes.card = 5 ∧
  ∃ S B : Lake,
    S ∈ v.lakes ∧
    B ∈ v.lakes ∧
    S ≠ B ∧
    (∀ fish : ℕ → Lake,
      fish 0 = v.start →
      (∀ i < 4, (fish i, fish (i + 1)) ∈ v.streams) →
      (fish 4 = S ∧ fish 4 = v.start) ∨ fish 4 = B)

theorem valley_of_five_lakes_streams (v : Valley) :
  v.valid → v.streams.card = 3 :=
sorry

end valley_of_five_lakes_streams_l1844_184469


namespace emily_total_songs_l1844_184448

/-- Calculates the total number of songs Emily has after buying more. -/
def total_songs (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Emily's total number of songs is 13 given the initial and bought amounts. -/
theorem emily_total_songs :
  total_songs 6 7 = 13 := by
  sorry

end emily_total_songs_l1844_184448


namespace original_bales_count_l1844_184413

theorem original_bales_count (bales_stacked bales_now : ℕ) 
  (h1 : bales_stacked = 26)
  (h2 : bales_now = 54) :
  bales_now - bales_stacked = 28 := by
  sorry

end original_bales_count_l1844_184413


namespace factor_implies_c_value_l1844_184458

/-- The polynomial P(x) -/
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + c*x + 15

/-- Theorem: If x - 3 is a factor of P(x), then c = -23 -/
theorem factor_implies_c_value (c : ℝ) : 
  (∀ x, P c x = 0 ↔ x = 3) → c = -23 := by
  sorry

end factor_implies_c_value_l1844_184458


namespace cube_difference_l1844_184400

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end cube_difference_l1844_184400


namespace starting_number_is_100_l1844_184415

/-- The starting number of a range ending at 400, where the average of the integers
    in this range is 100 greater than the average of the integers from 50 to 250. -/
def starting_number : ℤ :=
  let avg_50_to_250 := (50 + 250) / 2
  let avg_x_to_400 := avg_50_to_250 + 100
  2 * avg_x_to_400 - 400

theorem starting_number_is_100 : starting_number = 100 := by
  sorry

end starting_number_is_100_l1844_184415


namespace smallest_b_for_even_polynomial_l1844_184419

theorem smallest_b_for_even_polynomial : ∃ (b : ℕ+), 
  (∀ (x : ℤ), ∃ (k : ℤ), x^4 + (b : ℤ)^3 + (b : ℤ)^2 = 2 * k) ∧ 
  (∀ (b' : ℕ+), b' < b → ∃ (x : ℤ), ∀ (k : ℤ), x^4 + (b' : ℤ)^3 + (b' : ℤ)^2 ≠ 2 * k) :=
by sorry

end smallest_b_for_even_polynomial_l1844_184419


namespace sum_of_cubes_l1844_184459

theorem sum_of_cubes (x y z : ℕ+) :
  (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 504 →
  (x : ℕ) + y + z = 9 := by
  sorry

end sum_of_cubes_l1844_184459


namespace sum_of_common_ratios_l1844_184447

-- Define the common ratios and terms of the geometric sequences
variables {k p r : ℝ} {a₂ a₃ b₂ b₃ : ℝ}

-- Define the geometric sequences
def is_geometric_sequence (k p a₂ a₃ : ℝ) : Prop :=
  a₂ = k * p ∧ a₃ = k * p^2

-- State the theorem
theorem sum_of_common_ratios
  (h₁ : is_geometric_sequence k p a₂ a₃)
  (h₂ : is_geometric_sequence k r b₂ b₃)
  (h₃ : p ≠ r)
  (h₄ : k ≠ 0)
  (h₅ : a₃ - b₃ = 4 * (a₂ - b₂)) :
  p + r = 4 := by
sorry

end sum_of_common_ratios_l1844_184447


namespace one_slice_left_l1844_184480

/-- Represents the number of bread slices used each day of the week -/
structure WeeklyBreadUsage where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat

/-- Calculates the number of bread slices left after a week -/
def slicesLeft (initialSlices : Nat) (usage : WeeklyBreadUsage) : Nat :=
  initialSlices - (usage.monday + usage.tuesday + usage.wednesday + 
                   usage.thursday + usage.friday + usage.saturday + usage.sunday)

/-- Theorem stating that 1 slice of bread is left after the week -/
theorem one_slice_left (initialSlices : Nat) (usage : WeeklyBreadUsage) :
  initialSlices = 22 ∧
  usage.monday = 2 ∧
  usage.tuesday = 3 ∧
  usage.wednesday = 4 ∧
  usage.thursday = 1 ∧
  usage.friday = 3 ∧
  usage.saturday = 5 ∧
  usage.sunday = 3 →
  slicesLeft initialSlices usage = 1 := by
  sorry

#check one_slice_left

end one_slice_left_l1844_184480


namespace camryn_practice_schedule_l1844_184453

/-- Represents the number of days between Camryn's trumpet practices -/
def trumpet_interval : ℕ := 11

/-- Represents the number of days until Camryn practices both instruments again -/
def next_joint_practice : ℕ := 33

/-- Represents the number of days between Camryn's flute practices -/
def flute_interval : ℕ := 3

theorem camryn_practice_schedule :
  (trumpet_interval > 1) ∧
  (flute_interval > 1) ∧
  (flute_interval < trumpet_interval) ∧
  (next_joint_practice % trumpet_interval = 0) ∧
  (next_joint_practice % flute_interval = 0) :=
by sorry

end camryn_practice_schedule_l1844_184453


namespace ones_digit_of_largest_power_of_three_dividing_81_factorial_l1844_184440

/-- The largest power of 3 that divides n! -/
def largest_power_of_three_dividing_factorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27) + (n / 81)

/-- The ones digit of 3^n -/
def ones_digit_of_power_of_three (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem ones_digit_of_largest_power_of_three_dividing_81_factorial :
  ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 81) = 1 := by
  sorry

#eval ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 81)

end ones_digit_of_largest_power_of_three_dividing_81_factorial_l1844_184440


namespace smallest_multiplier_perfect_square_l1844_184414

theorem smallest_multiplier_perfect_square (x : ℕ+) :
  (∃ y : ℕ+, y = 2 ∧ 
    (∃ z : ℕ+, x * y = z^2) ∧
    (∀ w : ℕ+, w < y → ¬∃ v : ℕ+, x * w = v^2)) →
  x = 2 := by
sorry

end smallest_multiplier_perfect_square_l1844_184414


namespace range_of_m_l1844_184455

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x ≥ 2 → x^2 - 2*x + 1 ≥ m) → m ≤ 1 := by
  sorry

end range_of_m_l1844_184455


namespace depreciation_is_one_fourth_l1844_184492

/-- Represents the depreciation of a scooter over one year -/
def scooter_depreciation (initial_value : ℚ) (value_after_one_year : ℚ) : ℚ :=
  (initial_value - value_after_one_year) / initial_value

/-- Theorem stating that for the given initial value and value after one year,
    the depreciation fraction is 1/4 -/
theorem depreciation_is_one_fourth :
  scooter_depreciation 40000 30000 = 1/4 := by
  sorry

end depreciation_is_one_fourth_l1844_184492


namespace men_to_women_ratio_l1844_184489

/-- Represents a co-ed softball team -/
structure SoftballTeam where
  men : ℕ
  women : ℕ

/-- Properties of the softball team -/
def validTeam (team : SoftballTeam) : Prop :=
  team.women = team.men + 6 ∧ team.men + team.women = 16

theorem men_to_women_ratio (team : SoftballTeam) (h : validTeam team) :
  (team.men : ℚ) / team.women = 5 / 11 := by
  sorry

end men_to_women_ratio_l1844_184489


namespace hemisphere_center_of_mass_l1844_184432

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a hemisphere -/
structure Hemisphere where
  radius : ℝ

/-- Density function for the hemisphere -/
def density (p : Point3D) : ℝ :=
  sorry

/-- Center of mass of a hemisphere -/
def centerOfMass (h : Hemisphere) : Point3D :=
  sorry

/-- Theorem: The center of mass of a hemisphere with radius R and volume density
    proportional to the distance from the origin is located at (0, 0, 2R/5) -/
theorem hemisphere_center_of_mass (h : Hemisphere) :
  let com := centerOfMass h
  com.x = 0 ∧ com.y = 0 ∧ com.z = 2 * h.radius / 5 := by
  sorry

end hemisphere_center_of_mass_l1844_184432


namespace refrigerator_savings_l1844_184450

/-- Calculates the savings when paying cash for a refrigerator instead of installments --/
theorem refrigerator_savings (cash_price deposit installment_amount : ℕ) (num_installments : ℕ) :
  cash_price = 8000 →
  deposit = 3000 →
  installment_amount = 300 →
  num_installments = 30 →
  deposit + num_installments * installment_amount - cash_price = 4000 := by
  sorry

end refrigerator_savings_l1844_184450


namespace pages_copied_l1844_184421

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 4

/-- The amount available in dollars -/
def available_dollars : ℕ := 25

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The number of pages that can be copied for $25 is 625 -/
theorem pages_copied (cost_per_page : ℕ) (available_dollars : ℕ) (cents_per_dollar : ℕ) :
  (available_dollars * cents_per_dollar) / cost_per_page = 625 :=
sorry

end pages_copied_l1844_184421


namespace max_profit_at_optimal_price_l1844_184495

/-- Profit function for a store selling items -/
def profit_function (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (sales_increase_rate : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_price - price_reduction - cost_price) * (initial_sales + sales_increase_rate * price_reduction)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit_at_optimal_price 
  (cost_price : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : cost_price = 2)
  (h2 : initial_price = 13)
  (h3 : initial_sales = 500)
  (h4 : sales_increase_rate = 100) :
  ∃ (optimal_price_reduction : ℝ),
    optimal_price_reduction = 3 ∧
    profit_function cost_price initial_price initial_sales sales_increase_rate optimal_price_reduction = 6400 ∧
    ∀ (price_reduction : ℝ), 
      profit_function cost_price initial_price initial_sales sales_increase_rate price_reduction ≤ 
      profit_function cost_price initial_price initial_sales sales_increase_rate optimal_price_reduction :=
by
  sorry

#check max_profit_at_optimal_price

end max_profit_at_optimal_price_l1844_184495


namespace thirteenth_on_monday_l1844_184467

/-- Represents a day of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Represents a month of the year -/
inductive Month
  | january
  | february
  | march
  | april
  | may
  | june
  | july
  | august
  | september
  | october
  | november
  | december

/-- Returns the number of days in a given month -/
def daysInMonth (m : Month) : Nat :=
  match m with
  | .january => 31
  | .february => 28  -- Assuming non-leap year for simplicity
  | .march => 31
  | .april => 30
  | .may => 31
  | .june => 30
  | .july => 31
  | .august => 31
  | .september => 30
  | .october => 31
  | .november => 30
  | .december => 31

/-- Calculates the day of the week for the 13th of a given month, 
    given the day of the week for the 13th of the previous month -/
def dayOf13th (prevDay : DayOfWeek) (m : Month) : DayOfWeek :=
  sorry

/-- Theorem: In any year, there exists at least one month where the 13th falls on a Monday -/
theorem thirteenth_on_monday :
  ∀ (startDay : DayOfWeek), 
    ∃ (m : Month), dayOf13th startDay m = DayOfWeek.monday :=
  sorry

end thirteenth_on_monday_l1844_184467


namespace acute_angle_solution_l1844_184473

theorem acute_angle_solution (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) :
  Real.cos α * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1 →
  α = 40 * Real.pi / 180 := by
  sorry

end acute_angle_solution_l1844_184473


namespace hoopit_students_count_l1844_184425

/-- Represents the number of toes on each hand for Hoopits -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands for Hoopits -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes on each hand for Neglarts -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands for Neglarts -/
def neglart_hands : ℕ := 5

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that the number of Hoopit students on the bus is 7 -/
theorem hoopit_students_count : 
  ∃ (h : ℕ), h * (hoopit_toes_per_hand * hoopit_hands) + 
             neglart_students * (neglart_toes_per_hand * neglart_hands) = total_toes ∧ 
             h = 7 := by
  sorry

end hoopit_students_count_l1844_184425


namespace geometric_sequence_increasing_iff_first_three_increasing_l1844_184499

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The condition a₁ < a₂ < a₃ -/
def FirstThreeIncreasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_increasing_iff_first_three_increasing
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  IncreasingSequence a ↔ FirstThreeIncreasing a :=
sorry

end geometric_sequence_increasing_iff_first_three_increasing_l1844_184499


namespace geometric_sequence_sum_l1844_184474

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 2 + a 3 = 1 ∧
  a 2 + a 3 + a 4 = 2

/-- The theorem stating the sum of the 6th, 7th, and 8th terms -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end geometric_sequence_sum_l1844_184474


namespace part1_part2_l1844_184465

-- Define the inequality function
def f (k x : ℝ) : ℝ := (k^2 - 2*k - 3)*x^2 - (k + 1)*x - 1

-- Define the solution set M
def M (k : ℝ) : Set ℝ := {x : ℝ | f k x < 0}

-- Part 1: Range of positive integer k when 1 ∈ M
theorem part1 : 
  (∀ k : ℕ+, 1 ∈ M k ↔ k ∈ ({1, 2, 3, 4} : Set ℕ+)) :=
sorry

-- Part 2: Range of real k when M = ℝ
theorem part2 : 
  (∀ k : ℝ, M k = Set.univ ↔ k ∈ Set.Icc (-1) (11/5)) :=
sorry

end part1_part2_l1844_184465


namespace sum_product_inequality_l1844_184411

theorem sum_product_inequality (a b c x y z k : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hk1 : a + x = k) (hk2 : b + y = k) (hk3 : c + z = k) : 
  a * y + b * z + c * x < k^2 := by
sorry

end sum_product_inequality_l1844_184411


namespace train_length_l1844_184416

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (5 / 18) → 
  crossing_time = 30 →
  bridge_length = 240 →
  (train_speed * crossing_time) - bridge_length = 135 := by
  sorry

#check train_length

end train_length_l1844_184416


namespace max_temp_range_l1844_184441

/-- Given 5 temperatures with an average of 40 and a minimum of 30,
    the maximum possible range is 50. -/
theorem max_temp_range (temps : Fin 5 → ℝ) 
    (avg : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 40)
    (min : ∀ i, temps i ≥ 30) 
    (exists_min : ∃ i, temps i = 30) : 
    (∀ i j, temps i - temps j ≤ 50) ∧ 
    (∃ i j, temps i - temps j = 50) := by
  sorry

end max_temp_range_l1844_184441


namespace sqrt_equation_solution_l1844_184481

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (5 * x - 1) + Real.sqrt (x - 1) = 2 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l1844_184481


namespace divisors_of_8_factorial_l1844_184445

theorem divisors_of_8_factorial : Nat.card (Nat.divisors (Nat.factorial 8)) = 96 := by
  sorry

end divisors_of_8_factorial_l1844_184445


namespace pure_imaginary_ratio_l1844_184488

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 4 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -4 / 3 := by
sorry

end pure_imaginary_ratio_l1844_184488


namespace bacteria_fill_count_l1844_184476

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to fill a table with bacteria -/
def bacteriaFillWays (m n : ℕ) : ℕ :=
  2^(n-1) * (fib (2*n+1))^(m-1)

/-- Theorem: The number of ways to fill an m×n table with non-overlapping bacteria -/
theorem bacteria_fill_count (m n : ℕ) :
  bacteriaFillWays m n = 2^(n-1) * (fib (2*n+1))^(m-1) :=
by
  sorry

/-- Property: Bacteria have horizontal bodies of natural length -/
axiom bacteria_body_natural_length : True

/-- Property: Bacteria have nonnegative number of vertical feet -/
axiom bacteria_feet_nonnegative : True

/-- Property: Bacteria feet have nonnegative natural length -/
axiom bacteria_feet_natural_length : True

/-- Property: Bacteria do not overlap in the table -/
axiom bacteria_no_overlap : True

end bacteria_fill_count_l1844_184476


namespace line_intersects_extension_l1844_184485

/-- Given a line l: Ax + By + C = 0 and two points P₁ and P₂, 
    prove that l intersects with the extension of P₁P₂ under certain conditions. -/
theorem line_intersects_extension (A B C x₁ y₁ x₂ y₂ : ℝ) 
  (hAB : A ≠ 0 ∨ B ≠ 0)
  (hSameSide : (A * x₁ + B * y₁ + C) * (A * x₂ + B * y₂ + C) > 0)
  (hDistance : |A * x₁ + B * y₁ + C| > |A * x₂ + B * y₂ + C|) :
  ∃ (t : ℝ), t > 1 ∧ A * (x₁ + t * (x₂ - x₁)) + B * (y₁ + t * (y₂ - y₁)) + C = 0 :=
sorry

end line_intersects_extension_l1844_184485


namespace arithmetic_sequence_sum_l1844_184491

theorem arithmetic_sequence_sum (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence condition
  ((n + 1) * a (n + 1) = 4) →  -- sum of odd-numbered terms
  (n * a (n + 1) = 3) →  -- sum of even-numbered terms
  n = 3 := by
sorry

end arithmetic_sequence_sum_l1844_184491


namespace odd_function_property_l1844_184462

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f : ℝ → ℝ is increasing on [a,b] if x₁ ≤ x₂ implies f x₁ ≤ f x₂ for all x₁, x₂ in [a,b] -/
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ b → f x₁ ≤ f x₂

theorem odd_function_property (f : ℝ → ℝ) :
  IsOdd f →
  IncreasingOn f 3 7 →
  (∀ x ∈ Set.Icc 3 6, f x ≤ 8) →
  (∀ x ∈ Set.Icc 3 6, 1 ≤ f x) →
  f (-3) + 2 * f 6 = 15 := by
  sorry

end odd_function_property_l1844_184462


namespace subtraction_rearrangement_l1844_184461

theorem subtraction_rearrangement (a b c : ℤ) : a - b - c = a - (b + c) := by
  sorry

end subtraction_rearrangement_l1844_184461


namespace cube_volume_in_pyramid_l1844_184417

/-- A regular pyramid with a rectangular base and isosceles triangular lateral faces -/
structure RegularPyramid where
  base_length : ℝ
  base_width : ℝ
  lateral_faces_isosceles : Bool

/-- A cube placed inside the pyramid -/
structure InsideCube where
  side_length : ℝ

/-- The theorem stating the volume of the cube inside the pyramid -/
theorem cube_volume_in_pyramid (pyramid : RegularPyramid) (cube : InsideCube) : 
  pyramid.base_length = 2 →
  pyramid.base_width = 3 →
  pyramid.lateral_faces_isosceles = true →
  (cube.side_length * Real.sqrt 3 = Real.sqrt 13) →
  cube.side_length^3 = (39 * Real.sqrt 39) / 27 := by
  sorry

end cube_volume_in_pyramid_l1844_184417


namespace min_nSn_l1844_184463

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℤ  -- The sequence
  S : ℕ+ → ℤ  -- Sum of first n terms
  h4 : S 4 = -2
  h5 : S 5 = 0
  h6 : S 6 = 3

/-- The product of n and S_n -/
def nSn (seq : ArithmeticSequence) (n : ℕ+) : ℤ :=
  n * seq.S n

theorem min_nSn (seq : ArithmeticSequence) :
  ∃ (m : ℕ+), ∀ (n : ℕ+), nSn seq m ≤ nSn seq n ∧ nSn seq m = -9 :=
sorry

end min_nSn_l1844_184463


namespace circle_and_tangent_properties_l1844_184436

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 9

-- Define the center of the circle
def center (x y : ℝ) : Prop :=
  y = 2 * x ∧ x > 0 ∧ y > 0

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 4 * x + 3 * y = 25

theorem circle_and_tangent_properties :
  ∃ (cx cy : ℝ),
    -- The center lies on y = 2x in the first quadrant
    center cx cy ∧
    -- The circle passes through (1, -1)
    circle_C 1 (-1) ∧
    -- (4, 3) is outside the circle
    ¬ circle_C 4 3 ∧
    -- The tangent lines touch the circle at exactly one point each
    (∃ (tx ty : ℝ), circle_C tx ty ∧ tangent_line_1 tx) ∧
    (∃ (tx ty : ℝ), circle_C tx ty ∧ tangent_line_2 tx ty) :=
by sorry

end circle_and_tangent_properties_l1844_184436


namespace yw_equals_five_l1844_184438

/-- Triangle with sides a, b, c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point on the perimeter of a triangle --/
structure PerimeterPoint where
  distanceFromY : ℝ

/-- Definition of the meeting point of two ants crawling from X in opposite directions --/
def meetingPoint (t : Triangle) : PerimeterPoint :=
  { distanceFromY := 5 }

/-- Theorem stating that YW = 5 for the given triangle and ant movement --/
theorem yw_equals_five (t : Triangle) 
    (h1 : t.a = 7) 
    (h2 : t.b = 8) 
    (h3 : t.c = 9) : 
  (meetingPoint t).distanceFromY = 5 := by
  sorry

end yw_equals_five_l1844_184438


namespace max_value_of_sum_of_sqrt_differences_l1844_184430

theorem max_value_of_sum_of_sqrt_differences (x y z : ℝ) 
  (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  Real.sqrt (|x - y|) + Real.sqrt (|y - z|) + Real.sqrt (|z - x|) ≤ Real.sqrt 2 + 1 ∧
  ∃ x y z, x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ z ∈ Set.Icc 0 1 ∧
    Real.sqrt (|x - y|) + Real.sqrt (|y - z|) + Real.sqrt (|z - x|) = Real.sqrt 2 + 1 :=
by
  sorry

end max_value_of_sum_of_sqrt_differences_l1844_184430


namespace complex_on_imaginary_axis_l1844_184494

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (-2 + a * Complex.I) / (1 + Complex.I)
  (z.re = 0) ↔ (a = 2) := by
sorry

end complex_on_imaginary_axis_l1844_184494


namespace unique_perfect_between_primes_l1844_184442

/-- A number is perfect if the sum of its positive divisors equals twice the number. -/
def IsPerfect (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 2 * n

/-- The theorem stating that 6 is the only perfect number n such that n-1 and n+1 are prime. -/
theorem unique_perfect_between_primes :
  ∀ n : ℕ, IsPerfect n ∧ Nat.Prime (n - 1) ∧ Nat.Prime (n + 1) → n = 6 :=
by sorry

end unique_perfect_between_primes_l1844_184442


namespace roots_sum_of_squares_reciprocals_l1844_184454

theorem roots_sum_of_squares_reciprocals (α : ℝ) :
  let f (x : ℝ) := x^2 + x * Real.sin α + 1
  let g (x : ℝ) := x^2 + x * Real.cos α - 1
  ∀ a b c d : ℝ,
    f a = 0 → f b = 0 → g c = 0 → g d = 0 →
    1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 = 1 := by
  sorry

end roots_sum_of_squares_reciprocals_l1844_184454


namespace cylinder_min_surface_area_l1844_184401

/-- For a cylindrical tank with volume V, the surface area (without a lid) is minimized when the radius and height are both equal to ∛(V/π) -/
theorem cylinder_min_surface_area (V : ℝ) (h : V > 0) :
  let surface_area (r h : ℝ) := π * r^2 + 2 * π * r * h
  let volume (r h : ℝ) := π * r^2 * h
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (r' h' : ℝ), r' > 0 → h' > 0 → volume r' h' = V → 
      surface_area r' h' ≥ surface_area r r) ∧
    r = (V / π)^(1/3) :=
by sorry

end cylinder_min_surface_area_l1844_184401


namespace total_rectangles_count_l1844_184437

/-- Represents the number of rectangles a single cell can form -/
structure CellRectangles where
  count : Nat

/-- Represents a group of cells with the same rectangle-forming property -/
structure CellGroup where
  cells : Nat
  rectangles : CellRectangles

/-- Calculates the total number of rectangles for a cell group -/
def totalRectangles (group : CellGroup) : Nat :=
  group.cells * group.rectangles.count

/-- The main theorem stating the total number of rectangles -/
theorem total_rectangles_count 
  (total_cells : Nat)
  (group1 : CellGroup)
  (group2 : CellGroup)
  (h1 : total_cells = group1.cells + group2.cells)
  (h2 : total_cells = 40)
  (h3 : group1.cells = 36)
  (h4 : group1.rectangles.count = 4)
  (h5 : group2.cells = 4)
  (h6 : group2.rectangles.count = 8) :
  totalRectangles group1 + totalRectangles group2 = 176 := by
  sorry

#check total_rectangles_count

end total_rectangles_count_l1844_184437


namespace sqrt_square_fourteen_l1844_184405

theorem sqrt_square_fourteen : Real.sqrt (14^2) = 14 := by
  sorry

end sqrt_square_fourteen_l1844_184405


namespace fixed_point_of_linear_function_l1844_184471

/-- The function f(x) = ax - 3 + 3 always passes through the point (3, 4) for any real number a. -/
theorem fixed_point_of_linear_function (a : ℝ) : 
  let f := λ x : ℝ => a * x - 3 + 3
  f 3 = 4 := by sorry

end fixed_point_of_linear_function_l1844_184471


namespace distribute_four_students_three_groups_l1844_184420

/-- The number of ways to distribute n distinct students into k distinct groups,
    where each student is in exactly one group and each group has at least one member. -/
def distribute_students (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 4 distinct students into 3 distinct groups,
    where each student is in exactly one group and each group has at least one member, is 36. -/
theorem distribute_four_students_three_groups :
  distribute_students 4 3 = 36 := by sorry

end distribute_four_students_three_groups_l1844_184420


namespace average_wage_is_21_l1844_184482

def male_workers : ℕ := 20
def female_workers : ℕ := 15
def child_workers : ℕ := 5

def male_wage : ℕ := 25
def female_wage : ℕ := 20
def child_wage : ℕ := 8

def total_workers : ℕ := male_workers + female_workers + child_workers

def total_wages : ℕ := male_workers * male_wage + female_workers * female_wage + child_workers * child_wage

theorem average_wage_is_21 : total_wages / total_workers = 21 := by
  sorry

end average_wage_is_21_l1844_184482


namespace minimum_gloves_needed_l1844_184460

theorem minimum_gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) : 
  participants = 82 → gloves_per_participant = 2 → participants * gloves_per_participant = 164 := by
sorry

end minimum_gloves_needed_l1844_184460


namespace intersection_perpendicular_line_l1844_184424

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem intersection_perpendicular_line :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧
    (∀ (x y : ℝ), result_line x y → 
      ((x - x₀) * 2 + (y - y₀) * 1 = 0)) ∧
    result_line x₀ y₀ :=
sorry

end intersection_perpendicular_line_l1844_184424


namespace cos_alpha_value_l1844_184486

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 6) = 3 / 5) : 
  Real.cos α = (3 * Real.sqrt 3 + 4) / 10 := by
  sorry

end cos_alpha_value_l1844_184486


namespace largest_number_with_digits_3_2_sum_11_l1844_184446

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digits_3_2_sum_11 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 11 → n ≤ 32222 :=
by sorry

end largest_number_with_digits_3_2_sum_11_l1844_184446


namespace quadratic_roots_condition_l1844_184493

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - Real.sqrt m * x₁ + 1 = 0 ∧ x₂^2 - Real.sqrt m * x₂ + 1 = 0) →
  m > 2 ∧
  ∃ m₀ : ℝ, m₀ > 2 ∧ ¬(∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - Real.sqrt m₀ * x₁ + 1 = 0 ∧ x₂^2 - Real.sqrt m₀ * x₂ + 1 = 0) :=
by sorry

end quadratic_roots_condition_l1844_184493


namespace choose_three_from_eight_l1844_184418

theorem choose_three_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by sorry

end choose_three_from_eight_l1844_184418


namespace least_positive_h_divisible_by_1999_l1844_184466

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 3
  | (n + 2) => 8 * sequence_a (n + 1) + 9 * sequence_a n + 16

def is_divisible_by_1999 (x : ℤ) : Prop := ∃ k : ℤ, x = 1999 * k

theorem least_positive_h_divisible_by_1999 :
  ∀ n : ℕ, is_divisible_by_1999 (sequence_a (n + 18) - sequence_a n) ∧
  ∀ h : ℕ, h > 0 ∧ h < 18 → ∃ m : ℕ, ¬is_divisible_by_1999 (sequence_a (m + h) - sequence_a m) :=
by sorry

end least_positive_h_divisible_by_1999_l1844_184466


namespace minimum_employment_age_is_25_l1844_184431

/-- The minimum age required to be employed at the company -/
def minimum_employment_age : ℕ := 25

/-- Jane's current age -/
def jane_current_age : ℕ := 28

/-- Years until Dara reaches minimum employment age -/
def years_until_dara_reaches_minimum_age : ℕ := 14

/-- Years until Dara is half Jane's age -/
def years_until_dara_half_jane_age : ℕ := 6

theorem minimum_employment_age_is_25 :
  minimum_employment_age = 25 :=
by sorry

end minimum_employment_age_is_25_l1844_184431


namespace average_age_of_first_and_fifth_dog_l1844_184402

def dog_ages (age1 age2 age3 age4 age5 : ℕ) : Prop :=
  age1 = 10 ∧
  age2 = age1 - 2 ∧
  age3 = age2 + 4 ∧
  age4 * 2 = age3 ∧
  age5 = age4 + 20

theorem average_age_of_first_and_fifth_dog (age1 age2 age3 age4 age5 : ℕ) :
  dog_ages age1 age2 age3 age4 age5 →
  (age1 + age5) / 2 = 18 :=
by sorry

end average_age_of_first_and_fifth_dog_l1844_184402
