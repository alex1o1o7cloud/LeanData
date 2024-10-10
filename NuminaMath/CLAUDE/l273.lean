import Mathlib

namespace ab_equals_six_l273_27331

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l273_27331


namespace tangent_line_and_critical_point_l273_27384

/-- The function f(x) = (1/2)x^2 - ax - ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a*x - Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x - a - 1/x

theorem tangent_line_and_critical_point (a : ℝ) (h : a ≥ 0) :
  /- The equation of the tangent line to f(x) at x=1 when a=1 is y = -x + 1/2 -/
  (let y : ℝ → ℝ := fun x ↦ -x + 1/2
   f 1 1 = y 1 ∧ f' 1 1 = -1) ∧
  /- For any critical point x₀ of f(x), f(x₀) ≤ 1/2 -/
  ∀ x₀ > 0, f' a x₀ = 0 → f a x₀ ≤ 1/2 := by
  sorry

end tangent_line_and_critical_point_l273_27384


namespace no_solution_implies_a_leq_3_l273_27396

theorem no_solution_implies_a_leq_3 (a : ℝ) : 
  (∀ x : ℝ, ¬(x ≥ 3 ∧ x < a)) → a ≤ 3 := by
sorry

end no_solution_implies_a_leq_3_l273_27396


namespace fraction_evaluation_l273_27324

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by
  sorry

end fraction_evaluation_l273_27324


namespace delivery_fee_calculation_delivery_fee_is_twenty_l273_27369

theorem delivery_fee_calculation (sandwich_price : ℝ) (num_sandwiches : ℕ) 
  (tip_percentage : ℝ) (total_received : ℝ) (delivery_fee : ℝ) : Prop :=
  sandwich_price = 5 →
  num_sandwiches = 18 →
  tip_percentage = 0.1 →
  total_received = 121 →
  delivery_fee = 20 →
  total_received = (sandwich_price * num_sandwiches) + delivery_fee + 
    (tip_percentage * (sandwich_price * num_sandwiches + delivery_fee))

-- Proof
theorem delivery_fee_is_twenty :
  ∃ (delivery_fee : ℝ),
    delivery_fee_calculation 5 18 0.1 121 delivery_fee :=
by
  sorry

end delivery_fee_calculation_delivery_fee_is_twenty_l273_27369


namespace max_a_bound_l273_27312

theorem max_a_bound (a : ℝ) : 
  (∀ x > 0, (x^2 + 1) * Real.exp x ≥ a * x^2) ↔ a ≤ 2 * Real.exp 1 := by
sorry

end max_a_bound_l273_27312


namespace equation_equality_l273_27352

theorem equation_equality : (3 * 6 * 9) / 3 = (2 * 6 * 9) / 2 := by
  sorry

end equation_equality_l273_27352


namespace goose_eggs_count_l273_27306

theorem goose_eggs_count (
  total_eggs : ℕ
  ) (
  hatched_ratio : Rat
  ) (
  first_month_survival_ratio : Rat
  ) (
  first_year_death_ratio : Rat
  ) (
  first_year_survivors : ℕ
  ) : total_eggs = 2200 :=
  by
  have h1 : hatched_ratio = 2 / 3 := by sorry
  have h2 : first_month_survival_ratio = 3 / 4 := by sorry
  have h3 : first_year_death_ratio = 3 / 5 := by sorry
  have h4 : first_year_survivors = 110 := by sorry
  have h5 : ∀ e, e ≤ 1 := by sorry  -- No more than one goose hatched from each egg
  
  sorry

end goose_eggs_count_l273_27306


namespace f_difference_l273_27323

/-- The function f defined as f(x) = 5x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 1

/-- Theorem stating that f(x + h) - f(x) = h(10x + 5h - 2) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (10 * x + 5 * h - 2) := by
  sorry

end f_difference_l273_27323


namespace find_d_when_a_b_c_equal_l273_27358

theorem find_d_when_a_b_c_equal (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 2 = d + 3 * Real.sqrt (a + b + c - d) →
  a = b →
  b = c →
  d = 5/4 := by
sorry

end find_d_when_a_b_c_equal_l273_27358


namespace parcel_delivery_growth_l273_27342

/-- Represents the equation for parcel delivery growth over three months -/
theorem parcel_delivery_growth 
  (initial_delivery : ℕ) 
  (total_delivery : ℕ) 
  (growth_rate : ℝ) : 
  initial_delivery = 20000 → 
  total_delivery = 72800 → 
  2 + 2 * (1 + growth_rate) + 2 * (1 + growth_rate)^2 = 7.28 := by
  sorry

#check parcel_delivery_growth

end parcel_delivery_growth_l273_27342


namespace vector_perpendicular_l273_27336

/-- Given plane vectors a, b, and c, where c is perpendicular to (a + b), prove that t = -6/5 -/
theorem vector_perpendicular (a b c : ℝ × ℝ) (t : ℝ) :
  a = (1, 2) →
  b = (3, 4) →
  c = (t, t + 2) →
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) →
  t = -6/5 := by
  sorry

end vector_perpendicular_l273_27336


namespace count_integers_satisfying_inequality_l273_27390

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℤ), (∀ n ∈ S, (Real.sqrt (n + 1) ≤ Real.sqrt (3 * n + 2) ∧ 
    Real.sqrt (3 * n + 2) < Real.sqrt (2 * n + 7))) ∧ 
    S.card = 5 :=
by sorry

end count_integers_satisfying_inequality_l273_27390


namespace ascending_order_real_numbers_l273_27346

theorem ascending_order_real_numbers : -6 < (0 : ℝ) ∧ 0 < Real.sqrt 5 ∧ Real.sqrt 5 < Real.pi := by
  sorry

end ascending_order_real_numbers_l273_27346


namespace julia_grocery_purchase_l273_27318

/-- Represents the cost of items and the total bill for Julia's grocery purchase. -/
def grocery_bill (snickers_cost : ℚ) : ℚ :=
  let mms_cost := 2 * snickers_cost
  let pepsi_cost := 2 * mms_cost
  let bread_cost := 3 * pepsi_cost
  2 * snickers_cost + 3 * mms_cost + 4 * pepsi_cost + 5 * bread_cost

/-- Theorem stating the total cost of Julia's purchase and the additional amount she needs to pay. -/
theorem julia_grocery_purchase (snickers_cost : ℚ) (h : snickers_cost = 3/2) :
  grocery_bill snickers_cost = 126 ∧ grocery_bill snickers_cost - 100 = 26 := by
  sorry

#eval grocery_bill (3/2)

end julia_grocery_purchase_l273_27318


namespace square_difference_l273_27366

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l273_27366


namespace circumradius_of_special_triangle_l273_27322

/-- Given a triangle ABC with side lengths proportional to 7:5:3 and area 45√3,
    prove that the radius of its circumscribed circle is 14. -/
theorem circumradius_of_special_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a / b = 7 / 5 →
  b / c = 5 / 3 →
  (1 / 2) * a * b * Real.sin C = 45 * Real.sqrt 3 →
  R = (a / (2 * Real.sin A)) →
  R = 14 := by
  sorry

end circumradius_of_special_triangle_l273_27322


namespace circle_condition_l273_27344

/-- Represents a quadratic equation in two variables -/
structure QuadraticEquation :=
  (a b c d e f : ℝ)

/-- Checks if a QuadraticEquation represents a circle -/
def isCircle (eq : QuadraticEquation) : Prop :=
  eq.a = eq.b ∧ eq.d^2 + eq.e^2 - 4*eq.a*eq.f > 0

/-- The equation m^2x^2 + (m+2)y^2 + 2mx + m = 0 -/
def equation (m : ℝ) : QuadraticEquation :=
  ⟨m^2, m+2, 0, 2*m, 0, m⟩

/-- Theorem: The equation represents a circle if and only if m = -1 -/
theorem circle_condition :
  ∀ m : ℝ, isCircle (equation m) ↔ m = -1 :=
sorry

end circle_condition_l273_27344


namespace max_b_value_l273_27387

/-- The volume of the box -/
def box_volume : ℕ := 360

/-- Theorem stating the maximum possible value of b given the conditions -/
theorem max_b_value (a b c : ℕ) 
  (vol_eq : a * b * c = box_volume)
  (int_cond : 1 < c ∧ c < b ∧ b < a) : 
  b ≤ 10 := by
  sorry

end max_b_value_l273_27387


namespace vowel_initial_probability_is_7_26_l273_27354

/-- The set of all letters in the alphabet -/
def Alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

/-- The set of vowels, including Y and W -/
def Vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y', 'W'}

/-- A student's initials, represented by a single character -/
structure Initial where
  letter : Char
  letter_in_alphabet : letter ∈ Alphabet

/-- The class of students -/
def ClassInitials : Finset Initial := sorry

/-- The number of students in the class -/
axiom class_size : ClassInitials.card = 26

/-- All initials in the class are unique -/
axiom initials_unique : ∀ i j : Initial, i ∈ ClassInitials → j ∈ ClassInitials → i = j → i.letter = j.letter

/-- The probability of selecting a student with vowel initials -/
def vowel_initial_probability : ℚ :=
  (ClassInitials.filter (fun i => i.letter ∈ Vowels)).card / ClassInitials.card

/-- The main theorem: probability of selecting a student with vowel initials is 7/26 -/
theorem vowel_initial_probability_is_7_26 : vowel_initial_probability = 7 / 26 := by
  sorry

end vowel_initial_probability_is_7_26_l273_27354


namespace max_value_f_positive_three_distinct_roots_condition_l273_27305

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x > 0 then 1 - x^2 * Real.log x else Real.exp (-x - 2)

-- Part 1: Maximum value of f(x) for x > 0
theorem max_value_f_positive (x : ℝ) (h : x > 0) :
  f x ≤ 1 + 1 / (2 * Real.exp 1) :=
sorry

-- Part 2: Condition for three distinct real roots
theorem three_distinct_roots_condition (a b : ℝ) (h : a ≥ 0) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x + a * x^2 + b * x = 0 ∧
    f y + a * y^2 + b * y = 0 ∧
    f z + a * z^2 + b * z = 0) ↔
  b < -2 * Real.sqrt 2 ∨ b ≥ 1 / Real.exp 1 :=
sorry

end max_value_f_positive_three_distinct_roots_condition_l273_27305


namespace r_value_when_n_is_3_l273_27350

theorem r_value_when_n_is_3 : 
  let n : ℕ := 3
  let s := 2^n + 2
  let r := 4^s - 2*s
  r = 1048556 := by
sorry

end r_value_when_n_is_3_l273_27350


namespace dog_max_distance_dog_max_distance_is_22_l273_27326

/-- The maximum distance a dog can reach from the origin when secured at (6,8) with a 12-foot rope -/
theorem dog_max_distance : ℝ :=
  let dog_position : ℝ × ℝ := (6, 8)
  let rope_length : ℝ := 12
  let origin : ℝ × ℝ := (0, 0)
  let distance_to_origin : ℝ := Real.sqrt ((dog_position.1 - origin.1)^2 + (dog_position.2 - origin.2)^2)
  distance_to_origin + rope_length

theorem dog_max_distance_is_22 : dog_max_distance = 22 := by
  sorry

end dog_max_distance_dog_max_distance_is_22_l273_27326


namespace saras_house_difference_l273_27394

theorem saras_house_difference (sara_house : ℕ) (nada_house : ℕ) : 
  sara_house = 1000 → nada_house = 450 → sara_house - 2 * nada_house = 100 := by
  sorry

end saras_house_difference_l273_27394


namespace alpha_third_range_l273_27339

open Real Set

theorem alpha_third_range (α : ℝ) (h1 : sin α > 0) (h2 : cos α < 0) (h3 : sin (α/3) > cos (α/3)) :
  ∃ k : ℤ, α/3 ∈ (Set.Ioo (2*k*π + π/4) (2*k*π + π/3)) ∪ (Set.Ioo (2*k*π + 5*π/6) (2*k*π + π)) :=
sorry

end alpha_third_range_l273_27339


namespace smallest_number_of_eggs_l273_27375

theorem smallest_number_of_eggs (total_eggs : ℕ) (num_containers : ℕ) : 
  total_eggs > 150 →
  total_eggs = 12 * num_containers - 3 →
  (∀ n : ℕ, n < num_containers → 12 * n - 3 ≤ 150) →
  total_eggs = 153 := by
sorry

end smallest_number_of_eggs_l273_27375


namespace perpendicular_vectors_l273_27362

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c,
    then the first component of c is -3. -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) :
  a.1 = Real.sqrt 3 →
  a.2 = 1 →
  b.1 = 0 →
  b.2 = 1 →
  c.2 = Real.sqrt 3 →
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 →
  c.1 = -3 := by sorry

end perpendicular_vectors_l273_27362


namespace bank_balance_after_five_years_l273_27382

/-- Calculates the compound interest for a given principal, rate, and time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Represents the bank account balance after each year -/
def bankBalance : ℕ → ℝ
  | 0 => 5600
  | 1 => compoundInterest 5600 0.03 1
  | 2 => compoundInterest (bankBalance 1) 0.035 1
  | 3 => compoundInterest (bankBalance 2 + 2000) 0.04 1
  | 4 => compoundInterest (bankBalance 3) 0.045 1
  | 5 => compoundInterest (bankBalance 4) 0.05 1
  | _ => 0  -- For years beyond 5, return 0

theorem bank_balance_after_five_years :
  bankBalance 5 = 9094.20 := by
  sorry


end bank_balance_after_five_years_l273_27382


namespace unique_modular_congruence_l273_27398

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 100000 [ZMOD 9] ∧ n = 1 := by
  sorry

end unique_modular_congruence_l273_27398


namespace right_triangle_area_l273_27319

/-- The area of a right triangle with one leg measuring 6 and hypotenuse measuring 10 is 24. -/
theorem right_triangle_area : ∀ (a b c : ℝ), 
  a = 6 →
  c = 10 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 24 :=
by
  sorry

end right_triangle_area_l273_27319


namespace cookies_per_box_l273_27393

/-- Proof of the number of cookies per box in Brenda's banana pudding problem -/
theorem cookies_per_box 
  (num_trays : ℕ) 
  (cookies_per_tray : ℕ) 
  (cost_per_box : ℚ) 
  (total_cost : ℚ) 
  (h1 : num_trays = 3)
  (h2 : cookies_per_tray = 80)
  (h3 : cost_per_box = 7/2)
  (h4 : total_cost = 14) :
  (num_trays * cookies_per_tray) / (total_cost / cost_per_box) = 60 := by
  sorry

#eval (3 * 80) / (14 / (7/2)) -- Should evaluate to 60

end cookies_per_box_l273_27393


namespace square_side_length_l273_27316

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 81) (h2 : area = side ^ 2) :
  side = 9 := by
  sorry

end square_side_length_l273_27316


namespace video_game_earnings_l273_27364

/-- Given the conditions of Mike's video game selling scenario, prove the total earnings. -/
theorem video_game_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : 
  total_games = 16 → non_working_games = 8 → price_per_game = 7 → 
  (total_games - non_working_games) * price_per_game = 56 := by
  sorry

end video_game_earnings_l273_27364


namespace solution_set_inequality_l273_27399

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end solution_set_inequality_l273_27399


namespace intersection_distance_squared_l273_27372

/-- Given two circles in a 2D plane, one centered at (1,1) with radius 5 
and another centered at (1,-8) with radius √26, this theorem states that 
the square of the distance between their intersection points is 3128/81. -/
theorem intersection_distance_squared : 
  ∃ (C D : ℝ × ℝ), 
    ((C.1 - 1)^2 + (C.2 - 1)^2 = 25) ∧ 
    ((D.1 - 1)^2 + (D.2 - 1)^2 = 25) ∧
    ((C.1 - 1)^2 + (C.2 + 8)^2 = 26) ∧ 
    ((D.1 - 1)^2 + (D.2 + 8)^2 = 26) ∧
    ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 3128 / 81) :=
by sorry


end intersection_distance_squared_l273_27372


namespace largest_three_digit_product_l273_27300

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_three_digit_product (n x y : ℕ) :
  (100 ≤ n ∧ n < 1000) →
  (x < 10 ∧ y < 10) →
  isPrime x →
  isPrime (10 * x + y) →
  n = x * (10 * x + y) →
  n ≤ 553 :=
by sorry

end largest_three_digit_product_l273_27300


namespace ratio_odd_even_divisors_l273_27392

def M : ℕ := 18 * 18 * 125 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 14 = sum_even_divisors M := by sorry

end ratio_odd_even_divisors_l273_27392


namespace stating_head_start_for_tie_l273_27357

/-- Represents the race scenario -/
structure RaceScenario where
  course_length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- 
Calculates whether the race ends in a tie given a RaceScenario
-/
def is_tie (scenario : RaceScenario) : Prop :=
  scenario.course_length / scenario.speed_ratio = 
  (scenario.course_length - scenario.head_start)

/-- 
Theorem stating that for a 84-meter course where A is 4 times faster than B,
a 63-meter head start results in a tie
-/
theorem head_start_for_tie : 
  let scenario : RaceScenario := {
    course_length := 84,
    speed_ratio := 4,
    head_start := 63
  }
  is_tie scenario := by sorry

end stating_head_start_for_tie_l273_27357


namespace total_wheels_is_47_l273_27376

/-- The total number of wheels in Jordan's neighborhood -/
def total_wheels : ℕ :=
  let jordans_driveway := 
    2 * 4 + -- Two cars with 4 wheels each
    1 +     -- One car has a spare wheel
    3 * 2 + -- Three bikes with 2 wheels each
    1 +     -- One bike missing a rear wheel
    3 +     -- One bike with 2 main wheels and one training wheel
    2 +     -- Trash can with 2 wheels
    3 +     -- Tricycle with 3 wheels
    4 +     -- Wheelchair with 2 main wheels and 2 small front wheels
    4 +     -- Wagon with 4 wheels
    3       -- Pair of old roller skates with 3 wheels (one missing)
  let neighbors_driveway :=
    4 +     -- Pickup truck with 4 wheels
    2 +     -- Boat trailer with 2 wheels
    2 +     -- Motorcycle with 2 wheels
    4       -- ATV with 4 wheels
  jordans_driveway + neighbors_driveway

theorem total_wheels_is_47 : total_wheels = 47 := by
  sorry

end total_wheels_is_47_l273_27376


namespace total_volume_calculation_l273_27363

-- Define the dimensions of the rectangular parallelepiped
def box_length : ℝ := 2
def box_width : ℝ := 3
def box_height : ℝ := 4

-- Define the radius of half-spheres and cylinders
def sphere_radius : ℝ := 1
def cylinder_radius : ℝ := 1

-- Define the number of vertices and edges
def num_vertices : ℕ := 8
def num_edges : ℕ := 12

-- Theorem statement
theorem total_volume_calculation :
  let box_volume := box_length * box_width * box_height
  let half_sphere_volume := (num_vertices : ℝ) * (1/2) * (4/3) * Real.pi * sphere_radius^3
  let cylinder_volume := Real.pi * cylinder_radius^2 * 
    (2 * box_length + 2 * box_width + 2 * box_height)
  let total_volume := box_volume + half_sphere_volume + cylinder_volume
  total_volume = (72 + 112 * Real.pi) / 3 :=
by
  sorry

end total_volume_calculation_l273_27363


namespace print_statement_output_l273_27373

def print_output (a : ℕ) : String := s!"a={a}"

theorem print_statement_output (a : ℕ) (h : a = 10) : print_output a = "a=10" := by
  sorry

end print_statement_output_l273_27373


namespace students_history_or_geography_not_both_l273_27388

/-- The number of students taking both history and geography -/
def both : ℕ := 15

/-- The total number of students taking history -/
def history : ℕ := 30

/-- The number of students taking only geography -/
def only_geography : ℕ := 18

/-- Theorem: The number of students taking history or geography but not both is 33 -/
theorem students_history_or_geography_not_both : 
  (history - both) + only_geography = 33 := by sorry

end students_history_or_geography_not_both_l273_27388


namespace sandcastle_problem_l273_27349

theorem sandcastle_problem (mark_castles : ℕ) : 
  (mark_castles * 10 + mark_castles) +  -- Mark's castles and towers
  ((3 * mark_castles) * 5 + (3 * mark_castles)) = 580 -- Jeff's castles and towers
  → mark_castles = 20 := by
  sorry

end sandcastle_problem_l273_27349


namespace line_segment_params_sum_of_squares_l273_27361

/-- Given two points in 2D space, this function returns the parameters of the line segment connecting them. -/
def lineSegmentParams (p1 p2 : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem line_segment_params_sum_of_squares :
  let p1 : ℝ × ℝ := (-3, 6)
  let p2 : ℝ × ℝ := (4, 14)
  let (a, b, c, d) := lineSegmentParams p1 p2
  a^2 + b^2 + c^2 + d^2 = 158 := by sorry

end line_segment_params_sum_of_squares_l273_27361


namespace exists_term_with_nine_l273_27380

/-- An arithmetic progression with natural number first term and common difference -/
structure ArithmeticProgression :=
  (first_term : ℕ)
  (common_difference : ℕ)

/-- Function to check if a natural number contains the digit 9 -/
def contains_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10^k) % 10 = 9

/-- Theorem stating that there exists a term in the arithmetic progression containing the digit 9 -/
theorem exists_term_with_nine (ap : ArithmeticProgression) :
  ∃ n : ℕ, contains_nine (ap.first_term + n * ap.common_difference) :=
sorry

end exists_term_with_nine_l273_27380


namespace base7_subtraction_l273_27308

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- The statement to be proved -/
theorem base7_subtraction :
  let a := base7ToDecimal [2, 5, 3, 4]
  let b := base7ToDecimal [1, 4, 6, 6]
  decimalToBase7 (a - b) = [1, 0, 6, 5] := by
  sorry

end base7_subtraction_l273_27308


namespace cone_base_radius_l273_27309

/-- Given a semicircle with radius 6 cm forming the lateral surface of a cone,
    prove that the radius of the base circle of the cone is 3 cm. -/
theorem cone_base_radius (r : ℝ) (h : r = 6) : 
  2 * π * r / 2 = 2 * π * 3 := by sorry

end cone_base_radius_l273_27309


namespace intersection_A_B_l273_27359

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {-1, 1, 2, 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end intersection_A_B_l273_27359


namespace louise_wallet_amount_l273_27368

/-- The amount of money in Louise's wallet --/
def wallet_amount : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_toys, toy_price, num_bears, bear_price =>
    num_toys * toy_price + num_bears * bear_price

/-- Theorem stating the amount in Louise's wallet --/
theorem louise_wallet_amount :
  wallet_amount 28 10 20 15 = 580 := by
  sorry

end louise_wallet_amount_l273_27368


namespace students_playing_sport_b_l273_27337

/-- Given that there are 6 students playing sport A, and the number of students
    playing sport B is 4 times the number of students playing sport A,
    prove that 24 students play sport B. -/
theorem students_playing_sport_b (students_a : ℕ) (students_b : ℕ) : 
  students_a = 6 →
  students_b = 4 * students_a →
  students_b = 24 := by
  sorry

end students_playing_sport_b_l273_27337


namespace max_second_term_is_9_l273_27313

/-- An arithmetic sequence of three positive integers with sum 27 -/
structure ArithSeq27 where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_27 : a + (a + d) + (a + 2*d) = 27

/-- The second term of an arithmetic sequence -/
def second_term (seq : ArithSeq27) : ℕ := seq.a + seq.d

/-- Theorem: The maximum value of the second term in any ArithSeq27 is 9 -/
theorem max_second_term_is_9 : 
  ∀ seq : ArithSeq27, second_term seq ≤ 9 ∧ ∃ seq : ArithSeq27, second_term seq = 9 := by
  sorry

#check max_second_term_is_9

end max_second_term_is_9_l273_27313


namespace cylinder_radius_problem_l273_27377

/-- Given a cylinder with height 5 inches and radius r, 
    if increasing the radius by 4 inches or increasing the height by 4 inches 
    results in the same volume, then r = 5 + 3√5 -/
theorem cylinder_radius_problem (r : ℝ) : 
  (π * (r + 4)^2 * 5 = π * r^2 * 9) → r = 5 + 3 * Real.sqrt 5 := by
  sorry

end cylinder_radius_problem_l273_27377


namespace inverse_log_property_l273_27397

noncomputable section

variable (a : ℝ)
variable (a_pos : a > 0)
variable (a_ne_one : a ≠ 1)

def f (x : ℝ) := Real.log x / Real.log a

def f_inverse (x : ℝ) := a ^ x

theorem inverse_log_property (h : f_inverse a 2 = 9) : f a 9 + f a 6 = 2 := by
  sorry

#check inverse_log_property

end inverse_log_property_l273_27397


namespace series_convergence_l273_27391

/-- The series ∑(n=1 to ∞) [x^(2n-1) / ((n^2 + 1) * 3^n)] converges absolutely if and only if -√3 ≤ x ≤ √3 -/
theorem series_convergence (x : ℝ) : 
  (∑' n, (x^(2*n-1) / ((n^2 + 1) * 3^n))) ≠ 0 ↔ -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3 := by
  sorry

end series_convergence_l273_27391


namespace equation_proof_l273_27327

theorem equation_proof (x : ℝ) (h : x = 12) : (17.28 / x) / (3.6 * 0.2) = 2 := by
  sorry

end equation_proof_l273_27327


namespace system_of_equations_solution_l273_27370

theorem system_of_equations_solution (a b c : ℝ) :
  ∃ (x y z : ℝ), 
    (x + y + 2*z = a) ∧ 
    (x + 2*y + z = b) ∧ 
    (2*x + y + z = c) ∧
    (x = (3*c - a - b) / 4) ∧
    (y = (3*b - a - c) / 4) ∧
    (z = (3*a - b - c) / 4) := by
  sorry

end system_of_equations_solution_l273_27370


namespace expression_simplification_l273_27301

theorem expression_simplification (x y : ℝ) (h : x^2 ≠ y^2) :
  ((x^2 + y^2) / (x^2 - y^2)) + ((x^2 - y^2) / (x^2 + y^2)) = 2*(x^4 + y^4) / (x^4 - y^4) := by
  sorry

end expression_simplification_l273_27301


namespace age_difference_l273_27374

/-- Represents a person's age at different points in time -/
structure AgeRelation where
  current : ℕ
  future : ℕ

/-- The age relation between two people A and B -/
def age_relation (a b : AgeRelation) : Prop :=
  a.current - b.current = b.current - 10 ∧
  a.current - b.current = 25 - a.future

theorem age_difference (a b : AgeRelation) 
  (h : age_relation a b) : a.current - b.current = 5 := by
  sorry

end age_difference_l273_27374


namespace triangle_angles_l273_27389

theorem triangle_angles (a b c : ℝ) (ha : a = 3) (hb : b = Real.sqrt 11) (hc : c = 2 + Real.sqrt 5) :
  ∃ (A B C : ℝ), 
    (0 < A ∧ A < π) ∧ 
    (0 < B ∧ B < π) ∧ 
    (0 < C ∧ C < π) ∧ 
    A + B + C = π ∧
    B = C ∧
    A = π - 2*B := by
  sorry

end triangle_angles_l273_27389


namespace ten_thousandths_digit_of_seven_thirty_seconds_l273_27340

theorem ten_thousandths_digit_of_seven_thirty_seconds (x : ℚ) : 
  x = 7 / 32 → (x * 10000).floor % 10 = 7 := by
  sorry

end ten_thousandths_digit_of_seven_thirty_seconds_l273_27340


namespace marys_weight_l273_27351

-- Define the weights as real numbers
variable (mary_weight : ℝ)
variable (john_weight : ℝ)
variable (jamison_weight : ℝ)

-- Define the conditions
axiom john_weight_relation : john_weight = mary_weight + (1/4 * mary_weight)
axiom mary_jamison_relation : mary_weight = jamison_weight - 20
axiom total_weight : mary_weight + john_weight + jamison_weight = 540

-- Theorem to prove
theorem marys_weight : mary_weight = 160 := by
  sorry

end marys_weight_l273_27351


namespace odd_factorial_product_equals_sum_factorial_l273_27347

def oddFactorialProduct (m : ℕ) : ℕ := (List.range m).foldl (λ acc i => acc * Nat.factorial (2 * i + 1)) 1

def sumFirstNaturals (m : ℕ) : ℕ := m * (m + 1) / 2

theorem odd_factorial_product_equals_sum_factorial (m : ℕ) :
  oddFactorialProduct m = Nat.factorial (sumFirstNaturals m) ↔ m = 1 ∨ m = 2 ∨ m = 3 := by
  sorry

end odd_factorial_product_equals_sum_factorial_l273_27347


namespace root_of_equation_l273_27311

def combination (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def permutation (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem root_of_equation : ∃ (x : ℕ), 
  x > 6 ∧ 3 * (combination (x - 3) 4) = 5 * (permutation (x - 4) 2) ∧ x = 11 := by sorry

end root_of_equation_l273_27311


namespace square_root_special_form_l273_27343

theorem square_root_special_form :
  ∀ n : ℕ, 10 ≤ n ∧ n < 100 →
    (∃ a b : ℕ, n = 10 * a + b ∧ Real.sqrt n = a + Real.sqrt b) ↔
    (n = 64 ∨ n = 81) := by
  sorry

end square_root_special_form_l273_27343


namespace residue_mod_32_l273_27378

theorem residue_mod_32 : Int.mod (-1277) 32 = 3 := by
  sorry

end residue_mod_32_l273_27378


namespace parallelogram_with_inscribed_circle_is_rhombus_l273_27355

/-- A parallelogram is a quadrilateral with opposite sides parallel and equal. -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : sorry

/-- A circle is inscribed in a quadrilateral if it touches all four sides. -/
def has_inscribed_circle (p : Parallelogram) : Prop := sorry

/-- A rhombus is a parallelogram with all sides equal. -/
def is_rhombus (p : Parallelogram) : Prop := sorry

/-- Theorem: If a circle can be inscribed in a parallelogram, then the parallelogram is a rhombus. -/
theorem parallelogram_with_inscribed_circle_is_rhombus (p : Parallelogram) :
  has_inscribed_circle p → is_rhombus p := by
  sorry

end parallelogram_with_inscribed_circle_is_rhombus_l273_27355


namespace scrunchies_to_barrettes_ratio_l273_27383

/-- Represents the number of hair decorations Annie has --/
structure HairDecorations where
  barrettes : ℕ
  scrunchies : ℕ
  bobby_pins : ℕ

/-- Calculates the percentage of bobby pins in the total hair decorations --/
def bobby_pin_percentage (hd : HairDecorations) : ℚ :=
  (hd.bobby_pins : ℚ) / ((hd.barrettes + hd.scrunchies + hd.bobby_pins) : ℚ) * 100

/-- Theorem stating the ratio of scrunchies to barrettes --/
theorem scrunchies_to_barrettes_ratio (hd : HairDecorations) :
  hd.barrettes = 6 →
  hd.bobby_pins = hd.barrettes - 3 →
  bobby_pin_percentage hd = 14 →
  (hd.scrunchies : ℚ) / (hd.barrettes : ℚ) = 2 := by
  sorry

#check scrunchies_to_barrettes_ratio

end scrunchies_to_barrettes_ratio_l273_27383


namespace sequence_formula_l273_27317

/-- Given a sequence {a_n} where the sum of the first n terms S_n = 2^n - 1,
    prove that the general formula for the sequence is a_n = 2^(n-1) -/
theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 2^n - 1) : 
    ∀ n : ℕ, a n = 2^(n-1) := by
  sorry

end sequence_formula_l273_27317


namespace smallest_AC_l273_27303

/-- Represents a right triangle ABC with a point D on AC -/
structure RightTriangleWithPoint where
  AC : ℕ  -- Length of AC
  CD : ℕ  -- Length of CD
  bd_squared : ℕ  -- Square of length BD

/-- Defines the conditions for the right triangle and point D -/
def valid_triangle (t : RightTriangleWithPoint) : Prop :=
  t.AC > 0 ∧ t.CD > 0 ∧ t.CD < t.AC ∧ t.bd_squared = 36 ∧
  2 * t.AC * t.CD = t.CD * t.CD + t.bd_squared

/-- Theorem: The smallest possible value of AC is 6 -/
theorem smallest_AC :
  ∀ t : RightTriangleWithPoint, valid_triangle t → t.AC ≥ 6 :=
sorry

end smallest_AC_l273_27303


namespace trig_identity_proof_l273_27381

theorem trig_identity_proof (α : Real) (h1 : 0 < α) (h2 : α < π) (h3 : -Real.sin α = 2 * Real.cos α) :
  2 * (Real.sin α)^2 - Real.sin α * Real.cos α + (Real.cos α)^2 = 11/5 := by
  sorry

end trig_identity_proof_l273_27381


namespace power_of_power_l273_27321

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end power_of_power_l273_27321


namespace bank_a_investment_l273_27386

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  total_investment : ℝ
  bank_a_rate : ℝ
  bank_b_rate : ℝ
  bank_b_fee : ℝ
  years : ℕ
  final_amount : ℝ

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating the correct amount invested in Bank A -/
theorem bank_a_investment (scenario : InvestmentScenario) 
  (h1 : scenario.total_investment = 2000)
  (h2 : scenario.bank_a_rate = 0.04)
  (h3 : scenario.bank_b_rate = 0.06)
  (h4 : scenario.bank_b_fee = 50)
  (h5 : scenario.years = 3)
  (h6 : scenario.final_amount = 2430) :
  ∃ (bank_a_amount : ℝ),
    bank_a_amount = 1625 ∧
    compound_interest bank_a_amount scenario.bank_a_rate scenario.years +
    compound_interest (scenario.total_investment - scenario.bank_b_fee - bank_a_amount) scenario.bank_b_rate scenario.years =
    scenario.final_amount :=
  sorry

end bank_a_investment_l273_27386


namespace apollo_chariot_cost_l273_27348

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months before the price increase -/
def months_before_increase : ℕ := 6

/-- The initial price in golden apples -/
def initial_price : ℕ := 3

/-- The price increase factor -/
def price_increase_factor : ℕ := 2

/-- The total cost of chariot wheels for Apollo in golden apples for a year -/
def total_cost : ℕ := 
  (months_before_increase * initial_price) + 
  ((months_in_year - months_before_increase) * (initial_price * price_increase_factor))

/-- Theorem stating that the total cost for Apollo is 54 golden apples -/
theorem apollo_chariot_cost : total_cost = 54 := by
  sorry

end apollo_chariot_cost_l273_27348


namespace savings_after_twelve_months_l273_27329

/-- Represents the electricity pricing and consumption data for a user. -/
structure ElectricityData where
  originalPrice : ℚ
  valleyPrice : ℚ
  peakPrice : ℚ
  installationFee : ℚ
  monthlyConsumption : ℚ
  valleyConsumption : ℚ
  peakConsumption : ℚ
  months : ℕ

/-- Calculates the total savings after a given number of months for a user
    who has installed a peak-valley meter. -/
def totalSavings (data : ElectricityData) : ℚ :=
  let monthlyOriginalCost := data.monthlyConsumption * data.originalPrice
  let monthlyNewCost := data.valleyConsumption * data.valleyPrice + data.peakConsumption * data.peakPrice
  let monthlySavings := monthlyOriginalCost - monthlyNewCost
  let totalSavingsBeforeFee := monthlySavings * data.months
  totalSavingsBeforeFee - data.installationFee

/-- The main theorem stating that the total savings after 12 months is 236 yuan. -/
theorem savings_after_twelve_months :
  let data : ElectricityData := {
    originalPrice := 56/100,
    valleyPrice := 28/100,
    peakPrice := 56/100,
    installationFee := 100,
    monthlyConsumption := 200,
    valleyConsumption := 100,
    peakConsumption := 100,
    months := 12
  }
  totalSavings data = 236 := by sorry

end savings_after_twelve_months_l273_27329


namespace sin_negative_690_degrees_l273_27367

theorem sin_negative_690_degrees : Real.sin ((-690 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end sin_negative_690_degrees_l273_27367


namespace kaleb_shirts_l273_27379

-- Define the initial number of shirts
def initial_shirts : ℕ := 17

-- Define the number of shirts Kaleb would have after getting rid of 7
def remaining_shirts : ℕ := 10

-- Define the number of shirts Kaleb got rid of
def removed_shirts : ℕ := 7

-- Theorem to prove
theorem kaleb_shirts : initial_shirts = remaining_shirts + removed_shirts :=
by sorry

end kaleb_shirts_l273_27379


namespace gcd_factorial_nine_eleven_l273_27356

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_nine_eleven : 
  Nat.gcd (factorial 9) (factorial 11) = factorial 9 := by
  sorry

end gcd_factorial_nine_eleven_l273_27356


namespace triangle_area_ratio_specific_triangle_area_ratio_l273_27330

/-- The ratio of the areas of two triangles with the same base and different heights -/
theorem triangle_area_ratio (base : ℝ) (height1 height2 : ℝ) :
  base > 0 → height1 > 0 → height2 > 0 →
  (base * height1 / 2) / (base * height2 / 2) = height1 / height2 := by
  sorry

/-- The specific ratio of triangle areas for the given problem -/
theorem specific_triangle_area_ratio :
  let base := 3
  let height1 := 6.02
  let height2 := 2
  (base * height1 / 2) / (base * height2 / 2) = 3.01 := by
  sorry

end triangle_area_ratio_specific_triangle_area_ratio_l273_27330


namespace max_value_product_l273_27334

theorem max_value_product (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27/8 ∧
  ∃ x y z, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧
    (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) = 27/8 :=
by sorry

end max_value_product_l273_27334


namespace linear_function_composition_l273_27332

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k b : ℝ), k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 3) →
  (∀ x, f x = 2 * x + 1) ∨ (∀ x, f x = -2 * x - 3) :=
sorry

end linear_function_composition_l273_27332


namespace product_of_fractions_l273_27328

theorem product_of_fractions : (1 : ℚ) / 3 * 3 / 5 * 5 / 7 = 1 / 7 := by
  sorry

end product_of_fractions_l273_27328


namespace integer_root_of_special_polynomial_l273_27353

/-- Given a polynomial with integer coefficients of the form
    x^4 + b_3*x^3 + b_2*x^2 + b_1*x + 50,
    if s is an integer root of this polynomial and s^3 divides 50,
    then s = 1 or s = -1 -/
theorem integer_root_of_special_polynomial (b₃ b₂ b₁ s : ℤ) :
  (s^4 + b₃*s^3 + b₂*s^2 + b₁*s + 50 = 0) →
  (s^3 ∣ 50) →
  (s = 1 ∨ s = -1) :=
by sorry

end integer_root_of_special_polynomial_l273_27353


namespace min_value_squared_sum_l273_27325

theorem min_value_squared_sum (a b c : ℝ) (h : 2*a + 2*b + c = 8) :
  (a - 1)^2 + (b + 2)^2 + (c - 3)^2 ≥ 49/9 := by
  sorry

end min_value_squared_sum_l273_27325


namespace negation_of_universal_proposition_l273_27320

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 < 0) := by
  sorry

end negation_of_universal_proposition_l273_27320


namespace charles_learning_time_l273_27333

/-- The number of days it takes to learn one vowel, given the total days and number of vowels -/
def days_per_vowel (total_days : ℕ) (num_vowels : ℕ) : ℕ :=
  total_days / num_vowels

/-- Theorem stating that it takes 7 days to learn one vowel -/
theorem charles_learning_time :
  days_per_vowel 35 5 = 7 := by
  sorry

end charles_learning_time_l273_27333


namespace max_sales_price_l273_27385

/-- Represents the sales function for a product -/
def sales_function (x : ℝ) : ℝ := 400 - 20 * (x - 30)

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ := (x - 20) * (sales_function x)

/-- The unit purchase price of the product -/
def purchase_price : ℝ := 20

/-- The initial selling price of the product -/
def initial_price : ℝ := 30

/-- The initial sales volume in half a month -/
def initial_volume : ℝ := 400

/-- The price-volume relationship: change in volume per unit price increase -/
def price_volume_ratio : ℝ := -20

theorem max_sales_price : 
  ∃ (x : ℝ), x = 35 ∧ 
  ∀ (y : ℝ), profit_function y ≤ profit_function x :=
by sorry

end max_sales_price_l273_27385


namespace wine_barrels_l273_27360

theorem wine_barrels (a b : ℝ) : 
  (a + 8 = b) ∧ (b + 3 = 3 * (a - 3)) → a = 10 ∧ b = 18 := by
  sorry

end wine_barrels_l273_27360


namespace dog_tail_length_l273_27365

/-- Represents the length of a dog's body parts in inches -/
structure DogMeasurements where
  overall : ℝ
  body : ℝ
  head : ℝ
  tail : ℝ

/-- Theorem stating the length of a dog's tail given specific proportions -/
theorem dog_tail_length (d : DogMeasurements) 
  (h1 : d.overall = 30)
  (h2 : d.tail = d.body / 2)
  (h3 : d.head = d.body / 6)
  (h4 : d.overall = d.head + d.body + d.tail) : 
  d.tail = 6 := by
  sorry

#check dog_tail_length

end dog_tail_length_l273_27365


namespace tshirt_pricing_theorem_l273_27395

/-- Represents the cost and pricing information for two batches of T-shirts --/
structure TShirtBatches where
  first_batch_cost : ℕ
  second_batch_cost : ℕ
  quantity_ratio : ℚ
  price_difference : ℕ
  first_batch_selling_price : ℕ
  min_total_profit : ℕ

/-- Calculates the cost price of each T-shirt in the first batch --/
def cost_price_first_batch (b : TShirtBatches) : ℚ :=
  sorry

/-- Calculates the minimum selling price for the second batch --/
def min_selling_price_second_batch (b : TShirtBatches) : ℕ :=
  sorry

/-- Theorem stating the correct cost price and minimum selling price --/
theorem tshirt_pricing_theorem (b : TShirtBatches) 
  (h1 : b.first_batch_cost = 4000)
  (h2 : b.second_batch_cost = 5400)
  (h3 : b.quantity_ratio = 3/2)
  (h4 : b.price_difference = 5)
  (h5 : b.first_batch_selling_price = 70)
  (h6 : b.min_total_profit = 4060) :
  cost_price_first_batch b = 50 ∧ 
  min_selling_price_second_batch b = 66 :=
  sorry

end tshirt_pricing_theorem_l273_27395


namespace hyperbola_I_equation_hyperbola_II_equation_l273_27335

-- Part I
def hyperbola_I (x y : ℝ) : Prop :=
  let c : ℝ := 8  -- half of focal distance
  let e : ℝ := 4/3  -- eccentricity
  let a : ℝ := c/e
  let b : ℝ := Real.sqrt (c^2 - a^2)
  y^2/a^2 - x^2/b^2 = 1

theorem hyperbola_I_equation : 
  ∀ x y : ℝ, hyperbola_I x y ↔ y^2/36 - x^2/28 = 1 :=
sorry

-- Part II
def hyperbola_II (x y : ℝ) : Prop :=
  let c : ℝ := 6  -- distance from center to focus
  let a : ℝ := Real.sqrt (c^2/2)
  x^2/a^2 - y^2/a^2 = 1

theorem hyperbola_II_equation :
  ∀ x y : ℝ, hyperbola_II x y ↔ x^2/18 - y^2/18 = 1 :=
sorry

end hyperbola_I_equation_hyperbola_II_equation_l273_27335


namespace pen_cost_problem_l273_27310

theorem pen_cost_problem (total_students : Nat) (buyers : Nat) (pens_per_student : Nat) (pen_cost : Nat) :
  total_students = 32 →
  buyers > total_students / 2 →
  pens_per_student > 1 →
  pen_cost > pens_per_student →
  buyers * pens_per_student * pen_cost = 2116 →
  pen_cost = 23 := by
  sorry

end pen_cost_problem_l273_27310


namespace participant_selection_count_l273_27315

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_participants : ℕ := 4

def select_participants (boys girls participants : ℕ) : ℕ :=
  (Nat.choose boys 3 * Nat.choose girls 1) +
  (Nat.choose boys 2 * Nat.choose girls 2) +
  (Nat.choose boys 1 * Nat.choose girls 3)

theorem participant_selection_count :
  select_participants num_boys num_girls num_participants = 34 := by
  sorry

end participant_selection_count_l273_27315


namespace roots_of_x_squared_equals_x_l273_27345

theorem roots_of_x_squared_equals_x :
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end roots_of_x_squared_equals_x_l273_27345


namespace book_cost_solution_l273_27314

def book_cost_problem (x : ℕ) : Prop :=
  x > 0 ∧ 10 * x ≤ 1100 ∧ 11 * x > 1200

theorem book_cost_solution : ∃ (x : ℕ), book_cost_problem x ∧ x = 110 := by
  sorry

end book_cost_solution_l273_27314


namespace value_of_a_l273_27304

-- Define the conversion rate between paise and rupees
def paise_per_rupee : ℚ := 100

-- Define the given percentage as a rational number
def given_percentage : ℚ := 1 / 200

-- Define the given value in paise
def given_paise : ℚ := 85

-- Theorem statement
theorem value_of_a (a : ℚ) : given_percentage * a = given_paise → a = 170 := by
  sorry

end value_of_a_l273_27304


namespace even_function_iff_a_eq_one_l273_27341

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_iff_a_eq_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 1 := by sorry

end even_function_iff_a_eq_one_l273_27341


namespace average_score_is_106_l273_27371

/-- The average bowling score of three bowlers -/
def average_bowling_score (score1 score2 score3 : ℕ) : ℚ :=
  (score1 + score2 + score3 : ℚ) / 3

/-- Theorem: The average bowling score of three bowlers with scores 120, 113, and 85 is 106 -/
theorem average_score_is_106 :
  average_bowling_score 120 113 85 = 106 := by
  sorry

end average_score_is_106_l273_27371


namespace consecutive_primes_integral_roots_properties_l273_27307

-- Define consecutive primes
def consecutive_primes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

-- Define the quadratic equation with integral roots
def has_integral_roots (p q : ℕ) : Prop :=
  ∃ x y : ℤ, x^2 - (p + q : ℤ) * x + (p * q : ℤ) = 0 ∧
             y^2 - (p + q : ℤ) * y + (p * q : ℤ) = 0 ∧
             x ≠ y

theorem consecutive_primes_integral_roots_properties
  (p q : ℕ) (h1 : consecutive_primes p q) (h2 : has_integral_roots p q) :
  (∃ x y : ℤ, x + y = p + q ∧ Even (x + y)) ∧  -- Sum of roots is even
  (∀ x : ℤ, x^2 - (p + q : ℤ) * x + (p * q : ℤ) = 0 → x ≥ p) ∧  -- Each root ≥ p
  ¬Nat.Prime (p + q) :=  -- p+q is composite
by sorry

end consecutive_primes_integral_roots_properties_l273_27307


namespace expected_value_proof_l273_27338

/-- The expected value of winning (6-n)^2 dollars when rolling a fair 6-sided die -/
def expected_value : ℚ := 55 / 6

/-- A fair 6-sided die -/
def die : Finset ℕ := Finset.range 6

/-- The probability of rolling any number on a fair 6-sided die -/
def prob (n : ℕ) : ℚ := 1 / 6

/-- The winnings for rolling n on the die -/
def winnings (n : ℕ) : ℚ := (6 - n) ^ 2

theorem expected_value_proof :
  Finset.sum die (λ n => prob n * winnings n) = expected_value :=
sorry

end expected_value_proof_l273_27338


namespace cl2_moles_in_reaction_l273_27302

/-- Represents the stoichiometric coefficients of the reaction CH4 + 2Cl2 → CHCl3 + 4HCl -/
structure ReactionCoefficients where
  ch4 : ℕ
  cl2 : ℕ
  chcl3 : ℕ
  hcl : ℕ

/-- The balanced equation coefficients for the reaction -/
def balancedEquation : ReactionCoefficients :=
  { ch4 := 1, cl2 := 2, chcl3 := 1, hcl := 4 }

/-- Calculates the moles of Cl2 combined given the moles of CH4 and HCl -/
def molesOfCl2Combined (molesCH4 : ℕ) (molesHCl : ℕ) : ℕ :=
  (balancedEquation.cl2 * molesHCl) / balancedEquation.hcl

theorem cl2_moles_in_reaction (molesCH4 : ℕ) (molesHCl : ℕ) :
  molesCH4 = balancedEquation.ch4 ∧ molesHCl = balancedEquation.hcl →
  molesOfCl2Combined molesCH4 molesHCl = balancedEquation.cl2 :=
by
  sorry

end cl2_moles_in_reaction_l273_27302
