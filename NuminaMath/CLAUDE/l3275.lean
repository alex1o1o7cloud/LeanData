import Mathlib

namespace ing_catches_bo_l3275_327540

/-- The distance Bo jumps after n jumps -/
def bo_distance (n : ℕ) : ℕ := 6 * n

/-- The distance Ing jumps after n jumps -/
def ing_distance (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of jumps needed for Ing to catch Bo -/
def catch_up_jumps : ℕ := 11

theorem ing_catches_bo : 
  bo_distance catch_up_jumps = ing_distance catch_up_jumps :=
sorry

end ing_catches_bo_l3275_327540


namespace shooter_probability_l3275_327539

theorem shooter_probability (p : ℝ) (n k : ℕ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let prob_hit := p
  let num_shots := n
  let num_hits := k
  Nat.choose num_shots num_hits * prob_hit ^ num_hits * (1 - prob_hit) ^ (num_shots - num_hits) =
  Nat.choose 5 4 * (0.8 : ℝ) ^ 4 * (0.2 : ℝ) :=
by
  sorry

end shooter_probability_l3275_327539


namespace unique_solution_inequality_l3275_327572

theorem unique_solution_inequality (x : ℝ) : 
  x > 0 → x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x^3) ≥ 18 → x = 3 :=
by
  sorry

end unique_solution_inequality_l3275_327572


namespace geometric_sequence_sum_l3275_327518

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 + a 3 = 1) →
  (a 2 + a 3 + a 4 = 2) →
  (a 5 + a 6 + a 7 = 16) :=
by sorry

end geometric_sequence_sum_l3275_327518


namespace inequality_solution_minimum_value_minimum_value_condition_l3275_327532

-- Part 1: Inequality solution
theorem inequality_solution (x : ℝ) :
  (2 * x + 1) / (3 - x) ≥ 1 ↔ x ≤ 1 ∨ x > 2 :=
sorry

-- Part 2: Minimum value
theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) ≥ 25 :=
sorry

theorem minimum_value_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) = 25 ↔ x = 2/5 ∧ y = 3/5 :=
sorry

end inequality_solution_minimum_value_minimum_value_condition_l3275_327532


namespace range_of_x₁_l3275_327527

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the condition given in the problem
def Condition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1

-- Theorem statement
theorem range_of_x₁ (h_increasing : IsIncreasing f) (h_condition : Condition f) :
  ∀ x₁, (∃ x₂, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) ↔ x₁ > 1 :=
by sorry


end range_of_x₁_l3275_327527


namespace initial_stamp_ratio_l3275_327529

theorem initial_stamp_ratio (p q : ℕ) : 
  (p - 8 : ℚ) / (q + 8 : ℚ) = 6 / 5 →
  p - 8 = q + 8 →
  (p : ℚ) / q = 6 / 5 := by
sorry

end initial_stamp_ratio_l3275_327529


namespace square_1849_product_l3275_327525

theorem square_1849_product (x : ℤ) (h : x^2 = 1849) : (x + 2) * (x - 2) = 1845 := by
  sorry

end square_1849_product_l3275_327525


namespace imaginary_part_of_z_l3275_327575

def z : ℂ := 2 + Complex.I

theorem imaginary_part_of_z : z.im = 1 := by sorry

end imaginary_part_of_z_l3275_327575


namespace mushroom_picking_theorem_l3275_327591

/-- Calculates the total number of mushrooms picked over a three-day trip --/
def total_mushrooms (day1_revenue : ℕ) (day2_picked : ℕ) (price_per_mushroom : ℕ) : ℕ :=
  let day1_picked := day1_revenue / price_per_mushroom
  let day3_picked := 2 * day2_picked
  day1_picked + day2_picked + day3_picked

/-- The total number of mushrooms picked over three days is 65 --/
theorem mushroom_picking_theorem :
  total_mushrooms 58 12 2 = 65 := by
  sorry

#eval total_mushrooms 58 12 2

end mushroom_picking_theorem_l3275_327591


namespace equal_area_segment_property_l3275_327549

/-- A trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  area_ratio : (b + (b + 75)) / (b + 75 + (b + 150)) = 1 / 2  -- Midpoint segment divides areas in ratio 1:2

/-- The length of the segment parallel to bases dividing the trapezoid into equal areas -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  let x : ℝ := sorry
  x

/-- The main theorem -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(2812.5 + 112.5 * equal_area_segment t) / 100⌋ = ⌊(equal_area_segment t)^2 / 100⌋ := by
  sorry

end equal_area_segment_property_l3275_327549


namespace wrong_height_calculation_wrong_height_is_176_l3275_327559

/-- Given a class of boys with an incorrect average height and one boy's height recorded incorrectly,
    calculate the wrongly written height of that boy. -/
theorem wrong_height_calculation (n : ℕ) (initial_avg correct_avg actual_height : ℝ) : ℝ :=
  let wrong_height := actual_height + n * (initial_avg - correct_avg)
  wrong_height

/-- Prove that the wrongly written height of a boy is 176 cm given the specified conditions. -/
theorem wrong_height_is_176 :
  wrong_height_calculation 35 180 178 106 = 176 := by
  sorry

end wrong_height_calculation_wrong_height_is_176_l3275_327559


namespace additional_miles_with_bakery_stop_l3275_327524

/-- The additional miles driven with a bakery stop compared to without -/
theorem additional_miles_with_bakery_stop
  (apartment_to_bakery : ℕ)
  (bakery_to_grandma : ℕ)
  (grandma_to_apartment : ℕ)
  (h1 : apartment_to_bakery = 9)
  (h2 : bakery_to_grandma = 24)
  (h3 : grandma_to_apartment = 27) :
  (apartment_to_bakery + bakery_to_grandma + grandma_to_apartment) -
  (2 * grandma_to_apartment) = 6 :=
by sorry

end additional_miles_with_bakery_stop_l3275_327524


namespace simple_interest_principal_l3275_327569

/-- Simple interest calculation -/
theorem simple_interest_principal
  (rate : ℝ) (interest : ℝ) (time : ℝ)
  (h_rate : rate = 15)
  (h_interest : interest = 120)
  (h_time : time = 2) :
  (interest * 100) / (rate * time) = 400 := by
sorry

end simple_interest_principal_l3275_327569


namespace reciprocal_equality_l3275_327551

theorem reciprocal_equality (a b : ℝ) : 
  (1 / a = -8) → (1 / (-b) = 8) → (a = b) := by
  sorry

end reciprocal_equality_l3275_327551


namespace red_card_count_l3275_327550

theorem red_card_count (red_credit blue_credit total_cards total_credit : ℕ) 
  (h1 : red_credit = 3)
  (h2 : blue_credit = 5)
  (h3 : total_cards = 20)
  (h4 : total_credit = 84) :
  ∃ (red_cards blue_cards : ℕ),
    red_cards + blue_cards = total_cards ∧
    red_credit * red_cards + blue_credit * blue_cards = total_credit ∧
    red_cards = 8 := by
  sorry

end red_card_count_l3275_327550


namespace max_digits_product_5_4_l3275_327557

theorem max_digits_product_5_4 : 
  ∀ (a b : ℕ), 
    10000 ≤ a ∧ a ≤ 99999 →
    1000 ≤ b ∧ b ≤ 9999 →
    a * b < 1000000000 := by
  sorry

end max_digits_product_5_4_l3275_327557


namespace stream_speed_l3275_327594

/-- Given a boat that travels downstream and upstream, calculate the speed of the stream -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
                     (downstream_time upstream_time : ℝ) 
                     (h1 : downstream_distance = 72)
                     (h2 : upstream_distance = 30)
                     (h3 : downstream_time = 3)
                     (h4 : upstream_time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 7 := by
  sorry

end stream_speed_l3275_327594


namespace inequality_proof_l3275_327537

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + a) ≥ 3/2 := by
  sorry

end inequality_proof_l3275_327537


namespace kim_sweater_count_l3275_327568

/-- The number of sweaters Kim knit on Monday -/
def monday_sweaters : ℕ := 8

/-- The total number of sweaters Kim knit in the week -/
def total_sweaters : ℕ := 34

/-- The maximum number of sweaters Kim can knit in a day -/
def max_daily_sweaters : ℕ := 10

theorem kim_sweater_count :
  monday_sweaters ≤ max_daily_sweaters ∧
  monday_sweaters +
  (monday_sweaters + 2) +
  ((monday_sweaters + 2) - 4) +
  ((monday_sweaters + 2) - 4) +
  (monday_sweaters / 2) = total_sweaters :=
by sorry

end kim_sweater_count_l3275_327568


namespace isosceles_triangle_areas_sum_l3275_327507

/-- Represents a right triangle with sides a, b, and c --/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2

/-- Represents the areas of right isosceles triangles constructed on the sides of a right triangle --/
structure IsoscelesTriangleAreas (t : RightTriangle) where
  A : ℝ
  B : ℝ
  C : ℝ
  area_def_A : A = (1/2) * t.a^2
  area_def_B : B = (1/2) * t.b^2
  area_def_C : C = (1/2) * t.c^2

/-- Theorem: For a 5-12-13 right triangle with right isosceles triangles constructed on each side,
    the sum of the areas of the isosceles triangles on the two shorter sides
    equals the area of the isosceles triangle on the hypotenuse --/
theorem isosceles_triangle_areas_sum (t : RightTriangle)
  (h : t.a = 5 ∧ t.b = 12 ∧ t.c = 13)
  (areas : IsoscelesTriangleAreas t) :
  areas.A + areas.B = areas.C := by
  sorry

end isosceles_triangle_areas_sum_l3275_327507


namespace evaluate_expression_l3275_327514

theorem evaluate_expression : -(((16 / 4) * 6 - 50) + 5^2) = 1 := by
  sorry

end evaluate_expression_l3275_327514


namespace factorization_1_l3275_327566

theorem factorization_1 (a b x y : ℝ) : a * (x - y) + b * (y - x) = (a - b) * (x - y) := by
  sorry

end factorization_1_l3275_327566


namespace original_price_with_loss_l3275_327522

/-- Proves that given an article sold for 300 with a 50% loss, the original price was 600 -/
theorem original_price_with_loss (selling_price : ℝ) (loss_percent : ℝ) : 
  selling_price = 300 → loss_percent = 50 → 
  ∃ original_price : ℝ, 
    original_price = 600 ∧ 
    selling_price = original_price * (1 - loss_percent / 100) := by
  sorry

end original_price_with_loss_l3275_327522


namespace circle_equation_l3275_327516

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point is on the circle if its distance from the center equals the radius -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the y-axis if its distance from the y-axis equals its radius -/
def tangentToYAxis (c : Circle) : Prop :=
  |c.center.1| = c.radius

theorem circle_equation (c : Circle) (h : tangentToYAxis c) (h2 : c.center = (-2, 3)) :
  ∀ (x y : ℝ), onCircle c (x, y) ↔ (x + 2)^2 + (y - 3)^2 = 4 := by
  sorry

end circle_equation_l3275_327516


namespace vector_magnitude_problem_l3275_327588

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60 degrees in radians
  a = (2, 0) →
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) = Real.cos angle →
  ‖b‖ = 1 :=
by sorry

#check vector_magnitude_problem

end vector_magnitude_problem_l3275_327588


namespace stamp_collection_fraction_l3275_327595

/-- Given the stamp collection scenario, prove that KJ has half the stamps of AJ -/
theorem stamp_collection_fraction :
  ∀ (cj kj aj : ℕ) (f : ℚ),
  -- CJ has 5 more than twice the number of stamps that KJ has
  cj = 2 * kj + 5 →
  -- KJ has a certain fraction of the number of stamps AJ has
  kj = f * aj →
  -- The three boys have 930 stamps in total
  cj + kj + aj = 930 →
  -- AJ has 370 stamps
  aj = 370 →
  -- The fraction of stamps KJ has compared to AJ is 1/2
  f = 1/2 := by
sorry


end stamp_collection_fraction_l3275_327595


namespace abes_age_l3275_327593

theorem abes_age :
  ∀ (present_age : ℕ), 
    (present_age + (present_age - 7) = 37) → 
    present_age = 22 := by
  sorry

end abes_age_l3275_327593


namespace intersection_complement_equality_l3275_327599

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y, y = Real.log (1 - x^2)}
def B : Set ℝ := {y : ℝ | ∃ x, y = (4 : ℝ)^(x - 2)}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = Set.Ioc (-1) 0 := by sorry

end intersection_complement_equality_l3275_327599


namespace double_price_increase_l3275_327506

theorem double_price_increase (P : ℝ) (h : P > 0) : 
  P * (1 + 0.1) * (1 + 0.1) = P * (1 + 0.21) := by
sorry

end double_price_increase_l3275_327506


namespace solution_exists_l3275_327504

-- Define the functions f and g
def f (x : ℝ) := x^2 + 10
def g (x : ℝ) := x^2 - 6

-- State the theorem
theorem solution_exists (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 14) :
  a = Real.sqrt 8 ∨ a = 2 := by
sorry

end solution_exists_l3275_327504


namespace sum_m_n_in_interval_l3275_327580

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the theorem
theorem sum_m_n_in_interval (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (-5) 4) →
  (∀ y ∈ Set.Icc (-5) 4, ∃ x ∈ Set.Icc m n, f x = y) →
  m + n ∈ Set.Icc 1 5 := by
  sorry

end sum_m_n_in_interval_l3275_327580


namespace log_product_equality_l3275_327528

theorem log_product_equality : 
  ∀ (x : ℝ), x > 0 → 
  (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) * 
  (Real.log 6 / Real.log 5) * (Real.log 7 / Real.log 6) * 
  (Real.log 8 / Real.log 7) * (Real.log 9 / Real.log 8) * 
  (Real.log 10 / Real.log 9) * (Real.log 11 / Real.log 10) * 
  (Real.log 12 / Real.log 11) * (Real.log 13 / Real.log 12) * 
  (Real.log 14 / Real.log 13) * (Real.log 15 / Real.log 14) = 
  1 + Real.log 5 / Real.log 3 := by
sorry

end log_product_equality_l3275_327528


namespace ellipse_equation_l3275_327542

/-- Given an ellipse with eccentricity √7/4 and distance 4 from one endpoint of the minor axis to the right focus, prove its standard equation is x²/16 + y²/9 = 1 -/
theorem ellipse_equation (e : ℝ) (d : ℝ) (x y : ℝ) :
  e = Real.sqrt 7 / 4 →
  d = 4 →
  x^2 / 16 + y^2 / 9 = 1 := by
  sorry

end ellipse_equation_l3275_327542


namespace cubic_inequality_implies_value_range_l3275_327502

theorem cubic_inequality_implies_value_range (y : ℝ) : 
  y^3 - 6*y^2 + 11*y - 6 < 0 → 
  24 < y^3 + 6*y^2 + 11*y + 6 ∧ y^3 + 6*y^2 + 11*y + 6 < 120 := by
sorry

end cubic_inequality_implies_value_range_l3275_327502


namespace triangle_side_length_l3275_327562

theorem triangle_side_length (a b : ℝ) (A B : Real) :
  a = 10 →
  B = Real.pi / 3 →
  A = Real.pi / 4 →
  b = 10 * (Real.sin (Real.pi / 3) / Real.sin (Real.pi / 4)) :=
by sorry

end triangle_side_length_l3275_327562


namespace dancing_preference_theorem_l3275_327561

structure DancingPreference where
  like : Rat
  neutral : Rat
  dislike : Rat
  likeSayLike : Rat
  likeSayDislike : Rat
  dislikeSayLike : Rat
  dislikeSayDislike : Rat
  neutralSayLike : Rat
  neutralSayDislike : Rat

/-- The fraction of students who say they dislike dancing but actually like it -/
def fractionLikeSayDislike (pref : DancingPreference) : Rat :=
  (pref.like * pref.likeSayDislike) /
  (pref.like * pref.likeSayDislike + pref.dislike * pref.dislikeSayDislike + pref.neutral * pref.neutralSayDislike)

theorem dancing_preference_theorem (pref : DancingPreference) 
  (h1 : pref.like = 1/2)
  (h2 : pref.neutral = 3/10)
  (h3 : pref.dislike = 1/5)
  (h4 : pref.likeSayLike = 7/10)
  (h5 : pref.likeSayDislike = 3/10)
  (h6 : pref.dislikeSayLike = 1/5)
  (h7 : pref.dislikeSayDislike = 4/5)
  (h8 : pref.neutralSayLike = 2/5)
  (h9 : pref.neutralSayDislike = 3/5)
  : fractionLikeSayDislike pref = 15/49 := by
  sorry

end dancing_preference_theorem_l3275_327561


namespace marbles_remainder_l3275_327589

theorem marbles_remainder (r p g : ℕ) 
  (hr : r % 7 = 5) 
  (hp : p % 7 = 4) 
  (hg : g % 7 = 2) : 
  (r + p + g) % 7 = 4 := by
sorry

end marbles_remainder_l3275_327589


namespace combinations_equal_200_l3275_327521

/-- The number of varieties of gift bags -/
def gift_bags : ℕ := 10

/-- The number of colors of tissue paper -/
def tissue_papers : ℕ := 4

/-- The number of types of tags -/
def tags : ℕ := 5

/-- The total number of possible combinations -/
def total_combinations : ℕ := gift_bags * tissue_papers * tags

/-- Theorem stating that the total number of combinations is 200 -/
theorem combinations_equal_200 : total_combinations = 200 := by
  sorry

end combinations_equal_200_l3275_327521


namespace problem_statement_l3275_327509

theorem problem_statement (a b : ℕ) : 
  a = 105 → a^3 = 21 * 35 * 45 * b → b = 105 := by sorry

end problem_statement_l3275_327509


namespace andrews_age_l3275_327571

theorem andrews_age (andrew_age grandfather_age : ℕ) : 
  grandfather_age = 16 * andrew_age →
  grandfather_age - andrew_age = 60 →
  andrew_age = 4 := by
sorry

end andrews_age_l3275_327571


namespace inscribed_circle_ratio_l3275_327584

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is isosceles with base AB -/
def IsIsoscelesAB (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2

/-- Checks if a circle is inscribed in a triangle -/
def IsInscribed (c : Circle) (t : Triangle) : Prop := sorry

/-- Checks if a point is on a line segment -/
def IsOnSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Checks if a point is on a circle -/
def IsOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Calculates the distance between two points -/
def Distance (a : Point) (b : Point) : ℝ := sorry

/-- The main theorem -/
theorem inscribed_circle_ratio 
  (t : Triangle) 
  (c : Circle) 
  (M N : Point) 
  (k : ℝ) :
  IsIsoscelesAB t →
  IsInscribed c t →
  IsOnSegment M t.B t.C →
  IsOnCircle M c →
  IsOnSegment N t.A M →
  IsOnCircle N c →
  Distance t.A t.B / Distance t.B t.C = k →
  Distance M N / Distance t.A N = 2 * (2 - k) := by
  sorry

end inscribed_circle_ratio_l3275_327584


namespace total_cost_is_correct_l3275_327535

def phone_cost : ℝ := 2
def service_plan_monthly_cost : ℝ := 7
def service_plan_duration : ℕ := 4
def insurance_fee : ℝ := 10
def first_phone_tax_rate : ℝ := 0.05
def second_phone_tax_rate : ℝ := 0.03
def service_plan_discount_rate : ℝ := 0.20
def num_phones : ℕ := 2

def total_cost : ℝ :=
  let phone_total := phone_cost * num_phones
  let service_plan_total := service_plan_monthly_cost * service_plan_duration * num_phones
  let service_plan_discount := service_plan_total * service_plan_discount_rate
  let discounted_service_plan := service_plan_total - service_plan_discount
  let tax_total := (first_phone_tax_rate * phone_cost) + (second_phone_tax_rate * phone_cost)
  phone_total + discounted_service_plan + tax_total + insurance_fee

theorem total_cost_is_correct : total_cost = 58.96 := by
  sorry

end total_cost_is_correct_l3275_327535


namespace probability_non_red_face_l3275_327570

theorem probability_non_red_face (total_faces : ℕ) (red_faces : ℕ) (yellow_faces : ℕ) (blue_faces : ℕ) (green_faces : ℕ)
  (h1 : total_faces = 10)
  (h2 : red_faces = 5)
  (h3 : yellow_faces = 3)
  (h4 : blue_faces = 1)
  (h5 : green_faces = 1)
  (h6 : total_faces = red_faces + yellow_faces + blue_faces + green_faces) :
  (yellow_faces + blue_faces + green_faces : ℚ) / total_faces = 1 / 2 := by
  sorry

end probability_non_red_face_l3275_327570


namespace marble_remainder_l3275_327531

theorem marble_remainder (l j : ℕ) 
  (hl : l % 8 = 5) 
  (hj : j % 8 = 6) : 
  (l + j) % 8 = 3 := by
sorry

end marble_remainder_l3275_327531


namespace range_of_linear_function_l3275_327548

theorem range_of_linear_function (c : ℝ) (h : c ≠ 0) :
  let g : ℝ → ℝ := λ x ↦ c * x + 2
  let domain := Set.Icc (-1 : ℝ) 2
  let range := Set.image g domain
  range = if c > 0 
    then Set.Icc (-c + 2) (2 * c + 2)
    else Set.Icc (2 * c + 2) (-c + 2) := by
  sorry

end range_of_linear_function_l3275_327548


namespace end_behavior_of_g_l3275_327563

noncomputable def g (x : ℝ) : ℝ := -3 * x^4 + 5 * x^3 - 4

theorem end_behavior_of_g :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M) :=
sorry

end end_behavior_of_g_l3275_327563


namespace max_hot_dogs_is_3250_l3275_327587

/-- Represents the available pack sizes and their prices --/
structure PackInfo where
  size : Nat
  price : Rat

/-- The maximum number of hot dogs that can be purchased with the given budget --/
def maxHotDogs (packs : List PackInfo) (budget : Rat) : Nat :=
  sorry

/-- The available pack sizes and prices --/
def availablePacks : List PackInfo := [
  ⟨8, 155/100⟩,
  ⟨20, 305/100⟩,
  ⟨250, 2295/100⟩
]

/-- The budget in dollars --/
def totalBudget : Rat := 300

/-- Theorem stating that the maximum number of hot dogs that can be purchased is 3250 --/
theorem max_hot_dogs_is_3250 :
  maxHotDogs availablePacks totalBudget = 3250 := by sorry

end max_hot_dogs_is_3250_l3275_327587


namespace infinitely_many_n_with_large_prime_divisor_l3275_327555

theorem infinitely_many_n_with_large_prime_divisor :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ∃ p : ℕ, Nat.Prime p ∧ p > 2*n + Real.sqrt (2*n) ∧ p ∣ (n^2 + 1) :=
sorry

end infinitely_many_n_with_large_prime_divisor_l3275_327555


namespace factorization_theorem_l3275_327515

theorem factorization_theorem (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) = (a^2+b^2)*(b^2+c^2)*(c^2+a^2) :=
by sorry

end factorization_theorem_l3275_327515


namespace letters_in_mailboxes_l3275_327596

theorem letters_in_mailboxes :
  (number_of_ways : ℕ) →
  (number_of_letters : ℕ) →
  (number_of_mailboxes : ℕ) →
  (number_of_letters = 4) →
  (number_of_mailboxes = 3) →
  (number_of_ways = number_of_mailboxes ^ number_of_letters) :=
by sorry

end letters_in_mailboxes_l3275_327596


namespace square_of_binomial_with_sqrt_l3275_327577

theorem square_of_binomial_with_sqrt : 36^2 + 2 * 36 * Real.sqrt 49 + (Real.sqrt 49)^2 = 1849 := by
  sorry

end square_of_binomial_with_sqrt_l3275_327577


namespace triangle_inequality_l3275_327574

variable (A B C : ℝ) -- Angles of the triangle
variable (da db dc : ℝ) -- Distances from P to sides
variable (Ra Rb Rc : ℝ) -- Distances from P to vertices

-- Assume all variables are non-negative
variable (h1 : 0 ≤ A) (h2 : 0 ≤ B) (h3 : 0 ≤ C)
variable (h4 : 0 ≤ da) (h5 : 0 ≤ db) (h6 : 0 ≤ dc)
variable (h7 : 0 ≤ Ra) (h8 : 0 ≤ Rb) (h9 : 0 ≤ Rc)

-- Assume A, B, C form a valid triangle
variable (h10 : A + B + C = Real.pi)

theorem triangle_inequality (A B C da db dc Ra Rb Rc : ℝ)
  (h1 : 0 ≤ A) (h2 : 0 ≤ B) (h3 : 0 ≤ C)
  (h4 : 0 ≤ da) (h5 : 0 ≤ db) (h6 : 0 ≤ dc)
  (h7 : 0 ≤ Ra) (h8 : 0 ≤ Rb) (h9 : 0 ≤ Rc)
  (h10 : A + B + C = Real.pi) :
  3 * (da^2 + db^2 + dc^2) ≥ (Ra * Real.sin A)^2 + (Rb * Real.sin B)^2 + (Rc * Real.sin C)^2 :=
by sorry

end triangle_inequality_l3275_327574


namespace sum_of_factors_48_l3275_327565

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_48 : sum_of_factors 48 = 124 := by
  sorry

end sum_of_factors_48_l3275_327565


namespace tim_grew_44_cantaloupes_l3275_327533

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

/-- Proof that Tim grew 44 cantaloupes -/
theorem tim_grew_44_cantaloupes : tim_cantaloupes = 44 := by
  sorry

end tim_grew_44_cantaloupes_l3275_327533


namespace sufficient_not_necessary_condition_l3275_327579

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) :=
by sorry

end sufficient_not_necessary_condition_l3275_327579


namespace symmetric_points_sum_l3275_327583

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Given that point A(2, a) is symmetric to point B(b, -3) with respect to the x-axis,
    prove that a + b = 5 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_x_axis (2, a) (b, -3)) : a + b = 5 := by
  sorry

end symmetric_points_sum_l3275_327583


namespace geometric_sequence_increasing_ratio_l3275_327554

/-- A geometric sequence with first term less than zero and increasing terms has a common ratio between 0 and 1. -/
theorem geometric_sequence_increasing_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (h1 : a 1 < 0)  -- First term is negative
  (h2 : ∀ n : ℕ, a n < a (n + 1))  -- Sequence is strictly increasing
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q)  -- Definition of geometric sequence
  : 0 < q ∧ q < 1 := by
sorry

end geometric_sequence_increasing_ratio_l3275_327554


namespace perfect_game_score_l3275_327546

/-- Given that a perfect score is 21 points, prove that the total points
    after 3 perfect games is equal to 63. -/
theorem perfect_game_score (perfect_score : ℕ) (num_games : ℕ) :
  perfect_score = 21 → num_games = 3 → perfect_score * num_games = 63 := by
  sorry

end perfect_game_score_l3275_327546


namespace milk_mixture_water_content_l3275_327598

theorem milk_mixture_water_content 
  (initial_water_percentage : ℝ)
  (initial_milk_volume : ℝ)
  (pure_milk_volume : ℝ)
  (h1 : initial_water_percentage = 5)
  (h2 : initial_milk_volume = 10)
  (h3 : pure_milk_volume = 15) :
  let total_water := initial_water_percentage / 100 * initial_milk_volume
  let total_volume := initial_milk_volume + pure_milk_volume
  let final_water_percentage := total_water / total_volume * 100
  final_water_percentage = 2 := by
sorry

end milk_mixture_water_content_l3275_327598


namespace limit_fraction_to_one_third_l3275_327581

theorem limit_fraction_to_one_third :
  ∀ ε > 0, ∃ N : ℝ, ∀ n : ℝ, n > N → |((n + 20) / (3 * n + 1)) - (1 / 3)| < ε :=
by
  sorry

end limit_fraction_to_one_third_l3275_327581


namespace prob_divisible_by_five_prob_divisible_by_five_is_one_l3275_327541

/-- A three-digit positive integer with a ones digit of 5 -/
def ThreeDigitEndingIn5 : Type := { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ n % 10 = 5 }

/-- The probability that a number in ThreeDigitEndingIn5 is divisible by 5 -/
theorem prob_divisible_by_five (n : ThreeDigitEndingIn5) : ℚ :=
  1

/-- The probability that a number in ThreeDigitEndingIn5 is divisible by 5 is 1 -/
theorem prob_divisible_by_five_is_one : 
  ∀ n : ThreeDigitEndingIn5, prob_divisible_by_five n = 1 :=
sorry

end prob_divisible_by_five_prob_divisible_by_five_is_one_l3275_327541


namespace atomic_weight_sodium_l3275_327585

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def atomic_weight_chlorine : ℝ := 35.45

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def atomic_weight_oxygen : ℝ := 16.00

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight_compound : ℝ := 74.00

/-- Theorem stating that the atomic weight of sodium is 22.55 amu -/
theorem atomic_weight_sodium :
  molecular_weight_compound = atomic_weight_chlorine + atomic_weight_oxygen + 22.55 := by
  sorry

end atomic_weight_sodium_l3275_327585


namespace cos_graph_transformation_l3275_327508

theorem cos_graph_transformation (x : ℝ) : 
  4 * Real.cos (2 * x) = 4 * Real.cos (2 * (x - π/8) + π/4) :=
by sorry

end cos_graph_transformation_l3275_327508


namespace parents_gift_ratio_equal_l3275_327534

/-- Represents the spending on Christmas gifts -/
structure ChristmasGifts where
  sibling_cost : ℕ  -- Cost per sibling's gift
  num_siblings : ℕ  -- Number of siblings
  total_spent : ℕ  -- Total amount spent on all gifts
  parent_cost : ℕ  -- Cost per parent's gift

/-- Theorem stating that the ratio of gift values for Mia's parents is 1:1 -/
theorem parents_gift_ratio_equal (gifts : ChristmasGifts)
  (h1 : gifts.sibling_cost = 30)
  (h2 : gifts.num_siblings = 3)
  (h3 : gifts.total_spent = 150)
  (h4 : gifts.parent_cost = 30) :
  gifts.parent_cost / gifts.parent_cost = 1 := by
  sorry

#check parents_gift_ratio_equal

end parents_gift_ratio_equal_l3275_327534


namespace harlys_dogs_l3275_327547

theorem harlys_dogs (x : ℝ) : 
  (0.6 * x + 5 = 53) → x = 80 := by
  sorry

end harlys_dogs_l3275_327547


namespace E_and_G_complementary_l3275_327538

/-- The sample space of selecting 3 products from 100 products. -/
def Ω : Type := Unit

/-- The probability measure on the sample space. -/
def P : Ω → ℝ := sorry

/-- The event that all 3 selected products are non-defective. -/
def E : Set Ω := sorry

/-- The event that all 3 selected products are defective. -/
def F : Set Ω := sorry

/-- The event that at least one of the 3 selected products is defective. -/
def G : Set Ω := sorry

/-- The total number of products. -/
def total_products : ℕ := 100

/-- The number of defective products. -/
def defective_products : ℕ := 5

/-- The number of products selected. -/
def selected_products : ℕ := 3

theorem E_and_G_complementary :
  E ∪ G = Set.univ ∧ E ∩ G = ∅ :=
sorry

end E_and_G_complementary_l3275_327538


namespace no_integer_solutions_l3275_327597

theorem no_integer_solutions : ¬ ∃ (m n : ℤ), m + 2*n = 2*m*n - 3 := by
  sorry

end no_integer_solutions_l3275_327597


namespace problem_solution_l3275_327558

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |x + a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 3 ↔ x ≥ 1 ∨ x ≤ -1) ∧
  (∃ x : ℝ, f a x ≤ |a - 1| ↔ a ≤ 1/4) :=
sorry

end problem_solution_l3275_327558


namespace discount_difference_is_978_75_l3275_327536

/-- The initial invoice amount -/
def initial_amount : ℝ := 15000

/-- The single discount rate -/
def single_discount_rate : ℝ := 0.5

/-- The successive discount rates -/
def successive_discount_rates : List ℝ := [0.3, 0.15, 0.05]

/-- Calculate the amount after applying a single discount -/
def amount_after_single_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * (1 - rate)

/-- Calculate the amount after applying successive discounts -/
def amount_after_successive_discounts (amount : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) amount

/-- The difference between single discount and successive discounts -/
def discount_difference : ℝ :=
  amount_after_successive_discounts initial_amount successive_discount_rates -
  amount_after_single_discount initial_amount single_discount_rate

theorem discount_difference_is_978_75 :
  discount_difference = 978.75 := by sorry

end discount_difference_is_978_75_l3275_327536


namespace fencing_cost_is_105_rupees_l3275_327586

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  ratio : ℝ × ℝ

/-- Calculates the cost of fencing a rectangular field -/
def fencingCost (field : RectangularField) (costPerMeter : ℝ) : ℝ :=
  2 * (field.length + field.width) * costPerMeter

/-- Theorem: The cost of fencing a specific rectangular field is 105 rupees -/
theorem fencing_cost_is_105_rupees : 
  ∀ (field : RectangularField),
    field.ratio = (3, 4) →
    field.area = 10800 →
    fencingCost field 0.25 = 105 := by
  sorry

end fencing_cost_is_105_rupees_l3275_327586


namespace roots_of_quadratic_equation_l3275_327573

theorem roots_of_quadratic_equation (a b : ℝ) : 
  (a^2 + a - 5 = 0) → 
  (b^2 + b - 5 = 0) → 
  (a + b = -1) → 
  (a * b = -5) → 
  (2 * a^2 + a + b^2 = 16) := by
sorry

end roots_of_quadratic_equation_l3275_327573


namespace x_fifth_plus_72x_l3275_327545

theorem x_fifth_plus_72x (x : ℝ) (h : x^2 + 6*x = 12) : x^5 + 72*x = 2808*x - 4320 := by
  sorry

end x_fifth_plus_72x_l3275_327545


namespace jeremy_stroll_time_l3275_327564

/-- Proves that Jeremy's strolling time is 10 hours given his distance and speed -/
theorem jeremy_stroll_time (distance : ℝ) (speed : ℝ) (h1 : distance = 20) (h2 : speed = 2) :
  distance / speed = 10 := by
  sorry

end jeremy_stroll_time_l3275_327564


namespace ellipse_parameter_sum_l3275_327519

def ellipse (h k a b : ℝ) := fun (x y : ℝ) ↦ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem ellipse_parameter_sum :
  ∃ h k a b : ℝ,
    (∀ x y : ℝ, ellipse h k a b x y ↔ 
      Real.sqrt ((x - 0)^2 + (y - 0)^2) + Real.sqrt ((x - 6)^2 + (y - 0)^2) = 10) ∧
    h + k + a + b = 12 := by
  sorry

end ellipse_parameter_sum_l3275_327519


namespace at_op_difference_l3275_327517

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - y * x - 3 * x + 2 * y

-- State the theorem
theorem at_op_difference : at_op 9 5 - at_op 5 9 = -20 := by
  sorry

end at_op_difference_l3275_327517


namespace regular_polygon_sides_l3275_327567

theorem regular_polygon_sides : ∃ n : ℕ, 
  n > 0 ∧ 
  (360 : ℝ) / n = n - 9 ∧
  n = 24 := by
  sorry

end regular_polygon_sides_l3275_327567


namespace min_value_expression_l3275_327501

theorem min_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  4 ≤ a^2 + 2 * Real.sqrt (a * b) + Real.rpow (a^2 * b * c) (1/3) ∧
  ∃ a' b' c' : ℝ, 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    a'^2 + 2 * Real.sqrt (a' * b') + Real.rpow (a'^2 * b' * c') (1/3) = 4 :=
by sorry

end min_value_expression_l3275_327501


namespace second_player_can_always_win_l3275_327578

/-- Represents a square on the game board -/
inductive Square
| Empty : Square
| S : Square
| O : Square

/-- Represents the game board -/
def Board := Vector Square 2000

/-- Represents a player in the game -/
inductive Player
| First : Player
| Second : Player

/-- Checks if the game is over (SOS pattern found) -/
def is_game_over (board : Board) : Prop := sorry

/-- Represents a valid move in the game -/
structure Move where
  position : Fin 2000
  symbol : Square

/-- Applies a move to the board -/
def apply_move (board : Board) (move : Move) : Board := sorry

/-- Represents the game state -/
structure GameState where
  board : Board
  current_player : Player

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player -/
def is_winning_strategy (player : Player) (strategy : Strategy) : Prop := sorry

/-- The main theorem to prove -/
theorem second_player_can_always_win :
  ∃ (strategy : Strategy), is_winning_strategy Player.Second strategy := sorry

end second_player_can_always_win_l3275_327578


namespace minimum_speed_to_clear_building_l3275_327500

/-- The minimum speed required for a stone to clear a building -/
theorem minimum_speed_to_clear_building 
  (g H l : ℝ) (α : ℝ) (h_g : g > 0) (h_H : H > 0) (h_l : l > 0) 
  (h_α : 0 < α ∧ α < π / 2) : 
  ∃ (v₀ : ℝ), v₀ = Real.sqrt (g * (2 * H + l * (1 - Real.sin α) / Real.cos α)) ∧ 
  (∀ (v : ℝ), v > v₀ → 
    ∃ (trajectory : ℝ → ℝ), 
      (∀ x, trajectory x ≤ H + Real.tan α * (l - x)) ∧
      (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ l ∧ 
        trajectory x₁ = H ∧ trajectory x₂ = H + Real.tan α * (l - x₂))) :=
sorry

end minimum_speed_to_clear_building_l3275_327500


namespace sqrt_fraction_equality_l3275_327526

theorem sqrt_fraction_equality : 
  Real.sqrt ((16^10 + 2^30) / (16^6 + 2^35)) = 256 / Real.sqrt 2049 := by sorry

end sqrt_fraction_equality_l3275_327526


namespace venus_meal_cost_calculation_l3275_327523

/-- The cost per meal at Venus Hall -/
def venus_meal_cost : ℝ := 35

/-- The room rental cost for Caesar's -/
def caesars_room_cost : ℝ := 800

/-- The meal cost for Caesar's -/
def caesars_meal_cost : ℝ := 30

/-- The room rental cost for Venus Hall -/
def venus_room_cost : ℝ := 500

/-- The number of guests at which the costs are equal -/
def num_guests : ℝ := 60

theorem venus_meal_cost_calculation :
  caesars_room_cost + caesars_meal_cost * num_guests =
  venus_room_cost + venus_meal_cost * num_guests :=
by sorry

end venus_meal_cost_calculation_l3275_327523


namespace solve_marbles_problem_l3275_327590

def marbles_problem (initial : ℕ) (gifted : ℕ) (final : ℕ) : Prop :=
  ∃ (lost : ℕ), initial - lost - gifted = final

theorem solve_marbles_problem :
  marbles_problem 85 25 43 → (∃ (lost : ℕ), lost = 17) :=
by
  sorry

end solve_marbles_problem_l3275_327590


namespace snow_probability_l3275_327576

theorem snow_probability (p : ℝ) (h : p = 2/3) :
  1 - (1 - p)^3 = 26/27 := by sorry

end snow_probability_l3275_327576


namespace original_number_is_509_l3275_327513

theorem original_number_is_509 (subtracted_number : ℕ) : 
  (509 - subtracted_number) % 9 = 0 →
  subtracted_number ≥ 5 →
  ∀ n < subtracted_number, (509 - n) % 9 ≠ 0 →
  509 = 509 :=
by
  sorry

end original_number_is_509_l3275_327513


namespace peanuts_in_box_l3275_327552

/-- Given a box with an initial number of peanuts and a number of peanuts added,
    calculate the total number of peanuts in the box. -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: If there are 4 peanuts in a box and 2 more are added,
    the total number of peanuts in the box is 6. -/
theorem peanuts_in_box : total_peanuts 4 2 = 6 := by sorry

end peanuts_in_box_l3275_327552


namespace shaded_area_calculation_l3275_327520

theorem shaded_area_calculation (area_ABCD area_overlap : ℝ) 
  (h1 : area_ABCD = 196)
  (h2 : area_overlap = 1)
  (h3 : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b^2 = 4*a^2 ∧ a + b = Real.sqrt area_ABCD - Real.sqrt area_overlap) :
  ∃ (shaded_area : ℝ), shaded_area = 72 ∧ 
    shaded_area = area_ABCD - (((Real.sqrt area_ABCD - Real.sqrt area_overlap)/3)^2 + 4*((Real.sqrt area_ABCD - Real.sqrt area_overlap)/3)^2 - area_overlap) :=
by sorry

end shaded_area_calculation_l3275_327520


namespace some_zens_not_cens_l3275_327544

-- Define the sets
variable (U : Type) -- Universe set
variable (Zen : Set U) -- Set of Zens
variable (Ben : Set U) -- Set of Bens
variable (Cen : Set U) -- Set of Cens

-- Define the hypotheses
variable (h1 : Zen ⊆ Ben) -- All Zens are Bens
variable (h2 : ∃ x, x ∈ Ben ∧ x ∉ Cen) -- Some Bens are not Cens

-- Theorem to prove
theorem some_zens_not_cens : ∃ x, x ∈ Zen ∧ x ∉ Cen :=
sorry

end some_zens_not_cens_l3275_327544


namespace speed_conversion_l3275_327511

-- Define the conversion factors
def km_to_m : ℚ := 1000
def hour_to_sec : ℚ := 3600

-- Define the given speed in km/h
def speed_kmh : ℚ := 72

-- Define the conversion function
def kmh_to_ms (speed : ℚ) : ℚ :=
  speed * km_to_m / hour_to_sec

-- Theorem statement
theorem speed_conversion :
  kmh_to_ms speed_kmh = 20 := by
  sorry

end speed_conversion_l3275_327511


namespace four_digit_number_problem_l3275_327510

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem four_digit_number_problem (N : ℕ) 
  (h1 : is_four_digit N) 
  (h2 : (70000 + N) - (10 * N + 7) = 53208) : 
  N = 1865 := by sorry

end four_digit_number_problem_l3275_327510


namespace P_50_is_identity_l3275_327503

def P : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -4, -3]

theorem P_50_is_identity : P ^ 50 = 1 := by sorry

end P_50_is_identity_l3275_327503


namespace cost_of_thousand_gum_in_dollars_l3275_327582

/-- The cost of a single piece of gum in cents -/
def cost_of_one_gum : ℕ := 1

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of gum pieces we're calculating the cost for -/
def num_gum_pieces : ℕ := 1000

/-- Theorem: The cost of 1000 pieces of gum in dollars is 10.00 -/
theorem cost_of_thousand_gum_in_dollars : 
  (num_gum_pieces * cost_of_one_gum : ℚ) / cents_per_dollar = 10 := by
  sorry

end cost_of_thousand_gum_in_dollars_l3275_327582


namespace regular_fish_price_l3275_327530

/-- The regular price of fish per pound, given a 50% discount and half-pound package price -/
theorem regular_fish_price (discount_percent : ℚ) (discounted_half_pound_price : ℚ) : 
  discount_percent = 50 →
  discounted_half_pound_price = 3 →
  12 = (2 * discounted_half_pound_price) / (1 - discount_percent / 100) :=
by sorry

end regular_fish_price_l3275_327530


namespace max_gcd_abb_aba_l3275_327543

def abb (a b : ℕ) : ℕ := 100 * a + 11 * b

def aba (a b : ℕ) : ℕ := 101 * a + 10 * b

theorem max_gcd_abb_aba : 
  ∀ a b : ℕ, a ≠ b → a < 10 → b < 10 → 
  (∀ c d : ℕ, c ≠ d → c < 10 → d < 10 → 
    Nat.gcd (abb a b) (aba a b) ≥ Nat.gcd (abb c d) (aba c d)) → 
  Nat.gcd (abb a b) (aba a b) = 18 :=
by sorry

end max_gcd_abb_aba_l3275_327543


namespace valid_pairs_l3275_327560

def is_valid_pair (m n : ℕ) : Prop :=
  Nat.Prime m ∧ Nat.Prime n ∧ m < n ∧ n < 5 * m ∧ Nat.Prime (m + 3 * n)

theorem valid_pairs :
  ∀ m n : ℕ, is_valid_pair m n ↔ (m = 2 ∧ n = 3) ∨ (m = 2 ∧ n = 5) ∨ (m = 2 ∧ n = 7) :=
by sorry

end valid_pairs_l3275_327560


namespace M_subset_N_l3275_327592

-- Define the set M
def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2 : ℝ) * 180 + 45}

-- Define the set N
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4 : ℝ) * 180 + 45}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l3275_327592


namespace max_profit_l3275_327505

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

-- Define the sales volume function
def P (x : ℝ) : ℝ := 3 - 2 / (x + 1)

-- Define the profit function
def y (x : ℝ) : ℝ := 26 - 4 / (x + 1) - x

-- State the theorem
theorem max_profit (h : a > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ a → y x ≤ (if a ≥ 1 then 23 else 26 - 4 / (a + 1) - a)) ∧
  (if a ≥ 1 
   then y 1 = 23 
   else y a = 26 - 4 / (a + 1) - a) :=
sorry

end

end max_profit_l3275_327505


namespace judy_hits_percentage_l3275_327553

theorem judy_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 8)
  (h5 : total_hits ≥ home_runs + triples + doubles) :
  (((total_hits - (home_runs + triples + doubles)) : ℚ) / total_hits) * 100 = 74 := by
sorry

end judy_hits_percentage_l3275_327553


namespace milk_distribution_l3275_327556

def milk_problem (total_milk myeongseok_milk minjae_milk : Real) (mingu_extra : Real) : Prop :=
  let mingu_milk := myeongseok_milk + mingu_extra
  let friends_total := myeongseok_milk + mingu_milk + minjae_milk
  let remaining_milk := total_milk - friends_total
  (total_milk = 1) ∧ 
  (myeongseok_milk = 0.1) ∧ 
  (mingu_extra = 0.2) ∧ 
  (minjae_milk = 0.3) ∧ 
  (remaining_milk = 0.3)

theorem milk_distribution : 
  ∃ (total_milk myeongseok_milk minjae_milk mingu_extra : Real),
    milk_problem total_milk myeongseok_milk minjae_milk mingu_extra :=
by
  sorry

end milk_distribution_l3275_327556


namespace cubic_factorization_l3275_327512

theorem cubic_factorization (m : ℝ) : m^3 - 9*m = m*(m+3)*(m-3) := by sorry

end cubic_factorization_l3275_327512
