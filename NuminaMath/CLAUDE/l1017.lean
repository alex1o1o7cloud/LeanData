import Mathlib

namespace binary_decimal_base7_conversion_l1017_101753

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_decimal_base7_conversion :
  let binary := [true, false, true, true, false, true]
  binary_to_decimal binary = 45 ∧
  decimal_to_base7 45 = [6, 3] :=
by sorry

end binary_decimal_base7_conversion_l1017_101753


namespace contrapositive_equivalence_l1017_101772

theorem contrapositive_equivalence (a b : ℝ) :
  (∀ a b, a > b → a - 1 > b - 1) ↔ (∀ a b, a - 1 ≤ b - 1 → a ≤ b) :=
by sorry

end contrapositive_equivalence_l1017_101772


namespace sum_of_divisors_of_37_l1017_101794

theorem sum_of_divisors_of_37 (h : Nat.Prime 37) : 
  (Finset.filter (· ∣ 37) (Finset.range 38)).sum id = 38 := by
  sorry

end sum_of_divisors_of_37_l1017_101794


namespace work_hours_ratio_l1017_101720

def total_hours : ℕ := 157
def rebecca_hours : ℕ := 56

theorem work_hours_ratio (thomas_hours toby_hours : ℕ) : 
  thomas_hours + toby_hours + rebecca_hours = total_hours →
  toby_hours = thomas_hours + 10 →
  rebecca_hours = toby_hours - 8 →
  (toby_hours : ℚ) / (thomas_hours : ℚ) = 32 / 27 := by
  sorry

end work_hours_ratio_l1017_101720


namespace triangle_inequality_with_circumradius_l1017_101776

-- Define a structure for a triangle with its circumradius
structure Triangle :=
  (a b c : ℝ)
  (R : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hR : R > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (circumradius : R = (a * b * c) / (4 * area))
  (area : ℝ)
  (area_positive : area > 0)

-- State the theorem
theorem triangle_inequality_with_circumradius (t : Triangle) :
  1 / (t.a * t.b) + 1 / (t.b * t.c) + 1 / (t.c * t.a) ≥ 1 / (t.R ^ 2) :=
by sorry

end triangle_inequality_with_circumradius_l1017_101776


namespace negative_root_implies_a_less_than_neg_three_l1017_101789

theorem negative_root_implies_a_less_than_neg_three (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
by sorry

end negative_root_implies_a_less_than_neg_three_l1017_101789


namespace right_triangle_sin_y_l1017_101709

theorem right_triangle_sin_y (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 20) (h_b : b = 21) :
  let sin_y := a / c
  sin_y = 20 / 29 := by sorry

end right_triangle_sin_y_l1017_101709


namespace compound_interest_rate_l1017_101774

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P * (1 + r)^2 = 240 → 
  P * (1 + r) = 217.68707482993196 → 
  r = 0.1025 := by
sorry

end compound_interest_rate_l1017_101774


namespace gasoline_distribution_impossibility_l1017_101778

theorem gasoline_distribution_impossibility :
  ¬∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x + y + z = 50 ∧
  x = y + 10 ∧
  z + 26 = y :=
by sorry

end gasoline_distribution_impossibility_l1017_101778


namespace solve_equation_l1017_101792

theorem solve_equation : 
  ∃ x : ℚ, 64 + 5 * x / (180 / 3) = 65 ∧ x = 12 := by
  sorry

end solve_equation_l1017_101792


namespace angle_twice_complement_l1017_101795

theorem angle_twice_complement (x : ℝ) : 
  (x = 2 * (90 - x)) → x = 60 := by
  sorry

end angle_twice_complement_l1017_101795


namespace abs_m_minus_n_equals_three_l1017_101751

theorem abs_m_minus_n_equals_three (m n : ℝ) 
  (h1 : m * n = 4) 
  (h2 : m + n = 5) : 
  |m - n| = 3 := by
sorry

end abs_m_minus_n_equals_three_l1017_101751


namespace polynomial_divisibility_l1017_101724

theorem polynomial_divisibility (F : ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, F (x + y) - F x - F y = (x * y) * (F 1 - 1))
  (h2 : (F 2) % 5 = 0)
  (h3 : (F 5) % 2 = 0) :
  (F 7) % 10 = 0 := by
  sorry

end polynomial_divisibility_l1017_101724


namespace rectangle_area_is_9000_l1017_101700

/-- A rectangle WXYZ with given coordinates -/
structure Rectangle where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Z : ℝ × ℤ

/-- The area of a rectangle WXYZ -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- The theorem stating that the area of the given rectangle is 9000 -/
theorem rectangle_area_is_9000 (r : Rectangle) 
  (h1 : r.W = (2, 3))
  (h2 : r.X = (302, 23))
  (h3 : r.Z.1 = 4) :
  rectangleArea r = 9000 := by sorry

end rectangle_area_is_9000_l1017_101700


namespace inequality_solution_set_l1017_101740

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem inequality_solution_set (x : ℝ) :
  (0 < x ∧ f (Real.log x) + f (Real.log (1/x)) < 2 * f 1) ↔ (1/Real.exp 1 < x ∧ x < Real.exp 1) :=
sorry

end inequality_solution_set_l1017_101740


namespace total_miles_run_l1017_101735

theorem total_miles_run (xavier katie cole lily joe : ℝ) : 
  xavier = 3 * katie → 
  katie = 4 * cole → 
  lily = 5 * cole → 
  joe = 2 * lily → 
  xavier = 84 → 
  lily = 0.85 * joe → 
  xavier + katie + cole + lily + joe = 168.875 := by
sorry

end total_miles_run_l1017_101735


namespace carrie_pants_count_l1017_101743

/-- The number of pairs of pants Carrie bought -/
def pants_count : ℕ := 2

/-- The cost of a single shirt in dollars -/
def shirt_cost : ℕ := 8

/-- The cost of a single pair of pants in dollars -/
def pants_cost : ℕ := 18

/-- The cost of a single jacket in dollars -/
def jacket_cost : ℕ := 60

/-- The number of shirts Carrie bought -/
def shirts_count : ℕ := 4

/-- The number of jackets Carrie bought -/
def jackets_count : ℕ := 2

/-- The amount Carrie paid in dollars -/
def carrie_payment : ℕ := 94

theorem carrie_pants_count :
  shirts_count * shirt_cost + pants_count * pants_cost + jackets_count * jacket_cost = 2 * carrie_payment :=
by sorry

end carrie_pants_count_l1017_101743


namespace rationalize_denominator_l1017_101725

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 6 ∧ B = 4 ∧ C = -1 ∧ D = 1 ∧ E = 30 ∧ F = 12 := by
  sorry

end rationalize_denominator_l1017_101725


namespace objects_meeting_probability_l1017_101703

/-- The probability of two objects meeting in a coordinate plane --/
theorem objects_meeting_probability :
  let start_A : ℕ × ℕ := (0, 0)
  let start_B : ℕ × ℕ := (3, 5)
  let steps : ℕ := 5
  let prob_A_right : ℚ := 1/2
  let prob_A_up : ℚ := 1/2
  let prob_B_left : ℚ := 1/2
  let prob_B_down : ℚ := 1/2
  ∃ (meeting_prob : ℚ), meeting_prob = 31/128 := by
  sorry

end objects_meeting_probability_l1017_101703


namespace units_digit_of_k_squared_plus_two_to_k_l1017_101790

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2012^2 + 2^2012 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end units_digit_of_k_squared_plus_two_to_k_l1017_101790


namespace youngbin_line_position_l1017_101787

/-- Given a line of students with Youngbin in it, calculate the number of students in front of Youngbin. -/
def students_in_front (total : ℕ) (behind : ℕ) : ℕ :=
  total - behind - 1

/-- Theorem: There are 11 students in front of Youngbin given the problem conditions. -/
theorem youngbin_line_position : students_in_front 25 13 = 11 := by
  sorry

end youngbin_line_position_l1017_101787


namespace mixture_composition_l1017_101758

theorem mixture_composition (alcohol_volume : ℚ) (water_volume : ℚ) 
  (h1 : alcohol_volume = 3/5)
  (h2 : alcohol_volume / water_volume = 3/4) :
  water_volume = 4/5 := by
sorry

end mixture_composition_l1017_101758


namespace cost_of_900_candies_l1017_101770

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (num_candies : ℕ) : ℚ :=
  let candies_per_box : ℕ := 30
  let cost_per_box : ℚ := 7.5
  let discount_threshold : ℕ := 500
  let discount_rate : ℚ := 0.1
  let num_boxes : ℕ := num_candies / candies_per_box
  let discounted_cost_per_box : ℚ := if num_candies > discount_threshold then cost_per_box * (1 - discount_rate) else cost_per_box
  (num_boxes : ℚ) * discounted_cost_per_box

/-- The cost of 900 chocolate candies is $202.50 -/
theorem cost_of_900_candies : cost_of_candies 900 = 202.5 := by
  sorry

end cost_of_900_candies_l1017_101770


namespace canteen_to_bathroom_ratio_l1017_101734

/-- Represents the number of tables in the classroom -/
def num_tables : ℕ := 6

/-- Represents the number of students currently sitting at each table -/
def students_per_table : ℕ := 3

/-- Represents the number of girls who went to the bathroom -/
def girls_in_bathroom : ℕ := 3

/-- Represents the number of new groups added to the class -/
def new_groups : ℕ := 2

/-- Represents the number of students in each new group -/
def students_per_new_group : ℕ := 4

/-- Represents the number of countries from which foreign exchange students came -/
def num_countries : ℕ := 3

/-- Represents the number of foreign exchange students from each country -/
def students_per_country : ℕ := 3

/-- Represents the total number of students supposed to be in the class -/
def total_students : ℕ := 47

/-- Theorem stating the ratio of students who went to the canteen to girls who went to the bathroom -/
theorem canteen_to_bathroom_ratio :
  let students_present := num_tables * students_per_table
  let new_group_students := new_groups * students_per_new_group
  let foreign_students := num_countries * students_per_country
  let missing_students := girls_in_bathroom + new_group_students + foreign_students
  let canteen_students := total_students - students_present - missing_students
  (canteen_students : ℚ) / girls_in_bathroom = 3 := by
  sorry

end canteen_to_bathroom_ratio_l1017_101734


namespace roll_less_than_5_most_likely_l1017_101707

-- Define the probability of an event on a fair die
def prob (n : ℕ) : ℚ := n / 6

-- Define the events
def roll_6 : ℚ := prob 1
def roll_more_than_4 : ℚ := prob 2
def roll_less_than_4 : ℚ := prob 3
def roll_less_than_5 : ℚ := prob 4

-- Theorem statement
theorem roll_less_than_5_most_likely :
  roll_less_than_5 > roll_6 ∧
  roll_less_than_5 > roll_more_than_4 ∧
  roll_less_than_5 > roll_less_than_4 :=
sorry

end roll_less_than_5_most_likely_l1017_101707


namespace total_travel_time_l1017_101702

/-- Prove that the total time traveled is 4 hours -/
theorem total_travel_time (speed : ℝ) (distance_AB : ℝ) (h1 : speed = 60) (h2 : distance_AB = 120) :
  2 * distance_AB / speed = 4 := by
  sorry

end total_travel_time_l1017_101702


namespace min_dot_product_on_W_l1017_101750

/-- The trajectory W of point P -/
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 2 ∧ p.1 ≥ Real.sqrt 2}

/-- The dot product of two vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

/-- The origin point -/
def O : ℝ × ℝ := (0, 0)

theorem min_dot_product_on_W :
  ∀ A B : ℝ × ℝ, A ∈ W → B ∈ W → A ≠ B →
  ∀ C D : ℝ × ℝ, C ∈ W → D ∈ W →
  dot_product (C.1 - O.1, C.2 - O.2) (D.1 - O.1, D.2 - O.2) ≥
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) →
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) = 2 :=
by sorry

end min_dot_product_on_W_l1017_101750


namespace field_trip_absentees_prove_girls_absent_l1017_101775

/-- Given a field trip scenario, calculate the number of girls who couldn't join. -/
theorem field_trip_absentees (total_students : ℕ) (boys : ℕ) (girls_present : ℕ) : ℕ :=
  let girls_assigned := total_students - boys
  girls_assigned - girls_present

/-- Prove the number of girls who couldn't join the field trip. -/
theorem prove_girls_absent : field_trip_absentees 18 8 8 = 2 := by
  sorry

end field_trip_absentees_prove_girls_absent_l1017_101775


namespace triangle_area_l1017_101704

-- Define the triangle
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  cos_angle : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.side1 = 5 ∧ 
  t.side2 = 3 ∧ 
  5 * t.cos_angle^2 - 7 * t.cos_angle - 6 = 0

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t) : 
  (1/2) * t.side1 * t.side2 * Real.sqrt (1 - t.cos_angle^2) = 6 := by
  sorry

end triangle_area_l1017_101704


namespace correct_number_of_pupils_l1017_101783

/-- The number of pupils in a class where an error in one pupil's marks
    caused the class average to increase by half. -/
def number_of_pupils : ℕ :=
  let mark_increase : ℕ := 85 - 45
  let average_increase : ℚ := 1/2
  (2 * mark_increase : ℕ)

theorem correct_number_of_pupils :
  number_of_pupils = 80 :=
sorry

end correct_number_of_pupils_l1017_101783


namespace remainder_sum_l1017_101710

theorem remainder_sum (c d : ℤ) (hc : c % 60 = 58) (hd : d % 90 = 85) :
  (c + d) % 30 = 23 := by
  sorry

end remainder_sum_l1017_101710


namespace area_of_specific_trapezoid_l1017_101738

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The perimeter of the trapezoid -/
  perimeter : ℝ
  /-- The diagonal bisects the obtuse angle -/
  diagonalBisectsObtuseAngle : Prop

/-- The area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 96 -/
theorem area_of_specific_trapezoid :
  ∀ (t : IsoscelesTrapezoid),
    t.smallerBase = 3 ∧
    t.perimeter = 42 ∧
    t.diagonalBisectsObtuseAngle →
    areaOfTrapezoid t = 96 :=
  sorry

end area_of_specific_trapezoid_l1017_101738


namespace brand_comparison_l1017_101760

/-- Distribution of timing errors for brand A -/
def dist_A : List (ℝ × ℝ) := [(-1, 0.1), (0, 0.8), (1, 0.1)]

/-- Distribution of timing errors for brand B -/
def dist_B : List (ℝ × ℝ) := [(-2, 0.1), (-1, 0.2), (0, 0.4), (1, 0.2), (2, 0.1)]

/-- Expected value of a discrete random variable -/
def expected_value (dist : List (ℝ × ℝ)) : ℝ :=
  (dist.map (fun (x, p) => x * p)).sum

/-- Variance of a discrete random variable -/
def variance (dist : List (ℝ × ℝ)) : ℝ :=
  (dist.map (fun (x, p) => x^2 * p)).sum - (expected_value dist)^2

/-- Theorem stating the properties of brands A and B -/
theorem brand_comparison :
  expected_value dist_A = 0 ∧
  expected_value dist_B = 0 ∧
  variance dist_A = 0.2 ∧
  variance dist_B = 1.2 ∧
  variance dist_A < variance dist_B := by
  sorry

#check brand_comparison

end brand_comparison_l1017_101760


namespace cut_cube_properties_l1017_101773

/-- A cube with one corner cut off -/
structure CutCube where
  vertices : Finset (ℝ × ℝ × ℝ)
  faces : Finset (Finset (ℝ × ℝ × ℝ))

/-- Properties of a cube with one corner cut off -/
def is_valid_cut_cube (c : CutCube) : Prop :=
  c.vertices.card = 10 ∧ c.faces.card = 9

/-- Theorem: A cube with one corner cut off has 10 vertices and 9 faces -/
theorem cut_cube_properties (c : CutCube) (h : is_valid_cut_cube c) :
  c.vertices.card = 10 ∧ c.faces.card = 9 := by
  sorry


end cut_cube_properties_l1017_101773


namespace opposite_signs_and_greater_absolute_value_l1017_101705

theorem opposite_signs_and_greater_absolute_value (a b : ℝ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) ∧ 
  (a > 0 → |a| > |b|) ∧ 
  (b > 0 → |b| > |a|) := by
  sorry

end opposite_signs_and_greater_absolute_value_l1017_101705


namespace binomial_60_3_l1017_101719

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l1017_101719


namespace extremum_and_intersection_implies_m_range_l1017_101786

def f (x : ℝ) := x^3 - 3*x - 1

theorem extremum_and_intersection_implies_m_range :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1) ∨ f x ≥ f (-1)) →
  (∃ m : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) →
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) → 
  -3 < m ∧ m < 1 := by
sorry

end extremum_and_intersection_implies_m_range_l1017_101786


namespace square_difference_l1017_101785

theorem square_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 40 := by
sorry

end square_difference_l1017_101785


namespace sand_remaining_l1017_101791

/-- Given a truck with an initial amount of sand and an amount of sand lost during transit,
    prove that the remaining amount of sand is equal to the initial amount minus the lost amount. -/
theorem sand_remaining (initial_sand : ℝ) (sand_lost : ℝ) :
  initial_sand - sand_lost = initial_sand - sand_lost :=
by sorry

end sand_remaining_l1017_101791


namespace conic_is_ellipse_l1017_101762

/-- The equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-6)^2 + (y-4)^2) = 14

/-- The two focal points of the conic section --/
def focal_point1 : ℝ × ℝ := (0, -2)
def focal_point2 : ℝ × ℝ := (6, 4)

/-- Theorem stating that the given equation describes an ellipse --/
theorem conic_is_ellipse : ∃ (a b : ℝ) (center : ℝ × ℝ), 
  a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), conic_equation x y ↔ 
    ((x - center.1) / a)^2 + ((y - center.2) / b)^2 = 1 :=
sorry

end conic_is_ellipse_l1017_101762


namespace decimal_to_binary_98_l1017_101763

theorem decimal_to_binary_98 : 
  (98 : ℕ).digits 2 = [0, 1, 0, 0, 0, 1, 1] :=
sorry

end decimal_to_binary_98_l1017_101763


namespace equilateral_triangle_area_perimeter_ratio_l1017_101793

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by sorry

end equilateral_triangle_area_perimeter_ratio_l1017_101793


namespace ayen_exercise_time_l1017_101799

/-- Represents the total exercise time in minutes for a week -/
def weekly_exercise (
  weekday_jog : ℕ
  ) (tuesday_extra : ℕ) (friday_extra : ℕ) (saturday_jog : ℕ) (sunday_swim : ℕ) : ℚ :=
  let weekday_total := 3 * weekday_jog + (weekday_jog + tuesday_extra) + (weekday_jog + friday_extra)
  let jogging_total := weekday_total + saturday_jog
  let swimming_equivalent := (3 / 2) * sunday_swim
  (jogging_total + swimming_equivalent) / 60

/-- The theorem stating Ayen's total exercise time for the week -/
theorem ayen_exercise_time : 
  weekly_exercise 30 5 25 45 60 = (23 / 4) := by sorry

end ayen_exercise_time_l1017_101799


namespace piano_practice_minutes_l1017_101708

theorem piano_practice_minutes (practice_time_6days : ℕ) (practice_time_2days : ℕ) 
  (total_days : ℕ) (average_minutes : ℕ) :
  practice_time_6days = 100 →
  practice_time_2days = 80 →
  total_days = 9 →
  average_minutes = 100 →
  (6 * practice_time_6days + 2 * practice_time_2days + 
   (average_minutes * total_days - (6 * practice_time_6days + 2 * practice_time_2days))) / total_days = average_minutes :=
by
  sorry

end piano_practice_minutes_l1017_101708


namespace opposite_of_two_opposite_definition_l1017_101798

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 2 is -2
theorem opposite_of_two : opposite 2 = -2 := by
  -- The proof goes here
  sorry

-- Theorem proving the definition of opposite
theorem opposite_definition (x : ℝ) : x + opposite x = 0 := by
  -- The proof goes here
  sorry

end opposite_of_two_opposite_definition_l1017_101798


namespace six_less_than_twice_square_of_four_l1017_101727

theorem six_less_than_twice_square_of_four : (2 * 4^2) - 6 = 26 := by
  sorry

end six_less_than_twice_square_of_four_l1017_101727


namespace max_F_value_l1017_101722

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_four_digit : thousands ≥ 1 ∧ thousands ≤ 9

/-- Checks if a number is an "eternal number" -/
def is_eternal (n : FourDigitNumber) : Prop :=
  n.hundreds + n.tens + n.units = 12

/-- Swaps digits as described in the problem -/
def swap_digits (n : FourDigitNumber) : FourDigitNumber :=
  { thousands := n.hundreds
  , hundreds := n.thousands
  , tens := n.units
  , units := n.tens
  , is_four_digit := by sorry }

/-- Calculates F(M) as defined in the problem -/
def F (m : FourDigitNumber) : Int :=
  let n := swap_digits m
  let m_val := 1000 * m.thousands + 100 * m.hundreds + 10 * m.tens + m.units
  let n_val := 1000 * n.thousands + 100 * n.hundreds + 10 * n.tens + n.units
  (m_val - n_val) / 9

/-- Main theorem -/
theorem max_F_value (m : FourDigitNumber) 
  (h1 : is_eternal m)
  (h2 : m.thousands = m.hundreds - m.units)
  (h3 : (F m) % 9 = 0) :
  F m ≤ 9 ∧ ∃ (m' : FourDigitNumber), F m' = 9 :=
by sorry

end max_F_value_l1017_101722


namespace min_ones_in_sum_l1017_101730

/-- Count the number of '1's in the binary representation of an integer -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem min_ones_in_sum (a b : ℕ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (ca : countOnes a = 20041) 
  (cb : countOnes b = 20051) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ countOnes x = 20041 ∧ countOnes y = 20051 ∧ countOnes (x + y) = 1 := by
  sorry

end min_ones_in_sum_l1017_101730


namespace constant_phi_forms_cone_l1017_101756

/-- A point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The set of points satisfying φ = d -/
def ConstantPhiSet (d : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ = d}

/-- Definition of a cone in spherical coordinates -/
def IsCone (s : Set SphericalPoint) : Prop :=
  ∃ d : ℝ, s = ConstantPhiSet d

/-- Theorem: The set of points with constant φ forms a cone -/
theorem constant_phi_forms_cone (d : ℝ) :
  IsCone (ConstantPhiSet d) := by
  sorry

end constant_phi_forms_cone_l1017_101756


namespace fly_probabilities_l1017_101747

def fly_walk (n m : ℕ) : ℚ := (Nat.choose (n + m) n : ℚ) / (2 ^ (n + m))

def fly_walk_through (n₁ m₁ n₂ m₂ : ℕ) : ℚ :=
  (Nat.choose (n₁ + m₁) n₁ : ℚ) * (Nat.choose (n₂ + m₂) n₂) / (2 ^ (n₁ + m₁ + n₂ + m₂))

def fly_walk_circle : ℚ :=
  (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 * Nat.choose 9 4 : ℚ) / 2^18

theorem fly_probabilities :
  (fly_walk 8 10 = (Nat.choose 18 8 : ℚ) / 2^18) ∧
  (fly_walk_through 5 6 2 4 = ((Nat.choose 11 5 : ℚ) * Nat.choose 6 2) / 2^18) ∧
  (fly_walk_circle = (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 * Nat.choose 9 4 : ℚ) / 2^18) := by
  sorry

end fly_probabilities_l1017_101747


namespace triangle_incenter_inequality_l1017_101765

theorem triangle_incenter_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1/4 < ((a+b)*(b+c)*(c+a)) / ((a+b+c)^3) ∧ ((a+b)*(b+c)*(c+a)) / ((a+b+c)^3) ≤ 8/27 := by
  sorry

end triangle_incenter_inequality_l1017_101765


namespace product_of_three_numbers_l1017_101706

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 24)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 6 * z) :
  x * y * z = 126 := by sorry

end product_of_three_numbers_l1017_101706


namespace spiders_went_loose_l1017_101737

theorem spiders_went_loose (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (birds_sold puppies_adopted animals_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  initial_spiders = 15 →
  birds_sold = initial_birds / 2 →
  puppies_adopted = 3 →
  animals_left = 25 →
  initial_spiders - (animals_left - ((initial_birds - birds_sold) + (initial_puppies - puppies_adopted) + initial_cats)) = 7 := by
  sorry

end spiders_went_loose_l1017_101737


namespace cyclic_sum_inequality_l1017_101701

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b / (a + b + 2 * c) + b * c / (b + c + 2 * a) + c * a / (c + a + 2 * b)) ≤ (a + b + c) / 4 := by
  sorry

end cyclic_sum_inequality_l1017_101701


namespace find_a20_l1017_101714

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem find_a20 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 1 + a 3 + a 5 = 105) (h3 : a 2 + a 4 + a 6 = 99) : 
  a 20 = 1 := by sorry

end find_a20_l1017_101714


namespace age_and_marriage_relations_l1017_101733

-- Define the people
inductive Person : Type
| Roman : Person
| Oleg : Person
| Ekaterina : Person
| Zhanna : Person

-- Define the age relation
def olderThan : Person → Person → Prop := sorry

-- Define the marriage relation
def marriedTo : Person → Person → Prop := sorry

-- Theorem statement
theorem age_and_marriage_relations :
  -- Each person has a different age
  (∀ p q : Person, p ≠ q → (olderThan p q ∨ olderThan q p)) →
  -- Each husband is older than his wife
  (∀ p q : Person, marriedTo p q → olderThan p q) →
  -- Zhanna is older than Oleg
  olderThan Person.Zhanna Person.Oleg →
  -- There are exactly two married couples
  (∃! (p1 p2 q1 q2 : Person),
    p1 ≠ p2 ∧ q1 ≠ q2 ∧ p1 ≠ q1 ∧ p1 ≠ q2 ∧ p2 ≠ q1 ∧ p2 ≠ q2 ∧
    marriedTo p1 p2 ∧ marriedTo q1 q2) →
  -- Conclusion: Oleg is older than Ekaterina and Roman is the oldest and married to Zhanna
  olderThan Person.Oleg Person.Ekaterina ∧
  marriedTo Person.Roman Person.Zhanna ∧
  (∀ p : Person, p ≠ Person.Roman → olderThan Person.Roman p) :=
by sorry

end age_and_marriage_relations_l1017_101733


namespace season_games_count_l1017_101754

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The number of months in a season -/
def months_in_season : ℕ := 2

/-- The total number of baseball games in a season -/
def total_games : ℕ := games_per_month * months_in_season

theorem season_games_count : total_games = 14 := by
  sorry

end season_games_count_l1017_101754


namespace larger_number_of_pair_l1017_101768

theorem larger_number_of_pair (x y : ℝ) (h1 : x - y = 5) (h2 : x * y = 156) (h3 : x > y) :
  x = (5 + Real.sqrt 649) / 2 := by
  sorry

end larger_number_of_pair_l1017_101768


namespace seating_arrangement_l1017_101767

theorem seating_arrangement (n m : ℕ) (h1 : n = 6) (h2 : m = 4) : 
  (n.factorial / (n - m).factorial) = 360 := by
  sorry

end seating_arrangement_l1017_101767


namespace wine_bottle_prices_l1017_101736

-- Define the prices as real numbers
variable (A B C X Y : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := A + X = 3.50
def condition2 : Prop := B + X = 4.20
def condition3 : Prop := C + Y = 6.10
def condition4 : Prop := A = X + 1.50
def condition5 : Prop := B = X + 2.20
def condition6 : Prop := C = Y + 3.40

-- State the theorem to be proved
theorem wine_bottle_prices 
  (h1 : condition1 A X)
  (h2 : condition2 B X)
  (h3 : condition3 C Y)
  (h4 : condition4 A X)
  (h5 : condition5 B X)
  (h6 : condition6 C Y) :
  A = 2.50 ∧ B = 3.20 ∧ C = 4.75 ∧ X = 1.00 ∧ Y = 1.35 := by
  sorry

end wine_bottle_prices_l1017_101736


namespace ratio_composition_l1017_101717

theorem ratio_composition (a b c : ℚ) 
  (hab : a / b = 11 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 11 / 15 := by
  sorry

end ratio_composition_l1017_101717


namespace ruths_sandwiches_l1017_101711

theorem ruths_sandwiches (total : ℕ) (brother : ℕ) (first_cousin : ℕ) (other_cousins : ℕ) (left : ℕ) :
  total = 10 →
  brother = 2 →
  first_cousin = 2 →
  other_cousins = 2 →
  left = 3 →
  total - (brother + first_cousin + other_cousins + left) = 1 :=
by sorry

end ruths_sandwiches_l1017_101711


namespace largest_value_l1017_101726

theorem largest_value (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  (a - b > a) ∧ (a - b > a + b) ∧ (a - b > a * b) := by
  sorry

end largest_value_l1017_101726


namespace value_of_a_l1017_101723

theorem value_of_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 4) (h3 : c^2 / a = 4) : a = 2 := by
  sorry

end value_of_a_l1017_101723


namespace geometric_sequence_properties_l1017_101788

-- Define the geometric sequence and its sum
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)
def S (n : ℕ) : ℝ := (3^n - 1)

-- State the theorem
theorem geometric_sequence_properties :
  (a 1 + a 2 + a 3 = 26) ∧ 
  (S 6 = 728) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * 3^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → S (n + 1)^2 - S n * S (n + 2) = 4 * 3^n) :=
by sorry

end geometric_sequence_properties_l1017_101788


namespace cubic_root_sum_squares_l1017_101729

theorem cubic_root_sum_squares (a b c : ℝ) : 
  (a^3 - 4*a^2 + 7*a - 2 = 0) → 
  (b^3 - 4*b^2 + 7*b - 2 = 0) → 
  (c^3 - 4*c^2 + 7*c - 2 = 0) → 
  a^2 + b^2 + c^2 = 2 := by
sorry

end cubic_root_sum_squares_l1017_101729


namespace range_of_a_theorem_l1017_101759

-- Define the propositions p and q
def p (x : ℝ) : Prop := 4 / (x - 1) ≤ -1
def q (x a : ℝ) : Prop := x^2 - x < a^2 - a

-- Define the condition that ¬q is sufficient but not necessary for ¬p
def sufficient_not_necessary (a : ℝ) : Prop :=
  ∀ x, ¬(q x a) → ¬(p x) ∧ ∃ y, ¬(p y) ∧ q y a

-- Define the range of a
def range_of_a : Set ℝ := {a | a ∈ [0, 1] ∧ a ≠ 1/2}

-- State the theorem
theorem range_of_a_theorem :
  ∀ a, sufficient_not_necessary a ↔ a ∈ range_of_a :=
sorry

end range_of_a_theorem_l1017_101759


namespace absolute_value_inequality_l1017_101779

theorem absolute_value_inequality (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end absolute_value_inequality_l1017_101779


namespace largest_even_digit_multiple_of_8_l1017_101748

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

theorem largest_even_digit_multiple_of_8 :
  ∃ (n : ℕ), n = 8888 ∧
  has_only_even_digits n ∧
  n < 10000 ∧
  n % 8 = 0 ∧
  ∀ m : ℕ, has_only_even_digits m → m < 10000 → m % 8 = 0 → m ≤ n :=
sorry

end largest_even_digit_multiple_of_8_l1017_101748


namespace largest_n_satisfying_inequality_l1017_101741

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + n/8 + 1/8 < 1 ↔ n ≤ 4 :=
by sorry

end largest_n_satisfying_inequality_l1017_101741


namespace horner_method_v3_l1017_101749

def horner_polynomial (x : ℝ) : ℝ := 10 + 25*x - 8*x^2 + x^4 + 6*x^5 + 2*x^6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x + 6
  let v2 := v1 * x + 1
  v2 * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -36 :=
sorry

end horner_method_v3_l1017_101749


namespace right_triangle_max_ratio_l1017_101766

theorem right_triangle_max_ratio (a b c A : ℝ) : 
  a > 0 → b > 0 → c > 0 → A > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  A = (1/2) * a * b →  -- Area formula
  (a + b + A) / c ≤ (5/4) * Real.sqrt 2 := by
  sorry

end right_triangle_max_ratio_l1017_101766


namespace salt_solution_volume_l1017_101731

/-- Given a salt solution with a concentration of 15 grams per 1000 cubic centimeters,
    prove that 0.375 grams of salt corresponds to 25 cubic centimeters of solution. -/
theorem salt_solution_volume (concentration : ℝ) (volume : ℝ) (salt_amount : ℝ) :
  concentration = 15 / 1000 →
  salt_amount = 0.375 →
  volume * concentration = salt_amount →
  volume = 25 := by
  sorry

end salt_solution_volume_l1017_101731


namespace orange_juice_fraction_l1017_101782

theorem orange_juice_fraction (pitcher_capacity : ℚ) 
  (pitcher1_orange : ℚ) (pitcher1_apple : ℚ)
  (pitcher2_orange : ℚ) (pitcher2_apple : ℚ) :
  pitcher_capacity = 800 →
  pitcher1_orange = 1/4 →
  pitcher1_apple = 1/8 →
  pitcher2_orange = 1/5 →
  pitcher2_apple = 1/10 →
  (pitcher_capacity * pitcher1_orange + pitcher_capacity * pitcher2_orange) / (2 * pitcher_capacity) = 9/40 := by
  sorry

#check orange_juice_fraction

end orange_juice_fraction_l1017_101782


namespace raise_time_on_hoop_l1017_101739

/-- Time required to raise an object by a certain distance when wrapped around a rotating hoop -/
theorem raise_time_on_hoop (r : ℝ) (rpm : ℝ) (distance : ℝ) : 
  r > 0 → rpm > 0 → distance > 0 → 
  (distance / (2 * π * r)) * (60 / rpm) = 15 / π := by
  sorry

end raise_time_on_hoop_l1017_101739


namespace intersection_M_N_l1017_101716

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l1017_101716


namespace metallic_sheet_width_l1017_101796

/-- Given a rectangular metallic sheet with length 48 m and width w,
    if squares of 8 m are cut from each corner to form an open box with volume 5120 m³,
    then the width w of the original sheet is 36 m. -/
theorem metallic_sheet_width (w : ℝ) : 
  w > 0 →  -- Ensuring positive width
  (48 - 2 * 8) * (w - 2 * 8) * 8 = 5120 →  -- Volume equation
  w = 36 := by
sorry


end metallic_sheet_width_l1017_101796


namespace isosceles_triangle_quadratic_roots_l1017_101755

theorem isosceles_triangle_quadratic_roots (m n : ℝ) (k : ℝ) : 
  (m > 0 ∧ n > 0) →  -- positive side lengths
  (m = n ∨ m = 4 ∨ n = 4) →  -- isosceles condition
  (m ≠ n ∨ m ≠ 4) →  -- not equilateral
  (m + n > 4 ∧ m + 4 > n ∧ n + 4 > m) →  -- triangle inequality
  (m^2 - 6*m + k + 2 = 0) →  -- m is a root
  (n^2 - 6*n + k + 2 = 0) →  -- n is a root
  (k = 6 ∨ k = 7) :=
by sorry


end isosceles_triangle_quadratic_roots_l1017_101755


namespace square_difference_of_roots_l1017_101713

theorem square_difference_of_roots (α β : ℝ) : 
  (α^2 - 2*α - 4 = 0) → (β^2 - 2*β - 4 = 0) → (α - β)^2 = 20 := by
  sorry

end square_difference_of_roots_l1017_101713


namespace number_of_cube_nets_l1017_101777

/-- A net is a 2D arrangement of squares that can be folded to form a polyhedron -/
def Net : Type := Unit

/-- A cube net is a specific type of net that can be folded to form a cube -/
def CubeNet : Type := Net

/-- Function to count the number of distinct cube nets -/
def count_distinct_cube_nets : ℕ := sorry

/-- Theorem stating that the number of distinct cube nets is 11 -/
theorem number_of_cube_nets : count_distinct_cube_nets = 11 := by sorry

end number_of_cube_nets_l1017_101777


namespace max_rectangles_intersection_l1017_101721

/-- A rectangle in a plane --/
structure Rectangle where
  -- We don't need to define the specifics of a rectangle for this problem

/-- The number of intersection points between two rectangles --/
def intersection_points (r1 r2 : Rectangle) : ℕ := sorry

/-- The maximum number of intersection points between any two rectangles --/
def max_intersection_points : ℕ := sorry

/-- Theorem: The maximum number of intersection points between any two rectangles is 8 --/
theorem max_rectangles_intersection :
  max_intersection_points = 8 := by sorry

end max_rectangles_intersection_l1017_101721


namespace triangle_angle_measure_l1017_101769

theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 - c^2 + b^2 = -Real.sqrt 3 * a * b) : 
  Real.cos (Real.pi / 6) = (a^2 + b^2 - c^2) / (2 * a * b) := by
  sorry

end triangle_angle_measure_l1017_101769


namespace prize_points_l1017_101718

/-- The number of chocolate bunnies sold -/
def chocolate_bunnies : ℕ := 8

/-- The points per chocolate bunny -/
def points_per_bunny : ℕ := 100

/-- The number of Snickers bars needed -/
def snickers_bars : ℕ := 48

/-- The points per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- The total points needed for the prize -/
def total_points : ℕ := 2000

theorem prize_points :
  chocolate_bunnies * points_per_bunny + snickers_bars * points_per_snickers = total_points :=
by sorry

end prize_points_l1017_101718


namespace statement_a_statement_b_statement_c_incorrect_statement_d_main_theorem_l1017_101752

-- Statement A
theorem statement_a (x y : ℝ) (hx : x > 0) (hy : y > 0) : x / y + y / x ≥ 2 := by sorry

-- Statement B
theorem statement_b (x : ℝ) : (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2 := by sorry

-- Statement C (incorrect)
theorem statement_c_incorrect : ∃ x : ℝ, x > 0 ∧ x < 1 ∧ Real.log x / Real.log 10 + Real.log 10 / Real.log x < 2 := by sorry

-- Statement D
theorem statement_d (a : ℝ) (ha : a > 0) : (1 + a) * (1 + 1 / a) ≥ 4 := by sorry

-- Main theorem
theorem main_theorem : 
  (∀ x y : ℝ, x > 0 → y > 0 → x / y + y / x ≥ 2) ∧
  (∀ x : ℝ, (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2) ∧
  (∃ x : ℝ, x > 0 ∧ x < 1 ∧ Real.log x / Real.log 10 + Real.log 10 / Real.log x < 2) ∧
  (∀ a : ℝ, a > 0 → (1 + a) * (1 + 1 / a) ≥ 4) := by sorry

end statement_a_statement_b_statement_c_incorrect_statement_d_main_theorem_l1017_101752


namespace round_trip_percentage_l1017_101781

theorem round_trip_percentage (total_passengers : ℝ) 
  (h1 : total_passengers > 0) :
  let round_trip_with_car := 0.15 * total_passengers
  let round_trip_without_car := 0.6 * (round_trip_with_car / 0.4)
  (round_trip_with_car + round_trip_without_car) / total_passengers = 0.375 := by
sorry

end round_trip_percentage_l1017_101781


namespace mariams_neighborhood_homes_l1017_101757

/-- The number of homes in Mariam's neighborhood -/
def total_homes (homes_one_side : ℕ) (multiplier : ℕ) : ℕ :=
  homes_one_side + multiplier * homes_one_side

/-- Theorem stating the total number of homes in Mariam's neighborhood -/
theorem mariams_neighborhood_homes :
  total_homes 40 3 = 160 := by
  sorry

end mariams_neighborhood_homes_l1017_101757


namespace triangle_type_l1017_101715

theorem triangle_type (A : Real) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7/12) : 
  ∀ (B C : Real), 0 < B ∧ 0 < C → A + B + C = π → 
  (A < π/2 ∧ B < π/2 ∧ C < π/2) := by
  sorry

end triangle_type_l1017_101715


namespace knights_selection_ways_l1017_101771

/-- Represents the number of knights at the round table -/
def total_knights : ℕ := 12

/-- Represents the number of knights to be chosen -/
def knights_to_choose : ℕ := 5

/-- Represents the number of ways to choose knights in a linear arrangement -/
def linear_arrangements : ℕ := Nat.choose (total_knights - knights_to_choose + 1) knights_to_choose

/-- Represents the number of invalid arrangements (where first and last knights are adjacent) -/
def invalid_arrangements : ℕ := Nat.choose (total_knights - knights_to_choose - 1) (knights_to_choose - 2)

/-- Theorem stating the number of ways to choose knights under the given conditions -/
theorem knights_selection_ways : 
  linear_arrangements - invalid_arrangements = 36 := by sorry

end knights_selection_ways_l1017_101771


namespace avery_build_time_l1017_101712

theorem avery_build_time (tom_time : ℝ) (total_time : ℝ) : 
  tom_time = 4 →
  (1 / 2 + 1 / tom_time) + 1 / tom_time = 1 →
  2 = total_time :=
by sorry

end avery_build_time_l1017_101712


namespace equivalent_statements_l1017_101784

theorem equivalent_statements (A B : Prop) :
  ((A ∧ B) → ¬(A ∨ B)) ↔ ((A ∨ B) → ¬(A ∧ B)) := by
  sorry

end equivalent_statements_l1017_101784


namespace total_students_l1017_101764

/-- Represents the setup of students in lines -/
structure StudentLines where
  total_lines : ℕ
  students_per_line : ℕ
  left_position : ℕ
  right_position : ℕ

/-- Theorem stating the total number of students given the conditions -/
theorem total_students (setup : StudentLines) 
  (h1 : setup.total_lines = 5)
  (h2 : setup.left_position = 4)
  (h3 : setup.right_position = 9)
  (h4 : setup.students_per_line = setup.left_position + setup.right_position - 1) :
  setup.total_lines * setup.students_per_line = 60 := by
  sorry

end total_students_l1017_101764


namespace matrix_sum_theorem_l1017_101745

theorem matrix_sum_theorem (a b c : ℝ) 
  (h : a^4 + b^4 + c^4 - a^2*b^2 - a^2*c^2 - b^2*c^2 = 0) :
  (a^2 / (b^2 + c^2)) + (b^2 / (a^2 + c^2)) + (c^2 / (a^2 + b^2)) = 3/2 :=
by sorry

end matrix_sum_theorem_l1017_101745


namespace inequality_theorem_l1017_101742

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop :=
  |x^2 - 2*x - 8| ≤ 2 * |x - 4| * |x + 2|

-- Define the second condition for x > 1
def second_condition (x m : ℝ) : Prop :=
  x > 1 → x^2 - 2*x - 8 ≥ (m + 2)*x - m - 15

-- Theorem statement
theorem inequality_theorem :
  (∀ x : ℝ, inequality_condition x) ∧
  (∀ m : ℝ, (∀ x : ℝ, second_condition x m) → m ≤ 2) :=
sorry

end inequality_theorem_l1017_101742


namespace logarithmic_equation_solution_l1017_101732

theorem logarithmic_equation_solution (x : ℝ) (h1 : x > 1) :
  (Real.log x - 1) / Real.log 5 + 
  (Real.log (x^2 - 1)) / (Real.log 5 / 2) + 
  (Real.log (x - 1)) / (Real.log (1/5)) = 3 →
  x = Real.sqrt (5 * Real.sqrt 5 + 1) := by
sorry

end logarithmic_equation_solution_l1017_101732


namespace quiz_total_points_l1017_101728

/-- Represents a quiz with a specified number of questions, where each question after
    the first is worth a fixed number of points more than the preceding question. -/
structure Quiz where
  num_questions : ℕ
  point_increment : ℕ
  third_question_points : ℕ

/-- Calculates the total points for a given quiz. -/
def total_points (q : Quiz) : ℕ :=
  let first_question_points := q.third_question_points - 2 * q.point_increment
  let last_question_points := first_question_points + (q.num_questions - 1) * q.point_increment
  (first_question_points + last_question_points) * q.num_questions / 2

/-- Theorem stating that a quiz with 8 questions, where each question after the first
    is worth 4 points more than the preceding question, and the third question is
    worth 39 points, has a total of 360 points. -/
theorem quiz_total_points :
  ∀ (q : Quiz), q.num_questions = 8 ∧ q.point_increment = 4 ∧ q.third_question_points = 39 →
  total_points q = 360 :=
by
  sorry

end quiz_total_points_l1017_101728


namespace nonzero_real_solution_l1017_101797

theorem nonzero_real_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 1 / y = 12) (eq2 : y + 1 / x = 7 / 15) :
  x = 6 + 3 * Real.sqrt (8 / 7) ∨ x = 6 - 3 * Real.sqrt (8 / 7) :=
by sorry

end nonzero_real_solution_l1017_101797


namespace park_width_l1017_101744

/-- Given a rectangular park with specified length, tree density, and total number of trees,
    prove that the width of the park is as calculated. -/
theorem park_width (length : ℝ) (tree_density : ℝ) (total_trees : ℝ) (width : ℝ) : 
  length = 1000 →
  tree_density = 1 / 20 →
  total_trees = 100000 →
  width = total_trees / (length * tree_density) →
  width = 2000 :=
by sorry

end park_width_l1017_101744


namespace total_repair_time_l1017_101761

/-- Represents the time in minutes required for each repair task for different shoe types -/
structure ShoeRepairTime where
  buckle : ℕ
  strap : ℕ
  sole : ℕ

/-- Represents the number of shoes repaired in a session -/
structure SessionRepair where
  flat : ℕ
  sandal : ℕ
  highHeel : ℕ

/-- Calculates the total repair time for a given shoe type and quantity -/
def repairTime (time : ShoeRepairTime) (quantity : ℕ) : ℕ :=
  (time.buckle + time.strap + time.sole) * quantity

/-- Calculates the total repair time for a session -/
def sessionTime (flat : ShoeRepairTime) (sandal : ShoeRepairTime) (highHeel : ShoeRepairTime) (session : SessionRepair) : ℕ :=
  repairTime flat session.flat + repairTime sandal session.sandal + repairTime highHeel session.highHeel

theorem total_repair_time :
  let flat := ShoeRepairTime.mk 3 8 9
  let sandal := ShoeRepairTime.mk 4 5 0
  let highHeel := ShoeRepairTime.mk 6 12 10
  let session1 := SessionRepair.mk 6 4 3
  let session2 := SessionRepair.mk 4 7 5
  let breakTime := 15
  sessionTime flat sandal highHeel session1 + sessionTime flat sandal highHeel session2 + breakTime = 538 := by
  sorry

end total_repair_time_l1017_101761


namespace percent_relation_l1017_101780

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.14 * a) 
  (h2 : b = 0.35 * a) : 
  c = 0.4 * b := by
sorry

end percent_relation_l1017_101780


namespace unique_integer_solution_l1017_101746

theorem unique_integer_solution : 
  ∀ m n : ℕ+, 
    (m : ℚ) + n - (3 * m * n) / (m + n) = 2011 / 3 ↔ 
    ((m = 1144 ∧ n = 377) ∨ (m = 377 ∧ n = 1144)) := by
  sorry

end unique_integer_solution_l1017_101746
