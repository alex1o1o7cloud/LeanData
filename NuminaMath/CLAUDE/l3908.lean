import Mathlib

namespace a_fourth_plus_b_fourth_l3908_390861

theorem a_fourth_plus_b_fourth (a b : ℝ) 
  (h1 : a^2 - b^2 = 8) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 56 := by
sorry

end a_fourth_plus_b_fourth_l3908_390861


namespace range_of_t_l3908_390898

-- Define the quadratic function
def f (x t : ℝ) : ℝ := x^2 - 3*x + t

-- Define the solution set A
def A (t : ℝ) : Set ℝ := {x | f x t ≤ 0}

-- Define the condition for the intersection
def intersection_nonempty (t : ℝ) : Prop := 
  ∃ x, x ∈ A t ∧ x ≤ t

-- State the theorem
theorem range_of_t : 
  ∀ t : ℝ, intersection_nonempty t ↔ t ∈ Set.Icc 0 (9/4) :=
sorry

end range_of_t_l3908_390898


namespace systematic_sampling_theorem_l3908_390866

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstGroupSelection : ℕ) (groupNumber : ℕ) : ℕ :=
  firstGroupSelection + (groupNumber - 1) * (totalStudents / sampleSize)

/-- Theorem: In a systematic sampling of 400 students with a sample size of 20,
    if the selected number from the first group is 12,
    then the selected number from the 14th group is 272. -/
theorem systematic_sampling_theorem :
  systematicSample 400 20 12 14 = 272 := by
  sorry

#eval systematicSample 400 20 12 14

end systematic_sampling_theorem_l3908_390866


namespace f_19_equals_zero_l3908_390874

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period_two_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

-- Theorem statement
theorem f_19_equals_zero 
  (h1 : is_even f) 
  (h2 : has_period_two_negation f) : 
  f 19 = 0 := by sorry

end f_19_equals_zero_l3908_390874


namespace test_score_combination_l3908_390887

theorem test_score_combination :
  ∀ (x y z : ℕ),
    x + y + z = 6 →
    8 * x + 2 * y = 20 →
    x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end test_score_combination_l3908_390887


namespace female_muscovy_percentage_problem_l3908_390886

/-- The percentage of female Muscovy ducks -/
def female_muscovy_percentage (total_ducks : ℕ) (muscovy_percentage : ℚ) (female_muscovy : ℕ) : ℚ :=
  (female_muscovy : ℚ) / (muscovy_percentage * total_ducks) * 100

theorem female_muscovy_percentage_problem :
  female_muscovy_percentage 40 (1/2) 6 = 30 := by
  sorry

end female_muscovy_percentage_problem_l3908_390886


namespace repeating_decimal_sum_l3908_390855

def repeating_decimal (a b c : ℕ) : ℚ := (a * 100 + b * 10 + c : ℚ) / 999

theorem repeating_decimal_sum :
  repeating_decimal 2 3 4 - repeating_decimal 5 6 7 + repeating_decimal 8 9 1 = 186 / 333 := by
  sorry

end repeating_decimal_sum_l3908_390855


namespace john_change_theorem_l3908_390851

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total payment in cents given the number of each coin type -/
def total_payment (quarters dimes nickels : ℕ) : ℕ :=
  quarters * coin_value "quarter" + dimes * coin_value "dime" + nickels * coin_value "nickel"

/-- Calculates the change received given the total payment and the cost of the item -/
def change_received (payment cost : ℕ) : ℕ :=
  payment - cost

theorem john_change_theorem (candy_cost : ℕ) (h1 : candy_cost = 131) :
  change_received (total_payment 4 3 1) candy_cost = 4 := by
  sorry

end john_change_theorem_l3908_390851


namespace possible_values_of_d_over_a_l3908_390868

theorem possible_values_of_d_over_a (a d : ℝ) (h1 : a^2 - 6*a*d + 8*d^2 = 0) (h2 : a ≠ 0) :
  d/a = 1/2 ∨ d/a = 1/4 := by
sorry

end possible_values_of_d_over_a_l3908_390868


namespace james_total_spent_l3908_390822

def entry_fee : ℕ := 20
def num_rounds : ℕ := 2
def num_friends : ℕ := 5
def james_drinks : ℕ := 6
def drink_cost : ℕ := 6
def food_cost : ℕ := 14
def tip_percentage : ℚ := 30 / 100

def total_spent : ℕ := 163

theorem james_total_spent : 
  entry_fee + 
  (num_rounds * num_friends * drink_cost) + 
  (james_drinks * drink_cost) + 
  food_cost + 
  (((num_rounds * num_friends * drink_cost) + (james_drinks * drink_cost) + food_cost : ℚ) * tip_percentage).floor = 
  total_spent := by
  sorry

end james_total_spent_l3908_390822


namespace salt_mixture_percentage_l3908_390840

theorem salt_mixture_percentage : 
  let initial_volume : ℝ := 70
  let initial_concentration : ℝ := 0.20
  let added_volume : ℝ := 70
  let added_concentration : ℝ := 0.60
  let total_volume : ℝ := initial_volume + added_volume
  let final_concentration : ℝ := (initial_volume * initial_concentration + added_volume * added_concentration) / total_volume
  final_concentration = 0.40 := by sorry

end salt_mixture_percentage_l3908_390840


namespace mosquito_shadow_speed_l3908_390808

/-- The speed of a mosquito's shadow on the bottom of a water body. -/
def shadow_speed (v : ℝ) (cos_beta : ℝ) : Set ℝ :=
  {0, 2 * v * cos_beta}

/-- Theorem: Given the conditions of the mosquito problem, the speed of the shadow is either 0 m/s or 0.8 m/s. -/
theorem mosquito_shadow_speed 
  (v : ℝ) 
  (t : ℝ) 
  (h : ℝ) 
  (cos_theta : ℝ) 
  (cos_beta : ℝ) 
  (hv : v = 0.5)
  (ht : t = 20)
  (hh : h = 6)
  (hcos_theta : cos_theta = 0.6)
  (hcos_beta : cos_beta = 0.8)
  : shadow_speed v cos_beta = {0, 0.8} := by
  sorry

#check mosquito_shadow_speed

end mosquito_shadow_speed_l3908_390808


namespace smallest_dividend_l3908_390894

theorem smallest_dividend (q r : ℕ) (h1 : q = 12) (h2 : r = 3) :
  ∃ (a b : ℕ), a = b * q + r ∧ b > r ∧ ∀ (a' b' : ℕ), (a' = b' * q + r ∧ b' > r) → a ≤ a' :=
by sorry

end smallest_dividend_l3908_390894


namespace similar_triangles_problem_l3908_390837

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- The problem statement -/
theorem similar_triangles_problem 
  (t1 t2 : Triangle)  -- Two triangles
  (h1 : t1.area > t2.area)  -- t1 is the larger triangle
  (h2 : t1.area - t2.area = 32)  -- Area difference is 32
  (h3 : ∃ k : ℕ, t1.area / t2.area = k^2)  -- Ratio of areas is square of an integer
  (h4 : ∃ n : ℕ, t2.area = n)  -- Smaller triangle area is an integer
  (h5 : t2.side = 4)  -- Side of smaller triangle is 4
  : t1.side = 12 := by
  sorry

end similar_triangles_problem_l3908_390837


namespace square_equation_solution_l3908_390895

theorem square_equation_solution : 
  ∃ x : ℚ, ((3 * x + 15)^2 = 3 * (4 * x + 40)) ∧ (x = -5/3 ∨ x = -7) :=
by sorry

end square_equation_solution_l3908_390895


namespace percentage_women_red_hair_men_dark_hair_l3908_390820

theorem percentage_women_red_hair_men_dark_hair (
  women_fair_hair : Real) (women_dark_hair : Real) (women_red_hair : Real)
  (men_fair_hair : Real) (men_dark_hair : Real) (men_red_hair : Real)
  (h1 : women_fair_hair = 30)
  (h2 : women_dark_hair = 28)
  (h3 : women_red_hair = 12)
  (h4 : men_fair_hair = 20)
  (h5 : men_dark_hair = 35)
  (h6 : men_red_hair = 5)
  : women_red_hair + men_dark_hair = 47 := by
  sorry

end percentage_women_red_hair_men_dark_hair_l3908_390820


namespace candidate_vote_difference_l3908_390878

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 4500 →
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 1350 :=
by sorry

end candidate_vote_difference_l3908_390878


namespace even_product_probability_l3908_390864

def ten_sided_die := Finset.range 10

theorem even_product_probability :
  let outcomes := ten_sided_die.product ten_sided_die
  (outcomes.filter (fun (x, y) => (x + 1) * (y + 1) % 2 = 0)).card / outcomes.card = 3 / 4 := by
  sorry

end even_product_probability_l3908_390864


namespace parabola_translation_l3908_390843

/-- Represents a vertical translation of a function -/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f x + k

/-- Represents a horizontal translation of a function -/
def horizontalTranslation (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x + h)

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := fun x ↦ x^2

/-- The resulting parabola after translation -/
def resultingParabola : ℝ → ℝ := fun x ↦ (x + 1)^2 + 3

theorem parabola_translation :
  verticalTranslation (horizontalTranslation originalParabola 1) 3 = resultingParabola := by
  sorry

end parabola_translation_l3908_390843


namespace tessa_apples_left_l3908_390885

/-- The number of apples Tessa has left after making a pie -/
def apples_left (initial : ℝ) (gift : ℝ) (pie_requirement : ℝ) : ℝ :=
  initial + gift - pie_requirement

/-- Theorem: Given the initial conditions, Tessa will have 11.25 apples left -/
theorem tessa_apples_left :
  apples_left 10.0 5.5 4.25 = 11.25 := by
  sorry

end tessa_apples_left_l3908_390885


namespace registration_cost_per_vehicle_l3908_390804

theorem registration_cost_per_vehicle 
  (num_dirt_bikes : ℕ) 
  (cost_per_dirt_bike : ℕ) 
  (num_off_road : ℕ) 
  (cost_per_off_road : ℕ) 
  (total_cost : ℕ) 
  (h1 : num_dirt_bikes = 3)
  (h2 : cost_per_dirt_bike = 150)
  (h3 : num_off_road = 4)
  (h4 : cost_per_off_road = 300)
  (h5 : total_cost = 1825) :
  (total_cost - (num_dirt_bikes * cost_per_dirt_bike + num_off_road * cost_per_off_road)) / (num_dirt_bikes + num_off_road) = 25 := by
    sorry

end registration_cost_per_vehicle_l3908_390804


namespace smallest_common_factor_l3908_390805

theorem smallest_common_factor (n : ℕ) : n > 0 ∧ 
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 4)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 4))) →
  n = 1 := by sorry

end smallest_common_factor_l3908_390805


namespace min_sum_reciprocal_distances_l3908_390854

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through a point with a given angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Curve in polar form -/
structure PolarCurve where
  equation : ℝ → ℝ

/-- Function to calculate the minimum sum of reciprocal distances -/
noncomputable def minSumReciprocalDistances (l : Line) (c : PolarCurve) : ℝ :=
  sorry

/-- Theorem stating the minimum value of the sum of reciprocal distances -/
theorem min_sum_reciprocal_distances :
  let p := Point.mk 1 2
  let l := Line.mk p α
  let c := PolarCurve.mk (fun θ ↦ 6 * Real.sin θ)
  minSumReciprocalDistances l c = 2 * Real.sqrt 7 / 7 := by sorry

end min_sum_reciprocal_distances_l3908_390854


namespace last_four_digits_are_user_number_l3908_390806

/-- Represents a mobile phone number -/
structure MobilePhoneNumber where
  digits : Fin 11 → Nat
  network_id : Fin 3 → Nat
  area_code : Fin 3 → Nat
  user_number : Fin 4 → Nat

/-- The structure of a mobile phone number -/
def mobile_number_structure (m : MobilePhoneNumber) : Prop :=
  (∀ i : Fin 3, m.network_id i = m.digits i) ∧
  (∀ i : Fin 3, m.area_code i = m.digits (i + 3)) ∧
  (∀ i : Fin 4, m.user_number i = m.digits (i + 7))

/-- Theorem stating that the last 4 digits of a mobile phone number represent the user number -/
theorem last_four_digits_are_user_number (m : MobilePhoneNumber) 
  (h : mobile_number_structure m) : 
  ∀ i : Fin 4, m.user_number i = m.digits (i + 7) := by
  sorry

end last_four_digits_are_user_number_l3908_390806


namespace michael_work_days_l3908_390816

-- Define the work rates for Michael, Adam, and Lisa
def M : ℚ := 1 / 40
def A : ℚ := 1 / 60
def L : ℚ := 1 / 60

-- Define the total work as 1 unit
def total_work : ℚ := 1

-- Theorem stating the conditions and the result to be proved
theorem michael_work_days :
  -- Condition 1: Michael, Adam, and Lisa can do the work together in 15 days
  M + A + L = 1 / 15 →
  -- Condition 2: After 10 days of working together, 2/3 of the work is completed
  (M + A + L) * 10 = 2 / 3 →
  -- Condition 3: Adam and Lisa complete the remaining 1/3 of the work in 8 days
  (A + L) * 8 = 1 / 3 →
  -- Conclusion: Michael takes 40 days to complete the work separately
  total_work / M = 40 :=
by sorry


end michael_work_days_l3908_390816


namespace abs_inequality_implies_a_greater_than_two_l3908_390809

theorem abs_inequality_implies_a_greater_than_two :
  (∀ x : ℝ, |x + 3| - |x + 1| - 2 * a + 2 < 0) → a > 2 :=
by sorry

end abs_inequality_implies_a_greater_than_two_l3908_390809


namespace quadratic_function_range_l3908_390827

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The range of a function -/
def Range (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ y, y ∈ S ↔ ∃ x, f x = y

/-- The theorem statement -/
theorem quadratic_function_range
  (f g : ℝ → ℝ)
  (h1 : f = g)
  (h2 : QuadraticFunction f)
  (h3 : Range (f ∘ g) (Set.Ici 0)) :
  Range (fun x ↦ g x) (Set.Ici 0) := by
sorry

end quadratic_function_range_l3908_390827


namespace inequality_properties_l3908_390825

theorem inequality_properties (a b c : ℝ) :
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c : ℝ), a > b → a * (2^c) > b * (2^c)) :=
by sorry

end inequality_properties_l3908_390825


namespace drews_age_l3908_390872

theorem drews_age (sam_current_age : ℕ) (h1 : sam_current_age = 46) :
  ∃ drew_current_age : ℕ,
    drew_current_age = 12 ∧
    sam_current_age + 5 = 3 * (drew_current_age + 5) :=
by sorry

end drews_age_l3908_390872


namespace root_equation_implies_expression_value_l3908_390852

theorem root_equation_implies_expression_value (m : ℝ) : 
  2 * m^2 + 3 * m - 1 = 0 → 4 * m^2 + 6 * m - 2019 = -2017 := by
  sorry

end root_equation_implies_expression_value_l3908_390852


namespace b_95_mod_64_l3908_390860

/-- The sequence b_n defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The theorem stating that b_95 ≡ 48 (mod 64) -/
theorem b_95_mod_64 : b 95 ≡ 48 [ZMOD 64] := by
  sorry

end b_95_mod_64_l3908_390860


namespace equilateral_triangle_side_length_l3908_390824

/-- Three concentric circles with radii 1, 2, and 3 units -/
def circles := {r : ℝ | r = 1 ∨ r = 2 ∨ r = 3}

/-- A point on one of the circles -/
structure CirclePoint where
  x : ℝ
  y : ℝ
  r : ℝ
  on_circle : x^2 + y^2 = r^2
  radius_valid : r ∈ circles

/-- An equilateral triangle formed by points on the circles -/
structure EquilateralTriangle where
  A : CirclePoint
  B : CirclePoint
  C : CirclePoint
  equilateral : (A.x - B.x)^2 + (A.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2 ∧
                (B.x - C.x)^2 + (B.y - C.y)^2 = (C.x - A.x)^2 + (C.y - A.y)^2
  on_different_circles : A.r ≠ B.r ∧ B.r ≠ C.r ∧ C.r ≠ A.r

/-- The theorem stating that the side length of the equilateral triangle is √7 -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangle) : 
  (triangle.A.x - triangle.B.x)^2 + (triangle.A.y - triangle.B.y)^2 = 7 := by
  sorry

end equilateral_triangle_side_length_l3908_390824


namespace jed_cards_40_after_4_weeks_l3908_390850

/-- The number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + 6 * weeks - 2 * (weeks / 2)

/-- The theorem stating that Jed will have 40 cards after 4 weeks -/
theorem jed_cards_40_after_4_weeks :
  cards_after_weeks 20 4 = 40 := by sorry

end jed_cards_40_after_4_weeks_l3908_390850


namespace cyclic_equation_system_solution_l3908_390844

theorem cyclic_equation_system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : (x₃ + x₄ + x₅)^5 = 3*x₁)
  (h₂ : (x₄ + x₅ + x₁)^5 = 3*x₂)
  (h₃ : (x₅ + x₁ + x₂)^5 = 3*x₃)
  (h₄ : (x₁ + x₂ + x₃)^5 = 3*x₄)
  (h₅ : (x₂ + x₃ + x₄)^5 = 3*x₅)
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0) (pos₄ : x₄ > 0) (pos₅ : x₅ > 0) :
  x₁ = 1/3 ∧ x₂ = 1/3 ∧ x₃ = 1/3 ∧ x₄ = 1/3 ∧ x₅ = 1/3 := by
  sorry

end cyclic_equation_system_solution_l3908_390844


namespace stating_last_passenger_seat_probability_l3908_390849

/-- 
Represents the probability that the last passenger sits in their assigned seat
given n seats and n passengers, where the first passenger (Absent-Minded Scientist)
takes a random seat, and subsequent passengers follow the described seating rules.
-/
def last_passenger_correct_seat_prob (n : ℕ) : ℚ :=
  if n ≥ 2 then 1/2 else 0

/-- 
Theorem stating that for any number of seats n ≥ 2, the probability that 
the last passenger sits in their assigned seat is 1/2.
-/
theorem last_passenger_seat_probability (n : ℕ) (h : n ≥ 2) : 
  last_passenger_correct_seat_prob n = 1/2 := by
  sorry

end stating_last_passenger_seat_probability_l3908_390849


namespace line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l3908_390879

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem for the first line
theorem line_through_P_and_origin :
  ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (x + y = 0 ↔ (∃ t : ℝ, x = t * P.1 ∧ y = t * P.2)) :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ :
  ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (2 * x + y + 2 = 0 ↔ 
    (∃ t : ℝ, x = P.1 + t * 1 ∧ y = P.2 + t * (-2) ∧ 
    (1 * (-2) + 2 * 1 = 0))) :=
sorry

end line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l3908_390879


namespace outlets_per_room_l3908_390881

theorem outlets_per_room (total_rooms : ℕ) (total_outlets : ℕ) (h1 : total_rooms = 7) (h2 : total_outlets = 42) :
  total_outlets / total_rooms = 6 := by
  sorry

end outlets_per_room_l3908_390881


namespace divisible_by_nine_l3908_390807

theorem divisible_by_nine (A : ℕ) : A < 10 → (83 * 1000 + A * 100 + 5) % 9 = 0 ↔ A = 2 := by
  sorry

end divisible_by_nine_l3908_390807


namespace arithmetic_expression_equality_l3908_390836

theorem arithmetic_expression_equality : 1874 + 230 / 46 - 874 * 2 = 131 := by
  sorry

end arithmetic_expression_equality_l3908_390836


namespace merry_go_round_revolutions_l3908_390803

theorem merry_go_round_revolutions (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) (h1 : outer_radius = 30) (h2 : inner_radius = 10) 
  (h3 : outer_revolutions = 25) : 
  ∃ inner_revolutions : ℕ, 
    inner_revolutions * inner_radius * 2 * Real.pi = outer_revolutions * outer_radius * 2 * Real.pi ∧ 
    inner_revolutions = 75 := by
  sorry

end merry_go_round_revolutions_l3908_390803


namespace intersection_of_A_and_B_l3908_390880

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (Real.log x / Real.log 10)}
def B : Set ℝ := {x | 1 / x ≥ 1 / 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l3908_390880


namespace new_average_and_variance_l3908_390865

/-- Given three numbers with average 5 and variance 2, prove that adding 1 results in four numbers with average 4 and variance 4.5 -/
theorem new_average_and_variance 
  (x y z : ℝ) 
  (h_avg : (x + y + z) / 3 = 5)
  (h_var : ((x - 5)^2 + (y - 5)^2 + (z - 5)^2) / 3 = 2) :
  let new_numbers := [x, y, z, 1]
  ((x + y + z + 1) / 4 = 4) ∧ 
  (((x - 4)^2 + (y - 4)^2 + (z - 4)^2 + (1 - 4)^2) / 4 = 4.5) := by
sorry


end new_average_and_variance_l3908_390865


namespace first_fifth_mile_charge_l3908_390831

/-- Represents the charge structure of a taxi company -/
structure TaxiCharge where
  first_fifth_mile : ℝ
  per_additional_fifth : ℝ

/-- Calculates the total charge for a given distance -/
def total_charge (c : TaxiCharge) (distance : ℝ) : ℝ :=
  c.first_fifth_mile + c.per_additional_fifth * (distance * 5 - 1)

/-- Theorem stating the charge for the first 1/5 mile -/
theorem first_fifth_mile_charge (c : TaxiCharge) :
  c.per_additional_fifth = 0.40 →
  total_charge c 8 = 18.10 →
  c.first_fifth_mile = 2.50 := by
sorry

end first_fifth_mile_charge_l3908_390831


namespace original_number_proof_l3908_390892

theorem original_number_proof (x : ℝ) : 
  (1.25 * x - 0.70 * x = 22) → x = 40 := by
  sorry

end original_number_proof_l3908_390892


namespace solution_satisfies_system_l3908_390888

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * Real.log 3 / Real.log 2 + y = Real.log 18 / Real.log 2

def equation2 (x y : ℝ) : Prop := (5 : ℝ)^x = 25^y

-- Theorem statement
theorem solution_satisfies_system :
  equation1 2 1 ∧ equation2 2 1 := by
  sorry

end solution_satisfies_system_l3908_390888


namespace angle_through_point_l3908_390813

/-- 
Given an angle α whose terminal side passes through point P(1, 2) in a plane coordinate system,
prove that:
1. tan α = 2
2. (sin α + 2 cos α) / (2 sin α - cos α) = 4/3
-/
theorem angle_through_point (α : Real) : 
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.tan α = 2 ∧ (Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 4/3 := by
  sorry

end angle_through_point_l3908_390813


namespace exists_sum_of_five_squares_l3908_390833

theorem exists_sum_of_five_squares : 
  ∃ (n : ℕ) (a b c d e : ℤ), 
    (n : ℤ)^2 = a^2 + b^2 + c^2 + d^2 + e^2 ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
     c ≠ d ∧ c ≠ e ∧ 
     d ≠ e) ∧
    (a = 7 ∨ b = 7 ∨ c = 7 ∨ d = 7 ∨ e = 7) :=
by sorry

end exists_sum_of_five_squares_l3908_390833


namespace condo_units_count_l3908_390848

/-- Represents a condo development with regular and penthouse floors. -/
structure Condo where
  total_floors : Nat
  penthouse_floors : Nat
  regular_units : Nat
  penthouse_units : Nat

/-- Calculates the total number of units in a condo. -/
def total_units (c : Condo) : Nat :=
  (c.total_floors - c.penthouse_floors) * c.regular_units + c.penthouse_floors * c.penthouse_units

/-- Theorem stating that a condo with the given specifications has 256 units. -/
theorem condo_units_count : 
  let c : Condo := {
    total_floors := 23,
    penthouse_floors := 2,
    regular_units := 12,
    penthouse_units := 2
  }
  total_units c = 256 := by
  sorry

#check condo_units_count

end condo_units_count_l3908_390848


namespace rectangle_area_l3908_390899

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 112 → l * b = 588 := by
sorry

end rectangle_area_l3908_390899


namespace x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l3908_390859

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ ¬(x > 1)) :=
by sorry

end x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l3908_390859


namespace garrison_provisions_duration_l3908_390819

/-- The number of days provisions last for a garrison with reinforcements --/
def provisions_duration (initial_men : ℕ) (reinforcement_men : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_men * days_before_reinforcement + (initial_men + reinforcement_men) * days_after_reinforcement) / initial_men

/-- Theorem stating that given the problem conditions, the provisions were supposed to last 54 days initially --/
theorem garrison_provisions_duration :
  provisions_duration 2000 1600 18 20 = 54 := by
  sorry

end garrison_provisions_duration_l3908_390819


namespace max_value_sqrt_sum_l3908_390853

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -3/2)
  (z_ge : z ≥ -2) :
  (∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 2 ∧ 
   x₀ ≥ -1 ∧ y₀ ≥ -3/2 ∧ z₀ ≥ -2 ∧
   Real.sqrt (5 * x₀ + 5) + Real.sqrt (4 * y₀ + 6) + Real.sqrt (6 * z₀ + 10) = Real.sqrt 93) ∧
  (∀ (a b c : ℝ), a + b + c = 2 → 
   a ≥ -1 → b ≥ -3/2 → c ≥ -2 →
   Real.sqrt (5 * a + 5) + Real.sqrt (4 * b + 6) + Real.sqrt (6 * c + 10) ≤ Real.sqrt 93) :=
by sorry

end max_value_sqrt_sum_l3908_390853


namespace square_state_after_2010_transforms_l3908_390856

/-- Represents the four possible states of the square labeling -/
inductive SquareState
  | BADC
  | DCBA
  | ABCD
  | CDAB

/-- Applies one transformation (reflection then rotation) to the square -/
def transform (s : SquareState) : SquareState :=
  match s with
  | SquareState.BADC => SquareState.DCBA
  | SquareState.DCBA => SquareState.ABCD
  | SquareState.ABCD => SquareState.DCBA
  | SquareState.CDAB => SquareState.BADC

/-- Applies n transformations to the initial square state -/
def applyNTransforms (n : Nat) : SquareState :=
  match n with
  | 0 => SquareState.BADC
  | n + 1 => transform (applyNTransforms n)

theorem square_state_after_2010_transforms :
  applyNTransforms 2010 = SquareState.DCBA := by
  sorry

end square_state_after_2010_transforms_l3908_390856


namespace variance_of_specific_set_l3908_390863

theorem variance_of_specific_set (a : ℝ) : 
  (5 + 8 + a + 7 + 4) / 5 = a → 
  ((5 - a)^2 + (8 - a)^2 + (a - a)^2 + (7 - a)^2 + (4 - a)^2) / 5 = 2 := by
  sorry

end variance_of_specific_set_l3908_390863


namespace parabola_hyperbola_tangency_l3908_390869

theorem parabola_hyperbola_tangency (n : ℝ) : 
  (∃ x y : ℝ, y = x^2 + 6 ∧ y^2 - n*x^2 = 4 ∧ 
    ∀ x' y' : ℝ, y' = x'^2 + 6 → y'^2 - n*x'^2 = 4 → (x', y') = (x, y)) →
  (n = 12 + 4*Real.sqrt 7 ∨ n = 12 - 4*Real.sqrt 7) :=
sorry

end parabola_hyperbola_tangency_l3908_390869


namespace fraction_equality_l3908_390830

theorem fraction_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4) :
  (a + b + c) / (2*a + b - c) = 3 := by
sorry

end fraction_equality_l3908_390830


namespace complex_equation_solution_l3908_390802

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l3908_390802


namespace race_outcomes_l3908_390896

/-- The number of contestants in the race -/
def num_contestants : ℕ := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def podium_positions : ℕ := 3

/-- No ties are allowed in the race -/
axiom no_ties : True

/-- The number of different podium outcomes in the race -/
def podium_outcomes : ℕ := num_contestants * (num_contestants - 1) * (num_contestants - 2)

/-- Theorem: The number of different podium outcomes in the race is 120 -/
theorem race_outcomes : podium_outcomes = 120 := by
  sorry

end race_outcomes_l3908_390896


namespace simplify_expression_evaluate_expression_complex_expression_l3908_390818

-- Part 1
theorem simplify_expression (a b : ℝ) :
  2 * (a + b)^2 - 8 * (a + b)^2 + 3 * (a + b)^2 = -3 * (a + b)^2 := by sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (h : x^2 + 2*y = 4) :
  -3*x^2 - 6*y + 17 = 5 := by sorry

-- Part 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 3*b = 3) (h2 : 2*b - c = -5) (h3 : c - d = 9) :
  (a - c) + (2*b - d) - (2*b - c) = 7 := by sorry

end simplify_expression_evaluate_expression_complex_expression_l3908_390818


namespace obtuseTrianglesIn120Gon_l3908_390876

/-- The number of vertices in the regular polygon -/
def n : ℕ := 120

/-- A function that calculates the number of ways to choose three vertices 
    forming an obtuse triangle in a regular n-gon -/
def obtuseTrianglesCount (n : ℕ) : ℕ :=
  n * (n / 2 - 1) * (n / 2 - 2) / 2

/-- Theorem stating that the number of ways to choose three vertices 
    forming an obtuse triangle in a regular 120-gon is 205320 -/
theorem obtuseTrianglesIn120Gon : obtuseTrianglesCount n = 205320 := by
  sorry

end obtuseTrianglesIn120Gon_l3908_390876


namespace banana_cost_is_three_l3908_390826

/-- The cost of a single fruit item -/
structure FruitCost where
  apple : ℕ
  orange : ℕ
  banana : ℕ

/-- The quantity of fruits bought -/
structure FruitQuantity where
  apple : ℕ
  orange : ℕ
  banana : ℕ

/-- Calculate the discount based on the total number of fruits -/
def calculateDiscount (totalFruits : ℕ) : ℕ :=
  totalFruits / 5

/-- Calculate the total cost of fruits before discount -/
def calculateTotalCost (cost : FruitCost) (quantity : FruitQuantity) : ℕ :=
  cost.apple * quantity.apple + cost.orange * quantity.orange + cost.banana * quantity.banana

/-- The main theorem to prove -/
theorem banana_cost_is_three
  (cost : FruitCost)
  (quantity : FruitQuantity)
  (h1 : cost.apple = 1)
  (h2 : cost.orange = 2)
  (h3 : quantity.apple = 5)
  (h4 : quantity.orange = 3)
  (h5 : quantity.banana = 2)
  (h6 : calculateTotalCost cost quantity - calculateDiscount (quantity.apple + quantity.orange + quantity.banana) = 15) :
  cost.banana = 3 := by
  sorry

#check banana_cost_is_three

end banana_cost_is_three_l3908_390826


namespace log_equation_solution_l3908_390841

theorem log_equation_solution (p q : ℝ) (h : 0 < p) (h' : 0 < q) :
  Real.log p + 2 * Real.log q = Real.log (2 * p + q) → p = q / (q^2 - 2) :=
sorry

end log_equation_solution_l3908_390841


namespace initial_mushroom_amount_l3908_390823

-- Define the initial amount of mushrooms
def initial_amount : ℕ := sorry

-- Define the amount of mushrooms eaten
def eaten_amount : ℕ := 8

-- Define the amount of mushrooms left
def left_amount : ℕ := 7

-- Theorem stating that the initial amount is 15 pounds
theorem initial_mushroom_amount :
  initial_amount = eaten_amount + left_amount ∧ initial_amount = 15 := by
  sorry

end initial_mushroom_amount_l3908_390823


namespace product_sum_quotient_l3908_390890

theorem product_sum_quotient (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x * y = 9375 ∧ x + y = 400 → max x y / min x y = 15 := by
  sorry

end product_sum_quotient_l3908_390890


namespace a_perp_b_l3908_390834

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def are_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The given vectors -/
def a : ℝ × ℝ := (-5, 6)
def b : ℝ × ℝ := (6, 5)

/-- Theorem: Vectors a and b are perpendicular -/
theorem a_perp_b : are_perpendicular a b := by sorry

end a_perp_b_l3908_390834


namespace technicians_count_l3908_390889

/-- Represents the workshop scenario with given salary information --/
structure Workshop where
  totalWorkers : ℕ
  avgSalaryAll : ℚ
  avgSalaryTech : ℚ
  avgSalaryNonTech : ℚ

/-- Calculates the number of technicians in the workshop --/
def numTechnicians (w : Workshop) : ℚ :=
  (w.avgSalaryAll * w.totalWorkers - w.avgSalaryNonTech * w.totalWorkers) /
  (w.avgSalaryTech - w.avgSalaryNonTech)

/-- Theorem stating that the number of technicians is 7 --/
theorem technicians_count (w : Workshop) 
  (h1 : w.totalWorkers = 22)
  (h2 : w.avgSalaryAll = 850)
  (h3 : w.avgSalaryTech = 1000)
  (h4 : w.avgSalaryNonTech = 780) :
  numTechnicians w = 7 := by
  sorry

#eval numTechnicians ⟨22, 850, 1000, 780⟩

end technicians_count_l3908_390889


namespace complex_magnitude_l3908_390867

theorem complex_magnitude (z : ℂ) (h : z^2 = 4 - 3*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l3908_390867


namespace intersection_distance_squared_l3908_390871

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the intersection points
def intersection_points (A B M : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  line_l M.1 M.2 ∧ M.1 = 0

-- State the theorem
theorem intersection_distance_squared 
  (A B M : ℝ × ℝ) 
  (h : intersection_points A B M) : 
  (Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) + 
   Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2))^2 = 16 + 2 * Real.sqrt 3 :=
by sorry

end intersection_distance_squared_l3908_390871


namespace inheritance_calculation_l3908_390883

theorem inheritance_calculation (inheritance : ℝ) : 
  inheritance * 0.25 + (inheritance - inheritance * 0.25) * 0.15 = 15000 →
  inheritance = 41379 := by
sorry

end inheritance_calculation_l3908_390883


namespace f_is_quadratic_l3908_390838

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l3908_390838


namespace smallest_triangle_longer_leg_l3908_390828

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse_eq : hypotenuse = 2 * shorterLeg
  longerLeg_eq : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of four 30-60-90 triangles -/
def TriangleSequence (t1 t2 t3 t4 : Triangle30_60_90) : Prop :=
  t1.hypotenuse = 16 ∧
  t2.hypotenuse = t1.longerLeg ∧
  t3.hypotenuse = t2.longerLeg ∧
  t4.hypotenuse = t3.longerLeg ∧
  t2.hypotenuse = t1.hypotenuse / 2 ∧
  t3.hypotenuse = t2.hypotenuse / 2 ∧
  t4.hypotenuse = t3.hypotenuse / 2

theorem smallest_triangle_longer_leg
  (t1 t2 t3 t4 : Triangle30_60_90)
  (h : TriangleSequence t1 t2 t3 t4) :
  t4.longerLeg = 9 := by
  sorry

end smallest_triangle_longer_leg_l3908_390828


namespace victors_final_amount_l3908_390870

/-- Calculates the final amount of money Victor has after transactions -/
def final_amount (initial : ℕ) (allowance : ℕ) (additional : ℕ) (expense : ℕ) : ℕ :=
  initial + allowance + additional - expense

/-- Theorem stating that Victor's final amount is $203 -/
theorem victors_final_amount :
  final_amount 145 88 30 60 = 203 := by
  sorry

end victors_final_amount_l3908_390870


namespace cos_alpha_value_l3908_390815

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Icc 0 (Real.pi / 2)) 
  (h2 : Real.sin (α - Real.pi / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end cos_alpha_value_l3908_390815


namespace square_numbers_existence_l3908_390842

theorem square_numbers_existence (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ (q1 q2 : ℕ), q1.Prime ∧ q2.Prime ∧ q1 ≠ q2 ∧
    ¬(p^2 ∣ (q1^(p-1) - 1)) ∧ ¬(p^2 ∣ (q2^(p-1) - 1)) := by
  sorry

end square_numbers_existence_l3908_390842


namespace roots_of_quadratic_l3908_390846

theorem roots_of_quadratic (x : ℝ) : x^2 = 2*x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_l3908_390846


namespace linear_system_solution_l3908_390839

theorem linear_system_solution (x y m : ℝ) : 
  x + 2*y = m → 
  2*x - 3*y = 4 → 
  x + y = 7 → 
  m = 9 := by
sorry

end linear_system_solution_l3908_390839


namespace equation_transformation_l3908_390891

theorem equation_transformation (x y : ℝ) : x = y → x - 2 = y - 2 := by
  sorry

end equation_transformation_l3908_390891


namespace smallest_multiple_exceeding_100_l3908_390814

theorem smallest_multiple_exceeding_100 : ∃ (n : ℕ), 
  n > 0 ∧ 
  n % 45 = 0 ∧ 
  (n - 100) % 7 = 0 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 45 = 0 ∧ (m - 100) % 7 = 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_multiple_exceeding_100_l3908_390814


namespace not_p_sufficient_not_necessary_for_not_q_l3908_390862

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := x > 2

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) := by
  sorry

end not_p_sufficient_not_necessary_for_not_q_l3908_390862


namespace quadratic_inequality_solution_l3908_390884

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 24*x + 125

-- Define the lower and upper bounds of the solution interval
def lower_bound : ℝ := 6.71
def upper_bound : ℝ := 17.29

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x ≤ 9 ↔ lower_bound ≤ x ∧ x ≤ upper_bound := by
  sorry


end quadratic_inequality_solution_l3908_390884


namespace sum_of_different_geometric_not_geometric_l3908_390857

/-- Given two geometric sequences with different common ratios, their sum sequence is not a geometric sequence -/
theorem sum_of_different_geometric_not_geometric
  {α : Type*} [Field α]
  (a b : ℕ → α)
  (p q : α)
  (hp : p ≠ q)
  (ha : ∀ n, a (n + 1) = p * a n)
  (hb : ∀ n, b (n + 1) = q * b n)
  : ¬ (∃ r : α, ∀ n, (a (n + 1) + b (n + 1)) = r * (a n + b n)) :=
by sorry

end sum_of_different_geometric_not_geometric_l3908_390857


namespace salary_proof_l3908_390875

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := 6500

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.2

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.2

/-- Represents the new savings amount in Rupees after expense increase -/
def new_savings : ℝ := 260

theorem salary_proof :
  let original_expenses := monthly_salary * (1 - savings_rate)
  let new_expenses := original_expenses * (1 + expense_increase_rate)
  monthly_salary - new_expenses = new_savings := by
  sorry

#check salary_proof

end salary_proof_l3908_390875


namespace max_successful_throws_l3908_390877

/-- Represents the number of free throws attempted by Andrew -/
def andrew_throws : ℕ → ℕ := λ a => a

/-- Represents the number of free throws attempted by Beatrice -/
def beatrice_throws : ℕ → ℕ := λ b => b

/-- Represents the total number of free throws -/
def total_throws : ℕ := 105

/-- Represents the success rate of Andrew's free throws -/
def andrew_success_rate : ℚ := 1/3

/-- Represents the success rate of Beatrice's free throws -/
def beatrice_success_rate : ℚ := 3/5

/-- Calculates the total number of successful free throws -/
def total_successful_throws (a b : ℕ) : ℚ :=
  andrew_success_rate * a + beatrice_success_rate * b

/-- Theorem stating the maximum number of successful free throws -/
theorem max_successful_throws :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    a + b = total_throws ∧
    ∀ (x y : ℕ), 
      x > 0 → 
      y > 0 → 
      x + y = total_throws → 
      total_successful_throws a b ≥ total_successful_throws x y ∧
      total_successful_throws a b = 59 :=
sorry

end max_successful_throws_l3908_390877


namespace no_two_digit_number_satisfies_conditions_l3908_390832

theorem no_two_digit_number_satisfies_conditions : ¬ ∃ n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧         -- even
  n % 13 = 0 ∧        -- multiple of 13
  ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  ∃ k : ℕ, a * b = k * k  -- product of digits is a perfect square
  := by sorry

end no_two_digit_number_satisfies_conditions_l3908_390832


namespace zero_not_in_range_of_g_l3908_390835

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- This value doesn't matter as g is undefined at x = -3

theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end zero_not_in_range_of_g_l3908_390835


namespace other_communities_count_l3908_390812

/-- The number of students belonging to other communities in a school with given demographics -/
theorem other_communities_count (total : ℕ) (muslim hindu sikh buddhist christian jew : ℚ) 
  (h_total : total = 2500)
  (h_muslim : muslim = 28/100)
  (h_hindu : hindu = 26/100)
  (h_sikh : sikh = 12/100)
  (h_buddhist : buddhist = 10/100)
  (h_christian : christian = 6/100)
  (h_jew : jew = 4/100) :
  ↑total * (1 - (muslim + hindu + sikh + buddhist + christian + jew)) = 350 :=
by sorry

end other_communities_count_l3908_390812


namespace wire_length_l3908_390800

/-- Given that a 75-meter roll of wire weighs 15 kg, 
    this theorem proves that a roll weighing 5 kg is 25 meters long. -/
theorem wire_length (weight : ℝ) (length : ℝ) : 
  (75 : ℝ) / 15 = length / weight → weight = 5 → length = 25 := by
  sorry

end wire_length_l3908_390800


namespace cake_distribution_l3908_390829

theorem cake_distribution (total_pieces : ℕ) (num_friends : ℕ) (pieces_per_friend : ℕ) :
  total_pieces = 150 →
  num_friends = 50 →
  pieces_per_friend * num_friends = total_pieces →
  pieces_per_friend = 3 := by
sorry

end cake_distribution_l3908_390829


namespace no_four_identical_digits_in_powers_of_two_l3908_390847

theorem no_four_identical_digits_in_powers_of_two :
  ∀ n : ℕ, ¬ ∃ a : ℕ, a < 10 ∧ (2^n : ℕ) % 10000 = a * 1111 :=
sorry

end no_four_identical_digits_in_powers_of_two_l3908_390847


namespace bounded_fraction_exists_l3908_390893

theorem bounded_fraction_exists (C : ℝ) : ∃ C, ∀ k : ℤ, 
  |((k^8 - 2*k + 1) / (k^4 - 3))| < C :=
sorry

end bounded_fraction_exists_l3908_390893


namespace min_students_with_both_l3908_390858

theorem min_students_with_both (n : ℕ) (glasses watches both : ℕ → ℕ) :
  (∀ m : ℕ, m ≥ n → glasses m = (3 * m) / 8) →
  (∀ m : ℕ, m ≥ n → watches m = (5 * m) / 6) →
  (∀ m : ℕ, m ≥ n → glasses m + watches m - both m = m) →
  (∃ m : ℕ, m ≥ n ∧ both m = 5 ∧ ∀ k, k < m → ¬(glasses k = (3 * k) / 8 ∧ watches k = (5 * k) / 6)) :=
sorry

end min_students_with_both_l3908_390858


namespace solve_equations_l3908_390801

-- Define the equations
def equation1 (y : ℝ) : Prop := 2.4 * y - 9.8 = 1.4 * y - 9
def equation2 (x : ℝ) : Prop := x - 3 = (3/2) * x + 1

-- State the theorem
theorem solve_equations :
  (∃ y : ℝ, equation1 y ∧ y = 0.8) ∧
  (∃ x : ℝ, equation2 x ∧ x = -8) := by sorry

end solve_equations_l3908_390801


namespace arithmetic_sum_property_l3908_390811

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 7 = 2) →
  (a 4 + a 6 + a 8 = 3) :=
by
  sorry

end arithmetic_sum_property_l3908_390811


namespace parabola_points_order_l3908_390810

/-- Given a parabola y = 2(x-2)^2 + 1 and three points on it, 
    prove that the y-coordinates are in a specific order -/
theorem parabola_points_order (y₁ y₂ y₃ : ℝ) : 
  (y₁ = 2*(-3-2)^2 + 1) →  -- Point A(-3, y₁)
  (y₂ = 2*(3-2)^2 + 1) →   -- Point B(3, y₂)
  (y₃ = 2*(4-2)^2 + 1) →   -- Point C(4, y₃)
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end parabola_points_order_l3908_390810


namespace book_order_theorem_l3908_390821

/-- Represents the book "Journey to the West" -/
inductive JourneyToWest

/-- Represents the book "Morning Blossoms and Evening Blossoms" -/
inductive MorningBlossoms

/-- Represents the initial order of books -/
structure InitialOrder where
  mb_cost : ℕ
  jw_cost : ℕ
  mb_price_ratio : ℚ
  mb_quantity_diff : ℕ

/-- Represents the additional order constraints -/
structure AdditionalOrderConstraints where
  total_books : ℕ
  min_mb : ℕ
  max_cost : ℕ

/-- Calculates the unit prices based on the initial order -/
def calculate_unit_prices (order : InitialOrder) : ℚ × ℚ :=
  sorry

/-- Calculates the number of possible ordering schemes and the lowest total cost -/
def calculate_additional_order (constraints : AdditionalOrderConstraints) (mb_price jw_price : ℚ) : ℕ × ℕ :=
  sorry

theorem book_order_theorem (initial_order : InitialOrder) (constraints : AdditionalOrderConstraints) :
  initial_order.mb_cost = 14000 ∧
  initial_order.jw_cost = 7000 ∧
  initial_order.mb_price_ratio = 1.4 ∧
  initial_order.mb_quantity_diff = 300 ∧
  constraints.total_books = 10 ∧
  constraints.min_mb = 3 ∧
  constraints.max_cost = 124 →
  let (jw_price, mb_price) := calculate_unit_prices initial_order
  let (schemes, lowest_cost) := calculate_additional_order constraints mb_price jw_price
  jw_price = 10 ∧ mb_price = 14 ∧ schemes = 4 ∧ lowest_cost = 112 :=
sorry

end book_order_theorem_l3908_390821


namespace jerome_toy_cars_l3908_390882

/-- The number of toy cars Jerome has now -/
def total_cars (original : ℕ) (last_month : ℕ) (this_month : ℕ) : ℕ :=
  original + last_month + this_month

/-- Theorem: Jerome has 40 toy cars now -/
theorem jerome_toy_cars :
  let original := 25
  let last_month := 5
  let this_month := 2 * last_month
  total_cars original last_month this_month = 40 := by
sorry

end jerome_toy_cars_l3908_390882


namespace playful_not_brown_l3908_390873

structure Dog where
  playful : Prop
  brown : Prop
  knowsTricks : Prop
  canSwim : Prop

axiom all_playful_know_tricks : ∀ (d : Dog), d.playful → d.knowsTricks
axiom no_brown_can_swim : ∀ (d : Dog), d.brown → ¬d.canSwim
axiom cant_swim_dont_know_tricks : ∀ (d : Dog), ¬d.canSwim → ¬d.knowsTricks

theorem playful_not_brown : ∀ (d : Dog), d.playful → ¬d.brown := by
  sorry

end playful_not_brown_l3908_390873


namespace sin_function_parameters_l3908_390897

def period : ℝ := 8
def max_x : ℝ := 1

theorem sin_function_parameters (ω φ : ℝ) : 
  (2 * π / period = ω) → 
  (ω * max_x + φ = π / 2) → 
  (ω = π / 4 ∧ φ = π / 4) := by sorry

end sin_function_parameters_l3908_390897


namespace hyperbola_standard_equation_l3908_390845

-- Define the hyperbola equation
def hyperbola_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 16 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (4, 0)

-- Main theorem
theorem hyperbola_standard_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_asymptote : ∃ x y, asymptote_equation x y ∧ hyperbola_equation a b x y)
  (h_focus : ∃ x y, hyperbola_equation a b x y ∧ (x, y) = parabola_focus) :
  a^2 = 4 ∧ b^2 = 12 := by
  sorry

end hyperbola_standard_equation_l3908_390845


namespace smallest_n_for_candy_equation_l3908_390817

theorem smallest_n_for_candy_equation : ∃ (n : ℕ), n > 0 ∧
  (∀ (r g b y : ℕ), r > 0 ∧ g > 0 ∧ b > 0 ∧ y > 0 →
    (10 * r = 8 * g ∧ 8 * g = 9 * b ∧ 9 * b = 12 * y ∧ 12 * y = 18 * n) →
    (∀ (m : ℕ), m > 0 ∧ m < n →
      ¬(∃ (r' g' b' y' : ℕ), r' > 0 ∧ g' > 0 ∧ b' > 0 ∧ y' > 0 ∧
        10 * r' = 8 * g' ∧ 8 * g' = 9 * b' ∧ 9 * b' = 12 * y' ∧ 12 * y' = 18 * m))) ∧
  n = 20 :=
sorry

end smallest_n_for_candy_equation_l3908_390817
