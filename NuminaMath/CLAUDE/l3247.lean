import Mathlib

namespace shoe_discount_percentage_l3247_324796

def shoe_price : ℝ := 200
def shirts_price : ℝ := 160
def final_discount : ℝ := 0.05
def final_amount : ℝ := 285

theorem shoe_discount_percentage :
  ∃ (x : ℝ), 
    (shoe_price * (1 - x / 100) + shirts_price) * (1 - final_discount) = final_amount ∧
    x = 30 :=
  sorry

end shoe_discount_percentage_l3247_324796


namespace trigonometric_fraction_simplification_l3247_324704

theorem trigonometric_fraction_simplification (x : ℝ) :
  (2 + 3 * Real.sin x - 4 * Real.cos x) / (2 + 3 * Real.sin x + 2 * Real.cos x) 
  = (-1 + 3 * Real.sin (x/2) * Real.cos (x/2) + 4 * (Real.sin (x/2))^2) / 
    (2 + 3 * Real.sin (x/2) * Real.cos (x/2) - 2 * (Real.sin (x/2))^2) :=
by sorry

end trigonometric_fraction_simplification_l3247_324704


namespace arithmetic_sequence_sum_l3247_324758

/-- Given an arithmetic sequence with first term a₁ = 3, second term a₂ = 10,
    third term a₃ = 17, and sixth term a₆ = 38, prove that a₄ + a₅ = 55. -/
theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 38 →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = a 2 - a 1) →
  a 4 + a 5 = 55 := by
sorry

end arithmetic_sequence_sum_l3247_324758


namespace find_twelfth_number_l3247_324763

/-- Given a set of 12 numbers where the sum of the first 11 is known and the arithmetic mean of all 12 is known, find the 12th number. -/
theorem find_twelfth_number (sum_first_eleven : ℕ) (arithmetic_mean : ℚ) (h1 : sum_first_eleven = 137) (h2 : arithmetic_mean = 12) :
  ∃ x : ℕ, (sum_first_eleven + x : ℚ) / 12 = arithmetic_mean ∧ x = 7 :=
by sorry

end find_twelfth_number_l3247_324763


namespace quadratic_equation_solutions_l3247_324738

theorem quadratic_equation_solutions :
  (∀ x : ℝ, 3 * x^2 - 6 * x - 2 = 0 ↔ x = (3 + Real.sqrt 15) / 3 ∨ x = (3 - Real.sqrt 15) / 3) ∧
  (∀ x : ℝ, x^2 + 6 * x + 8 = 0 ↔ x = -2 ∨ x = -4) := by
  sorry

end quadratic_equation_solutions_l3247_324738


namespace sum_of_powers_l3247_324757

theorem sum_of_powers (a b : ℝ) 
  (h1 : (1 / (a + b)) ^ 2003 = 1)
  (h2 : (-a + b) ^ 2005 = 1) :
  a ^ 2003 + b ^ 2004 = 1 := by
sorry

end sum_of_powers_l3247_324757


namespace non_monotonic_range_l3247_324733

/-- A function f is not monotonic on an interval if there exists a point in the interval where f' is zero --/
def NotMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, deriv f x = 0

/-- The main theorem --/
theorem non_monotonic_range (k : ℝ) :
  NotMonotonic (fun x => x^3 - 12*x) (k - 1) (k + 1) →
  k ∈ Set.union (Set.Ioo (-3) (-1)) (Set.Ioo 1 3) := by
  sorry

end non_monotonic_range_l3247_324733


namespace factorial_ratio_equals_seven_and_half_l3247_324703

theorem factorial_ratio_equals_seven_and_half :
  (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / 
  (Nat.factorial 9 * Nat.factorial 8 : ℚ) = 15 / 2 := by
  sorry

end factorial_ratio_equals_seven_and_half_l3247_324703


namespace smallest_twin_egg_number_l3247_324778

def is_twin_egg_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧
  ((n / 100) % 10 = (n / 10) % 10)

def exchange_digits (n : ℕ) : ℕ :=
  (n % 100) * 100 + (n / 100)

def F (m : ℕ) : ℚ :=
  (m - exchange_digits m : ℚ) / 11

theorem smallest_twin_egg_number :
  ∃ (m : ℕ),
    is_twin_egg_number m ∧
    (m / 1000 ≠ (m / 100) % 10) ∧
    ∃ (k : ℕ), F m / 54 = (k : ℚ) ^ 2 ∧
    ∀ (n : ℕ),
      is_twin_egg_number n ∧
      (n / 1000 ≠ (n / 100) % 10) ∧
      (∃ (j : ℕ), F n / 54 = (j : ℚ) ^ 2) →
      m ≤ n ∧
    m = 7117 :=
by sorry


end smallest_twin_egg_number_l3247_324778


namespace digit_sum_proof_l3247_324761

theorem digit_sum_proof (P Q R : ℕ) : 
  P ∈ Finset.range 9 → 
  Q ∈ Finset.range 9 → 
  R ∈ Finset.range 9 → 
  P + P + P = 2022 → 
  P + Q + R = 15 := by
sorry

end digit_sum_proof_l3247_324761


namespace largest_house_number_l3247_324750

def phone_number : List Nat := [2, 7, 1, 3, 1, 4, 7]

def sum_digits (digits : List Nat) : Nat :=
  digits.sum

def is_distinct (digits : List Nat) : Prop :=
  digits.length = digits.eraseDups.length

def is_valid_house_number (house : List Nat) : Prop :=
  house.length = 4 ∧ 
  is_distinct house ∧
  sum_digits house = sum_digits phone_number

theorem largest_house_number : 
  ∀ house : List Nat, is_valid_house_number house → 
  house.foldl (fun acc d => acc * 10 + d) 0 ≤ 9871 :=
by sorry

end largest_house_number_l3247_324750


namespace product_in_fourth_quadrant_l3247_324779

/-- Given two complex numbers Z₁ and Z₂, prove that their product Z is in the fourth quadrant -/
theorem product_in_fourth_quadrant (Z₁ Z₂ : ℂ) (h₁ : Z₁ = 3 + I) (h₂ : Z₂ = 1 - I) :
  let Z := Z₁ * Z₂
  (Z.re > 0 ∧ Z.im < 0) := by sorry

end product_in_fourth_quadrant_l3247_324779


namespace percentage_difference_l3247_324793

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.4)) :
  (y - x) / x = 0.4 := by sorry

end percentage_difference_l3247_324793


namespace race_heartbeats_l3247_324705

/-- Calculates the total number of heartbeats during a race given the heart rate, race distance, and pace. -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Theorem stating that given specific conditions, the total number of heartbeats during a race is 28800. -/
theorem race_heartbeats :
  let heart_rate : ℕ := 160  -- beats per minute
  let race_distance : ℕ := 30  -- miles
  let pace : ℕ := 6  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 28800 :=
by sorry

end race_heartbeats_l3247_324705


namespace son_work_time_l3247_324795

def work_problem (man_time son_time combined_time : ℝ) : Prop :=
  man_time > 0 ∧ son_time > 0 ∧ combined_time > 0 ∧
  1 / man_time + 1 / son_time = 1 / combined_time

theorem son_work_time (man_time combined_time : ℝ) 
  (h1 : man_time = 5)
  (h2 : combined_time = 4)
  : ∃ (son_time : ℝ), work_problem man_time son_time combined_time ∧ son_time = 20 := by
  sorry

end son_work_time_l3247_324795


namespace loss_percentage_calculation_l3247_324730

theorem loss_percentage_calculation (purchase_price selling_price : ℚ) : 
  purchase_price = 490 → 
  selling_price = 465.5 → 
  (purchase_price - selling_price) / purchase_price * 100 = 5 := by
sorry

end loss_percentage_calculation_l3247_324730


namespace no_y_intercepts_l3247_324717

/-- A parabola defined by x = 2y^2 - 3y + 7 -/
def parabola (y : ℝ) : ℝ := 2 * y^2 - 3 * y + 7

/-- A y-intercept occurs when x = 0 -/
def is_y_intercept (y : ℝ) : Prop := parabola y = 0

/-- The parabola has no y-intercepts -/
theorem no_y_intercepts : ¬∃ y : ℝ, is_y_intercept y := by
  sorry

end no_y_intercepts_l3247_324717


namespace calculate_upstream_speed_l3247_324712

/-- The speed of a man rowing in a river -/
structure RowerSpeed where
  stillWater : ℝ
  downstream : ℝ
  upstream : ℝ

/-- Theorem: Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem calculate_upstream_speed (s : RowerSpeed) 
  (h1 : s.stillWater = 40)
  (h2 : s.downstream = 45) :
  s.upstream = 35 := by
  sorry

end calculate_upstream_speed_l3247_324712


namespace min_value_of_expression_l3247_324721

theorem min_value_of_expression (x : ℝ) :
  (x^2 - 4*x + 3) * (x^2 + 4*x + 3) ≥ -16 ∧
  ∃ y : ℝ, (y^2 - 4*y + 3) * (y^2 + 4*y + 3) = -16 :=
by sorry

end min_value_of_expression_l3247_324721


namespace mode_is_97_l3247_324725

/-- Represents a test score with its frequency -/
structure ScoreFrequency where
  score : Nat
  frequency : Nat

/-- Definition of the dataset from the stem-and-leaf plot -/
def testScores : List ScoreFrequency := [
  ⟨75, 2⟩, ⟨81, 2⟩, ⟨82, 3⟩, ⟨89, 2⟩, ⟨93, 1⟩, ⟨94, 2⟩, ⟨97, 4⟩,
  ⟨106, 1⟩, ⟨112, 2⟩, ⟨114, 3⟩, ⟨120, 1⟩
]

/-- Definition of mode: the score with the highest frequency -/
def isMode (s : ScoreFrequency) (scores : List ScoreFrequency) : Prop :=
  ∀ t ∈ scores, s.frequency ≥ t.frequency

/-- Theorem stating that 97 is the mode of the test scores -/
theorem mode_is_97 : ∃ s ∈ testScores, s.score = 97 ∧ isMode s testScores := by
  sorry

end mode_is_97_l3247_324725


namespace siblings_age_sum_l3247_324722

theorem siblings_age_sum (R D S J : ℕ) : 
  R = D + 6 →
  D = S + 8 →
  J = R - 5 →
  R + 8 = 2 * (S + 8) →
  J + 10 = (D + 10) / 2 + 4 →
  (R - 3) + (D - 3) + (S - 3) + (J - 3) = 43 :=
by sorry

end siblings_age_sum_l3247_324722


namespace map_to_actual_distance_l3247_324790

/-- Given a map distance and scale, calculate the actual distance between two cities. -/
theorem map_to_actual_distance 
  (map_distance : ℝ) 
  (scale : ℝ) 
  (h1 : map_distance = 88) 
  (h2 : scale = 15) : 
  map_distance * scale = 1320 := by
  sorry

end map_to_actual_distance_l3247_324790


namespace car_truck_difference_l3247_324710

theorem car_truck_difference (total_vehicles trucks : ℕ) 
  (h1 : total_vehicles = 69)
  (h2 : trucks = 21)
  (h3 : total_vehicles > 2 * trucks) : 
  total_vehicles - 2 * trucks = 27 := by
  sorry

end car_truck_difference_l3247_324710


namespace prop_c_prop_d_l3247_324788

-- Proposition C
theorem prop_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  1/(2*a) + 1/b ≥ 4 := by sorry

-- Proposition D
theorem prop_d (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 4) :
  ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 4 →
    x^2/(x+1) + y^2/(y+1) ≥ m ∧ 
    (∃ (u v : ℝ), u > 0 ∧ v > 0 ∧ u + v = 4 ∧ u^2/(u+1) + v^2/(v+1) = m) ∧
    m = 8/3 := by sorry

end prop_c_prop_d_l3247_324788


namespace units_digit_of_seven_to_six_to_five_l3247_324700

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 9 := by
  sorry

end units_digit_of_seven_to_six_to_five_l3247_324700


namespace remaining_cooking_time_l3247_324782

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 6
def cooking_time_per_potato : ℕ := 8

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 72 := by
  sorry

end remaining_cooking_time_l3247_324782


namespace no_valid_numbers_l3247_324749

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def digitSum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem no_valid_numbers : ¬ ∃ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  digitSum n = 27 ∧
  isEven ((n / 10) % 10) ∧
  isEven n :=
sorry

end no_valid_numbers_l3247_324749


namespace negation_equivalence_l3247_324711

theorem negation_equivalence : 
  (¬ ∃ (x : ℝ), x^2 - x + 1 ≤ 0) ↔ (∀ (x : ℝ), x^2 - x + 1 > 0) := by
  sorry

end negation_equivalence_l3247_324711


namespace contrapositive_equivalence_l3247_324783

theorem contrapositive_equivalence :
  (∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), a ≤ 0 → ab ≤ 0) :=
by sorry

end contrapositive_equivalence_l3247_324783


namespace bill_purchase_percentage_bill_specific_problem_l3247_324760

/-- The problem of determining the percentage by which Bill could have purchased a product for less -/
theorem bill_purchase_percentage (original_profit_rate : ℝ) (new_profit_rate : ℝ) 
  (original_selling_price : ℝ) (additional_profit : ℝ) : ℝ :=
  let original_cost := original_selling_price / (1 + original_profit_rate)
  let new_selling_price := original_selling_price + additional_profit
  let percentage_less := 1 - (new_selling_price / ((1 + new_profit_rate) * original_cost))
  percentage_less * 100

/-- Proof of the specific problem instance -/
theorem bill_specific_problem : 
  bill_purchase_percentage 0.1 0.3 549.9999999999995 35 = 10 := by
  sorry

end bill_purchase_percentage_bill_specific_problem_l3247_324760


namespace chocolate_box_problem_l3247_324713

theorem chocolate_box_problem (total : ℕ) (caramels : ℕ) (nougats : ℕ) (truffles : ℕ) (peanut_clusters : ℕ) :
  total = 50 →
  caramels = 3 →
  nougats = 2 * caramels →
  truffles = caramels + (truffles - caramels) →
  peanut_clusters = (64 * total) / 100 →
  total = caramels + nougats + truffles + peanut_clusters →
  truffles - caramels = 6 := by
sorry

end chocolate_box_problem_l3247_324713


namespace exist_integers_with_gcd_property_l3247_324741

theorem exist_integers_with_gcd_property :
  ∃ (a : Fin 2011 → ℕ+), (∀ i j, i < j → a i < a j) ∧
    (∀ i j, i < j → Nat.gcd (a i) (a j) = (a j) - (a i)) := by
  sorry

end exist_integers_with_gcd_property_l3247_324741


namespace savings_after_purchase_l3247_324716

/-- Calculates the amount left in savings after buying sweaters, scarves, and mittens for a family --/
theorem savings_after_purchase (sweater_price scarf_price mitten_price : ℕ) 
  (family_members total_savings : ℕ) : 
  sweater_price = 35 →
  scarf_price = 25 →
  mitten_price = 15 →
  family_members = 10 →
  total_savings = 800 →
  total_savings - (sweater_price + scarf_price + mitten_price) * family_members = 50 := by
  sorry

end savings_after_purchase_l3247_324716


namespace part_I_part_II_l3247_324746

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 + 3*y^2 = 4
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  ellipse t.A.1 t.A.2 ∧ 
  ellipse t.B.1 t.B.2 ∧
  line_l t.C.1 t.C.2 ∧
  (t.B.2 - t.A.2) / (t.B.1 - t.A.1) = 1

-- Theorem for part I
theorem part_I (t : Triangle) (h : triangle_conditions t) 
  (h_origin : t.A.1 = 0 ∧ t.A.2 = 0) :
  (∃ (AB_length area : ℝ), 
    AB_length = 2 * Real.sqrt 2 ∧ 
    area = 2) :=
sorry

-- Theorem for part II
theorem part_II (t : Triangle) (h : triangle_conditions t) 
  (h_right_angle : (t.B.1 - t.A.1) * (t.C.1 - t.A.1) + (t.B.2 - t.A.2) * (t.C.2 - t.A.2) = 0)
  (h_max_AC : ∀ (t' : Triangle), triangle_conditions t' → 
    (t'.C.1 - t'.A.1)^2 + (t'.C.2 - t'.A.2)^2 ≤ (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) :
  (∃ (m : ℝ), m = -1 ∧ t.B.2 - t.A.2 = t.B.1 - t.A.1 ∧ t.A.2 = t.A.1 + m) :=
sorry

end part_I_part_II_l3247_324746


namespace square_difference_divided_l3247_324736

theorem square_difference_divided : (111^2 - 102^2) / 9 = 213 := by
  sorry

end square_difference_divided_l3247_324736


namespace members_playing_both_l3247_324754

/-- The number of members who play both badminton and tennis in a sports club -/
theorem members_playing_both (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 30)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 19)
  (h_neither : neither = 3) :
  badminton + tennis - (total - neither) = 9 := by
  sorry

end members_playing_both_l3247_324754


namespace original_number_proof_l3247_324770

theorem original_number_proof (x : ℝ) (h : 1 - 1/x = 5/2) : x = 2/3 := by
  sorry

end original_number_proof_l3247_324770


namespace existence_of_good_subset_l3247_324708

def M : Finset ℕ := Finset.range 20

def is_valid_function (f : Finset ℕ → ℕ) : Prop :=
  ∀ S : Finset ℕ, S ⊆ M → S.card = 9 → f S ∈ M

theorem existence_of_good_subset (f : Finset ℕ → ℕ) (h : is_valid_function f) :
  ∃ T : Finset ℕ, T ⊆ M ∧ T.card = 10 ∧ ∀ k ∈ T, f (T \ {k}) ≠ k := by
  sorry

#check existence_of_good_subset

end existence_of_good_subset_l3247_324708


namespace cubic_function_properties_l3247_324714

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x + 4

/-- The function reaches an extreme value at x = 2 -/
def extreme_at_2 (a b : ℝ) : Prop := f a b 2 = -4/3

/-- The derivative of f is zero at x = 2 -/
def derivative_zero_at_2 (a b : ℝ) : Prop := 3 * a * 2^2 - b = 0

theorem cubic_function_properties (a b : ℝ) 
  (h1 : extreme_at_2 a b) 
  (h2 : derivative_zero_at_2 a b) :
  (∀ x, f a b x = (1/3) * x^3 - 4 * x + 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a b x ≥ f a b y) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a b x ≤ f a b y) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f a b x = 28/3) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f a b x = -4/3) :=
sorry

end cubic_function_properties_l3247_324714


namespace common_tangents_exist_curves_intersect_at_angles_l3247_324726

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines an ellipse with the equation 16x^2 + 25y^2 = 400 -/
def is_on_ellipse (p : Point) : Prop :=
  16 * p.x^2 + 25 * p.y^2 = 400

/-- Defines a circle with the equation x^2 + y^2 = 20 -/
def is_on_circle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 20

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is tangent to the ellipse -/
def is_tangent_to_ellipse (l : Line) : Prop :=
  ∃ p : Point, is_on_ellipse p ∧ l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a line is tangent to the circle -/
def is_tangent_to_circle (l : Line) : Prop :=
  ∃ p : Point, is_on_circle p ∧ l.a * p.x + l.b * p.y + l.c = 0

/-- Theorem stating that there exist common tangents to the ellipse and circle -/
theorem common_tangents_exist : 
  ∃ l : Line, is_tangent_to_ellipse l ∧ is_tangent_to_circle l :=
sorry

/-- Calculates the angle between two curves at an intersection point -/
noncomputable def angle_between_curves (p : Point) : ℝ :=
sorry

/-- Theorem stating that the ellipse and circle intersect at certain angles -/
theorem curves_intersect_at_angles : 
  ∃ p : Point, is_on_ellipse p ∧ is_on_circle p ∧ angle_between_curves p ≠ 0 :=
sorry

end common_tangents_exist_curves_intersect_at_angles_l3247_324726


namespace parallel_lines_a_value_l3247_324723

/-- Two lines are parallel if their slopes are equal but not equal to the ratio of their constants -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ / n₁ = m₂ / n₂ ∧ m₁ / n₁ ≠ c₁ / c₂

theorem parallel_lines_a_value (a : ℝ) :
  parallel (3 + a) 4 (5 - 3*a) 2 (5 + a) 8 → a = -7 := by
  sorry

end parallel_lines_a_value_l3247_324723


namespace price_increase_percentage_l3247_324759

def original_price : ℝ := 300
def new_price : ℝ := 390

theorem price_increase_percentage :
  (new_price - original_price) / original_price * 100 = 30 := by
  sorry

end price_increase_percentage_l3247_324759


namespace initial_bushes_count_l3247_324762

/-- The number of new bushes that grow between each pair of neighboring bushes every hour. -/
def new_bushes_per_hour : ℕ := 2

/-- The total number of hours of growth. -/
def total_hours : ℕ := 3

/-- The total number of bushes after the growth period. -/
def final_bush_count : ℕ := 190

/-- Calculate the number of bushes after one hour of growth. -/
def bushes_after_one_hour (initial_bushes : ℕ) : ℕ :=
  initial_bushes + new_bushes_per_hour * (initial_bushes - 1)

/-- Calculate the number of bushes after the total growth period. -/
def bushes_after_growth (initial_bushes : ℕ) : ℕ :=
  (bushes_after_one_hour^[total_hours]) initial_bushes

/-- The theorem stating that 8 is the correct initial number of bushes. -/
theorem initial_bushes_count : 
  ∃ (n : ℕ), n > 0 ∧ bushes_after_growth n = final_bush_count ∧ 
  ∀ (m : ℕ), m ≠ n → bushes_after_growth m ≠ final_bush_count :=
sorry

end initial_bushes_count_l3247_324762


namespace parabola_tangent_theorem_l3247_324747

/-- Parabola C₁ with equation x² = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on the parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : x^2 = 2 * C.p * y

/-- External point M -/
structure ExternalPoint (C : Parabola) where
  a : ℝ
  y : ℝ
  hy : y = -2 * C.p

/-- Theorem stating the main results -/
theorem parabola_tangent_theorem (C : Parabola) (M : ExternalPoint C) :
  -- Part 1: If a line through focus with x-intercept 2 intersects C₁ at Q and N 
  -- such that |Q'N'| = 2√5, then p = 2
  (∃ (Q N : ParabolaPoint C), 
    (Q.x / 2 + 2 * Q.y / C.p = 1) ∧ 
    (N.x / 2 + 2 * N.y / C.p = 1) ∧ 
    ((Q.x - N.x)^2 = 20)) →
  C.p = 2 ∧
  -- Part 2: If A and B are tangent points, then k₁ · k₂ = -4
  (∀ (A B : ParabolaPoint C),
    (A.y - M.y = (A.x / C.p) * (A.x - M.a)) →
    (B.y - M.y = (B.x / C.p) * (B.x - M.a)) →
    ((A.x / C.p) * (B.x / C.p) = -4)) := by
  sorry

end parabola_tangent_theorem_l3247_324747


namespace book_cost_is_300_divided_by_num_books_l3247_324751

/-- Represents the cost of lawn mowing and video games -/
structure Costs where
  lawn_price : ℕ
  video_game_price : ℕ

/-- Represents Kenny's lawn mowing and purchasing activities -/
structure KennyActivities where
  costs : Costs
  lawns_mowed : ℕ
  video_games_bought : ℕ

/-- Calculates the cost of each book based on Kenny's activities -/
def book_cost (activities : KennyActivities) (num_books : ℕ) : ℚ :=
  let total_earned := activities.costs.lawn_price * activities.lawns_mowed
  let spent_on_games := activities.costs.video_game_price * activities.video_games_bought
  let remaining_for_books := total_earned - spent_on_games
  (remaining_for_books : ℚ) / num_books

/-- Theorem stating that the cost of each book is $300 divided by the number of books -/
theorem book_cost_is_300_divided_by_num_books 
  (activities : KennyActivities) 
  (num_books : ℕ) 
  (h1 : activities.costs.lawn_price = 15)
  (h2 : activities.costs.video_game_price = 45)
  (h3 : activities.lawns_mowed = 35)
  (h4 : activities.video_games_bought = 5)
  (h5 : num_books > 0) :
  book_cost activities num_books = 300 / num_books :=
by
  sorry

#check book_cost_is_300_divided_by_num_books

end book_cost_is_300_divided_by_num_books_l3247_324751


namespace range_of_m_l3247_324784

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) → m < 2 := by
  sorry

end range_of_m_l3247_324784


namespace polynomial_division_remainder_l3247_324772

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 58 = (X - 3) * q + 19 := by sorry

end polynomial_division_remainder_l3247_324772


namespace range_of_quadratic_expression_l3247_324748

theorem range_of_quadratic_expression (x y : ℝ) :
  (4 * x^2 - 2 * Real.sqrt 3 * x * y + 4 * y^2 = 13) →
  (10 - 4 * Real.sqrt 3 ≤ x^2 + 4 * y^2) ∧ (x^2 + 4 * y^2 ≤ 10 + 4 * Real.sqrt 3) := by
  sorry

end range_of_quadratic_expression_l3247_324748


namespace bank_account_deposit_fraction_l3247_324798

theorem bank_account_deposit_fraction (B : ℝ) (f : ℝ) : 
  B > 0 →
  (3/5) * B = B - 400 →
  600 + f * 600 = 750 →
  f = 1/4 := by
sorry

end bank_account_deposit_fraction_l3247_324798


namespace range_of_m_l3247_324756

-- Define the propositions
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x m : ℝ) : Prop := x > m

-- Define the theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(q x m) → ¬(p x)) ∧ ¬(¬(p x) → ¬(q x m))) →
  (∀ m : ℝ, m ≥ 1 ↔ ∃ x : ℝ, p x ∧ q x m) :=
sorry

end range_of_m_l3247_324756


namespace solution_set_range_of_a_l3247_324780

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 1|

-- Theorem for the solution of f(x) ≤ 9
theorem solution_set (x : ℝ) : f x ≤ 9 ↔ x ∈ Set.Icc (-2) 4 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc 0 2, f x = -x^2 + a) ↔ a ∈ Set.Icc (19/4) 7 := by sorry

end solution_set_range_of_a_l3247_324780


namespace ellipse_properties_l3247_324728

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The equation of the ellipse in the form x²/a² + y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- Checks if a point (x, y) lies on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  e.equation x y

/-- The focal distance of the ellipse -/
def Ellipse.focalDistance (e : Ellipse) : ℝ := 2

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) 
    (h1 : e.focalDistance = 2)
    (h2 : e.contains (-1) (3/2)) : 
  (∃ a b : ℝ, e.equation = fun x y ↦ x^2/a^2 + y^2/b^2 = 1 ∧ a = 2 ∧ b^2 = 3) ∧
  (∀ x y : ℝ, e.contains x y ↔ x^2/4 + y^2/3 = 1) ∧
  (e.contains 2 0 ∧ e.contains (-2) 0 ∧ e.contains 0 (Real.sqrt 3) ∧ e.contains 0 (-Real.sqrt 3)) ∧
  (∃ majorAxis : ℝ, majorAxis = 4) ∧
  (∃ minorAxis : ℝ, minorAxis = 2 * Real.sqrt 3) ∧
  (∃ eccentricity : ℝ, eccentricity = 1/2) :=
sorry

end ellipse_properties_l3247_324728


namespace certain_number_proof_l3247_324752

theorem certain_number_proof : ∃ x : ℝ, 0.60 * x = 0.45 * 30 + 16.5 ∧ x = 50 := by
  sorry

end certain_number_proof_l3247_324752


namespace number_exceeds_value_l3247_324744

theorem number_exceeds_value (n : ℕ) (v : ℕ) (h : n = 69) : 
  n = v + 3 * (86 - n) → v = 18 := by
sorry

end number_exceeds_value_l3247_324744


namespace binomial_divisibility_l3247_324766

theorem binomial_divisibility (k : ℕ) (h : k ≥ 2) :
  ∃ m : ℤ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = m * 2^(3*k) ∧
  ¬∃ n : ℤ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = n * 2^(3*k+1) :=
sorry

end binomial_divisibility_l3247_324766


namespace quadratic_equations_root_range_l3247_324781

/-- The range of real numbers for a, such that at most two of the given three quadratic equations do not have real roots -/
theorem quadratic_equations_root_range : 
  {a : ℝ | (∃ x : ℝ, x^2 - a*x + 9 = 0) ∨ 
           (∃ x : ℝ, x^2 + a*x - 2*a = 0) ∨ 
           (∃ x : ℝ, x^2 + (a+1)*x + 9/4 = 0)} = 
  {a : ℝ | a ≤ -4 ∨ a ≥ 0} := by sorry

end quadratic_equations_root_range_l3247_324781


namespace decimal_255_to_octal_l3247_324734

-- Define a function to convert decimal to octal
def decimal_to_octal (n : ℕ) : List ℕ :=
  sorry

-- Theorem statement
theorem decimal_255_to_octal :
  decimal_to_octal 255 = [3, 7, 7] := by
  sorry

end decimal_255_to_octal_l3247_324734


namespace expand_and_simplify_product_l3247_324776

theorem expand_and_simplify_product (x : ℝ) :
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end expand_and_simplify_product_l3247_324776


namespace medicine_tablets_l3247_324789

theorem medicine_tablets (tablets_A tablets_B min_extraction : ℕ) : 
  tablets_A = 10 →
  min_extraction = 15 →
  min_extraction = tablets_A + 2 + (tablets_B - 2) →
  tablets_B = 5 := by
sorry

end medicine_tablets_l3247_324789


namespace trapezoid_construction_uniqueness_l3247_324787

-- Define the necessary types
def Line : Type := ℝ → ℝ → Prop
def Point : Type := ℝ × ℝ
def Direction : Type := ℝ × ℝ

-- Define the trapezoid structure
structure Trapezoid where
  side1 : Line
  side2 : Line
  diag1_start : Point
  diag1_end : Point
  diag2_direction : Direction

-- Define the theorem
theorem trapezoid_construction_uniqueness 
  (side1 side2 : Line)
  (E F : Point)
  (diag2_dir : Direction) :
  ∃! (trap : Trapezoid), 
    trap.side1 = side1 ∧ 
    trap.side2 = side2 ∧ 
    trap.diag1_start = E ∧ 
    trap.diag1_end = F ∧ 
    trap.diag2_direction = diag2_dir :=
sorry

end trapezoid_construction_uniqueness_l3247_324787


namespace range_of_m_l3247_324742

def p (x : ℝ) : Prop := abs (2 * x + 1) ≤ 3

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m : 
  ∀ m : ℝ, (m > 0 ∧ 
    (∀ x : ℝ, p x → q x m) ∧ 
    (∃ x : ℝ, ¬(p x) ∧ q x m)) ↔ 
  m ≥ 3 := by sorry

end range_of_m_l3247_324742


namespace perpendicular_lines_theorem_l3247_324785

/-- Given two perpendicular lines and their point of intersection, prove that m + n - p = 0 --/
theorem perpendicular_lines_theorem (m n p : ℝ) : 
  (∀ x y, m * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + n = 0) →  -- Lines are perpendicular
  (m * 1 + 4 * p - 2 = 0) →  -- (1, p) is on the first line
  (2 * 1 - 5 * p + n = 0) →  -- (1, p) is on the second line
  m + n - p = 0 := by
  sorry

end perpendicular_lines_theorem_l3247_324785


namespace theresa_julia_multiple_l3247_324737

/-- The number of video games Tory has -/
def tory_games : ℕ := 6

/-- The number of video games Julia has -/
def julia_games : ℕ := tory_games / 3

/-- The number of video games Theresa has -/
def theresa_games : ℕ := 11

/-- The multiple of video games Theresa has compared to Julia -/
def multiple : ℕ := (theresa_games - 5) / julia_games

theorem theresa_julia_multiple :
  multiple = 3 :=
sorry

end theresa_julia_multiple_l3247_324737


namespace table_runner_coverage_l3247_324768

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) (coverage_percentage : ℝ) (three_layer_area : ℝ) :
  total_runner_area = 212 →
  table_area = 175 →
  coverage_percentage = 0.80 →
  three_layer_area = 24 →
  ∃ (two_layer_area : ℝ),
    two_layer_area = 48 ∧
    two_layer_area + three_layer_area + (coverage_percentage * table_area - two_layer_area - three_layer_area) = coverage_percentage * table_area ∧
    two_layer_area + three_layer_area = total_runner_area - (coverage_percentage * table_area - two_layer_area - three_layer_area) :=
by sorry

end table_runner_coverage_l3247_324768


namespace tickets_left_kaleb_tickets_left_l3247_324797

theorem tickets_left (initial_tickets : ℕ) (ticket_cost : ℕ) (spent_on_ride : ℕ) : ℕ :=
  let tickets_used := spent_on_ride / ticket_cost
  initial_tickets - tickets_used

theorem kaleb_tickets_left :
  tickets_left 6 9 27 = 3 := by
  sorry

end tickets_left_kaleb_tickets_left_l3247_324797


namespace expression_equality_l3247_324740

theorem expression_equality : (-1)^2023 - Real.sqrt 9 + |1 - Real.sqrt 2| - ((-8) ^ (1/3 : ℝ)) = Real.sqrt 2 - 3 := by
  sorry

end expression_equality_l3247_324740


namespace farmer_profit_l3247_324745

/-- Calculate the profit for a group of piglets -/
def profit_for_group (num_piglets : ℕ) (months : ℕ) (price : ℕ) : ℕ :=
  num_piglets * price - num_piglets * 12 * months

/-- Calculate the total profit for all piglet groups -/
def total_profit : ℕ :=
  profit_for_group 2 12 350 +
  profit_for_group 3 15 400 +
  profit_for_group 2 18 450 +
  profit_for_group 1 21 500

/-- The farmer's profit from selling 8 piglets is $1788 -/
theorem farmer_profit : total_profit = 1788 := by
  sorry

end farmer_profit_l3247_324745


namespace existence_of_irrational_sum_l3247_324769

theorem existence_of_irrational_sum (n : ℕ) (a : Fin n → ℝ) :
  ∃ (x : ℝ), ∀ (i : Fin n), Irrational (x + a i) := by
  sorry

end existence_of_irrational_sum_l3247_324769


namespace bus_riders_percentage_l3247_324794

/-- Represents the scenario of introducing a bus service in Johnstown --/
structure BusScenario where
  population : Nat
  car_pollution : Nat
  bus_pollution : Nat
  bus_capacity : Nat
  carbon_reduction : Nat

/-- Calculates the percentage of people who now take the bus --/
def percentage_bus_riders (scenario : BusScenario) : Rat :=
  let cars_removed := scenario.carbon_reduction / scenario.car_pollution
  (cars_removed : Rat) / scenario.population * 100

/-- Theorem stating that the percentage of people who now take the bus is 12.5% --/
theorem bus_riders_percentage (scenario : BusScenario) 
  (h1 : scenario.population = 80)
  (h2 : scenario.car_pollution = 10)
  (h3 : scenario.bus_pollution = 100)
  (h4 : scenario.bus_capacity = 40)
  (h5 : scenario.carbon_reduction = 100) :
  percentage_bus_riders scenario = 25/2 := by
  sorry

#eval percentage_bus_riders {
  population := 80,
  car_pollution := 10,
  bus_pollution := 100,
  bus_capacity := 40,
  carbon_reduction := 100
}

end bus_riders_percentage_l3247_324794


namespace hyperbola_condition_ellipse_x_foci_condition_l3247_324719

-- Define the curve C
def C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - k) + p.2^2 / (k - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  (4 - k) * (k - 1) < 0

-- Define what it means for C to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (k : ℝ) : Prop :=
  k - 1 > 0 ∧ 4 - k > 0 ∧ 4 - k > k - 1

-- Theorem 1: If C is a hyperbola, then k < 1 or k > 4
theorem hyperbola_condition (k : ℝ) :
  is_hyperbola k → k < 1 ∨ k > 4 :=
by sorry

-- Theorem 2: If C is an ellipse with foci on the x-axis, then 1 < k < 2.5
theorem ellipse_x_foci_condition (k : ℝ) :
  is_ellipse_x_foci k → 1 < k ∧ k < 2.5 :=
by sorry

end hyperbola_condition_ellipse_x_foci_condition_l3247_324719


namespace min_value_expression_l3247_324775

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (8 / (x + 1)) + (1 / y) ≥ 25 / 3 :=
by sorry

end min_value_expression_l3247_324775


namespace mountain_distances_l3247_324735

/-- Given a mountainous region with points A, B, and C, where:
    - The horizontal projection of BC is 2400 m
    - Peak B is 800 m higher than C
    - The elevation angle of AB is 20°
    - The elevation angle of AC is 2°
    - The angle between AB' and AC' (where B' and C' are horizontal projections) is 60°
    
    This theorem states that:
    - The horizontal projection of AB is approximately 2426 m
    - The horizontal projection of AC is approximately 2374 m
    - The height difference between B and A is approximately 883.2 m
-/
theorem mountain_distances (BC_proj : ℝ) (B_height_diff : ℝ) (AB_angle : ℝ) (AC_angle : ℝ) (ABC_angle : ℝ)
  (h_BC_proj : BC_proj = 2400)
  (h_B_height_diff : B_height_diff = 800)
  (h_AB_angle : AB_angle = 20 * π / 180)
  (h_AC_angle : AC_angle = 2 * π / 180)
  (h_ABC_angle : ABC_angle = 60 * π / 180) :
  ∃ (AB_proj AC_proj BA_height : ℝ),
    (abs (AB_proj - 2426) < 1) ∧
    (abs (AC_proj - 2374) < 1) ∧
    (abs (BA_height - 883.2) < 0.1) := by
  sorry

end mountain_distances_l3247_324735


namespace not_all_problems_solvable_by_algorithm_l3247_324791

/-- Represents a problem that can be solved computationally -/
def Problem : Type := Unit

/-- Represents an algorithm -/
def Algorithm : Type := Unit

/-- Represents the characteristic that an algorithm is executed step by step -/
def stepwise (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that each step of an algorithm yields a unique result -/
def uniqueStepResult (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are effective for a class of problems -/
def effectiveForClass (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are mechanical -/
def mechanical (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms can require repetitive calculation -/
def repetitiveCalculation (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are a universal method -/
def universalMethod (a : Algorithm) : Prop := sorry

/-- Theorem stating that not all problems can be solved by algorithms -/
theorem not_all_problems_solvable_by_algorithm : 
  ¬ (∀ (p : Problem), ∃ (a : Algorithm), 
    stepwise a ∧ 
    uniqueStepResult a ∧ 
    effectiveForClass a ∧ 
    mechanical a ∧ 
    repetitiveCalculation a ∧ 
    universalMethod a) := by sorry


end not_all_problems_solvable_by_algorithm_l3247_324791


namespace inequality_range_l3247_324709

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) ↔ m ∈ Set.Iic (-3) :=
by sorry

end inequality_range_l3247_324709


namespace inner_square_is_square_l3247_324755

/-- A point on a line segment -/
structure PointOnSegment (A B : ℝ × ℝ) where
  point : ℝ × ℝ
  on_segment : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ point = (1 - t) • A + t • B

/-- A square defined by its four vertices -/
structure Square (A B C D : ℝ × ℝ) where
  is_square : sorry  -- Definition of a square

/-- Points dividing sides of a square in the same ratio -/
def divide_sides_equally (ABCD : Square A B C D) 
  (N : PointOnSegment A B) (K : PointOnSegment B C) 
  (L : PointOnSegment C D) (M : PointOnSegment D A) : Prop :=
  sorry  -- Definition of dividing sides equally

theorem inner_square_is_square 
  (ABCD : Square A B C D)
  (N : PointOnSegment A B) (K : PointOnSegment B C) 
  (L : PointOnSegment C D) (M : PointOnSegment D A)
  (h : divide_sides_equally ABCD N K L M) :
  Square N.point K.point L.point M.point :=
sorry

end inner_square_is_square_l3247_324755


namespace can_display_properties_l3247_324715

/-- Represents a triangular display of cans. -/
structure CanDisplay where
  totalCans : ℕ
  canWeight : ℕ

/-- Calculates the number of rows in the display. -/
def numberOfRows (d : CanDisplay) : ℕ :=
  Nat.sqrt d.totalCans

/-- Calculates the total weight of the display in kg. -/
def totalWeight (d : CanDisplay) : ℕ :=
  d.totalCans * d.canWeight

/-- Theorem stating the properties of the specific can display. -/
theorem can_display_properties (d : CanDisplay) 
  (h1 : d.totalCans = 225)
  (h2 : d.canWeight = 5) :
  numberOfRows d = 15 ∧ totalWeight d = 1125 := by
  sorry

end can_display_properties_l3247_324715


namespace no_pairs_50_75_six_pairs_50_600_l3247_324707

-- Define the function to count pairs satisfying the conditions
def countPairs (gcd : Nat) (lcm : Nat) : Nat :=
  (Finset.filter (fun p : Nat × Nat => 
    p.1.gcd p.2 = gcd ∧ p.1.lcm p.2 = lcm) (Finset.product (Finset.range (lcm + 1)) (Finset.range (lcm + 1)))).card

-- Theorem for the first part
theorem no_pairs_50_75 : countPairs 50 75 = 0 := by sorry

-- Theorem for the second part
theorem six_pairs_50_600 : countPairs 50 600 = 6 := by sorry

end no_pairs_50_75_six_pairs_50_600_l3247_324707


namespace sum_of_solutions_quadratic_l3247_324786

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + 16 = 12*x₁ ∧ x₂^2 + 16 = 12*x₂ ∧ 
   (∀ y : ℝ, y^2 + 16 = 12*y → y = x₁ ∨ y = x₂) ∧
   x₁ + x₂ = 12) :=
by sorry

end sum_of_solutions_quadratic_l3247_324786


namespace right_triangle_tan_G_l3247_324720

theorem right_triangle_tan_G (FG HG FH : ℝ) (h1 : FG = 17) (h2 : HG = 15) 
  (h3 : FG^2 = FH^2 + HG^2) : 
  FH / HG = 8 / 15 := by
sorry

end right_triangle_tan_G_l3247_324720


namespace puppies_left_l3247_324753

theorem puppies_left (initial : ℕ) (given_away : ℕ) (h1 : initial = 7) (h2 : given_away = 5) :
  initial - given_away = 2 := by
  sorry

end puppies_left_l3247_324753


namespace A_difference_max_l3247_324727

def A (a b : ℕ+) : ℚ := (a + 3) / 12

theorem A_difference_max :
  (∃ a₁ b₁ a₂ b₂ : ℕ+, 
    A a₁ b₁ = 15 / (26 - b₁) ∧
    A a₂ b₂ = 15 / (26 - b₂) ∧
    ∀ a b : ℕ+, A a b = 15 / (26 - b) → 
      A a₁ b₁ ≤ A a b ∧ A a b ≤ A a₂ b₂) →
  A a₂ b₂ - A a₁ b₁ = 57 / 4 :=
sorry

end A_difference_max_l3247_324727


namespace angle_is_15_degrees_l3247_324701

-- Define the triangle MIT
structure Triangle :=
  (M I T : ℝ × ℝ)

-- Define the points X, Y, O, P
structure Points :=
  (X Y O P : ℝ × ℝ)

def angle_MOP (t : Triangle) (p : Points) : ℝ :=
  sorry  -- The actual calculation of the angle

-- Main theorem
theorem angle_is_15_degrees 
  (t : Triangle) 
  (p : Points) 
  (h1 : t.M.1 = 0 ∧ t.M.2 = 0)  -- Assume M is at (0,0)
  (h2 : t.I.1 = 12 ∧ t.I.2 = 0)  -- MI = 12
  (h3 : (t.M.1 - p.X.1)^2 + (t.M.2 - p.X.2)^2 = 4)  -- MX = 2
  (h4 : (t.I.1 - p.Y.1)^2 + (t.I.2 - p.Y.2)^2 = 4)  -- YI = 2
  (h5 : p.O = ((t.M.1 + t.I.1)/2, (t.M.2 + t.I.2)/2))  -- O is midpoint of MI
  (h6 : p.P = ((p.X.1 + p.Y.1)/2, (p.X.2 + p.Y.2)/2))  -- P is midpoint of XY
  : angle_MOP t p = 15 * π / 180 :=
sorry

end angle_is_15_degrees_l3247_324701


namespace min_sum_distances_min_sum_distances_equality_l3247_324739

theorem min_sum_distances (u v : ℝ) : 
  Real.sqrt (u^2 + v^2) + Real.sqrt ((u - 1)^2 + v^2) + 
  Real.sqrt (u^2 + (v - 1)^2) + Real.sqrt ((u - 1)^2 + (v - 1)^2) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem min_sum_distances_equality : 
  Real.sqrt ((1/2)^2 + (1/2)^2) + Real.sqrt ((1/2 - 1)^2 + (1/2)^2) + 
  Real.sqrt ((1/2)^2 + (1/2 - 1)^2) + Real.sqrt ((1/2 - 1)^2 + (1/2 - 1)^2) = 2 * Real.sqrt 2 :=
by sorry

end min_sum_distances_min_sum_distances_equality_l3247_324739


namespace luncheon_cost_theorem_l3247_324767

/-- Represents the cost of a luncheon item -/
structure LuncheonItem where
  price : ℚ

/-- Represents a luncheon order -/
structure Luncheon where
  sandwiches : ℕ
  coffee : ℕ
  pie : ℕ
  total : ℚ

/-- The theorem to be proved -/
theorem luncheon_cost_theorem (s : LuncheonItem) (c : LuncheonItem) (p : LuncheonItem) 
  (l1 : Luncheon) (l2 : Luncheon) : 
  l1.sandwiches = 2 ∧ l1.coffee = 5 ∧ l1.pie = 2 ∧ l1.total = 25/4 ∧
  l2.sandwiches = 5 ∧ l2.coffee = 8 ∧ l2.pie = 3 ∧ l2.total = 121/10 →
  s.price + c.price + p.price = 31/20 := by
  sorry

#eval 31/20  -- This should evaluate to 1.55

end luncheon_cost_theorem_l3247_324767


namespace inequality_proof_l3247_324774

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a + b + c ≤ 2) : 
  Real.sqrt (b^2 + a*c) + Real.sqrt (a^2 + b*c) + Real.sqrt (c^2 + a*b) ≤ 3 := by
  sorry

end inequality_proof_l3247_324774


namespace pet_store_dogs_l3247_324764

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs as 3:5 and 18 cats, there are 30 dogs -/
theorem pet_store_dogs : calculate_dogs 3 5 18 = 30 := by
  sorry

end pet_store_dogs_l3247_324764


namespace unique_solution_condition_l3247_324702

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 6)*(x - 4) = -33 + k*x) ↔ (k = -6 + 6*Real.sqrt 3 ∨ k = -6 - 6*Real.sqrt 3) := by
  sorry

end unique_solution_condition_l3247_324702


namespace min_value_theorem_l3247_324777

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ ∃ m : ℕ+, (m : ℝ) / 3 + 27 / (m : ℝ) = 6 := by
  sorry

end min_value_theorem_l3247_324777


namespace paul_bought_45_cookies_l3247_324718

/-- The number of cookies Paul bought -/
def paul_cookies : ℕ := 45

/-- The number of cookies Paula bought -/
def paula_cookies : ℕ := paul_cookies - 3

/-- The total number of cookies bought by Paul and Paula -/
def total_cookies : ℕ := 87

/-- Theorem stating that Paul bought 45 cookies given the conditions -/
theorem paul_bought_45_cookies : 
  paul_cookies = 45 ∧ 
  paula_cookies = paul_cookies - 3 ∧ 
  paul_cookies + paula_cookies = total_cookies :=
sorry

end paul_bought_45_cookies_l3247_324718


namespace johns_annual_savings_l3247_324729

/-- Calculates the annual savings for John's new apartment situation -/
theorem johns_annual_savings
  (former_rent_per_sqft : ℝ)
  (former_apartment_size : ℝ)
  (new_apartment_cost : ℝ)
  (h1 : former_rent_per_sqft = 2)
  (h2 : former_apartment_size = 750)
  (h3 : new_apartment_cost = 2800)
  : (former_rent_per_sqft * former_apartment_size - new_apartment_cost / 2) * 12 = 1200 := by
  sorry

#check johns_annual_savings

end johns_annual_savings_l3247_324729


namespace tenth_row_fifth_column_l3247_324765

/-- Calculates the sum of the first n natural numbers -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the triangular arrangement of natural numbers -/
def triangularArrangement (row : ℕ) (col : ℕ) : ℕ :=
  triangularSum (row - 1) + col

/-- The number in the 10th row and 5th column of the triangular arrangement -/
theorem tenth_row_fifth_column :
  triangularArrangement 10 5 = 101 := by
  sorry

end tenth_row_fifth_column_l3247_324765


namespace first_book_cost_l3247_324706

/-- The cost of Shelby's first book given her spending at the book fair -/
theorem first_book_cost (initial_amount : ℕ) (second_book_cost : ℕ) (poster_cost : ℕ) (num_posters : ℕ) :
  initial_amount = 20 →
  second_book_cost = 4 →
  poster_cost = 4 →
  num_posters = 2 →
  ∃ (first_book_cost : ℕ),
    first_book_cost + second_book_cost + (num_posters * poster_cost) = initial_amount ∧
    first_book_cost = 8 :=
by sorry

end first_book_cost_l3247_324706


namespace negation_of_implication_l3247_324773

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end negation_of_implication_l3247_324773


namespace hyperbola_m_value_l3247_324792

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 + x^2/m = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

-- Theorem statement
theorem hyperbola_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, hyperbola_equation x y m ↔ asymptote_equation x y) → m = -3 :=
by sorry

end hyperbola_m_value_l3247_324792


namespace consecutive_sum_theorem_l3247_324732

theorem consecutive_sum_theorem (n : ℕ) (h : n ≥ 6) :
  ∃ (k a : ℕ), k ≥ 3 ∧ n = k * a + k * (k - 1) / 2 :=
by sorry

end consecutive_sum_theorem_l3247_324732


namespace quadratic_distinct_roots_l3247_324724

theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 2) * x₁^2 + 2 * x₁ - 1 = 0 ∧ (k - 2) * x₂^2 + 2 * x₂ - 1 = 0) ↔
  (k > 1 ∧ k ≠ 2) :=
by sorry

end quadratic_distinct_roots_l3247_324724


namespace quadratic_roots_property_l3247_324799

theorem quadratic_roots_property (α β : ℝ) : 
  α ≠ β →
  α^2 + 3*α - 1 = 0 →
  β^2 + 3*β - 1 = 0 →
  α^2 + 4*α + β = -2 := by
sorry

end quadratic_roots_property_l3247_324799


namespace x_axis_segment_range_l3247_324743

/-- Definition of a quadratic function -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ -a * x^2 + 2 * b * x - c

/-- Definition of the centrally symmetric function with respect to (0,0) -/
def CentrallySymmetricFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + 2 * b * x + c

/-- Theorem about the range of x-axis segment length for the centrally symmetric function -/
theorem x_axis_segment_range
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : a + b + c = 0)
  (h2 : (2*c + b - a) * (2*c + b + 3*a) < 0) :
  ∃ (x₁ x₂ : ℝ), CentrallySymmetricFunction a b c x₁ = 0 ∧
                 CentrallySymmetricFunction a b c x₂ = 0 ∧
                 Real.sqrt 3 < |x₁ - x₂| ∧
                 |x₁ - x₂| < 2 * Real.sqrt 7 := by
  sorry


end x_axis_segment_range_l3247_324743


namespace undefined_expression_l3247_324771

theorem undefined_expression (a : ℝ) : 
  ¬ (∃ x : ℝ, x = (a + 3) / (a^2 - 9)) ↔ a = -3 ∨ a = 3 := by
  sorry

end undefined_expression_l3247_324771


namespace failed_students_l3247_324731

theorem failed_students (total : ℕ) (passed_percentage : ℚ) 
  (h1 : total = 804)
  (h2 : passed_percentage = 75 / 100) :
  ↑total * (1 - passed_percentage) = 201 := by
  sorry

end failed_students_l3247_324731
