import Mathlib

namespace building_residents_contradiction_l1328_132835

theorem building_residents_contradiction (chess : ℕ) (arkhangelsk : ℕ) (airplane : ℕ)
  (chess_airplane : ℕ) (arkhangelsk_airplane : ℕ) (chess_arkhangelsk : ℕ)
  (chess_arkhangelsk_airplane : ℕ) :
  chess = 25 →
  arkhangelsk = 30 →
  airplane = 28 →
  chess_airplane = 18 →
  arkhangelsk_airplane = 17 →
  chess_arkhangelsk = 16 →
  chess_arkhangelsk_airplane = 15 →
  chess + arkhangelsk + airplane - chess_arkhangelsk - chess_airplane - arkhangelsk_airplane + chess_arkhangelsk_airplane > 45 :=
by sorry

end building_residents_contradiction_l1328_132835


namespace shirts_per_minute_l1328_132805

/-- Given an industrial machine that makes 12 shirts in 6 minutes,
    prove that it makes 2 shirts per minute. -/
theorem shirts_per_minute :
  let total_shirts : ℕ := 12
  let total_minutes : ℕ := 6
  let shirts_per_minute : ℚ := total_shirts / total_minutes
  shirts_per_minute = 2 := by
  sorry

end shirts_per_minute_l1328_132805


namespace debby_water_bottles_l1328_132862

theorem debby_water_bottles (initial_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ) 
  (h1 : initial_bottles = 301)
  (h2 : bottles_per_day = 144)
  (h3 : remaining_bottles = 157) :
  (initial_bottles - remaining_bottles) / bottles_per_day = 1 :=
sorry

end debby_water_bottles_l1328_132862


namespace ceiling_floor_difference_l1328_132802

theorem ceiling_floor_difference (x : ℝ) : 
  (⌈x⌉ : ℝ) + (⌊x⌋ : ℝ) = 2 * x → (⌈x⌉ : ℝ) - (⌊x⌋ : ℝ) = 1 := by
  sorry

end ceiling_floor_difference_l1328_132802


namespace function_symmetry_l1328_132860

theorem function_symmetry (f : ℝ → ℝ) (x : ℝ) : f (x - 1) = f (1 - x) := by
  sorry

end function_symmetry_l1328_132860


namespace book_distribution_count_l1328_132853

def distribute_books (total_books : ℕ) (min_per_location : ℕ) : ℕ :=
  (total_books - 2 * min_per_location + 1)

theorem book_distribution_count :
  distribute_books 8 2 = 5 := by
sorry

end book_distribution_count_l1328_132853


namespace impossible_coin_probabilities_l1328_132899

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end impossible_coin_probabilities_l1328_132899


namespace number_calculation_l1328_132806

theorem number_calculation : ∃ x : ℚ, x = 2/15 + 1/5 + 1/2 :=
by sorry

end number_calculation_l1328_132806


namespace rectangle_area_change_l1328_132811

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := L / 2
  let new_area := L * B / 2
  new_length * B = new_area → B = B :=
by sorry

end rectangle_area_change_l1328_132811


namespace irrational_approximation_l1328_132839

theorem irrational_approximation (k : ℝ) (ε : ℝ) 
  (h_irr : Irrational k) (h_pos : ε > 0) :
  ∃ (m n : ℤ), |m * k - n| < ε :=
sorry

end irrational_approximation_l1328_132839


namespace right_triangle_inradius_l1328_132876

/-- A right triangle with side lengths 9, 12, and 15 has an inradius of 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →  -- Given side lengths
  a^2 + b^2 = c^2 →          -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 3 :=
by sorry

end right_triangle_inradius_l1328_132876


namespace problem_statement_l1328_132827

theorem problem_statement :
  (∀ a : ℝ, Real.exp a ≥ a + 1) ∧
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) := by
  sorry

end problem_statement_l1328_132827


namespace current_age_of_D_l1328_132812

theorem current_age_of_D (a b c d : ℕ) : 
  a + b + c + d = 108 →
  a - b = 12 →
  c - (a - 34) = 3 * (d - (a - 34)) →
  d = 13 := by
sorry

end current_age_of_D_l1328_132812


namespace popsicle_stick_ratio_l1328_132808

-- Define the number of popsicle sticks for each person
def steve_sticks : ℕ := 12
def total_sticks : ℕ := 108

-- Define the relationship between Sam and Sid's sticks
def sam_sticks (sid_sticks : ℕ) : ℕ := 3 * sid_sticks

-- Theorem to prove
theorem popsicle_stick_ratio :
  ∃ (sid_sticks : ℕ),
    sid_sticks > 0 ∧
    sam_sticks sid_sticks + sid_sticks + steve_sticks = total_sticks ∧
    sid_sticks = 2 * steve_sticks :=
by sorry

end popsicle_stick_ratio_l1328_132808


namespace geometric_sequence_formula_l1328_132872

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n) →
  (∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1)) :=
by sorry

end geometric_sequence_formula_l1328_132872


namespace complement_B_A_when_m_2_range_of_m_for_necessary_not_sufficient_l1328_132854

def A : Set ℝ := {x | 4 < x ∧ x ≤ 8}
def B (m : ℝ) : Set ℝ := {x | 5 - m^2 ≤ x ∧ x ≤ 5 + m^2}

theorem complement_B_A_when_m_2 :
  {x : ℝ | 1 ≤ x ∧ x ≤ 4 ∨ 8 < x ∧ x ≤ 9} = (B 2) \ A := by sorry

theorem range_of_m_for_necessary_not_sufficient :
  {m : ℝ | A ⊆ B m ∧ A ≠ B m} = {m : ℝ | -1 < m ∧ m < 1} := by sorry

end complement_B_A_when_m_2_range_of_m_for_necessary_not_sufficient_l1328_132854


namespace number_difference_l1328_132866

theorem number_difference (x y : ℚ) 
  (sum_eq : x + y = 40)
  (triple_minus_quad : 3 * y - 4 * x = 10) :
  abs (y - x) = 60 / 7 := by
  sorry

end number_difference_l1328_132866


namespace business_value_calculation_l1328_132849

theorem business_value_calculation (man_share : ℚ) (sold_fraction : ℚ) (sale_price : ℚ) : 
  man_share = 2/3 → 
  sold_fraction = 3/4 → 
  sale_price = 6500 → 
  (sale_price / sold_fraction) / man_share = 39000 :=
by
  sorry

end business_value_calculation_l1328_132849


namespace arithmetic_sequence_product_l1328_132890

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  ArithmeticSequence a → a 4 * a 5 = 24 → a 3 * a 6 = 16 := by
  sorry

end arithmetic_sequence_product_l1328_132890


namespace range_of_a_l1328_132871

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else (a-3)*x + 4*a

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  (0 < a ∧ a ≤ 3/4) :=
by sorry

end range_of_a_l1328_132871


namespace max_candy_remainder_l1328_132861

theorem max_candy_remainder (n : ℕ) : 
  ∃ (k : ℕ), n^2 = 5 * k + 4 ∧ 
  ∀ (m : ℕ), n^2 = 5 * m + (n^2 % 5) → n^2 % 5 ≤ 4 :=
sorry

end max_candy_remainder_l1328_132861


namespace digit_puzzle_l1328_132847

def is_not_zero (d : Nat) : Prop := d ≠ 0
def is_even (d : Nat) : Prop := d % 2 = 0
def is_five (d : Nat) : Prop := d = 5
def is_not_six (d : Nat) : Prop := d ≠ 6
def is_less_than_seven (d : Nat) : Prop := d < 7

theorem digit_puzzle (d : Nat) 
  (h_range : d ≤ 9)
  (h_statements : ∃! (s : Fin 5), ¬(
    match s with
    | 0 => is_not_zero d
    | 1 => is_even d
    | 2 => is_five d
    | 3 => is_not_six d
    | 4 => is_less_than_seven d
  )) :
  ¬(is_even d) :=
sorry

end digit_puzzle_l1328_132847


namespace pizza_burger_overlap_l1328_132851

theorem pizza_burger_overlap (total : ℕ) (pizza : ℕ) (burger : ℕ) 
  (h_total : total = 200)
  (h_pizza : pizza = 125)
  (h_burger : burger = 115) :
  pizza + burger - total = 40 := by
  sorry

end pizza_burger_overlap_l1328_132851


namespace intersection_union_ratio_l1328_132804

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  short_diagonal : ℝ
  long_diagonal : ℝ

/-- The rotation of a rhombus by 90 degrees -/
def rotate_90 (r : Rhombus) : Rhombus := r

/-- The intersection of a rhombus and its 90 degree rotation -/
def intersection (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The union of a rhombus and its 90 degree rotation -/
def union (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in 2D space -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The ratio of the intersection area to the union area is 1/2023 -/
theorem intersection_union_ratio (r : Rhombus) 
  (h1 : r.short_diagonal = 1) 
  (h2 : r.long_diagonal = 2023) : 
  area (intersection r) / area (union r) = 1 / 2023 := by sorry

end intersection_union_ratio_l1328_132804


namespace workshop_workers_l1328_132894

theorem workshop_workers (total_average : ℝ) (tech_count : ℕ) (tech_average : ℝ) (nontech_average : ℝ) : 
  total_average = 9500 → 
  tech_count = 7 → 
  tech_average = 12000 → 
  nontech_average = 6000 → 
  ∃ (total_workers : ℕ), total_workers = 12 ∧ 
    (total_workers : ℝ) * total_average = 
      (tech_count : ℝ) * tech_average + 
      ((total_workers - tech_count) : ℝ) * nontech_average :=
by sorry

end workshop_workers_l1328_132894


namespace trajectory_and_circle_properties_l1328_132818

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the fixed line l
def line_l (x : ℝ) : Prop := x = 4

-- Define point F as the intersection of parabola and line l
def point_F : ℝ × ℝ := (2, 4)

-- Define the condition for point P
def condition_P (P Q F : ℝ × ℝ) : Prop :=
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  let PF := (F.1 - P.1, F.2 - P.2)
  (PQ.1 + Real.sqrt 2 * PF.1, PQ.2 + Real.sqrt 2 * PF.2) • (PQ.1 - Real.sqrt 2 * PF.1, PQ.2 - Real.sqrt 2 * PF.2) = 0

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 8/3

-- Define the range of |AB|
def range_AB (ab : ℝ) : Prop := 4 * Real.sqrt 6 / 3 ≤ ab ∧ ab ≤ 2 * Real.sqrt 3

theorem trajectory_and_circle_properties :
  ∀ (P : ℝ × ℝ),
  (∃ (Q : ℝ × ℝ), line_l Q.1 ∧ condition_P P Q point_F) →
  trajectory_C P.1 P.2 ∧
  (∀ (A B : ℝ × ℝ),
    (circle_O A.1 A.2 ∧ line_l A.1 ∧ trajectory_C A.1 A.2) →
    (circle_O B.1 B.2 ∧ line_l B.1 ∧ trajectory_C B.1 B.2) →
    A ≠ B →
    (let O := (0, 0); let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2);
     (A.1 * B.1 + A.2 * B.2 = 0) → range_AB AB)) :=
by sorry

end trajectory_and_circle_properties_l1328_132818


namespace library_books_end_of_month_l1328_132833

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 300)
  (h2 : loaned_books = 160)
  (h3 : return_rate = 65 / 100) :
  initial_books - loaned_books + (return_rate * loaned_books).floor = 244 :=
by sorry

end library_books_end_of_month_l1328_132833


namespace circle_passes_through_points_l1328_132846

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of our specific circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

theorem circle_passes_through_points :
  ∃ (c : Circle),
    (∀ (x y : ℝ), circle_equation x y ↔ c.contains (x, y)) ∧
    c.contains (0, 0) ∧
    c.contains (4, 0) ∧
    c.contains (-1, 1) := by
  sorry

end circle_passes_through_points_l1328_132846


namespace machine_total_time_l1328_132893

/-- The total time a machine worked, including downtime, given its production rates and downtime -/
theorem machine_total_time
  (time_A : ℕ) (shirts_A : ℕ) (time_B : ℕ) (shirts_B : ℕ) (downtime : ℕ)
  (h_A : time_A = 75 ∧ shirts_A = 13)
  (h_B : time_B = 5 ∧ shirts_B = 3)
  (h_downtime : downtime = 120) :
  time_A + time_B + downtime = 200 := by
  sorry


end machine_total_time_l1328_132893


namespace perpendicular_line_through_point_l1328_132898

/-- Given line equation -/
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

/-- Candidate line equation -/
def candidate_line (x y : ℝ) : Prop := 3 * x + 2 * y - 4 = 0

/-- Point that the candidate line should pass through -/
def point : ℝ × ℝ := (2, -1)

theorem perpendicular_line_through_point :
  (candidate_line point.1 point.2) ∧ 
  (∀ (x y : ℝ), given_line x y → 
    (3 * 2 + 2 * (-3) = 0)) := by sorry

end perpendicular_line_through_point_l1328_132898


namespace davids_english_marks_l1328_132865

def davidsMathMarks : ℕ := 65
def davidsPhysicsMarks : ℕ := 82
def davidsChemistryMarks : ℕ := 67
def davidsBiologyMarks : ℕ := 85
def davidsAverageMarks : ℕ := 76
def totalSubjects : ℕ := 5

theorem davids_english_marks :
  ∃ (englishMarks : ℕ), 
    (englishMarks + davidsMathMarks + davidsPhysicsMarks + davidsChemistryMarks + davidsBiologyMarks) / totalSubjects = davidsAverageMarks ∧
    englishMarks = 81 :=
by sorry

end davids_english_marks_l1328_132865


namespace unique_function_property_l1328_132896

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 1) = f x + 1)
  (h3 : ∀ x ≠ 0, f (1 / x) = f x / x^2) :
  ∀ x, f x = x := by
sorry

end unique_function_property_l1328_132896


namespace unique_divisibility_by_99_l1328_132868

-- Define the structure of the number N
def N (a b : ℕ) : ℕ := a * 10^9 + 2018 * 10^5 + b * 10^4 + 2019

-- Define the divisibility condition
def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

-- State the theorem
theorem unique_divisibility_by_99 :
  ∃! (a b : ℕ), a < 10 ∧ b < 10 ∧ is_divisible_by_99 (N a b) :=
sorry

end unique_divisibility_by_99_l1328_132868


namespace smallest_positive_multiple_of_45_l1328_132886

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by sorry

end smallest_positive_multiple_of_45_l1328_132886


namespace cherries_theorem_l1328_132856

def cherries_problem (initial_cherries : ℕ) (difference : ℕ) : ℕ :=
  initial_cherries - difference

theorem cherries_theorem (initial_cherries : ℕ) (difference : ℕ) 
  (h1 : initial_cherries = 16) (h2 : difference = 10) :
  cherries_problem initial_cherries difference = 6 := by
  sorry

end cherries_theorem_l1328_132856


namespace marble_probability_value_l1328_132820

/-- The probability of having one white and one blue marble left when drawing
    marbles randomly from a bag containing 3 blue and 5 white marbles until 2 are left -/
def marble_probability : ℚ :=
  let total_marbles : ℕ := 8
  let blue_marbles : ℕ := 3
  let white_marbles : ℕ := 5
  let marbles_drawn : ℕ := 6
  let favorable_outcomes : ℕ := Nat.choose white_marbles white_marbles * Nat.choose blue_marbles (blue_marbles - 1)
  let total_outcomes : ℕ := Nat.choose total_marbles marbles_drawn
  (favorable_outcomes : ℚ) / total_outcomes

/-- Theorem stating that the probability of having one white and one blue marble left
    is equal to 3/28 -/
theorem marble_probability_value : marble_probability = 3 / 28 := by
  sorry

end marble_probability_value_l1328_132820


namespace bridget_sarah_cents_difference_bridget_sarah_solution_l1328_132834

theorem bridget_sarah_cents_difference : ℕ → ℕ → ℕ → Prop :=
  fun total sarah_cents difference =>
    total = 300 ∧
    sarah_cents = 125 ∧
    difference = total - 2 * sarah_cents

theorem bridget_sarah_solution :
  ∃ (difference : ℕ), bridget_sarah_cents_difference 300 125 difference ∧ difference = 50 :=
by
  sorry

end bridget_sarah_cents_difference_bridget_sarah_solution_l1328_132834


namespace geometric_sequence_problem_l1328_132842

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → 
  (∃ r : ℝ, 160 * r = b ∧ b * r = 108 / 64) → 
  b = 15 * Real.sqrt 6 := by
sorry

end geometric_sequence_problem_l1328_132842


namespace min_value_reciprocal_sum_l1328_132850

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end min_value_reciprocal_sum_l1328_132850


namespace ellipse_satisfies_conditions_l1328_132848

/-- An ellipse with foci on the y-axis, focal distance 4, and passing through (3,2) -/
def ellipse_equation (x y : ℝ) : Prop :=
  y^2 / 16 + x^2 / 12 = 1

/-- The focal distance of the ellipse -/
def focal_distance : ℝ := 4

/-- A point on the ellipse -/
def point_on_ellipse : ℝ × ℝ := (3, 2)

/-- Theorem stating that the ellipse equation satisfies the given conditions -/
theorem ellipse_satisfies_conditions :
  (∀ x y, ellipse_equation x y → (x = point_on_ellipse.1 ∧ y = point_on_ellipse.2)) ∧
  (∃ f₁ f₂ : ℝ, f₁ = -f₂ ∧ f₁^2 = (focal_distance/2)^2 ∧
    ∀ x y, ellipse_equation x y →
      (x^2 + (y - f₁)^2)^(1/2) + (x^2 + (y - f₂)^2)^(1/2) = 2 * (16^(1/2))) :=
sorry

end ellipse_satisfies_conditions_l1328_132848


namespace hotel_flat_fee_l1328_132892

/-- Given a hotel's pricing structure and two customers' stays, calculate the flat fee for the first night. -/
theorem hotel_flat_fee (linda_total linda_nights bob_total bob_nights : ℕ) 
  (h1 : linda_total = 205)
  (h2 : linda_nights = 4)
  (h3 : bob_total = 350)
  (h4 : bob_nights = 7) :
  ∃ (flat_fee nightly_rate : ℕ),
    flat_fee + (linda_nights - 1) * nightly_rate = linda_total ∧
    flat_fee + (bob_nights - 1) * nightly_rate = bob_total ∧
    flat_fee = 60 := by
  sorry

#check hotel_flat_fee

end hotel_flat_fee_l1328_132892


namespace fifth_odd_with_odd_factors_is_81_l1328_132867

/-- A function that returns true if a number is a perfect square, false otherwise -/
def is_perfect_square (n : ℕ) : Bool := sorry

/-- A function that returns true if a number has an odd number of factors, false otherwise -/
def has_odd_factors (n : ℕ) : Bool := is_perfect_square n

/-- A function that returns the nth odd integer with an odd number of factors -/
def nth_odd_with_odd_factors (n : ℕ) : ℕ := sorry

theorem fifth_odd_with_odd_factors_is_81 :
  nth_odd_with_odd_factors 5 = 81 := by sorry

end fifth_odd_with_odd_factors_is_81_l1328_132867


namespace train_length_problem_l1328_132864

/-- Given a train traveling at constant speed through a tunnel and over a bridge,
    prove that the length of the train is 200m. -/
theorem train_length_problem (tunnel_length : ℝ) (tunnel_time : ℝ) (bridge_length : ℝ) (bridge_time : ℝ)
    (h1 : tunnel_length = 860)
    (h2 : tunnel_time = 22)
    (h3 : bridge_length = 790)
    (h4 : bridge_time = 33)
    (h5 : (bridge_length + x) / bridge_time = (tunnel_length - x) / tunnel_time) :
    x = 200 := by
  sorry

#check train_length_problem

end train_length_problem_l1328_132864


namespace sqrt_n_plus_9_equals_25_l1328_132884

theorem sqrt_n_plus_9_equals_25 (n : ℝ) : Real.sqrt (n + 9) = 25 → n = 616 := by
  sorry

end sqrt_n_plus_9_equals_25_l1328_132884


namespace ratio_value_l1328_132807

theorem ratio_value (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end ratio_value_l1328_132807


namespace ten_person_round_robin_matches_l1328_132878

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: A 10-person round-robin tournament has 45 matches -/
theorem ten_person_round_robin_matches :
  roundRobinMatches 10 = 45 := by
  sorry

#eval roundRobinMatches 10  -- Should output 45

end ten_person_round_robin_matches_l1328_132878


namespace cos_sum_squared_one_solutions_l1328_132803

theorem cos_sum_squared_one_solutions (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = π/2 + k*π ∨ 
            x = π/4 + 2*k*π ∨ 
            x = 3*π/4 + 2*k*π ∨ 
            x = π/6 + 2*k*π ∨ 
            x = 5*π/6 + 2*k*π) :=
by sorry

end cos_sum_squared_one_solutions_l1328_132803


namespace inverse_variation_problem_l1328_132814

/-- Given that a² varies inversely with b⁴, and a = 7 when b = 2, 
    prove that a² = 3.0625 when b = 4 -/
theorem inverse_variation_problem (a b : ℝ) (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → a^2 * x^4 = k) →  -- a² varies inversely with b⁴
  (7^2 * 2^4 = k) →                             -- a = 7 when b = 2
  (a^2 * 4^4 = k) →                             -- condition for b = 4
  a^2 = 3.0625                                  -- conclusion
:= by sorry

end inverse_variation_problem_l1328_132814


namespace sine_shift_to_cosine_l1328_132821

open Real

theorem sine_shift_to_cosine (x : ℝ) :
  let f (t : ℝ) := sin (2 * t + π / 6)
  let g (t : ℝ) := f (t + π / 6)
  g x = cos (2 * x) :=
by sorry

end sine_shift_to_cosine_l1328_132821


namespace maggie_fish_books_l1328_132895

/-- The number of fish books Maggie bought -/
def fish_books : ℕ := sorry

/-- The total amount Maggie spent -/
def total_spent : ℕ := 170

/-- The number of plant books Maggie bought -/
def plant_books : ℕ := 9

/-- The number of science magazines Maggie bought -/
def science_magazines : ℕ := 10

/-- The cost of each book -/
def book_cost : ℕ := 15

/-- The cost of each magazine -/
def magazine_cost : ℕ := 2

theorem maggie_fish_books : 
  fish_books = 1 := by sorry

end maggie_fish_books_l1328_132895


namespace prob_red_then_green_l1328_132831

/-- A bag containing one red ball and one green ball -/
structure Bag :=
  (red : Nat)
  (green : Nat)

/-- The initial state of the bag -/
def initial_bag : Bag :=
  { red := 1, green := 1 }

/-- A draw from the bag -/
inductive Draw
  | Red
  | Green

/-- The probability of drawing a specific sequence of two balls -/
def prob_draw (first second : Draw) : ℚ :=
  1 / 4

/-- Theorem: The probability of drawing a red ball first and a green ball second is 1/4 -/
theorem prob_red_then_green :
  prob_draw Draw.Red Draw.Green = 1 / 4 := by
  sorry

end prob_red_then_green_l1328_132831


namespace complex_root_magnitude_l1328_132826

theorem complex_root_magnitude (z : ℂ) : z^2 + 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_root_magnitude_l1328_132826


namespace mail_cost_theorem_l1328_132824

def cost_per_package : ℕ := 5
def number_of_parents : ℕ := 2
def number_of_brothers : ℕ := 3
def children_per_brother : ℕ := 2

def total_relatives : ℕ := 
  number_of_parents + number_of_brothers + 
  number_of_brothers * (1 + 1 + children_per_brother)

def total_cost : ℕ := total_relatives * cost_per_package

theorem mail_cost_theorem : total_cost = 70 := by
  sorry

end mail_cost_theorem_l1328_132824


namespace marks_animals_legs_count_l1328_132823

theorem marks_animals_legs_count :
  let kangaroo_count : ℕ := 23
  let goat_count : ℕ := 3 * kangaroo_count
  let kangaroo_legs : ℕ := 2
  let goat_legs : ℕ := 4
  kangaroo_count * kangaroo_legs + goat_count * goat_legs = 322 := by
  sorry

end marks_animals_legs_count_l1328_132823


namespace carltons_shirts_l1328_132809

theorem carltons_shirts (shirts : ℕ) (vests : ℕ) (outfits : ℕ) : 
  vests = 2 * shirts → 
  outfits = vests * shirts → 
  outfits = 18 → 
  shirts = 3 := by
sorry

end carltons_shirts_l1328_132809


namespace multiples_of_10_and_12_within_100_l1328_132888

theorem multiples_of_10_and_12_within_100 : 
  ∃! n : ℕ, n ≤ 100 ∧ 10 ∣ n ∧ 12 ∣ n :=
by
  -- The proof would go here
  sorry

end multiples_of_10_and_12_within_100_l1328_132888


namespace fifteenth_student_age_l1328_132874

/-- Given a class of 15 students with an average age of 15 years,
    where 8 students have an average age of 14 years and 6 students
    have an average age of 16 years, the age of the 15th student is 17 years. -/
theorem fifteenth_student_age
  (total_students : Nat)
  (total_average_age : ℚ)
  (group1_students : Nat)
  (group1_average_age : ℚ)
  (group2_students : Nat)
  (group2_average_age : ℚ)
  (h1 : total_students = 15)
  (h2 : total_average_age = 15)
  (h3 : group1_students = 8)
  (h4 : group1_average_age = 14)
  (h5 : group2_students = 6)
  (h6 : group2_average_age = 16) :
  (total_students * total_average_age) - (group1_students * group1_average_age) - (group2_students * group2_average_age) = 17 := by
  sorry


end fifteenth_student_age_l1328_132874


namespace unique_solution_cos_arctan_sin_arccos_l1328_132841

theorem unique_solution_cos_arctan_sin_arccos (z : ℝ) :
  (∃! z : ℝ, 0 ≤ z ∧ z ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos z))) = z) ∧
  (Real.cos (Real.arctan (Real.sin (Real.arccos (Real.sqrt 2 / 2)))) = Real.sqrt 2 / 2) :=
by sorry

end unique_solution_cos_arctan_sin_arccos_l1328_132841


namespace exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l1328_132838

/-- The measure of the exterior angle DEF in a configuration where a regular pentagon
    and a regular octagon share a side. -/
theorem exterior_angle_pentagon_octagon : ℝ :=
  let pentagon_interior_angle : ℝ := 180 * (5 - 2) / 5
  let octagon_interior_angle : ℝ := 180 * (8 - 2) / 8
  let sum_of_angles_at_E : ℝ := 360
  117

/-- Proof that the exterior angle DEF measures 117° when a regular pentagon ABCDE
    and a regular octagon AEFGHIJK share a side AE in a plane. -/
theorem exterior_angle_pentagon_octagon_is_117 :
  exterior_angle_pentagon_octagon = 117 := by
  sorry

end exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l1328_132838


namespace number_problem_l1328_132840

theorem number_problem (X Y Z : ℝ) 
  (h1 : X - Y = 3500)
  (h2 : (3/5) * X = (2/3) * Y)
  (h3 : 0.097 * Y = Real.sqrt Z) :
  X = 35000 ∧ Y = 31500 ∧ Z = 9333580.25 := by
  sorry

end number_problem_l1328_132840


namespace sufficient_but_not_necessary_l1328_132869

def sequence_a (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => (sequence_a a₁ n) ^ 2

def monotonically_increasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (n + 1) > f n

theorem sufficient_but_not_necessary :
  (∀ a₁ : ℝ, a₁ = 2 → monotonically_increasing (sequence_a a₁)) ∧
  (∃ a₁ : ℝ, a₁ ≠ 2 ∧ monotonically_increasing (sequence_a a₁)) :=
by sorry

end sufficient_but_not_necessary_l1328_132869


namespace choose_one_book_result_l1328_132889

/-- The number of ways to choose one book from a collection of Chinese, English, and Math books -/
def choose_one_book (chinese : ℕ) (english : ℕ) (math : ℕ) : ℕ :=
  chinese + english + math

/-- Theorem: Choosing one book from 10 Chinese, 7 English, and 5 Math books has 22 possibilities -/
theorem choose_one_book_result : choose_one_book 10 7 5 = 22 := by
  sorry

end choose_one_book_result_l1328_132889


namespace wrappers_collection_proof_l1328_132885

/-- The number of wrappers collected by Andy -/
def andy_wrappers : ℕ := 34

/-- The number of wrappers collected by Max -/
def max_wrappers : ℕ := 15

/-- The number of wrappers collected by Zoe -/
def zoe_wrappers : ℕ := 25

/-- The total number of wrappers collected by all three friends -/
def total_wrappers : ℕ := andy_wrappers + max_wrappers + zoe_wrappers

theorem wrappers_collection_proof : total_wrappers = 74 := by
  sorry

end wrappers_collection_proof_l1328_132885


namespace real_part_of_complex_product_l1328_132813

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 + Complex.I) * (2 - Complex.I) ∧ z.re = 3 := by
  sorry

end real_part_of_complex_product_l1328_132813


namespace vectors_not_coplanar_l1328_132837

/-- Prove that the given vectors are not coplanar -/
theorem vectors_not_coplanar (a b c : ℝ × ℝ × ℝ) :
  a = (-7, 10, -5) →
  b = (0, -2, -1) →
  c = (-2, 4, -1) →
  ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end vectors_not_coplanar_l1328_132837


namespace football_players_count_l1328_132858

theorem football_players_count (cricket_players hockey_players softball_players total_players : ℕ) 
  (h1 : cricket_players = 10)
  (h2 : hockey_players = 12)
  (h3 : softball_players = 13)
  (h4 : total_players = 51) :
  total_players - (cricket_players + hockey_players + softball_players) = 16 := by
  sorry

end football_players_count_l1328_132858


namespace arrangements_with_restrictions_total_arrangements_prove_total_arrangements_l1328_132836

-- Define the number of people
def n : ℕ := 5

-- Define the function to calculate permutations
def permutations (k : ℕ) : ℕ := Nat.factorial k

-- Define the function to calculate arrangements
def arrangements (n k : ℕ) : ℕ := permutations n / permutations (n - k)

-- Theorem statement
theorem arrangements_with_restrictions :
  arrangements n n - 2 * arrangements (n - 1) (n - 1) + arrangements (n - 2) (n - 2) = 78 := by
  sorry

-- The result we want to prove
theorem total_arrangements : ℕ := 78

-- The main theorem
theorem prove_total_arrangements :
  arrangements n n - 2 * arrangements (n - 1) (n - 1) + arrangements (n - 2) (n - 2) = total_arrangements := by
  sorry

end arrangements_with_restrictions_total_arrangements_prove_total_arrangements_l1328_132836


namespace system_solutions_l1328_132881

/-- The system of equations has only two solutions -/
theorem system_solutions :
  ∀ x y z : ℝ,
  (x + y * z = 2 ∧ y + x * z = 2 ∧ z + x * y = 2) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end system_solutions_l1328_132881


namespace unique_satisfying_pair_satisfying_pair_is_negative_one_zero_l1328_132801

/-- Predicate that checks if a pair (m, n) satisfies the condition for all (x, y) -/
def satisfies_condition (m n : ℝ) : Prop :=
  ∀ x y : ℝ, y ≠ 0 → x / y = m → (x + y)^2 = n

/-- Theorem stating that (-1, 0) is the only pair satisfying the condition -/
theorem unique_satisfying_pair :
  ∃! p : ℝ × ℝ, satisfies_condition p.1 p.2 ∧ p = (-1, 0) := by
  sorry

/-- Corollary: If (m, n) satisfies the condition, then m = -1 and n = 0 -/
theorem satisfying_pair_is_negative_one_zero (m n : ℝ) :
  satisfies_condition m n → m = -1 ∧ n = 0 := by
  sorry

end unique_satisfying_pair_satisfying_pair_is_negative_one_zero_l1328_132801


namespace coin_ratio_l1328_132883

theorem coin_ratio (pennies nickels dimes quarters : ℕ) 
  (h1 : nickels = 5 * dimes)
  (h2 : pennies = 3 * nickels)
  (h3 : pennies = 120)
  (h4 : pennies + 5 * nickels + 10 * dimes + 25 * quarters = 800) :
  quarters = 2 * dimes := by
  sorry

end coin_ratio_l1328_132883


namespace no_three_points_property_H_l1328_132843

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of property H for a line intersecting the ellipse C -/
def property_H (A B M : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ C M.1 M.2 ∧
  M.1 = 3/5 * A.1 + 4/5 * B.1 ∧
  M.2 = 3/5 * A.2 + 4/5 * B.2

/-- Main theorem: No three distinct points on C form lines all having property H -/
theorem no_three_points_property_H :
  ¬ ∃ (P Q R : ℝ × ℝ),
    C P.1 P.2 ∧ C Q.1 Q.2 ∧ C R.1 R.2 ∧
    P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
    (∃ M₁, property_H P Q M₁) ∧
    (∃ M₂, property_H Q R M₂) ∧
    (∃ M₃, property_H R P M₃) :=
sorry

end no_three_points_property_H_l1328_132843


namespace class_project_total_l1328_132863

/-- Calculates the total amount gathered for a class project with discounts and fees -/
theorem class_project_total (total_students : ℕ) (full_price : ℚ) 
  (full_paying : ℕ) (high_merit : ℕ) (financial_needs : ℕ) (special_discount : ℕ)
  (high_merit_discount : ℚ) (financial_needs_discount : ℚ) (special_discount_rate : ℚ)
  (admin_fee : ℚ) :
  total_students = 35 →
  full_price = 50 →
  full_paying = 20 →
  high_merit = 5 →
  financial_needs = 7 →
  special_discount = 3 →
  high_merit_discount = 25 / 100 →
  financial_needs_discount = 1 / 2 →
  special_discount_rate = 10 / 100 →
  admin_fee = 100 →
  (full_paying * full_price + 
   high_merit * (full_price * (1 - high_merit_discount)) +
   financial_needs * (full_price * financial_needs_discount) +
   special_discount * (full_price * (1 - special_discount_rate))) - admin_fee = 1397.5 := by
  sorry


end class_project_total_l1328_132863


namespace connecting_point_on_line_connecting_point_on_line_x_plus_1_connecting_point_area_and_distance_l1328_132891

-- Define the concept of a "connecting point"
def is_connecting_point (P Q : ℝ × ℝ) : Prop :=
  Q.1 = P.1 ∧ 
  ((P.1 ≥ 0 ∧ Q.2 = P.2) ∨ (P.1 < 0 ∧ Q.2 = -P.2))

-- Part 1
theorem connecting_point_on_line (k : ℝ) (A A' : ℝ × ℝ) :
  k ≠ 0 →
  A.2 = k * A.1 →
  is_connecting_point A A' →
  A' = (-2, -6) →
  k = -3 :=
sorry

-- Part 2
theorem connecting_point_on_line_x_plus_1 (m : ℝ) (B B' : ℝ × ℝ) :
  B.2 = B.1 + 1 →
  is_connecting_point B B' →
  B' = (m, 2) →
  (m ≥ 0 → B = (1, 2)) ∧
  (m < 0 → B = (-3, -2)) :=
sorry

-- Part 3
theorem connecting_point_area_and_distance (P C C' : ℝ × ℝ) :
  P = (1, 0) →
  C.2 = -2 * C.1 + 2 →
  is_connecting_point C C' →
  abs ((P.1 - C.1) * (C'.2 - C.2)) / 2 = 18 →
  Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = 3 * Real.sqrt 5 :=
sorry

end connecting_point_on_line_connecting_point_on_line_x_plus_1_connecting_point_area_and_distance_l1328_132891


namespace race_time_differences_l1328_132810

/-- Race competition with three competitors --/
structure RaceCompetition where
  distance : ℝ
  time_A : ℝ
  time_B : ℝ
  time_C : ℝ

/-- Calculate time difference between two competitors --/
def timeDifference (t1 t2 : ℝ) : ℝ := t2 - t1

/-- Theorem stating the time differences between competitors --/
theorem race_time_differences (race : RaceCompetition) 
  (h_distance : race.distance = 250)
  (h_time_A : race.time_A = 40)
  (h_time_B : race.time_B = 50)
  (h_time_C : race.time_C = 55) : 
  (timeDifference race.time_A race.time_B = 10) ∧ 
  (timeDifference race.time_A race.time_C = 15) ∧ 
  (timeDifference race.time_B race.time_C = 5) := by
  sorry

end race_time_differences_l1328_132810


namespace inverse_proportion_problem_l1328_132877

/-- Given that x and y are inversely proportional, and when their sum is 50, x is three times y,
    prove that y = -39.0625 when x = -12 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (∀ x y, x * y = k) →  -- x and y are inversely proportional
  (∃ x y, x + y = 50 ∧ x = 3 * y) →  -- when their sum is 50, x is three times y
  (x = -12 → y = -39.0625) :=  -- prove that y = -39.0625 when x = -12
by sorry

end inverse_proportion_problem_l1328_132877


namespace intersection_sum_l1328_132873

/-- Given two lines y = nx + 3 and y = 5x + c that intersect at (4, 11), prove that n + c = -7 -/
theorem intersection_sum (n c : ℝ) : 
  (4 * n + 3 = 11) → (5 * 4 + c = 11) → n + c = -7 := by
  sorry

end intersection_sum_l1328_132873


namespace jerome_toy_car_ratio_l1328_132816

/-- Proves that the ratio of toy cars Jerome bought this month to last month is 2:1 -/
theorem jerome_toy_car_ratio :
  let original_cars : ℕ := 25
  let cars_bought_last_month : ℕ := 5
  let total_cars_now : ℕ := 40
  let cars_bought_this_month : ℕ := total_cars_now - original_cars - cars_bought_last_month
  cars_bought_this_month / cars_bought_last_month = 2 := by
  sorry

end jerome_toy_car_ratio_l1328_132816


namespace power_function_through_point_l1328_132817

/-- A power function that passes through the point (2, √2) -/
def f (x : ℝ) : ℝ := x^(1/2)

/-- Theorem: For the power function f(x) that passes through (2, √2), f(5) = √5 -/
theorem power_function_through_point (h : f 2 = Real.sqrt 2) : f 5 = Real.sqrt 5 := by
  sorry

end power_function_through_point_l1328_132817


namespace solution_set_for_a_4_range_of_a_for_all_x_geq_4_l1328_132855

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

theorem solution_set_for_a_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

theorem range_of_a_for_all_x_geq_4 :
  (∀ x : ℝ, f a x ≥ 4) → (a ≤ -3 ∨ a ≥ 5) := by sorry

end solution_set_for_a_4_range_of_a_for_all_x_geq_4_l1328_132855


namespace inequality_proof_l1328_132819

theorem inequality_proof (m n : ℝ) (h : m > n) : 1 - 2*m < 1 - 2*n := by
  sorry

end inequality_proof_l1328_132819


namespace marcus_car_mileage_l1328_132845

/-- Calculates the final mileage of a car after a road trip --/
def final_mileage (initial_mileage : ℕ) (tank_capacity : ℕ) (fuel_efficiency : ℕ) (refills : ℕ) : ℕ :=
  initial_mileage + tank_capacity * refills * fuel_efficiency

/-- Theorem stating the final mileage of Marcus' car after the road trip --/
theorem marcus_car_mileage :
  final_mileage 1728 20 30 2 = 2928 := by
  sorry

#eval final_mileage 1728 20 30 2

end marcus_car_mileage_l1328_132845


namespace trapezoid_ratio_satisfies_equation_l1328_132828

/-- Represents a trapezoid with a point inside dividing it into four triangles -/
structure TrapezoidWithPoint where
  AB : ℝ
  CD : ℝ
  area_PCD : ℝ
  area_PAD : ℝ
  area_PBC : ℝ
  area_PAB : ℝ
  h_AB_gt_CD : AB > CD
  h_areas : area_PCD = 3 ∧ area_PAD = 5 ∧ area_PBC = 6 ∧ area_PAB = 8

/-- The ratio of AB to CD satisfies a specific quadratic equation -/
theorem trapezoid_ratio_satisfies_equation (t : TrapezoidWithPoint) :
  let k := t.AB / t.CD
  k^2 + (22/6) * k + 16/6 = 0 := by
  sorry

end trapezoid_ratio_satisfies_equation_l1328_132828


namespace jamie_coin_problem_l1328_132822

/-- The number of nickels (and dimes and quarters) in Jamie's jar -/
def num_coins : ℕ := 33

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of coins in Jamie's jar in cents -/
def total_value : ℕ := 1320

theorem jamie_coin_problem :
  num_coins * nickel_value + num_coins * dime_value + num_coins * quarter_value = total_value :=
by sorry

end jamie_coin_problem_l1328_132822


namespace parabola_focus_coordinates_l1328_132882

/-- Given a parabola with equation x = 2py^2 where p > 0, its focus has coordinates (1/(8p), 0) -/
theorem parabola_focus_coordinates (p : ℝ) (hp : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | x = 2 * p * y^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (1 / (8 * p), 0) := by
  sorry

end parabola_focus_coordinates_l1328_132882


namespace total_votes_is_1375_l1328_132857

/-- Represents the election results with given conditions -/
structure ElectionResults where
  winners_votes : ℕ  -- Combined majority of winners
  spoiled_votes : ℕ  -- Number of spoiled votes
  final_percentages : List ℚ  -- Final round percentages for top three candidates

/-- Calculates the total number of votes cast in the election -/
def total_votes (results : ElectionResults) : ℕ :=
  results.winners_votes + results.spoiled_votes

/-- Theorem stating that the total number of votes is 1375 given the conditions -/
theorem total_votes_is_1375 (results : ElectionResults) 
  (h1 : results.winners_votes = 1050)
  (h2 : results.spoiled_votes = 325)
  (h3 : results.final_percentages = [41/100, 34/100, 25/100]) :
  total_votes results = 1375 := by
  sorry

#eval total_votes { winners_votes := 1050, spoiled_votes := 325, final_percentages := [41/100, 34/100, 25/100] }

end total_votes_is_1375_l1328_132857


namespace biscuit_boxes_combination_exists_l1328_132880

theorem biscuit_boxes_combination_exists : ∃ (a b c d e : ℕ), 16*a + 17*b + 23*c + 39*d + 40*e = 100 := by
  sorry

end biscuit_boxes_combination_exists_l1328_132880


namespace rectangular_prism_sum_l1328_132844

/-- Represents a rectangular prism -/
structure RectangularPrism where
  -- We don't need to define any specific properties here

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, vertices, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_edges rp + num_vertices rp + num_faces rp = 26 := by
  sorry

end rectangular_prism_sum_l1328_132844


namespace vector_decomposition_l1328_132815

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![3, 3, -1]
def p : Fin 3 → ℝ := ![3, 1, 0]
def q : Fin 3 → ℝ := ![-1, 2, 1]
def r : Fin 3 → ℝ := ![-1, 0, 2]

/-- Theorem: x can be decomposed as p + q - r -/
theorem vector_decomposition : x = p + q - r := by
  sorry

end vector_decomposition_l1328_132815


namespace chocolate_bars_count_l1328_132830

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 18

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 28

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * bars_per_small_box

theorem chocolate_bars_count : total_chocolate_bars = 504 := by
  sorry

end chocolate_bars_count_l1328_132830


namespace inequality_proof_l1328_132879

theorem inequality_proof (x : ℝ) : 2 ≤ (3 * x^2 - 6 * x + 6) / (x^2 - x + 1) ∧ (3 * x^2 - 6 * x + 6) / (x^2 - x + 1) ≤ 6 := by
  sorry

end inequality_proof_l1328_132879


namespace chess_tournament_games_l1328_132852

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n.choose 2) = 45 := by sorry

end chess_tournament_games_l1328_132852


namespace apple_problem_l1328_132870

/-- Proves that given the conditions of the apple problem, each child originally had 15 apples -/
theorem apple_problem (num_children : Nat) (apples_eaten : Nat) (apples_sold : Nat) (apples_left : Nat) :
  num_children = 5 →
  apples_eaten = 8 →
  apples_sold = 7 →
  apples_left = 60 →
  ∃ x : Nat, num_children * x - apples_eaten - apples_sold = apples_left ∧ x = 15 := by
  sorry

end apple_problem_l1328_132870


namespace chocolate_factory_order_completion_l1328_132859

/-- Represents the number of days required to complete an order of candies. -/
def days_to_complete_order (candies_per_hour : ℕ) (hours_per_day : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_hour + hours_per_day - 1) / hours_per_day

/-- Theorem stating that it takes 8 days to complete the order under given conditions. -/
theorem chocolate_factory_order_completion :
  days_to_complete_order 50 10 4000 = 8 := by
  sorry

#eval days_to_complete_order 50 10 4000

end chocolate_factory_order_completion_l1328_132859


namespace no_adjacent_standing_probability_l1328_132875

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people sitting around the circular table. -/
def numPeople : ℕ := 10

/-- The probability of getting the desired outcome (no two adjacent people standing). -/
def probability : ℚ := validArrangements numPeople / 2^numPeople

/-- Theorem stating that the probability of no two adjacent people standing
    in a circular arrangement of 10 people, each flipping a fair coin, is 123/1024. -/
theorem no_adjacent_standing_probability :
  probability = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l1328_132875


namespace seventh_root_unity_product_l1328_132829

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end seventh_root_unity_product_l1328_132829


namespace max_length_sum_l1328_132897

/-- The length of an integer is the number of positive prime factors, not necessarily distinct, whose product is equal to the integer. -/
def length (k : ℕ) : ℕ := sorry

/-- A number is prime if it has exactly two factors -/
def isPrime (p : ℕ) : Prop := sorry

theorem max_length_sum :
  ∀ x y z : ℕ,
  x > 1 → y > 1 → z > 1 →
  (∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ x = p * q) →
  (∃ p q r : ℕ, isPrime p ∧ isPrime q ∧ isPrime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ y = p * q * r) →
  x + 3 * y + 5 * z < 5000 →
  length x + length y + length z ≤ 14 :=
sorry

end max_length_sum_l1328_132897


namespace mode_of_visual_acuity_l1328_132825

-- Define the visual acuity values and their frequencies
def visual_acuity : List ℝ := [4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
def frequencies : List ℕ := [2, 3, 6, 9, 12, 8, 5, 3]

-- Define a function to find the mode
def mode (values : List ℝ) (freqs : List ℕ) : ℝ :=
  let pairs := List.zip values freqs
  let maxFreq := List.foldl (fun acc (_, f) => max acc f) 0 pairs
  let modes := List.filter (fun (_, f) => f == maxFreq) pairs
  (List.head! modes).1

-- Theorem: The mode of visual acuity is 4.7
theorem mode_of_visual_acuity :
  mode visual_acuity frequencies = 4.7 :=
by sorry

end mode_of_visual_acuity_l1328_132825


namespace chandelier_illumination_probability_chandelier_illumination_probability_is_correct_l1328_132800

/-- The probability of a chandelier with 3 parallel-connected bulbs being illuminated, 
    given that the probability of each bulb working properly is 0.7 -/
theorem chandelier_illumination_probability : ℝ :=
  let p : ℝ := 0.7  -- probability of each bulb working properly
  let num_bulbs : ℕ := 3  -- number of bulbs in parallel connection
  1 - (1 - p) ^ num_bulbs

/-- Proof that the probability of the chandelier being illuminated is 0.973 -/
theorem chandelier_illumination_probability_is_correct : 
  chandelier_illumination_probability = 0.973 := by
  sorry


end chandelier_illumination_probability_chandelier_illumination_probability_is_correct_l1328_132800


namespace john_lift_weight_l1328_132832

/-- Calculates the final weight John can lift after training and using a magical bracer -/
def final_lift_weight (initial_weight : ℕ) (weight_increase : ℕ) (bracer_multiplier : ℕ) : ℕ :=
  let after_training := initial_weight + weight_increase
  let bracer_increase := after_training * bracer_multiplier
  after_training + bracer_increase

/-- Proves that John can lift 2800 pounds after training and using the magical bracer -/
theorem john_lift_weight :
  final_lift_weight 135 265 6 = 2800 := by
  sorry

end john_lift_weight_l1328_132832


namespace orchid_rose_difference_l1328_132887

theorem orchid_rose_difference (initial_roses initial_orchids final_roses final_orchids : ℕ) :
  initial_roses = 7 →
  initial_orchids = 12 →
  final_roses = 11 →
  final_orchids = 20 →
  final_orchids - final_roses = 9 := by
sorry

end orchid_rose_difference_l1328_132887
