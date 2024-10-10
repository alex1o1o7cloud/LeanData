import Mathlib

namespace perpendicular_lines_slope_l1567_156780

/-- Given two lines l₁ and l₂ in the xy-plane:
    l₁: mx + y - 1 = 0
    l₂: x - 2y + 5 = 0
    If l₁ is perpendicular to l₂, then m = 2. -/
theorem perpendicular_lines_slope (m : ℝ) : 
  (∀ x y, mx + y - 1 = 0 → x - 2*y + 5 = 0 → (mx + y - 1 = 0 ∧ x - 2*y + 5 = 0) → m = 2) := by
  sorry

end perpendicular_lines_slope_l1567_156780


namespace f_extrema_on_interval_l1567_156739

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem f_extrema_on_interval :
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc 1 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc 1 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc 1 3, f x₂ = max) ∧
    min = 1 ∧ max = 5 := by
  sorry

end f_extrema_on_interval_l1567_156739


namespace dodecagon_enclosure_l1567_156721

/-- The number of sides of the central polygon -/
def m : ℕ := 12

/-- The number of smaller polygons enclosing the central polygon -/
def num_enclosing : ℕ := 12

/-- The number of smaller polygons meeting at each vertex of the central polygon -/
def num_meeting : ℕ := 3

/-- The number of sides of each smaller polygon -/
def n : ℕ := 12

/-- The interior angle of a regular polygon with m sides -/
def interior_angle (m : ℕ) : ℚ := (m - 2) * 180 / m

/-- The exterior angle of a regular polygon with m sides -/
def exterior_angle (m : ℕ) : ℚ := 180 - interior_angle m

/-- Theorem stating that n must be 12 for the given configuration -/
theorem dodecagon_enclosure :
  exterior_angle m = num_meeting * (exterior_angle n / num_meeting) :=
sorry

end dodecagon_enclosure_l1567_156721


namespace greatest_square_with_nine_factors_l1567_156777

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem greatest_square_with_nine_factors :
  ∃ n : ℕ, n = 196 ∧
    n < 200 ∧
    is_perfect_square n ∧
    count_factors n = 9 ∧
    ∀ m : ℕ, m < 200 → is_perfect_square m → count_factors m = 9 → m ≤ n :=
sorry

end greatest_square_with_nine_factors_l1567_156777


namespace geometric_sequence_quadratic_roots_l1567_156708

/-- Given that 2, b, a form a geometric sequence in order, prove that the equation ax^2 + bx + 1/3 = 0 has exactly 2 real roots -/
theorem geometric_sequence_quadratic_roots
  (b a : ℝ)
  (h_geometric : ∃ (q : ℝ), b = 2 * q ∧ a = 2 * q^2) :
  (∃! (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + 1/3 = 0 ∧ a * y^2 + b * y + 1/3 = 0) :=
sorry

end geometric_sequence_quadratic_roots_l1567_156708


namespace sum_f_positive_l1567_156786

def f (x : ℝ) := x^3 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end sum_f_positive_l1567_156786


namespace friend_lunch_cost_l1567_156770

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 19 →
  difference = 3 →
  friend_cost = total / 2 + difference / 2 →
  friend_cost = 11 := by
sorry

end friend_lunch_cost_l1567_156770


namespace combined_work_time_l1567_156769

theorem combined_work_time (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a = 21 → b = 6 → c = 12 → (1 / (1/a + 1/b + 1/c) : ℝ) = 84/25 := by
  sorry

end combined_work_time_l1567_156769


namespace lillian_sugar_at_home_l1567_156726

/-- The number of cups of sugar needed for cupcake batter -/
def batterSugar (cupcakes : ℕ) : ℕ := cupcakes / 12

/-- The number of cups of sugar needed for cupcake frosting -/
def frostingSugar (cupcakes : ℕ) : ℕ := 2 * (cupcakes / 12)

/-- The total number of cups of sugar needed for cupcakes -/
def totalSugarNeeded (cupcakes : ℕ) : ℕ := batterSugar cupcakes + frostingSugar cupcakes

theorem lillian_sugar_at_home (cupcakes sugarBought sugarAtHome : ℕ) :
  cupcakes = 60 →
  sugarBought = 12 →
  totalSugarNeeded cupcakes = sugarBought + sugarAtHome →
  sugarAtHome = 3 := by
  sorry

#check lillian_sugar_at_home

end lillian_sugar_at_home_l1567_156726


namespace lindas_savings_l1567_156717

theorem lindas_savings (savings : ℝ) : 
  (2 / 3 : ℝ) * savings + 250 = savings → savings = 750 := by
  sorry

end lindas_savings_l1567_156717


namespace cubic_and_quadratic_sum_l1567_156744

theorem cubic_and_quadratic_sum (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (prod_eq : x * y = 12) : 
  x^3 + y^3 = 224 ∧ x^2 + y^2 = 40 := by
sorry

end cubic_and_quadratic_sum_l1567_156744


namespace boys_in_class_l1567_156788

/-- Given a class with 10 girls, prove that if there are 780 ways to select 1 girl and 2 boys
    when choosing 3 students at random, then the number of boys in the class is 13. -/
theorem boys_in_class (num_girls : ℕ) (num_ways : ℕ) : 
  num_girls = 10 →
  num_ways = 780 →
  (∃ num_boys : ℕ, 
    num_ways = (num_girls.choose 1) * (num_boys.choose 2) ∧
    num_boys = 13) :=
by sorry

end boys_in_class_l1567_156788


namespace min_value_of_expression_l1567_156719

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (1 : ℝ) / (2 * b - 3) = -a / (2 * b)) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ (1 : ℝ) / (2 * y - 3) = -x / (2 * y) → 2 * a + 3 * b ≤ 2 * x + 3 * y) ∧
  (2 * a + 3 * b = 25 / 2) :=
sorry

end min_value_of_expression_l1567_156719


namespace complex_square_multiply_i_l1567_156753

theorem complex_square_multiply_i : (1 - Complex.I)^2 * Complex.I = 2 := by
  sorry

end complex_square_multiply_i_l1567_156753


namespace sequence_third_term_l1567_156701

/-- Given a sequence {a_n} with general term a_n = 3n - 5, prove that a_3 = 4 -/
theorem sequence_third_term (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 5) : a 3 = 4 := by
  sorry

end sequence_third_term_l1567_156701


namespace answer_choices_per_mc_question_l1567_156748

/-- The number of ways to answer 3 true-false questions where all answers cannot be the same -/
def true_false_combinations : ℕ := 6

/-- The total number of ways to write the answer key -/
def total_combinations : ℕ := 96

/-- The number of multiple-choice questions -/
def num_mc_questions : ℕ := 2

theorem answer_choices_per_mc_question :
  ∃ n : ℕ, n > 0 ∧ true_false_combinations * n^num_mc_questions = total_combinations :=
sorry

end answer_choices_per_mc_question_l1567_156748


namespace lily_milk_problem_l1567_156723

theorem lily_milk_problem (initial_milk : ℚ) (milk_given : ℚ) (milk_left : ℚ) : 
  initial_milk = 5 → milk_given = 18/7 → milk_left = initial_milk - milk_given → milk_left = 17/7 := by
  sorry

end lily_milk_problem_l1567_156723


namespace mariela_get_well_cards_l1567_156729

theorem mariela_get_well_cards (total : ℝ) (from_home : ℝ) (from_country : ℝ)
  (h1 : total = 403.0)
  (h2 : from_home = 287.0)
  (h3 : total = from_home + from_country) :
  from_country = 116.0 := by
  sorry

end mariela_get_well_cards_l1567_156729


namespace train_speed_calculation_l1567_156750

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 250 ∧ bridge_length = 120 ∧ time = 20 →
  (train_length + bridge_length) / time = 18.5 := by
  sorry

end train_speed_calculation_l1567_156750


namespace initial_gasohol_volume_l1567_156707

/-- Represents the composition of a fuel mixture -/
structure FuelMixture where
  ethanol : ℝ  -- Percentage of ethanol
  gasoline : ℝ  -- Percentage of gasoline

/-- Represents the state of the fuel tank -/
structure FuelTank where
  volume : ℝ  -- Total volume in liters
  mixture : FuelMixture  -- Composition of the mixture

def initial_mixture : FuelMixture := { ethanol := 0.05, gasoline := 0.95 }
def desired_mixture : FuelMixture := { ethanol := 0.10, gasoline := 0.90 }
def ethanol_added : ℝ := 2.5

theorem initial_gasohol_volume (initial : FuelTank) (final : FuelTank) :
  initial.mixture = initial_mixture →
  final.mixture = desired_mixture →
  final.volume = initial.volume + ethanol_added →
  final.volume * final.mixture.ethanol = initial.volume * initial.mixture.ethanol + ethanol_added →
  initial.volume = 45 := by
  sorry

end initial_gasohol_volume_l1567_156707


namespace quadratic_minimum_l1567_156713

theorem quadratic_minimum (p q : ℝ) : 
  (∃ (y : ℝ → ℝ), (∀ x, y x = x^2 + p*x + q) ∧ 
   (∃ x₀, ∀ x, y x₀ ≤ y x) ∧ 
   (∃ x₁, y x₁ = 0)) →
  q = p^2 / 4 := by
sorry

end quadratic_minimum_l1567_156713


namespace tan_half_product_squared_l1567_156738

theorem tan_half_product_squared (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) : 
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 26 / 7 := by
  sorry

end tan_half_product_squared_l1567_156738


namespace absolute_value_equation_one_negative_root_l1567_156724

theorem absolute_value_equation_one_negative_root (a : ℝ) : 
  (∃! x : ℝ, x < 0 ∧ |x| = a * x + 1) → a > 1 := by
  sorry

end absolute_value_equation_one_negative_root_l1567_156724


namespace concert_attendance_l1567_156725

/-- The number of buses used for the concert. -/
def num_buses : ℕ := 8

/-- The number of students each bus can carry. -/
def students_per_bus : ℕ := 45

/-- The total number of students who went to the concert. -/
def total_students : ℕ := num_buses * students_per_bus

theorem concert_attendance : total_students = 360 := by
  sorry

end concert_attendance_l1567_156725


namespace complex_location_l1567_156728

theorem complex_location (z : ℂ) (h : (1 + Complex.I * Real.sqrt 3) * z = Complex.I * (2 * Real.sqrt 3)) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_location_l1567_156728


namespace efgh_is_parallelogram_l1567_156711

-- Define the types for points and quadrilaterals
variable (Point : Type) (Quadrilateral : Type)

-- Define the property of being a convex quadrilateral
variable (is_convex_quadrilateral : Quadrilateral → Prop)

-- Define the property of forming an equilateral triangle
variable (forms_equilateral_triangle : Point → Point → Point → Prop)

-- Define the property of a triangle being directed outward or inward
variable (is_outward : Point → Point → Point → Quadrilateral → Prop)
variable (is_inward : Point → Point → Point → Quadrilateral → Prop)

-- Define the property of being a parallelogram
variable (is_parallelogram : Point → Point → Point → Point → Prop)

-- Theorem statement
theorem efgh_is_parallelogram 
  (A B C D E F G H : Point) (Q : Quadrilateral) :
  is_convex_quadrilateral Q →
  forms_equilateral_triangle A B E →
  forms_equilateral_triangle B C F →
  forms_equilateral_triangle C D G →
  forms_equilateral_triangle D A H →
  is_outward A B E Q →
  is_outward C D G Q →
  is_inward B C F Q →
  is_inward D A H Q →
  is_parallelogram E F G H :=
by sorry

end efgh_is_parallelogram_l1567_156711


namespace curve_is_line_segment_l1567_156784

/-- Parametric curve defined by x = 3t² + 4 and y = t² - 2, where 0 ≤ t ≤ 3 -/
def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + 4, t^2 - 2)

/-- The range of the parameter t -/
def t_range : Set ℝ := {t : ℝ | 0 ≤ t ∧ t ≤ 3}

/-- The set of points on the curve -/
def curve_points : Set (ℝ × ℝ) :=
  {p | ∃ t ∈ t_range, p = parametric_curve t}

/-- Theorem: The curve is a line segment -/
theorem curve_is_line_segment :
  ∃ a b c : ℝ, a ≠ 0 ∧ curve_points = {p : ℝ × ℝ | a * p.1 + b * p.2 = c} ∩
    {p : ℝ × ℝ | ∃ t ∈ t_range, p = parametric_curve t} :=
by sorry

end curve_is_line_segment_l1567_156784


namespace coefficient_of_y_l1567_156774

theorem coefficient_of_y (b : ℝ) : 
  (5 * (2 : ℝ)^2 - b * 2 + 55 = 59) → b = 8 := by
  sorry

end coefficient_of_y_l1567_156774


namespace leila_marathon_distance_l1567_156799

/-- Represents the total distance covered in marathons -/
structure MarathonDistance where
  miles : ℕ
  yards : ℕ

/-- Calculates the total distance covered in multiple marathons -/
def totalDistance (numMarathons : ℕ) (marathonMiles : ℕ) (marathonYards : ℕ) (yardsPerMile : ℕ) : MarathonDistance :=
  sorry

/-- Theorem stating the total distance covered by Leila in her marathons -/
theorem leila_marathon_distance :
  let numMarathons : ℕ := 15
  let marathonMiles : ℕ := 26
  let marathonYards : ℕ := 385
  let yardsPerMile : ℕ := 1760
  let result := totalDistance numMarathons marathonMiles marathonYards yardsPerMile
  result.miles = 393 ∧ result.yards = 495 ∧ result.yards < yardsPerMile :=
by sorry

end leila_marathon_distance_l1567_156799


namespace five_students_four_lectures_l1567_156740

/-- The number of ways students can choose lectures --/
def number_of_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 5 students choosing from 4 lectures results in 4^5 choices --/
theorem five_students_four_lectures :
  number_of_choices 5 4 = 4^5 := by
  sorry

end five_students_four_lectures_l1567_156740


namespace percentage_of_160_l1567_156732

theorem percentage_of_160 : (3 / 8 : ℚ) / 100 * 160 = 0.6 := by
  sorry

end percentage_of_160_l1567_156732


namespace longest_side_is_72_l1567_156749

/-- A rectangle with specific properties --/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 120
  area_eq : length * width = 2880

/-- The longest side of a SpecialRectangle is 72 --/
theorem longest_side_is_72 (rect : SpecialRectangle) : 
  max rect.length rect.width = 72 := by
  sorry

#check longest_side_is_72

end longest_side_is_72_l1567_156749


namespace smallest_exponent_sum_l1567_156757

theorem smallest_exponent_sum (p q r s : ℕ+) 
  (h_eq : (3^(p:ℕ))^2 + (3^(q:ℕ))^3 + (3^(r:ℕ))^5 = (3^(s:ℕ))^7) : 
  (p:ℕ) + q + r + s ≥ 106 := by
  sorry

end smallest_exponent_sum_l1567_156757


namespace conditions_implications_l1567_156773

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between A, B, C, and D
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_C : C → B
axiom C_nec_not_suff_D : (D → C) ∧ ¬(C → D)

-- State the theorem to be proved
theorem conditions_implications :
  -- B is a necessary but not sufficient condition for A
  ((B → A) ∧ ¬(A → B)) ∧
  -- A is a sufficient but not necessary condition for C
  ((A → C) ∧ ¬(C → A)) ∧
  -- D is neither a sufficient nor necessary condition for A
  (¬(D → A) ∧ ¬(A → D)) := by
  sorry

end conditions_implications_l1567_156773


namespace plane_angle_in_right_triangle_l1567_156745

/-- Given a right triangle and a plane through its hypotenuse, 
    this theorem relates the angles the plane makes with the triangle and its legs. -/
theorem plane_angle_in_right_triangle 
  (α β : Real) 
  (h_α : 0 < α ∧ α < π / 2) 
  (h_β : 0 < β ∧ β < π / 2) : 
  ∃ γ, γ = Real.arcsin (Real.sqrt (Real.sin (α + β) * Real.sin (α - β))) ∧ 
           0 ≤ γ ∧ γ ≤ π / 2 := by
  sorry


end plane_angle_in_right_triangle_l1567_156745


namespace intersection_A_B_union_complement_A_B_range_of_m_l1567_156781

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3 * m}

-- State the theorems
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 5} := by sorry

theorem union_complement_A_B : (Set.univ \ A) ∪ B = {x | -2 < x ∧ x < 5} := by sorry

theorem range_of_m (m : ℝ) : B ∩ C m = C m → m < -1/2 := by sorry

end intersection_A_B_union_complement_A_B_range_of_m_l1567_156781


namespace inscribed_circle_radius_l1567_156754

theorem inscribed_circle_radius (XY XZ YZ : ℝ) (h1 : XY = 26) (h2 : XZ = 15) (h3 : YZ = 17) :
  let s := (XY + XZ + YZ) / 2
  let area := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  area / s = 2 * Real.sqrt 42 / 29 := by sorry

end inscribed_circle_radius_l1567_156754


namespace ball_costs_and_max_purchase_l1567_156779

/-- Represents the cost of basketballs and soccer balls -/
structure BallCosts where
  basketball : ℕ
  soccer : ℕ

/-- Represents the purchase constraints -/
structure PurchaseConstraints where
  total_balls : ℕ
  max_cost : ℕ

/-- Theorem stating the correct costs and maximum number of basketballs -/
theorem ball_costs_and_max_purchase 
  (costs : BallCosts) 
  (constraints : PurchaseConstraints) : 
  (2 * costs.basketball + 3 * costs.soccer = 310) → 
  (5 * costs.basketball + 2 * costs.soccer = 500) → 
  (constraints.total_balls = 60) → 
  (constraints.max_cost = 4000) → 
  (costs.basketball = 80 ∧ costs.soccer = 50 ∧ 
   (∀ m : ℕ, m * costs.basketball + (constraints.total_balls - m) * costs.soccer ≤ constraints.max_cost → m ≤ 33)) := by
  sorry

end ball_costs_and_max_purchase_l1567_156779


namespace subset_sum_exists_l1567_156762

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 ∧ 
  (∀ n ∈ nums, n < 100) ∧ 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
sorry

end subset_sum_exists_l1567_156762


namespace james_class_size_l1567_156715

theorem james_class_size (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n = 4 * k - 1) ∧
  (∃ k : ℕ, n = 5 * k - 2) ∧
  (∃ k : ℕ, n = 6 * k - 3) →
  n = 123 ∨ n = 183 := by
sorry

end james_class_size_l1567_156715


namespace parabola_directrix_l1567_156706

/-- Given a parabola y = -3x^2 + 6x - 5, prove that its directrix is y = -23/12 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 6 * x - 5
  ∃ k : ℝ, k = -23/12 ∧ ∀ x y : ℝ, f x = y →
    ∃ h : ℝ, h > 0 ∧ (x - 1)^2 + (y + 2 - k)^2 = (y + 2 - (k + h))^2 :=
by sorry

end parabola_directrix_l1567_156706


namespace translation_office_staff_count_l1567_156796

/-- The number of people working at a translation office -/
def translation_office_staff : ℕ :=
  let english_only : ℕ := 8
  let german_only : ℕ := 8
  let russian_only : ℕ := 8
  let english_german : ℕ := 1
  let german_russian : ℕ := 2
  let english_russian : ℕ := 3
  let all_three : ℕ := 1
  english_only + german_only + russian_only + english_german + german_russian + english_russian + all_three

/-- Theorem stating the number of people working at the translation office -/
theorem translation_office_staff_count : translation_office_staff = 31 := by
  sorry

end translation_office_staff_count_l1567_156796


namespace f_derivative_at_2_l1567_156727

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 7) * (x - 8) * (x - 9) * (x - 10)

theorem f_derivative_at_2 : 
  deriv f 2 = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 := by
  sorry

end f_derivative_at_2_l1567_156727


namespace train_length_calculation_train_B_length_l1567_156720

/-- The length of a train given the length of another train, their speeds, and the time they take to cross each other when moving in opposite directions. -/
theorem train_length_calculation (length_A : ℝ) (speed_A speed_B : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_A_ms := speed_A * 1000 / 3600
  let speed_B_ms := speed_B * 1000 / 3600
  let relative_speed := speed_A_ms + speed_B_ms
  let total_distance := relative_speed * crossing_time
  total_distance - length_A

/-- The length of Train B is approximately 299.95 meters. -/
theorem train_B_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_length_calculation 200 120 80 9 - 299.95| < ε :=
sorry

end train_length_calculation_train_B_length_l1567_156720


namespace minimum_buses_needed_minimum_buses_for_field_trip_l1567_156703

theorem minimum_buses_needed 
  (total_students : ℕ) 
  (regular_capacity : ℕ) 
  (reduced_capacity : ℕ) 
  (reduced_buses : ℕ) : ℕ :=
  let remaining_students := total_students - (reduced_capacity * reduced_buses)
  let regular_buses_needed := (remaining_students + regular_capacity - 1) / regular_capacity
  regular_buses_needed + reduced_buses

theorem minimum_buses_for_field_trip : 
  minimum_buses_needed 1234 45 30 3 = 29 := by
  sorry

end minimum_buses_needed_minimum_buses_for_field_trip_l1567_156703


namespace reptile_house_count_l1567_156733

/-- The number of animals in the Rain Forest exhibit -/
def rain_forest_animals : ℕ := 7

/-- The number of animals in the Reptile House -/
def reptile_house_animals : ℕ := 3 * rain_forest_animals - 5

theorem reptile_house_count : reptile_house_animals = 16 := by
  sorry

end reptile_house_count_l1567_156733


namespace quadratic_equation_solution_l1567_156785

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 3 ∧ x₂ = -2) ∧ 
  ((2 * x₁ - 1)^2 - 25 = 0) ∧ 
  ((2 * x₂ - 1)^2 - 25 = 0) := by
  sorry

end quadratic_equation_solution_l1567_156785


namespace k_value_proof_l1567_156714

theorem k_value_proof (k : ℤ) 
  (h1 : (0.0004040404 : ℝ) * (10 : ℝ) ^ (k : ℝ) > 1000000)
  (h2 : (0.0004040404 : ℝ) * (10 : ℝ) ^ (k : ℝ) < 10000000) : 
  k = 11 := by
  sorry

end k_value_proof_l1567_156714


namespace rectangle_area_increase_l1567_156755

theorem rectangle_area_increase (length : ℝ) (breadth : ℝ) 
  (h1 : length = 40)
  (h2 : breadth = 20)
  (h3 : length = 2 * breadth) :
  (length - 5) * (breadth + 5) - length * breadth = 75 := by
  sorry

end rectangle_area_increase_l1567_156755


namespace moving_circle_trajectory_l1567_156722

/-- Fixed circle C -/
def C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

/-- Fixed line L -/
def L (x : ℝ) : Prop := x = 1

/-- Moving circle P -/
def P (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

/-- P is externally tangent to C -/
def externally_tangent (x y r : ℝ) : Prop :=
  (x + 2 - 1)^2 + y^2 = (r + 1)^2

/-- P is tangent to L -/
def tangent_to_L (x y r : ℝ) : Prop := x - r = 1

/-- Trajectory of the center of P -/
def trajectory (x y : ℝ) : Prop := y^2 = -8*x

theorem moving_circle_trajectory :
  ∀ x y r : ℝ,
  C x y ∧ L 1 ∧ P x y r ∧ externally_tangent x y r ∧ tangent_to_L x y r →
  trajectory x y :=
sorry

end moving_circle_trajectory_l1567_156722


namespace earlier_usage_time_correct_l1567_156782

/-- Represents a beer barrel with two taps -/
structure BeerBarrel where
  capacity : ℕ
  midwayTapRate : ℕ  -- minutes per litre
  bottomTapRate : ℕ  -- minutes per litre

/-- Calculates how much earlier the lower tap was used than usual -/
def earlierUsageTime (barrel : BeerBarrel) (usageTime : ℕ) : ℕ :=
  let drawnAmount := usageTime / barrel.bottomTapRate
  let midwayAmount := barrel.capacity / 2
  let remainingAmount := barrel.capacity - drawnAmount
  let excessAmount := remainingAmount - midwayAmount
  excessAmount * barrel.midwayTapRate

theorem earlier_usage_time_correct (barrel : BeerBarrel) (usageTime : ℕ) :
  barrel.capacity = 36 ∧ 
  barrel.midwayTapRate = 6 ∧ 
  barrel.bottomTapRate = 4 ∧ 
  usageTime = 16 →
  earlierUsageTime barrel usageTime = 84 := by
  sorry

#eval earlierUsageTime ⟨36, 6, 4⟩ 16

end earlier_usage_time_correct_l1567_156782


namespace bridge_length_l1567_156760

/-- The length of a bridge given train characteristics and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time_s : Real) :
  train_length = 130 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time_s = 30 →
  ∃ (bridge_length : Real),
    bridge_length = 245 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time_s :=
by sorry

end bridge_length_l1567_156760


namespace candy_packing_problem_l1567_156789

theorem candy_packing_problem (n : ℕ) : 
  n % 10 = 6 ∧ 
  n % 15 = 11 ∧ 
  200 ≤ n ∧ n ≤ 250 → 
  n = 206 ∨ n = 236 := by
sorry

end candy_packing_problem_l1567_156789


namespace minimize_quadratic_l1567_156768

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The theorem states that x = 3 minimizes the quadratic function f(x) = 3x^2 - 18x + 7 -/
theorem minimize_quadratic :
  ∃ (x_min : ℝ), x_min = 3 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
sorry

end minimize_quadratic_l1567_156768


namespace moment_of_inertia_unit_mass_moment_of_inertia_arbitrary_mass_l1567_156746

/-- The moment of inertia of a system of points -/
noncomputable def moment_of_inertia {n : ℕ} (a : Fin n → Fin n → ℝ) (m : Fin n → ℝ) : ℝ :=
  let total_mass := (Finset.univ.sum m)
  (1 / total_mass) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => m i * m j * (a i j)^2)))

/-- Theorem: Moment of inertia for unit masses -/
theorem moment_of_inertia_unit_mass {n : ℕ} (a : Fin n → Fin n → ℝ) :
  moment_of_inertia a (λ _ => 1) = 
  (1 / n) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => (a i j)^2))) :=
sorry

/-- Theorem: Moment of inertia for arbitrary masses -/
theorem moment_of_inertia_arbitrary_mass {n : ℕ} (a : Fin n → Fin n → ℝ) (m : Fin n → ℝ) :
  moment_of_inertia a m = 
  (1 / (Finset.univ.sum m)) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => m i * m j * (a i j)^2))) :=
sorry

end moment_of_inertia_unit_mass_moment_of_inertia_arbitrary_mass_l1567_156746


namespace polynomial_remainder_l1567_156778

/-- The polynomial p(x) = x^3 - 2x^2 + x + 1 -/
def p (x : ℝ) : ℝ := x^3 - 2*x^2 + x + 1

/-- The remainder when p(x) is divided by (x-4) -/
def remainder : ℝ := p 4

theorem polynomial_remainder : remainder = 37 := by
  sorry

end polynomial_remainder_l1567_156778


namespace picnic_men_count_l1567_156736

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  total : ℕ
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Conditions for the picnic attendance -/
def picnicConditions (p : PicnicAttendance) : Prop :=
  p.total = 200 ∧
  p.men = p.women + 20 ∧
  p.adults = p.children + 20 ∧
  p.adults = p.men + p.women ∧
  p.total = p.men + p.women + p.children

/-- Theorem: Given the conditions, the number of men at the picnic is 65 -/
theorem picnic_men_count (p : PicnicAttendance) :
  picnicConditions p → p.men = 65 := by
  sorry

end picnic_men_count_l1567_156736


namespace no_real_solutions_l1567_156787

theorem no_real_solutions : ∀ x : ℝ, (x^2000 / 2001 + 2 * Real.sqrt 3 * x^2 - 2 * Real.sqrt 5 * x + Real.sqrt 3) ≠ 0 := by
  sorry

end no_real_solutions_l1567_156787


namespace hyperbola_equation_l1567_156758

/-- Represents a hyperbola with a given asymptote and a point it passes through -/
structure Hyperbola where
  asymptote_slope : ℝ
  point : ℝ × ℝ

/-- The standard form of a hyperbola equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard equation of a hyperbola given its asymptote and a point -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 1/2)
    (h_point : h.point = (2 * Real.sqrt 2, 1)) :
    standard_equation 4 1 (h.point.1) (h.point.2) :=
  sorry

end hyperbola_equation_l1567_156758


namespace parabola_intersection_theorem_l1567_156735

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus with slope k
def line (k x y : ℝ) : Prop := y = k*(x - 2)

-- Define the intersection points
def intersection (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ line k p.1 p.2}

-- Define the condition AF = 2FB
def point_condition (A B : ℝ × ℝ) : Prop :=
  (4 - A.1, -A.2) = (2*(B.1 - 4), 2*B.2)

-- Theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  ∃ (A B : ℝ × ℝ), A ∈ intersection k ∧ B ∈ intersection k ∧
  point_condition A B → |k| = 2*Real.sqrt 2 :=
sorry

end parabola_intersection_theorem_l1567_156735


namespace f_monotone_increasing_l1567_156783

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f is monotonically increasing on ℝ
theorem f_monotone_increasing : 
  ∀ (x y : ℝ), x < y → f x < f y := by
  sorry

end f_monotone_increasing_l1567_156783


namespace ramsey_r33_l1567_156759

-- Define a type for the colors of the edges
inductive Color
| Red
| Blue

-- Define the graph type
def Graph := Fin 6 → Fin 6 → Color

-- Define what it means for three vertices to form a monochromatic triangle
def IsMonochromaticTriangle (g : Graph) (v1 v2 v3 : Fin 6) : Prop :=
  g v1 v2 = g v2 v3 ∧ g v2 v3 = g v3 v1

-- State the theorem
theorem ramsey_r33 (g : Graph) :
  (∀ (v1 v2 : Fin 6), v1 ≠ v2 → g v1 v2 = g v2 v1) →  -- Symmetry condition
  (∃ (v1 v2 v3 : Fin 6), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ IsMonochromaticTriangle g v1 v2 v3) :=
by
  sorry

end ramsey_r33_l1567_156759


namespace six_number_list_product_l1567_156737

theorem six_number_list_product (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_order : a₁ ≤ a₂ ∧ a₂ ≤ a₃ ∧ a₃ ≤ a₄ ∧ a₄ ≤ a₅ ∧ a₅ ≤ a₆)
  (h_remove_largest : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 - 1)
  (h_remove_smallest : (a₂ + a₃ + a₄ + a₅ + a₆) / 5 = (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 + 1)
  (h_remove_both : (a₂ + a₃ + a₄ + a₅) / 4 = 20) :
  a₁ * a₆ = 375 := by
sorry

end six_number_list_product_l1567_156737


namespace min_value_expression_l1567_156763

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 3) :
  a^2 + 4*a*b + 8*b^2 + 10*b*c + 3*c^2 ≥ 27 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 3 ∧
    a₀^2 + 4*a₀*b₀ + 8*b₀^2 + 10*b₀*c₀ + 3*c₀^2 = 27 :=
by sorry

end min_value_expression_l1567_156763


namespace blue_socks_count_l1567_156797

/-- Represents the number of pairs of socks Luis bought -/
structure SockPurchase where
  red : ℕ
  blue : ℕ

/-- Represents the cost of socks in dollars -/
structure SockCost where
  red : ℕ
  blue : ℕ

/-- Calculates the total cost of the sock purchase -/
def totalCost (purchase : SockPurchase) (cost : SockCost) : ℕ :=
  purchase.red * cost.red + purchase.blue * cost.blue

theorem blue_socks_count (purchase : SockPurchase) (cost : SockCost) :
  purchase.red = 4 →
  cost.red = 3 →
  cost.blue = 5 →
  totalCost purchase cost = 42 →
  purchase.blue = 6 := by
  sorry

end blue_socks_count_l1567_156797


namespace even_function_m_value_l1567_156752

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Given f(x) = x^2 + (m+2)x + 3 is an even function, prove that m = -2 -/
theorem even_function_m_value (m : ℝ) :
  IsEven (fun x => x^2 + (m+2)*x + 3) → m = -2 := by
  sorry

end even_function_m_value_l1567_156752


namespace N_properties_l1567_156765

def N : ℕ := 2^2022 + 1

theorem N_properties :
  (∃ k : ℕ, N = 65 * k) ∧
  (∃ a b c d : ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ N = a * b * c * d) := by
  sorry

end N_properties_l1567_156765


namespace fraction_zero_implies_x_negative_one_l1567_156705

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x + 1) / (x - 1) = 0 ∧ x ≠ 1 → x = -1 := by
  sorry

end fraction_zero_implies_x_negative_one_l1567_156705


namespace equilateral_triangle_perimeter_l1567_156743

theorem equilateral_triangle_perimeter (s : ℝ) (h_positive : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l1567_156743


namespace pizza_pepperoni_ratio_l1567_156793

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_pepperoni : ℕ)

/-- Represents a slice of pizza -/
structure PizzaSlice :=
  (pepperoni : ℕ)

def cut_pizza (p : Pizza) (slice1_pepperoni : ℕ) : PizzaSlice × PizzaSlice :=
  let slice1 := PizzaSlice.mk slice1_pepperoni
  let slice2 := PizzaSlice.mk (p.total_pepperoni - slice1_pepperoni)
  (slice1, slice2)

def pepperoni_ratio (slice1 : PizzaSlice) (slice2 : PizzaSlice) : ℚ :=
  slice1.pepperoni / slice2.pepperoni

theorem pizza_pepperoni_ratio :
  let original_pizza := Pizza.mk 40
  let (jellys_slice, other_slice) := cut_pizza original_pizza 10
  let jellys_slice_after_loss := PizzaSlice.mk (jellys_slice.pepperoni - 1)
  pepperoni_ratio jellys_slice_after_loss other_slice = 3 / 10 := by
  sorry

end pizza_pepperoni_ratio_l1567_156793


namespace min_distance_to_curve_l1567_156731

theorem min_distance_to_curve :
  let f (x y : ℝ) := Real.sqrt (x^2 + y^2)
  let g (x y : ℝ) := 6*x + 8*y - 4*x^2
  ∃ (min : ℝ), min = Real.sqrt 2061 / 8 ∧
    (∀ x y : ℝ, g x y = 48 → f x y ≥ min) ∧
    (∃ x y : ℝ, g x y = 48 ∧ f x y = min) :=
by sorry

end min_distance_to_curve_l1567_156731


namespace base_b_is_ten_l1567_156702

/-- Given that 1304 in base b, when squared, equals 99225 in base b, prove that b = 10 -/
theorem base_b_is_ten (b : ℕ) (h : b > 1) : 
  (b^3 + 3*b^2 + 4)^2 = 9*b^4 + 9*b^3 + 2*b^2 + 2*b + 5 → b = 10 := by
  sorry

#check base_b_is_ten

end base_b_is_ten_l1567_156702


namespace blue_cross_coverage_l1567_156718

/-- Represents a circular flag with overlapping crosses -/
structure CircularFlag where
  /-- The total area of the flag -/
  total_area : ℝ
  /-- The area covered by the blue cross -/
  blue_cross_area : ℝ
  /-- The area covered by the red cross -/
  red_cross_area : ℝ
  /-- The area covered by both crosses combined -/
  combined_crosses_area : ℝ
  /-- The red cross is half the width of the blue cross -/
  red_half_blue : blue_cross_area = 2 * red_cross_area
  /-- The combined area of both crosses is 50% of the flag's area -/
  combined_half_total : combined_crosses_area = 0.5 * total_area
  /-- The red cross covers 20% of the flag's area -/
  red_fifth_total : red_cross_area = 0.2 * total_area

/-- Theorem stating that the blue cross alone covers 30% of the flag's area -/
theorem blue_cross_coverage (flag : CircularFlag) : 
  flag.blue_cross_area = 0.3 * flag.total_area := by
  sorry

end blue_cross_coverage_l1567_156718


namespace barbed_wire_height_l1567_156734

theorem barbed_wire_height (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) :
  area = 3136 →
  cost_per_meter = 1 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 666 →
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let wire_length := perimeter - (↑num_gates * gate_width)
  let wire_cost := wire_length * cost_per_meter
  let height := (total_cost - wire_cost) / wire_length
  height = 2 := by sorry

end barbed_wire_height_l1567_156734


namespace x_value_in_set_l1567_156756

theorem x_value_in_set (x : ℝ) : -2 ∈ ({3, 5, x, x^2 + 3*x} : Set ℝ) → x = -1 := by
  sorry

end x_value_in_set_l1567_156756


namespace triangle_problem_l1567_156742

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_a : a = Real.sqrt 5)
  (h_b : b = 3)
  (h_sin_C : Real.sin C = 2 * Real.sin A) : 
  c = 2 * Real.sqrt 5 ∧ Real.sin (2 * A - Real.pi / 4) = Real.sqrt 2 / 10 := by
  sorry

end triangle_problem_l1567_156742


namespace spending_problem_solution_l1567_156767

def spending_problem (initial_money : ℝ) : Prop :=
  let remaining_after_first := initial_money - (initial_money / 2 + 200)
  let spent_at_second := remaining_after_first / 2 + 300
  initial_money - (initial_money / 2 + 200) - spent_at_second = 350

theorem spending_problem_solution :
  spending_problem 3000 := by sorry

end spending_problem_solution_l1567_156767


namespace remove_layer_from_10x10x10_cube_l1567_156741

/-- Represents a cube made of smaller cubes -/
structure Cube where
  side_length : ℕ
  total_cubes : ℕ

/-- Calculates the number of remaining cubes after removing one layer -/
def remaining_cubes (c : Cube) : ℕ :=
  c.total_cubes - (c.side_length * c.side_length)

/-- Theorem: For a 10x10x10 cube, removing one layer leaves 900 cubes -/
theorem remove_layer_from_10x10x10_cube :
  let c : Cube := { side_length := 10, total_cubes := 1000 }
  remaining_cubes c = 900 := by
  sorry

end remove_layer_from_10x10x10_cube_l1567_156741


namespace circle_and_tangents_l1567_156761

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line tangent to the circle
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 4 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 3)

-- Define the two possible tangent lines through P
def tangent_line1 (x : ℝ) : Prop := x = 2
def tangent_line2 (x y : ℝ) : Prop := 5 * x - 12 * y + 26 = 0

theorem circle_and_tangents :
  -- The circle is tangent to the given line
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line x y) ∧
  -- The circle passes through only one point of the line
  (∀ (x y : ℝ), circle_equation x y → tangent_line x y → 
    ∀ (x' y' : ℝ), x' ≠ x ∨ y' ≠ y → circle_equation x' y' → ¬tangent_line x' y') ∧
  -- The two tangent lines pass through P and are tangent to the circle
  (tangent_line1 point_P.1 ∨ tangent_line2 point_P.1 point_P.2) ∧
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line1 x) ∧
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line2 x y) ∧
  -- There are no other tangent lines through P
  (∀ (f : ℝ → ℝ), f point_P.1 = point_P.2 → 
    (∃ (x y : ℝ), circle_equation x y ∧ y = f x) →
    (∀ x, f x = point_P.2 + (x - point_P.1) * 5 / 12 ∨ f x = point_P.2)) :=
sorry

end circle_and_tangents_l1567_156761


namespace function_value_theorem_l1567_156771

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2 where f(-5) = m, prove that f(5) = -m + 4 -/
theorem function_value_theorem (a b c m : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^5 + c * x^3 + 2
  f (-5) = m → f 5 = -m + 4 := by
sorry

end function_value_theorem_l1567_156771


namespace handshakes_at_reunion_l1567_156751

/-- Represents a family reunion with married couples -/
structure FamilyReunion where
  couples : ℕ
  people_per_couple : ℕ := 2

/-- Calculates the total number of handshakes at a family reunion -/
def total_handshakes (reunion : FamilyReunion) : ℕ :=
  let total_people := reunion.couples * reunion.people_per_couple
  let handshakes_per_person := total_people - 1 - 1 - (3 * reunion.people_per_couple)
  (total_people * handshakes_per_person) / 2

/-- Theorem: The total number of handshakes at a specific family reunion is 64 -/
theorem handshakes_at_reunion :
  let reunion : FamilyReunion := { couples := 8 }
  total_handshakes reunion = 64 := by
  sorry

end handshakes_at_reunion_l1567_156751


namespace ordering_of_numbers_l1567_156712

theorem ordering_of_numbers (a b : ℝ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hab : a + b < 0) : 
  b < -a ∧ -a < 0 ∧ 0 < a ∧ a < -b :=
sorry

end ordering_of_numbers_l1567_156712


namespace smallest_fraction_greater_than_three_fourths_l1567_156747

theorem smallest_fraction_greater_than_three_fourths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 3 / 4 →
    (73 : ℚ) / 97 ≤ (a : ℚ) / b :=
by sorry

end smallest_fraction_greater_than_three_fourths_l1567_156747


namespace fraction_simplification_l1567_156794

theorem fraction_simplification (a b x : ℝ) 
  (h1 : x = b / a) 
  (h2 : a ≠ b) 
  (h3 : a ≠ 0) : 
  (2 * a + b) / (2 * a - b) = (2 + x) / (2 - x) := by
  sorry

end fraction_simplification_l1567_156794


namespace commute_time_difference_commute_time_difference_is_two_l1567_156709

/-- The difference in commute time between walking and taking the train -/
theorem commute_time_difference : ℝ :=
  let distance : ℝ := 1.5  -- miles
  let walking_speed : ℝ := 3  -- mph
  let train_speed : ℝ := 20  -- mph
  let additional_train_time : ℝ := 23.5  -- minutes

  let walking_time : ℝ := distance / walking_speed * 60  -- minutes
  let train_travel_time : ℝ := distance / train_speed * 60  -- minutes
  let total_train_time : ℝ := train_travel_time + additional_train_time

  walking_time - total_train_time

/-- The commute time difference is 2 minutes -/
theorem commute_time_difference_is_two : commute_time_difference = 2 := by
  sorry

end commute_time_difference_commute_time_difference_is_two_l1567_156709


namespace geometric_sequence_sum_l1567_156764

/-- A positive geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r > 0, ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 3 →
  a 3 + a 4 = 12 →
  a 4 + a 5 = 24 := by
sorry

end geometric_sequence_sum_l1567_156764


namespace complex_product_equality_l1567_156792

theorem complex_product_equality (x : ℂ) (h : x = Complex.exp (Complex.I * π / 9)) : 
  (2*x + x^3) * (2*x^3 + x^9) * (2*x^6 + x^18) * (2*x^9 + x^27) * (2*x^12 + x^36) * (2*x^15 + x^45) = 549 := by
  sorry

end complex_product_equality_l1567_156792


namespace fence_painting_problem_l1567_156790

/-- Given a fence of 360 square feet to be painted by three people in the ratio 3:5:2,
    prove that the person with the smallest share paints 72 square feet. -/
theorem fence_painting_problem (total_area : ℝ) (ratio_a ratio_b ratio_c : ℕ) :
  total_area = 360 →
  ratio_a = 3 →
  ratio_b = 5 →
  ratio_c = 2 →
  (ratio_a + ratio_b + ratio_c : ℝ) * (total_area / (ratio_a + ratio_b + ratio_c : ℝ) * ratio_c) = 72 :=
by sorry

end fence_painting_problem_l1567_156790


namespace shaded_area_between_triangles_l1567_156791

/-- The area of the shaded region between two back-to-back isosceles triangles -/
theorem shaded_area_between_triangles (b h x₀ : ℝ) :
  b > 0 → h > 0 →
  let x₁ := x₀ - b / 2
  let x₂ := x₀ + b / 2
  let y := h
  (x₂ - x₁) * y = 280 :=
by
  sorry

#check shaded_area_between_triangles 12 10 10

end shaded_area_between_triangles_l1567_156791


namespace prism_dimension_is_five_l1567_156716

/-- Represents a rectangular prism with dimensions n × n × 2n -/
structure RectangularPrism (n : ℕ) where
  length : ℕ := n
  width : ℕ := n
  height : ℕ := 2 * n

/-- The number of unit cubes obtained by cutting the prism -/
def num_unit_cubes (n : ℕ) : ℕ := 2 * n^3

/-- The total number of faces of all unit cubes -/
def total_faces (n : ℕ) : ℕ := 6 * num_unit_cubes n

/-- The number of blue faces (painted faces of the original prism) -/
def blue_faces (n : ℕ) : ℕ := 2 * n^2 + 4 * (2 * n^2)

/-- Theorem stating that if one-sixth of the total faces are blue, then n = 5 -/
theorem prism_dimension_is_five (n : ℕ) :
  (blue_faces n : ℚ) / (total_faces n : ℚ) = 1 / 6 → n = 5 := by
  sorry

end prism_dimension_is_five_l1567_156716


namespace set_intersection_equals_greater_equal_one_l1567_156795

-- Define the sets S and T
def S : Set ℝ := {x | x^2 - x ≥ 0}
def T : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem set_intersection_equals_greater_equal_one :
  S ∩ T = {x : ℝ | x ≥ 1} := by sorry

end set_intersection_equals_greater_equal_one_l1567_156795


namespace final_passengers_count_l1567_156776

/-- The number of people on the bus after all stops -/
def final_passengers : ℕ :=
  let initial := 110
  let stop1 := initial - 20 + 15
  let stop2 := stop1 - 34 + 17
  let stop3 := stop2 - 18 + 7
  let stop4 := stop3 - 29 + 19
  let stop5 := stop4 - 11 + 13
  let stop6 := stop5 - 15 + 8
  let stop7 := stop6 - 13 + 5
  let stop8 := stop7 - 6 + 0
  stop8

/-- Theorem stating that the final number of passengers is 48 -/
theorem final_passengers_count : final_passengers = 48 := by
  sorry

end final_passengers_count_l1567_156776


namespace average_with_added_number_l1567_156798

theorem average_with_added_number (x : ℝ) : 
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 := by sorry

end average_with_added_number_l1567_156798


namespace course_selection_schemes_l1567_156700

theorem course_selection_schemes (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 2) :
  (Nat.choose (n - k) m) + (k * Nat.choose (n - k) (m - 1)) = 36 := by
  sorry

end course_selection_schemes_l1567_156700


namespace cube_root_negative_a_l1567_156704

theorem cube_root_negative_a (a : ℝ) : 
  ((-a : ℝ) ^ (1/3 : ℝ) = Real.sqrt 2) → (a ^ (1/3 : ℝ) = -Real.sqrt 2) := by
  sorry

end cube_root_negative_a_l1567_156704


namespace right_triangle_and_multiplicative_inverse_l1567_156772

theorem right_triangle_and_multiplicative_inverse :
  (35^2 + 312^2 = 313^2) ∧ 
  (520 * 2026 % 4231 = 1) := by
sorry

end right_triangle_and_multiplicative_inverse_l1567_156772


namespace base_conversion_theorem_l1567_156710

theorem base_conversion_theorem : 
  ∃! n : ℕ, ∃ S : Finset ℕ, 
    (∀ c ∈ S, c ≥ 2 ∧ c^3 ≤ 250 ∧ 250 < c^4) ∧ 
    Finset.card S = n ∧ 
    n = 3 := by
sorry

end base_conversion_theorem_l1567_156710


namespace lcm_of_180_and_504_l1567_156730

theorem lcm_of_180_and_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end lcm_of_180_and_504_l1567_156730


namespace right_triangle_acute_angles_l1567_156775

-- Define a right triangle with two acute angles
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  is_right_triangle : angle1 + angle2 = 90

-- Define the condition that the ratio of the two acute angles is 3:1
def angle_ratio (t : RightTriangle) : Prop :=
  t.angle1 / t.angle2 = 3

-- Theorem statement
theorem right_triangle_acute_angles 
  (t : RightTriangle) 
  (h : angle_ratio t) : 
  (t.angle1 = 67.5 ∧ t.angle2 = 22.5) ∨ (t.angle1 = 22.5 ∧ t.angle2 = 67.5) :=
sorry

end right_triangle_acute_angles_l1567_156775


namespace larger_divided_by_smaller_l1567_156766

theorem larger_divided_by_smaller (L S : ℕ) (h1 : L - S = 2395) (h2 : S = 476) (h3 : L % S = 15) :
  L / S = 6 := by
  sorry

end larger_divided_by_smaller_l1567_156766
