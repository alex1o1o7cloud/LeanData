import Mathlib

namespace cyclic_difference_sum_lower_bound_l3495_349504

theorem cyclic_difference_sum_lower_bound 
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
                a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
                a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
                a₆ ≠ a₇) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + 
  (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 := by
  sorry


end cyclic_difference_sum_lower_bound_l3495_349504


namespace x_range_and_max_y_over_x_l3495_349538

/-- Circle C with center (4,3) and radius 3 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + (p.2 - 3)^2 = 9}

/-- A point P on circle C -/
def P : ℝ × ℝ := sorry

/-- P is on circle C -/
axiom hP : P ∈ C

theorem x_range_and_max_y_over_x :
  (1 ≤ P.1 ∧ P.1 ≤ 7) ∧
  ∀ Q ∈ C, Q.2 / Q.1 ≤ 24 / 7 := by sorry

end x_range_and_max_y_over_x_l3495_349538


namespace participation_plans_l3495_349599

/-- The number of students -/
def total_students : ℕ := 4

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The number of subjects -/
def subjects : ℕ := 3

/-- The number of students that can be freely selected -/
def free_selection : ℕ := total_students - 1

theorem participation_plans :
  (Nat.choose free_selection (selected_students - 1)) * (Nat.factorial subjects) = 18 := by
  sorry

end participation_plans_l3495_349599


namespace half_coverage_days_l3495_349577

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 58

/-- Represents the daily growth factor of the lily pad patch -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days to cover half the lake is one less than the full coverage days -/
theorem half_coverage_days : 
  ∃ (days : ℕ), days = full_coverage_days - 1 ∧ 
  (daily_growth_factor : ℚ) * ((1 : ℚ) / 2) = 1 := by
  sorry

end half_coverage_days_l3495_349577


namespace abs_neg_2023_l3495_349560

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_neg_2023_l3495_349560


namespace toothpick_15th_stage_l3495_349549

def toothpick_sequence (n : ℕ) : ℕ :=
  5 + 3 * (n - 1)

theorem toothpick_15th_stage :
  toothpick_sequence 15 = 47 := by
  sorry

end toothpick_15th_stage_l3495_349549


namespace partial_fraction_sum_zero_l3495_349590

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_zero_l3495_349590


namespace sum_of_reciprocal_relations_l3495_349583

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = -5) : 
  x + y = -16/9 := by
  sorry

end sum_of_reciprocal_relations_l3495_349583


namespace modulus_of_complex_fraction_l3495_349578

theorem modulus_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.abs (2 * i / (i - 1)) = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l3495_349578


namespace largest_gcd_of_sum_1001_l3495_349554

theorem largest_gcd_of_sum_1001 :
  ∃ (a b : ℕ+), a + b = 1001 ∧
  ∀ (c d : ℕ+), c + d = 1001 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 143 :=
sorry

end largest_gcd_of_sum_1001_l3495_349554


namespace min_transportation_fee_l3495_349559

/-- Represents the transportation problem with given parameters -/
structure TransportProblem where
  total_goods : ℕ
  large_truck_capacity : ℕ
  large_truck_cost : ℕ
  small_truck_capacity : ℕ
  small_truck_cost : ℕ

/-- Calculates the transportation cost for a given number of large and small trucks -/
def transportation_cost (p : TransportProblem) (large_trucks : ℕ) (small_trucks : ℕ) : ℕ :=
  large_trucks * p.large_truck_cost + small_trucks * p.small_truck_cost

/-- Checks if a combination of trucks can transport all goods -/
def can_transport_all (p : TransportProblem) (large_trucks : ℕ) (small_trucks : ℕ) : Prop :=
  large_trucks * p.large_truck_capacity + small_trucks * p.small_truck_capacity ≥ p.total_goods

/-- Theorem stating that the minimum transportation fee is 1800 yuan -/
theorem min_transportation_fee (p : TransportProblem) 
    (h1 : p.total_goods = 20)
    (h2 : p.large_truck_capacity = 7)
    (h3 : p.large_truck_cost = 600)
    (h4 : p.small_truck_capacity = 4)
    (h5 : p.small_truck_cost = 400) :
    (∀ large_trucks small_trucks, can_transport_all p large_trucks small_trucks →
      transportation_cost p large_trucks small_trucks ≥ 1800) ∧
    (∃ large_trucks small_trucks, can_transport_all p large_trucks small_trucks ∧
      transportation_cost p large_trucks small_trucks = 1800) :=
  sorry


end min_transportation_fee_l3495_349559


namespace unique_integer_square_less_than_double_l3495_349511

theorem unique_integer_square_less_than_double :
  ∃! x : ℤ, x^2 < 2*x :=
by
  -- Proof goes here
  sorry

end unique_integer_square_less_than_double_l3495_349511


namespace colored_cube_covers_plane_l3495_349589

/-- A cube with colored middle squares on each face -/
structure ColoredCube where
  a : ℕ
  b : ℕ
  c : ℕ

/-- An infinite plane with unit squares -/
def Plane := ℕ × ℕ

/-- A point on the plane is colorable if the cube can land on it with its colored face -/
def isColorable (cube : ColoredCube) (point : Plane) : Prop := sorry

/-- The main theorem: If any two sides of the cube are relatively prime, 
    then every point on the plane is colorable -/
theorem colored_cube_covers_plane (cube : ColoredCube) :
  (Nat.gcd (2 * cube.a + 1) (2 * cube.b + 1) = 1 ∨
   Nat.gcd (2 * cube.b + 1) (2 * cube.c + 1) = 1 ∨
   Nat.gcd (2 * cube.a + 1) (2 * cube.c + 1) = 1) →
  ∀ (point : Plane), isColorable cube point := by
  sorry

end colored_cube_covers_plane_l3495_349589


namespace boys_count_proof_l3495_349568

/-- Given a total number of eyes and the number of eyes per boy, 
    calculate the number of boys. -/
def number_of_boys (total_eyes : ℕ) (eyes_per_boy : ℕ) : ℕ :=
  total_eyes / eyes_per_boy

theorem boys_count_proof (total_eyes : ℕ) (eyes_per_boy : ℕ) 
  (h1 : total_eyes = 46) (h2 : eyes_per_boy = 2) : 
  number_of_boys total_eyes eyes_per_boy = 23 := by
  sorry

#eval number_of_boys 46 2

end boys_count_proof_l3495_349568


namespace m_range_l3495_349593

theorem m_range : ∃ m : ℝ, m = Real.sqrt 5 - 1 ∧ 1 < m ∧ m < 2 := by
  sorry

end m_range_l3495_349593


namespace sin_product_identity_l3495_349500

theorem sin_product_identity (α β : ℝ) :
  Real.sin α * Real.sin β = (Real.sin ((α + β) / 2))^2 - (Real.sin ((α - β) / 2))^2 := by
  sorry

end sin_product_identity_l3495_349500


namespace sarka_age_l3495_349542

/-- Represents the ages of three sisters and their mother -/
structure FamilyAges where
  sarka : ℕ
  liba : ℕ
  eliska : ℕ
  mother : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.liba = ages.sarka + 3 ∧
  ages.eliska = ages.sarka + 8 ∧
  ages.mother = ages.sarka + 29 ∧
  (ages.sarka + ages.liba + ages.eliska + ages.mother) / 4 = 21

/-- The theorem stating Šárka's age -/
theorem sarka_age :
  ∃ (ages : FamilyAges), problem_conditions ages ∧ ages.sarka = 11 := by
  sorry

end sarka_age_l3495_349542


namespace park_area_is_3750_l3495_349598

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The perimeter of the park in meters -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- The area of the park in square meters -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.width

/-- The cost of fencing per meter in dollars -/
def fencingCostPerMeter : ℝ := 0.80

/-- The total cost of fencing the park in dollars -/
def totalFencingCost (park : RectangularPark) : ℝ :=
  perimeter park * fencingCostPerMeter

theorem park_area_is_3750 (park : RectangularPark) 
    (h : totalFencingCost park = 200) : area park = 3750 := by
  sorry

end park_area_is_3750_l3495_349598


namespace tan_alpha_half_implies_expression_equals_two_l3495_349591

theorem tan_alpha_half_implies_expression_equals_two (α : Real) 
  (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := by
  sorry

end tan_alpha_half_implies_expression_equals_two_l3495_349591


namespace common_factor_polynomial_l3495_349509

theorem common_factor_polynomial (a b c : ℤ) :
  ∃ (k : ℤ), (12 * a * b^3 * c + 8 * a^3 * b) = k * (4 * a * b) ∧ k ≠ 0 :=
sorry

end common_factor_polynomial_l3495_349509


namespace five_people_six_chairs_l3495_349539

/-- The number of ways to arrange n people in m chairs in a row -/
def arrangePeopleInChairs (n : ℕ) (m : ℕ) : ℕ :=
  (m - n + 1).factorial

theorem five_people_six_chairs :
  arrangePeopleInChairs 5 6 = 720 := by
  sorry

end five_people_six_chairs_l3495_349539


namespace total_pet_time_is_108_minutes_l3495_349582

-- Define the time spent on each activity
def dog_walk_play_time : ℚ := 1/2
def dog_feed_time : ℚ := 1/5
def cat_play_time : ℚ := 1/4
def cat_feed_time : ℚ := 1/10

-- Define the number of times each activity is performed daily
def dog_walk_play_frequency : ℕ := 2
def dog_feed_frequency : ℕ := 1
def cat_play_frequency : ℕ := 2
def cat_feed_frequency : ℕ := 1

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_pet_time_is_108_minutes :
  (dog_walk_play_time * dog_walk_play_frequency +
   dog_feed_time * dog_feed_frequency +
   cat_play_time * cat_play_frequency +
   cat_feed_time * cat_feed_frequency) * minutes_per_hour = 108 := by
  sorry

end total_pet_time_is_108_minutes_l3495_349582


namespace fixed_point_theorem_l3495_349525

/-- The parabola y^2 = 2px where p > 0 -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The moving line y = kx + b where k ≠ 0 and b ≠ 0 -/
def movingLine (k b x y : ℝ) : Prop := y = k*x + b ∧ k ≠ 0 ∧ b ≠ 0

/-- The slopes of OA and OB multiply to √3 -/
def slopeProduct (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = Real.sqrt 3

/-- The theorem stating that the line always passes through a fixed point -/
theorem fixed_point_theorem (p k b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ k₁ k₂,
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    movingLine k b x₁ y₁ ∧ movingLine k b x₂ y₂ ∧
    slopeProduct k₁ k₂) →
  movingLine k b (-2 * Real.sqrt 3 * p / 3) 0 :=
sorry

end fixed_point_theorem_l3495_349525


namespace triplet_sum_to_two_l3495_349569

theorem triplet_sum_to_two :
  -- Triplet A
  (1/4 : ℚ) + (1/4 : ℚ) + (3/2 : ℚ) = 2 ∧
  -- Triplet B
  (3 : ℤ) + (-1 : ℤ) + (0 : ℤ) = 2 ∧
  -- Triplet C
  (0.2 : ℝ) + (0.7 : ℝ) + (1.1 : ℝ) = 2 ∧
  -- Triplet D
  (2.2 : ℝ) + (-0.5 : ℝ) + (0.5 : ℝ) ≠ 2 ∧
  -- Triplet E
  (3/5 : ℚ) + (4/5 : ℚ) + (1/5 : ℚ) ≠ 2 := by
  sorry

end triplet_sum_to_two_l3495_349569


namespace impossibleEvent_l3495_349574

/-- A fair dice with faces numbered 1 to 6 -/
def Dice : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The event of getting a number divisible by 10 when rolling the dice -/
def DivisibleBy10 (n : ℕ) : Prop := n % 10 = 0

/-- Theorem: The event of rolling a number divisible by 10 is impossible -/
theorem impossibleEvent : ∀ n ∈ Dice, ¬ DivisibleBy10 n := by
  sorry

end impossibleEvent_l3495_349574


namespace points_cover_rectangles_l3495_349520

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  left : ℝ
  bottom : ℝ
  width : ℝ
  height : ℝ

/-- The unit square -/
def unitSquare : Rectangle := { left := 0, bottom := 0, width := 1, height := 1 }

/-- Check if a point is inside a rectangle -/
def isInside (p : Point) (r : Rectangle) : Prop :=
  r.left ≤ p.x ∧ p.x ≤ r.left + r.width ∧
  r.bottom ≤ p.y ∧ p.y ≤ r.bottom + r.height

/-- Check if a rectangle is inside another rectangle -/
def isContained (inner outer : Rectangle) : Prop :=
  outer.left ≤ inner.left ∧ inner.left + inner.width ≤ outer.left + outer.width ∧
  outer.bottom ≤ inner.bottom ∧ inner.bottom + inner.height ≤ outer.bottom + outer.height

/-- The main theorem -/
theorem points_cover_rectangles : ∃ (points : Finset Point),
  points.card ≤ 1600 ∧
  ∀ (r : Rectangle),
    isContained r unitSquare →
    r.width * r.height = 0.005 →
    ∃ (p : Point), p ∈ points ∧ isInside p r := by
  sorry

end points_cover_rectangles_l3495_349520


namespace tan_315_degrees_l3495_349524

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end tan_315_degrees_l3495_349524


namespace nonstudent_ticket_price_l3495_349563

theorem nonstudent_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (student_price : ℕ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 821)
  (h2 : total_revenue = 1933)
  (h3 : student_price = 2)
  (h4 : student_tickets = 530)
  (h5 : student_tickets < total_tickets) :
  let nonstudent_tickets : ℕ := total_tickets - student_tickets
  let nonstudent_price : ℕ := (total_revenue - student_price * student_tickets) / nonstudent_tickets
  nonstudent_price = 3 := by
sorry

end nonstudent_ticket_price_l3495_349563


namespace sum_of_reciprocals_l3495_349522

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 := by
sorry

end sum_of_reciprocals_l3495_349522


namespace tower_of_hanoi_correct_l3495_349550

/-- Minimum number of moves required to solve the Tower of Hanoi problem with n discs -/
def tower_of_hanoi (n : ℕ) : ℕ :=
  2^n - 1

/-- Theorem: The minimum number of moves for the Tower of Hanoi problem with n discs is 2^n - 1 -/
theorem tower_of_hanoi_correct (n : ℕ) : tower_of_hanoi n = 2^n - 1 := by
  sorry

end tower_of_hanoi_correct_l3495_349550


namespace monday_rain_inches_l3495_349541

/-- Proves that the number of inches of rain collected on Monday is 4 -/
theorem monday_rain_inches (
  gallons_per_inch : ℝ)
  (tuesday_rain : ℝ)
  (price_per_gallon : ℝ)
  (total_revenue : ℝ)
  (h1 : gallons_per_inch = 15)
  (h2 : tuesday_rain = 3)
  (h3 : price_per_gallon = 1.2)
  (h4 : total_revenue = 126)
  : ∃ (monday_rain : ℝ), monday_rain = 4 ∧
    gallons_per_inch * (monday_rain + tuesday_rain) * price_per_gallon = total_revenue :=
by
  sorry

end monday_rain_inches_l3495_349541


namespace increasing_function_unique_root_l3495_349588

/-- An increasing function on ℝ has exactly one root -/
theorem increasing_function_unique_root (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) :
  ∃! x, f x = 0 :=
sorry

end increasing_function_unique_root_l3495_349588


namespace simplify_expression_l3495_349534

theorem simplify_expression (x : ℝ) : 5*x + 9*x^2 + 8 - (6 - 5*x - 3*x^2) = 12*x^2 + 10*x + 2 := by
  sorry

end simplify_expression_l3495_349534


namespace min_tests_for_passing_probability_l3495_349546

theorem min_tests_for_passing_probability (p : ℝ) (threshold : ℝ) : 
  (p = 3/4) → (threshold = 0.99) → 
  (∀ k : ℕ, k < 4 → 1 - (1 - p)^k ≤ threshold) ∧ 
  (1 - (1 - p)^4 > threshold) := by
sorry

end min_tests_for_passing_probability_l3495_349546


namespace relationship_abc_l3495_349505

theorem relationship_abc :
  let a : ℝ := (0.9 : ℝ) ^ (0.3 : ℝ)
  let b : ℝ := (1.2 : ℝ) ^ (0.3 : ℝ)
  let c : ℝ := (0.5 : ℝ) ^ (-0.3 : ℝ)
  c > b ∧ b > a := by
  sorry

end relationship_abc_l3495_349505


namespace triangle_abc_proof_l3495_349508

theorem triangle_abc_proof (c : ℝ) (A C : ℝ) 
  (h_c : c = 10)
  (h_A : A = 45 * π / 180)
  (h_C : C = 30 * π / 180) :
  ∃ (a b B : ℝ),
    a = 10 * Real.sqrt 2 ∧
    b = 5 * (Real.sqrt 2 + Real.sqrt 6) ∧
    B = 105 * π / 180 := by
  sorry

end triangle_abc_proof_l3495_349508


namespace right_triangle_squares_area_l3495_349557

theorem right_triangle_squares_area (a b : ℝ) (ha : a = 3) (hb : b = 9) :
  let c := Real.sqrt (a^2 + b^2)
  a^2 + b^2 + c^2 + (1/2 * a * b) = 193.5 := by
  sorry

end right_triangle_squares_area_l3495_349557


namespace salesman_visits_l3495_349552

def S : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | n + 3 => S (n + 2) + S (n + 1) + S n

theorem salesman_visits (n : ℕ) : S 12 = 927 := by
  sorry

end salesman_visits_l3495_349552


namespace major_axis_coincide_condition_l3495_349570

/-- Represents the coefficients of a general ellipse equation -/
structure EllipseCoefficients where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if the major axis coincides with a conjugate diameter -/
def majorAxisCoincideWithConjugateDiameter (c : EllipseCoefficients) : Prop :=
  c.A * c.E - c.B * c.D = 0 ∧ 2 * c.B^2 + (c.A - c.C) * c.A = 0

/-- Theorem stating the conditions for the major axis to coincide with a conjugate diameter -/
theorem major_axis_coincide_condition (c : EllipseCoefficients) :
  majorAxisCoincideWithConjugateDiameter c ↔
  (c.A * c.E - c.B * c.D = 0 ∧ 2 * c.B^2 + (c.A - c.C) * c.A = 0) :=
by sorry

end major_axis_coincide_condition_l3495_349570


namespace paris_cafe_contribution_l3495_349597

/-- Represents the currency exchange problem in the Paris cafe --/
theorem paris_cafe_contribution
  (pastry_cost : ℝ)
  (emily_dollars : ℝ)
  (exchange_rate : ℝ)
  (h_pastry_cost : pastry_cost = 8)
  (h_emily_dollars : emily_dollars = 10)
  (h_exchange_rate : exchange_rate = 1.20)
  : ∃ (berengere_contribution : ℝ),
    berengere_contribution = 0 ∧
    emily_dollars / exchange_rate + berengere_contribution ≥ pastry_cost :=
by sorry

end paris_cafe_contribution_l3495_349597


namespace equation_solution_l3495_349561

theorem equation_solution (m : ℕ+) : 
  (∃ x : ℕ+, x ≠ 8 ∧ (m * x : ℚ) / (x - 8 : ℚ) = ((4 * m + x) : ℚ) / (x - 8 : ℚ)) ↔ 
  m = 3 ∨ m = 5 := by
  sorry

end equation_solution_l3495_349561


namespace bakery_storage_l3495_349579

theorem bakery_storage (sugar flour baking_soda : ℕ) : 
  (sugar : ℚ) / flour = 5 / 2 →
  (flour : ℚ) / baking_soda = 10 / 1 →
  (flour : ℚ) / (baking_soda + 60) = 8 / 1 →
  sugar = 6000 := by
sorry


end bakery_storage_l3495_349579


namespace triangle_perimeter_not_72_l3495_349595

theorem triangle_perimeter_not_72 (a b c : ℝ) : 
  a = 20 → b = 15 → a + b > c → a + c > b → b + c > a → a + b + c ≠ 72 := by
  sorry

end triangle_perimeter_not_72_l3495_349595


namespace alley_width_l3495_349523

/-- Given a ladder of length a in an alley, making angles of 60° and 45° with the ground on opposite walls, 
    the width of the alley w is equal to (√3 * a) / 2. -/
theorem alley_width (a : ℝ) (w : ℝ) (h : ℝ) (k : ℝ) : 
  a > 0 → 
  k = a * (1 / 2) → 
  h = a * (Real.sqrt 2 / 2) → 
  w ^ 2 = h ^ 2 + k ^ 2 → 
  w = (Real.sqrt 3 * a) / 2 := by
sorry

end alley_width_l3495_349523


namespace little_john_money_distribution_l3495_349584

theorem little_john_money_distribution (initial_amount spent_on_sweets final_amount : ℚ) 
  (num_friends : ℕ) (h1 : initial_amount = 20.10) (h2 : spent_on_sweets = 1.05) 
  (h3 : final_amount = 17.05) (h4 : num_friends = 2) : 
  (initial_amount - final_amount - spent_on_sweets) / num_friends = 1 := by
  sorry

end little_john_money_distribution_l3495_349584


namespace symmetry_implies_values_l3495_349596

/-- Two lines are symmetric about y = x if and only if they are inverse functions of each other -/
axiom symmetry_iff_inverse (f g : ℝ → ℝ) : 
  (∀ x y, f y = x ↔ g x = y) ↔ (∀ x, f (g x) = x ∧ g (f x) = x)

/-- The line ax - y + 2 = 0 -/
def line1 (a : ℝ) (x : ℝ) : ℝ := a * x + 2

/-- The line 3x - y - b = 0 -/
def line2 (b : ℝ) (x : ℝ) : ℝ := 3 * x - b

theorem symmetry_implies_values (a b : ℝ) : 
  (∀ x y, line1 a y = x ↔ line2 b x = y) → a = 1/3 ∧ b = 6 := by
  sorry

end symmetry_implies_values_l3495_349596


namespace limit_example_l3495_349540

/-- The limit of (2x^2 - 5x + 2) / (x - 1/2) as x approaches 1/2 is -3 -/
theorem limit_example (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, x ≠ 1/2 → |x - 1/2| < δ →
    |((2 * x^2 - 5 * x + 2) / (x - 1/2)) + 3| < ε :=
by sorry

end limit_example_l3495_349540


namespace three_times_x_not_much_different_from_two_l3495_349548

theorem three_times_x_not_much_different_from_two :
  ∃ (x : ℝ), 3 * x - 2 ≤ -1 :=
by sorry

end three_times_x_not_much_different_from_two_l3495_349548


namespace f_derivative_l3495_349501

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - (x + 1)

-- State the theorem
theorem f_derivative (x : ℝ) : 
  (deriv f) x = 4 * x + 3 := by sorry

end f_derivative_l3495_349501


namespace martha_total_savings_l3495_349575

/-- Represents Martha's savings plan for a month --/
structure SavingsPlan where
  daily_allowance : ℝ
  week1_savings : List ℝ
  week2_savings : List ℝ
  week3_savings : List ℝ
  week4_savings : List ℝ
  week1_expense : ℝ
  week2_expense : ℝ
  week3_expense : ℝ
  week4_expense : ℝ

/-- Calculates the total savings for a given week --/
def weekly_savings (savings : List ℝ) (expense : ℝ) : ℝ :=
  savings.sum - expense

/-- Calculates the total savings for the month --/
def total_monthly_savings (plan : SavingsPlan) : ℝ :=
  weekly_savings plan.week1_savings plan.week1_expense +
  weekly_savings plan.week2_savings plan.week2_expense +
  weekly_savings plan.week3_savings plan.week3_expense +
  weekly_savings plan.week4_savings plan.week4_expense

/-- Martha's specific savings plan --/
def martha_plan : SavingsPlan :=
  { daily_allowance := 15
  , week1_savings := [6, 6, 6, 6, 6, 6, 4.5]
  , week2_savings := [7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 6]
  , week3_savings := [9, 9, 9, 9, 7.5, 9, 9]
  , week4_savings := [10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 9]
  , week1_expense := 20
  , week2_expense := 30
  , week3_expense := 40
  , week4_expense := 50
  }

/-- Theorem stating that Martha's total savings at the end of the month is $106 --/
theorem martha_total_savings :
  total_monthly_savings martha_plan = 106 := by
  sorry

end martha_total_savings_l3495_349575


namespace sara_minus_lucas_sum_l3495_349562

def sara_list := List.range 50

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def lucas_list := sara_list.map replace_three_with_two

theorem sara_minus_lucas_sum : 
  sara_list.sum - lucas_list.sum = 105 := by sorry

end sara_minus_lucas_sum_l3495_349562


namespace students_not_eating_lunch_l3495_349580

theorem students_not_eating_lunch (total_students : ℕ) 
  (cafeteria_students : ℕ) (h1 : total_students = 60) 
  (h2 : cafeteria_students = 10) : 
  total_students - (3 * cafeteria_students + cafeteria_students) = 20 := by
  sorry

end students_not_eating_lunch_l3495_349580


namespace intersection_of_M_and_N_l3495_349592

def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_of_M_and_N :
  M ∩ N = {y | y ≥ 1} := by sorry

end intersection_of_M_and_N_l3495_349592


namespace race_speed_calculation_l3495_349530

/-- Given two teams racing on a 300-mile course, where one team finishes 3 hours earlier
    and has an average speed 5 mph greater than the other, prove that the slower team's
    average speed is 20 mph. -/
theorem race_speed_calculation (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 300 ∧ time_diff = 3 ∧ speed_diff = 5 →
  ∃ (speed_e : ℝ) (time_e : ℝ),
    speed_e > 0 ∧
    time_e > 0 ∧
    distance = speed_e * time_e ∧
    distance = (speed_e + speed_diff) * (time_e - time_diff) ∧
    speed_e = 20 :=
by sorry

end race_speed_calculation_l3495_349530


namespace perpendicular_lines_k_values_l3495_349517

theorem perpendicular_lines_k_values (k : ℝ) : 
  (∀ x y : ℝ, (k - 1) * x + (2 * k + 3) * y - 2 = 0 ∧ 
               k * x + (1 - k) * y - 3 = 0 → 
               ((k - 1) * k + (2 * k + 3) * (1 - k) = 0)) → 
  k = 1 ∨ k = -3 :=
sorry

end perpendicular_lines_k_values_l3495_349517


namespace carl_first_six_probability_l3495_349567

/-- The probability of rolling a 6 on a single die roll -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a single die roll -/
def prob_not_six : ℚ := 1 - prob_six

/-- The sequence of probabilities for Carl rolling the first 6 on his nth turn -/
def carl_first_six (n : ℕ) : ℚ := (prob_not_six ^ (3 * n - 1)) * prob_six

/-- The sum of the geometric series representing the probability of Carl rolling the first 6 -/
def probability_carl_first_six : ℚ := (carl_first_six 1) / (1 - (prob_not_six ^ 3))

theorem carl_first_six_probability :
  probability_carl_first_six = 25 / 91 :=
sorry

end carl_first_six_probability_l3495_349567


namespace point_on_line_l3495_349581

/-- Given a line passing through points (0,2) and (-4,-1), prove that if (t,7) lies on this line, then t = 20/3 -/
theorem point_on_line (t : ℝ) : 
  (∀ x y : ℝ, (y - 2) / x = (-1 - 2) / (-4 - 0)) → -- Line through (0,2) and (-4,-1)
  ((7 - 2) / t = (-1 - 2) / (-4 - 0)) →             -- (t,7) lies on the line
  t = 20 / 3 := by
sorry

end point_on_line_l3495_349581


namespace novel_to_history_ratio_l3495_349547

theorem novel_to_history_ratio :
  let science_pages : ℕ := 600
  let history_pages : ℕ := 300
  let novel_pages : ℕ := science_pages / 4
  novel_pages.gcd history_pages = novel_pages →
  (novel_pages / novel_pages.gcd history_pages) = 1 ∧
  (history_pages / novel_pages.gcd history_pages) = 2 :=
by sorry

end novel_to_history_ratio_l3495_349547


namespace blue_pencil_length_l3495_349572

/-- Given a pencil with a total length of 6 cm, a purple part of 3 cm, and a black part of 2 cm,
    prove that the length of the blue part is 1 cm. -/
theorem blue_pencil_length (total : ℝ) (purple : ℝ) (black : ℝ) (blue : ℝ)
    (h_total : total = 6)
    (h_purple : purple = 3)
    (h_black : black = 2)
    (h_sum : total = purple + black + blue) :
    blue = 1 := by
  sorry

end blue_pencil_length_l3495_349572


namespace pb_cookie_probability_l3495_349506

/-- Represents the number of peanut butter cookies Jenny brought -/
def jenny_pb : ℕ := 40

/-- Represents the number of chocolate chip cookies Jenny brought -/
def jenny_cc : ℕ := 50

/-- Represents the number of peanut butter cookies Marcus brought -/
def marcus_pb : ℕ := 30

/-- Represents the number of lemon cookies Marcus brought -/
def marcus_lemon : ℕ := 20

/-- Represents the total number of cookies -/
def total_cookies : ℕ := jenny_pb + jenny_cc + marcus_pb + marcus_lemon

/-- Represents the total number of peanut butter cookies -/
def total_pb : ℕ := jenny_pb + marcus_pb

/-- Theorem stating that the probability of selecting a peanut butter cookie is 50% -/
theorem pb_cookie_probability : 
  (total_pb : ℚ) / total_cookies * 100 = 50 := by sorry

end pb_cookie_probability_l3495_349506


namespace average_bicycling_speed_l3495_349576

/-- Calculates the average bicycling speed given the conditions of the problem -/
theorem average_bicycling_speed (total_distance : ℝ) (bicycle_time : ℝ) (run_speed : ℝ) (total_time : ℝ) :
  total_distance = 20 →
  bicycle_time = 12 / 60 →
  run_speed = 8 →
  total_time = 117 / 60 →
  let run_time := total_time - bicycle_time
  let run_distance := run_speed * run_time
  let bicycle_distance := total_distance - run_distance
  bicycle_distance / bicycle_time = 30 := by
  sorry

#check average_bicycling_speed

end average_bicycling_speed_l3495_349576


namespace middle_term_coefficient_l3495_349516

/-- Given a binomial expansion (x^2 - 2/x)^n where the 5th term is constant,
    prove that the coefficient of the middle term is -160 -/
theorem middle_term_coefficient
  (x : ℝ) (n : ℕ)
  (h_constant : ∃ k : ℝ, (n.choose 4) * (x^2)^(n-4) * (-2/x)^4 = k) :
  ∃ m : ℕ, m = (n+1)/2 ∧ (n.choose (m-1)) * (x^2)^(m-1) * (-2/x)^(n-m+1) = -160 * x^(2*m-n-1) :=
sorry

end middle_term_coefficient_l3495_349516


namespace parabola_p_value_l3495_349544

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = 2 * C.p * x

/-- Theorem: Value of p for a parabola given specific point conditions -/
theorem parabola_p_value (C : Parabola) (A : PointOnParabola C)
  (h1 : Real.sqrt ((A.x - C.p/2)^2 + A.y^2) = 12)  -- Distance from A to focus is 12
  (h2 : A.x = 9)  -- Distance from A to y-axis is 9
  : C.p = 6 := by
  sorry

end parabola_p_value_l3495_349544


namespace solve_for_t_l3495_349551

theorem solve_for_t (s t : ℚ) 
  (eq1 : 12 * s + 7 * t = 165)
  (eq2 : s = t + 3) : 
  t = 129 / 19 := by
  sorry

end solve_for_t_l3495_349551


namespace remainder_problem_l3495_349528

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 19) → N % 13 = 6 := by
sorry

end remainder_problem_l3495_349528


namespace sqrt_sum_equality_system_of_equations_solution_l3495_349513

-- Problem 1
theorem sqrt_sum_equality : |Real.sqrt 2 - Real.sqrt 5| + 2 * Real.sqrt 2 = Real.sqrt 5 + Real.sqrt 2 := by
  sorry

-- Problem 2
theorem system_of_equations_solution :
  ∃ (x y : ℝ), 4 * x + y = 15 ∧ 3 * x - 2 * y = 3 ∧ x = 3 ∧ y = 3 := by
  sorry

end sqrt_sum_equality_system_of_equations_solution_l3495_349513


namespace exists_circle_with_n_lattice_points_l3495_349535

/-- A point with integer coordinates in the plane -/
def LatticePoint := ℤ × ℤ

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of lattice points on the circumference of a circle -/
def latticePointsOnCircle (c : Circle) : ℕ :=
  sorry

/-- For every natural number n, there exists a circle with exactly n lattice points on its circumference -/
theorem exists_circle_with_n_lattice_points (n : ℕ) :
  ∃ c : Circle, latticePointsOnCircle c = n :=
sorry

end exists_circle_with_n_lattice_points_l3495_349535


namespace gcf_360_150_l3495_349526

theorem gcf_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcf_360_150_l3495_349526


namespace no_true_propositions_l3495_349594

theorem no_true_propositions : 
  let prop1 := ∀ x : ℝ, x^2 - 3*x + 2 = 0
  let prop2 := ∃ x : ℚ, x^2 = 2
  let prop3 := ∃ x : ℝ, x^2 + 1 = 0
  let prop4 := ∀ x : ℝ, 4*x^2 > 2*x - 1 + 3*x^2
  ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4 :=
by
  sorry

#check no_true_propositions

end no_true_propositions_l3495_349594


namespace shed_width_calculation_l3495_349531

theorem shed_width_calculation (backyard_length backyard_width shed_length sod_area : ℝ)
  (h1 : backyard_length = 20)
  (h2 : backyard_width = 13)
  (h3 : shed_length = 3)
  (h4 : sod_area = 245)
  (h5 : backyard_length * backyard_width - sod_area = shed_length * shed_width) :
  shed_width = 5 := by
  sorry

end shed_width_calculation_l3495_349531


namespace product_digit_sum_base_8_l3495_349571

def base_8_to_decimal (n : ℕ) : ℕ := sorry

def decimal_to_base_8 (n : ℕ) : ℕ := sorry

def sum_of_digits_base_8 (n : ℕ) : ℕ := sorry

theorem product_digit_sum_base_8 :
  let a := 35
  let b := 21
  let product := (base_8_to_decimal a) * (base_8_to_decimal b)
  sum_of_digits_base_8 (decimal_to_base_8 product) = 21
  := by sorry

end product_digit_sum_base_8_l3495_349571


namespace square_side_sum_l3495_349533

theorem square_side_sum (b d : ℕ) : 
  15^2 = b^2 + 10^2 + d^2 → (b + d = 13 ∨ b + d = 15) :=
by sorry

end square_side_sum_l3495_349533


namespace triangle_symmetric_negative_three_four_l3495_349502

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a * b - a - b + 1

-- Theorem statement
theorem triangle_symmetric_negative_three_four : triangle (-3) 4 = triangle 4 (-3) := by
  sorry

end triangle_symmetric_negative_three_four_l3495_349502


namespace inequality_solution_l3495_349545

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - a*x - 6*a^2 > 0

-- Define the solution set
def solution_set (x₁ x₂ : ℝ) : Set ℝ := {x | x < x₁ ∨ x > x₂}

theorem inequality_solution (a : ℝ) (x₁ x₂ : ℝ) :
  a < 0 →
  (∀ x, inequality x a ↔ x ∈ solution_set x₁ x₂) →
  x₂ - x₁ = 5 * Real.sqrt 2 →
  a = -Real.sqrt 2 := by
  sorry

end inequality_solution_l3495_349545


namespace locus_of_perpendicular_foot_l3495_349536

/-- Given a plane P (z = 0), points A on P and O not on P, prove that the locus of points H,
    where H is the foot of the perpendicular from O to any line in P through A,
    forms a circle with the given equation. -/
theorem locus_of_perpendicular_foot (a b d e f : ℝ) :
  let P : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  let A : ℝ × ℝ × ℝ := (a, b, 0)
  let O : ℝ × ℝ × ℝ := (d, e, f)
  let H := {h : ℝ × ℝ × ℝ | ∃ (u v : ℝ),
    h = ((a * (u^2 + v^2) + (d*u + e*v - a*u - b*v)*u) / (u^2 + v^2),
         (b * (u^2 + v^2) + (d*u + e*v - a*u - b*v)*v) / (u^2 + v^2),
         0)}
  ∀ (x y : ℝ), (x, y, 0) ∈ H ↔ x^2 + y^2 - (a+d)*x - (b+e)*y + a*d + b*e = 0 :=
by sorry

end locus_of_perpendicular_foot_l3495_349536


namespace factor_condition_l3495_349515

theorem factor_condition (t : ℚ) : 
  (∃ k : ℚ, (X - t) * k = 3 * X^2 + 10 * X - 8) ↔ (t = 2/3 ∨ t = -4) :=
by sorry

end factor_condition_l3495_349515


namespace joshua_friends_count_l3495_349555

def total_skittles : ℕ := 40
def skittles_per_friend : ℕ := 8

theorem joshua_friends_count : 
  total_skittles / skittles_per_friend = 5 := by sorry

end joshua_friends_count_l3495_349555


namespace triple_layer_area_is_six_l3495_349553

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the arrangement of carpets -/
structure CarpetArrangement where
  hallSize : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area covered by all three carpets in the given arrangement -/
def tripleLayerArea (arrangement : CarpetArrangement) : ℝ :=
  sorry

/-- Theorem stating that the area covered by all three carpets is 6 square meters -/
theorem triple_layer_area_is_six (arrangement : CarpetArrangement) 
  (h1 : arrangement.hallSize = 10)
  (h2 : arrangement.carpet1 = ⟨6, 8⟩)
  (h3 : arrangement.carpet2 = ⟨6, 6⟩)
  (h4 : arrangement.carpet3 = ⟨5, 7⟩) :
  tripleLayerArea arrangement = 6 := by
  sorry

end triple_layer_area_is_six_l3495_349553


namespace mac_running_rate_l3495_349558

/-- The running rate of Apple in miles per hour -/
def apple_rate : ℝ := 3

/-- The race distance in miles -/
def race_distance : ℝ := 24

/-- The time difference between Apple and Mac in minutes -/
def time_difference : ℝ := 120

/-- Mac's running rate in miles per hour -/
def mac_rate : ℝ := 4

/-- Theorem stating that given the conditions, Mac's running rate is 4 miles per hour -/
theorem mac_running_rate : 
  let apple_time := race_distance / apple_rate * 60  -- Apple's time in minutes
  let mac_time := apple_time - time_difference       -- Mac's time in minutes
  mac_rate = race_distance / (mac_time / 60) := by
sorry

end mac_running_rate_l3495_349558


namespace area_ratio_S₂_to_S₁_l3495_349527

-- Define the sets S₁ and S₂
def S₁ : Set (ℝ × ℝ) := {p | Real.log (1 + p.1^2 + p.2^2) ≤ 1 + Real.log (p.1 + p.2)}
def S₂ : Set (ℝ × ℝ) := {p | Real.log (2 + p.1^2 + p.2^2) ≤ 2 + Real.log (p.1 + p.2)}

-- Define the areas of S₁ and S₂
noncomputable def area_S₁ : ℝ := Real.pi * 49
noncomputable def area_S₂ : ℝ := Real.pi * 4998

-- Theorem statement
theorem area_ratio_S₂_to_S₁ : area_S₂ / area_S₁ = 102 := by sorry

end area_ratio_S₂_to_S₁_l3495_349527


namespace geese_survival_l3495_349529

/-- Given the following conditions:
  1. 500 goose eggs were laid
  2. 2/3 of eggs hatched
  3. 3/4 of hatched geese survived the first month
  4. 2/5 of geese that survived the first month survived the first year
Prove that 100 geese survived the first year -/
theorem geese_survival (total_eggs : ℕ) (hatch_rate first_month_rate first_year_rate : ℚ) :
  total_eggs = 500 →
  hatch_rate = 2/3 →
  first_month_rate = 3/4 →
  first_year_rate = 2/5 →
  (total_eggs : ℚ) * hatch_rate * first_month_rate * first_year_rate = 100 := by
  sorry

#eval (500 : ℚ) * (2/3) * (3/4) * (2/5)

end geese_survival_l3495_349529


namespace perpendicular_lines_m_values_l3495_349521

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 1 = 0 ∧ m^2 * x - 2 * y - 1 = 0 → 
    (1 : ℝ) * (-m^2 : ℝ) = -1) → 
  m = 0 ∨ m = 2 := by
sorry

end perpendicular_lines_m_values_l3495_349521


namespace factor_expression_l3495_349512

theorem factor_expression (x : ℝ) : 35 * x^11 + 49 * x^22 = 7 * x^11 * (5 + 7 * x^11) := by
  sorry

end factor_expression_l3495_349512


namespace largest_m_for_inequality_l3495_349543

theorem largest_m_for_inequality : ∃ m : ℕ+, 
  (m = 27) ∧ 
  (∀ n : ℕ+, n ≤ m → (2*n + 1)/(3*n + 8) < (Real.sqrt 5 - 1)/2 ∧ (Real.sqrt 5 - 1)/2 < (n + 7)/(2*n + 1)) ∧
  (∀ m' : ℕ+, m' > m → ∃ n : ℕ+, n ≤ m' ∧ ((2*n + 1)/(3*n + 8) ≥ (Real.sqrt 5 - 1)/2 ∨ (Real.sqrt 5 - 1)/2 ≥ (n + 7)/(2*n + 1))) :=
by sorry

end largest_m_for_inequality_l3495_349543


namespace factor_polynomial_l3495_349537

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 50 * x^10 = 25 * x^7 * (3 - 2 * x^3) := by
  sorry

end factor_polynomial_l3495_349537


namespace integral_one_plus_cos_over_pi_half_interval_l3495_349503

theorem integral_one_plus_cos_over_pi_half_interval :
  ∫ x in (-π/2)..(π/2), (1 + Real.cos x) = π + 2 := by sorry

end integral_one_plus_cos_over_pi_half_interval_l3495_349503


namespace least_number_divisible_by_seven_with_remainder_one_l3495_349510

theorem least_number_divisible_by_seven_with_remainder_one : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → n % k = 1) ∧ 
  n % 7 = 0 ∧
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → m % k = 1) ∨ m % 7 ≠ 0) ∧
  n = 301 :=
by
  sorry

end least_number_divisible_by_seven_with_remainder_one_l3495_349510


namespace smallest_four_digit_divisible_by_53_l3495_349532

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 := by
sorry

end smallest_four_digit_divisible_by_53_l3495_349532


namespace quadratic_root_relation_l3495_349565

theorem quadratic_root_relation (p q : ℝ) :
  (∃ α : ℝ, (α^2 + p*α + q = 0) ∧ ((2*α)^2 + p*(2*α) + q = 0)) →
  2*p^2 = 9*q :=
by sorry

end quadratic_root_relation_l3495_349565


namespace security_compromise_l3495_349518

/-- Represents the security level of a system -/
inductive SecurityLevel
  | High
  | Medium
  | Low

/-- Represents a file type -/
inductive FileType
  | Secure
  | Suspicious

/-- Represents a website -/
structure Website where
  trusted : Bool

/-- Represents a user action -/
inductive UserAction
  | ShareInfo
  | DownloadFile (fileType : FileType)

/-- Represents the state of a system after a user action -/
structure SystemState where
  securityLevel : SecurityLevel

/-- Defines how a user action affects the system state -/
def updateSystemState (website : Website) (action : UserAction) (initialState : SystemState) : SystemState :=
  match website.trusted, action with
  | true, _ => initialState
  | false, UserAction.ShareInfo => ⟨SecurityLevel.Low⟩
  | false, UserAction.DownloadFile FileType.Suspicious => ⟨SecurityLevel.Low⟩
  | false, UserAction.DownloadFile FileType.Secure => initialState

theorem security_compromise (website : Website) (action : UserAction) (initialState : SystemState) :
  ¬website.trusted →
  (action = UserAction.ShareInfo ∨ (∃ (ft : FileType), action = UserAction.DownloadFile ft ∧ ft = FileType.Suspicious)) →
  (updateSystemState website action initialState).securityLevel = SecurityLevel.Low :=
by sorry


end security_compromise_l3495_349518


namespace paycheck_calculation_l3495_349573

def biweekly_gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

theorem paycheck_calculation :
  biweekly_gross_pay * (1 - retirement_rate) - tax_deduction = 740 := by
  sorry

end paycheck_calculation_l3495_349573


namespace vector_b_solution_l3495_349586

def vector_a : ℝ × ℝ := (1, -2)

theorem vector_b_solution (b : ℝ × ℝ) :
  (b.1 * vector_a.2 = b.2 * vector_a.1) →  -- parallel condition
  (b.1^2 + b.2^2 = 20) →                   -- magnitude condition
  (b = (2, -4) ∨ b = (-2, 4)) :=
by sorry

end vector_b_solution_l3495_349586


namespace intersection_A_B_l3495_349556

def A : Set ℕ := {x | 0 < x ∧ x < 6}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_A_B : A ∩ B = {2, 4} := by sorry

end intersection_A_B_l3495_349556


namespace min_value_fraction_l3495_349564

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (a + 2) * (b + 2) / (16 * a * b) ≥ 1 := by
  sorry

end min_value_fraction_l3495_349564


namespace remainder_theorem_l3495_349566

/-- A polynomial of the form Ax^6 + Bx^4 + Cx^2 + 5 -/
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 5

/-- The remainder when p(x) is divided by x-2 is 13 -/
def remainder_condition (A B C : ℝ) : Prop := p A B C 2 = 13

theorem remainder_theorem (A B C : ℝ) (h : remainder_condition A B C) :
  p A B C (-2) = 13 := by sorry

end remainder_theorem_l3495_349566


namespace triangle_altitude_area_theorem_l3495_349519

/-- Definition of a triangle with altitudes and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  area : ℝ

/-- Theorem stating the existence and non-existence of triangles with specific properties -/
theorem triangle_altitude_area_theorem :
  (∃ t : Triangle, t.ha < 1 ∧ t.hb < 1 ∧ t.hc < 1 ∧ t.area > 2) ∧
  (¬ ∃ t : Triangle, t.ha > 2 ∧ t.hb > 2 ∧ t.hc > 2 ∧ t.area < 1) :=
by sorry

end triangle_altitude_area_theorem_l3495_349519


namespace equal_pay_implies_hours_constraint_l3495_349587

/-- Represents the payment structure and hours worked for Harry and James -/
structure WorkData where
  x : ℝ  -- hourly rate
  h : ℝ  -- Harry's normal hours
  y : ℝ  -- Harry's overtime hours

/-- The theorem states that if Harry and James were paid the same amount,
    and James worked 41 hours, then h + 2y = 42 -/
theorem equal_pay_implies_hours_constraint (data : WorkData) :
  data.x * data.h + 2 * data.x * data.y = data.x * 40 + 2 * data.x * 1 →
  data.h + 2 * data.y = 42 := by
  sorry

#check equal_pay_implies_hours_constraint

end equal_pay_implies_hours_constraint_l3495_349587


namespace pencils_given_equals_nine_l3495_349585

/-- The number of pencils in one stroke -/
def pencils_per_stroke : ℕ := 12

/-- The number of strokes Namjoon had -/
def namjoon_strokes : ℕ := 2

/-- The number of pencils Namjoon had left after giving some to Yoongi -/
def pencils_left : ℕ := 15

/-- The number of pencils Namjoon gave to Yoongi -/
def pencils_given_to_yoongi : ℕ := namjoon_strokes * pencils_per_stroke - pencils_left

theorem pencils_given_equals_nine : pencils_given_to_yoongi = 9 := by
  sorry

end pencils_given_equals_nine_l3495_349585


namespace eight_points_on_circle_theorem_l3495_349507

/-- A point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- The theorem statement -/
theorem eight_points_on_circle_theorem
  (p : ℕ) (n : ℕ) (points : Finset IntPoint) :
  Nat.Prime p →
  p % 2 = 1 →
  n > 0 →
  points.card = 8 →
  (∀ pt ∈ points, ∃ (x y : ℤ), pt = ⟨x, y⟩) →
  (∃ (center : IntPoint) (r : ℤ), r^2 = (p^n)^2 / 4 ∧
    ∀ pt ∈ points, (pt.x - center.x)^2 + (pt.y - center.y)^2 = r^2) →
  ∃ (a b c : IntPoint), a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∃ (ab bc ca : ℤ),
      ab = (a.x - b.x)^2 + (a.y - b.y)^2 ∧
      bc = (b.x - c.x)^2 + (b.y - c.y)^2 ∧
      ca = (c.x - a.x)^2 + (c.y - a.y)^2 ∧
      ab % p^(n+1) = 0 ∧ bc % p^(n+1) = 0 ∧ ca % p^(n+1) = 0 :=
by sorry

end eight_points_on_circle_theorem_l3495_349507


namespace graduating_class_size_l3495_349514

theorem graduating_class_size 
  (geometry : ℕ) 
  (biology : ℕ) 
  (overlap_diff : ℕ) 
  (h1 : geometry = 144) 
  (h2 : biology = 119) 
  (h3 : overlap_diff = 88) :
  geometry + biology - min geometry biology = 263 :=
by sorry

end graduating_class_size_l3495_349514
