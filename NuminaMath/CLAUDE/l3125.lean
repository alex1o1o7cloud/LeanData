import Mathlib

namespace sector_area_l3125_312599

/-- Theorem: Area of a circular sector with central angle 2π/3 and arc length 2 --/
theorem sector_area (r : ℝ) (h1 : (2 * π / 3) * r = 2) : 
  (1 / 2) * r^2 * (2 * π / 3) = 3 / π := by
  sorry


end sector_area_l3125_312599


namespace constant_n_value_l3125_312526

theorem constant_n_value (m n : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + m) = x^2 + n*x + 12) : n = 7 := by
  sorry

end constant_n_value_l3125_312526


namespace volume_cube_inscribed_sphere_l3125_312587

/-- The volume of a cube inscribed in a sphere -/
theorem volume_cube_inscribed_sphere (R : ℝ) (h : R > 0) :
  ∃ (V : ℝ), V = (8 / 9) * Real.sqrt 3 * R^3 ∧ V > 0 := by sorry

end volume_cube_inscribed_sphere_l3125_312587


namespace group_size_l3125_312558

theorem group_size (total : ℕ) 
  (h1 : (total : ℚ) / 5 = (0.12 * total + 64 : ℚ)) : total = 800 := by
  sorry

end group_size_l3125_312558


namespace hot_dogs_remainder_l3125_312572

theorem hot_dogs_remainder : 16789537 % 5 = 2 := by
  sorry

end hot_dogs_remainder_l3125_312572


namespace range_of_a_l3125_312520

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (f = λ x => x * |x^2 - a|) →
  (∃ x ∈ Set.Icc 1 2, f x < 2) →
  -1 < a ∧ a < 5 := by sorry

end range_of_a_l3125_312520


namespace choir_members_count_l3125_312564

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 8 = 5 := by
  sorry

end choir_members_count_l3125_312564


namespace distance_to_origin_of_point_on_parabola_l3125_312501

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * C.p * x

theorem distance_to_origin_of_point_on_parabola
  (C : Parabola)
  (A : PointOnParabola C)
  (h1 : Real.sqrt ((A.x - C.p/2)^2 + A.y^2) = 6)  -- Distance from A to focus is 6
  (h2 : A.x = 3)  -- Distance from A to y-axis is 3
  : Real.sqrt (A.x^2 + A.y^2) = 3 * Real.sqrt 5 := by
  sorry

end distance_to_origin_of_point_on_parabola_l3125_312501


namespace focus_of_given_parabola_l3125_312583

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  /-- The equation of the parabola in the form y = a(x - h)^2 + k -/
  equation : ℝ → ℝ
  /-- The coefficient 'a' determines the direction and width of the parabola -/
  a : ℝ
  /-- The horizontal shift of the vertex -/
  h : ℝ
  /-- The vertical shift of the vertex -/
  k : ℝ

/-- The focus of a parabola is a point from which all points on the parabola are equidistant to the directrix. -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Given parabola y = (x-3)^2 + 2 -/
def given_parabola : Parabola where
  equation := fun x ↦ (x - 3)^2 + 2
  a := 1
  h := 3
  k := 2

/-- Theorem: The focus of the parabola y = (x-3)^2 + 2 is at the point (3, 9/4) -/
theorem focus_of_given_parabola :
  focus given_parabola = (3, 9/4) := by sorry

end focus_of_given_parabola_l3125_312583


namespace sum_of_solutions_quadratic_l3125_312509

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ p q : ℝ, 16 - 4*x - x^2 = 0 ∧ x = p ∨ x = q) → 
  (∃ p q : ℝ, 16 - 4*p - p^2 = 0 ∧ 16 - 4*q - q^2 = 0 ∧ p + q = 4) :=
sorry

end sum_of_solutions_quadratic_l3125_312509


namespace system_solution_l3125_312527

theorem system_solution :
  ∀ (x y z : ℝ),
    (x + 1) * y * z = 12 ∧
    (y + 1) * z * x = 4 ∧
    (z + 1) * x * y = 4 →
    ((x = 1/3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end system_solution_l3125_312527


namespace train_platform_passing_time_l3125_312529

/-- Given a train of length 250 meters that passes a pole in 10 seconds
    and a platform in 60 seconds, prove that the time taken to pass
    only the platform is 50 seconds. -/
theorem train_platform_passing_time
  (train_length : ℝ)
  (pole_passing_time : ℝ)
  (platform_total_passing_time : ℝ)
  (h1 : train_length = 250)
  (h2 : pole_passing_time = 10)
  (h3 : platform_total_passing_time = 60) :
  let train_speed := train_length / pole_passing_time
  let platform_length := train_speed * platform_total_passing_time - train_length
  platform_length / train_speed = 50 := by
sorry

end train_platform_passing_time_l3125_312529


namespace quadratic_tangent_to_x_axis_l3125_312573

/-- A quadratic function f(x) = ax^2 + bx + c is tangent to the x-axis
    if and only if c = b^2 / (4a) -/
theorem quadratic_tangent_to_x_axis (a b c : ℝ) (h : c = b^2 / (4 * a)) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, x ≠ x₀ → f x > 0 :=
by sorry

end quadratic_tangent_to_x_axis_l3125_312573


namespace special_function_inequality_l3125_312549

/-- A function satisfying the given differential inequality -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  diff_twice : ∀ x ∈ domain, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ (deriv f) x
  ineq : ∀ x ∈ domain, x * (deriv^[2] f x) > f x

/-- The main theorem -/
theorem special_function_inequality (φ : SpecialFunction) (x₁ x₂ : ℝ) 
    (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) : 
    φ.f x₁ + φ.f x₂ < φ.f (x₁ + x₂) := by
  sorry

end special_function_inequality_l3125_312549


namespace intersection_range_l3125_312598

theorem intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ 
  -3 ≤ a ∧ a ≤ 1 :=
by sorry

end intersection_range_l3125_312598


namespace composite_probability_six_dice_l3125_312518

/-- The number of sides on a standard die -/
def dieSize : Nat := 6

/-- The number of dice rolled -/
def numDice : Nat := 6

/-- The set of possible outcomes when rolling a die -/
def dieOutcomes : Finset Nat := Finset.range dieSize

/-- The total number of possible outcomes when rolling 6 dice -/
def totalOutcomes : Nat := dieSize ^ numDice

/-- A function that determines if a number is prime -/
def isPrime (n : Nat) : Bool := sorry

/-- A function that determines if a number is composite -/
def isComposite (n : Nat) : Bool := n > 1 ∧ ¬(isPrime n)

/-- The number of outcomes where the product is not composite -/
def nonCompositeOutcomes : Nat := 19

/-- The probability of rolling a composite product -/
def compositeProb : Rat := (totalOutcomes - nonCompositeOutcomes) / totalOutcomes

theorem composite_probability_six_dice :
  compositeProb = 46637 / 46656 := by sorry

end composite_probability_six_dice_l3125_312518


namespace square_difference_of_integers_l3125_312556

theorem square_difference_of_integers (a b : ℤ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 60) (h4 : a - b = 16) : 
  a^2 - b^2 = 960 := by
sorry

end square_difference_of_integers_l3125_312556


namespace ava_watched_hours_l3125_312567

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of minutes Ava watched television
def ava_watched_minutes : ℕ := 240

-- Theorem to prove
theorem ava_watched_hours : ava_watched_minutes / minutes_per_hour = 4 := by
  sorry

end ava_watched_hours_l3125_312567


namespace ellipse_hyperbola_tangent_l3125_312597

/-- The value of n for which the ellipse 2x^2 + 3y^2 = 6 and the hyperbola 3x^2 - n(y-1)^2 = 3 are tangent -/
def tangent_n : ℝ := -6

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y n : ℝ) : Prop := 3 * x^2 - n * (y - 1)^2 = 3

/-- Two curves are tangent if they intersect at exactly one point -/
def are_tangent (f g : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, f p.1 p.2 ∧ g p.1 p.2

theorem ellipse_hyperbola_tangent :
  are_tangent (λ x y => is_on_ellipse x y) (λ x y => is_on_hyperbola x y tangent_n) :=
sorry

end ellipse_hyperbola_tangent_l3125_312597


namespace tangent_line_at_point_one_four_l3125_312578

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 2

/-- The derivative of the parabola function -/
def f' (x : ℝ) : ℝ := 2*x + 1

theorem tangent_line_at_point_one_four :
  let x₀ : ℝ := 1
  let y₀ : ℝ := 4
  -- The point (1,4) lies on the parabola
  (f x₀ = y₀) →
  -- The slope of the tangent line at (1,4) is 3
  (f' x₀ = 3) ∧
  -- The equation of the tangent line is 3x - y + 1 = 0
  (∀ x y, y - y₀ = f' x₀ * (x - x₀) ↔ 3*x - y + 1 = 0) :=
by sorry

end tangent_line_at_point_one_four_l3125_312578


namespace rectangle_no_integer_points_l3125_312575

-- Define the rectangle type
structure Rectangle where
  a : ℝ
  b : ℝ
  h : a < b

-- Define the property of having no integer points
def hasNoIntegerPoints (r : Rectangle) : Prop :=
  ∀ x y : ℤ, ¬(0 ≤ x ∧ x ≤ r.b ∧ 0 ≤ y ∧ y ≤ r.a)

-- Theorem statement
theorem rectangle_no_integer_points (r : Rectangle) :
  hasNoIntegerPoints r ↔ min r.a r.b < 1 := by sorry

end rectangle_no_integer_points_l3125_312575


namespace cube_root_problem_l3125_312546

theorem cube_root_problem (a b c : ℝ) : 
  (3 * a + 21) ^ (1/3) = 3 → 
  (4 * a - b - 1) ^ (1/2) = 2 → 
  c ^ (1/2) = c → 
  a = 2 ∧ b = 3 ∧ c = 0 ∧ (3 * a + 10 * b + c) ^ (1/2) = 6 ∨ (3 * a + 10 * b + c) ^ (1/2) = -6 :=
by sorry

end cube_root_problem_l3125_312546


namespace odd_function_implies_m_n_equal_one_f_is_decreasing_k_range_l3125_312539

noncomputable def f (m n x : ℝ) : ℝ := (m - 3^x) / (n + 3^x)

theorem odd_function_implies_m_n_equal_one 
  (h : ∀ x, f m n x = -f m n (-x)) : m = 1 ∧ n = 1 := by sorry

theorem f_is_decreasing : 
  ∀ x y, x < y → f 1 1 x > f 1 1 y := by sorry

theorem k_range (t : ℝ) (h1 : t ∈ Set.Icc 0 4) 
  (h2 : f 1 1 (k - 2*t^2) + f 1 1 (4*t - 2*t^2) < 0) : 
  k > -1 := by sorry

end odd_function_implies_m_n_equal_one_f_is_decreasing_k_range_l3125_312539


namespace albert_betty_age_ratio_l3125_312596

/-- Given the ages of Albert, Mary, and Betty, prove that the ratio of Albert's age to Betty's age is 4:1 -/
theorem albert_betty_age_ratio :
  ∀ (albert mary betty : ℕ),
  albert = 2 * mary →
  mary = albert - 8 →
  betty = 4 →
  (albert : ℚ) / betty = 4 / 1 := by
sorry

end albert_betty_age_ratio_l3125_312596


namespace y_intercept_of_line_l3125_312580

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (a b : ℝ) : ℝ := b

/-- Given a line with equation y = 2x - 1, prove that its y-intercept is -1. -/
theorem y_intercept_of_line (x y : ℝ) (h : y = 2 * x - 1) : y_intercept 2 (-1) = -1 := by
  sorry

end y_intercept_of_line_l3125_312580


namespace max_product_under_constraint_max_product_achievable_l3125_312582

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 9 * a^2 + 16 * b^2 = 25) : a * b ≤ 25 / 24 := by
  sorry

theorem max_product_achievable (ε : ℝ) (hε : ε > 0) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 9 * a^2 + 16 * b^2 = 25 ∧ a * b > 25 / 24 - ε := by
  sorry

end max_product_under_constraint_max_product_achievable_l3125_312582


namespace max_profit_l3125_312561

/-- Profit function for price greater than 120 yuan -/
def profit_above (x : ℝ) : ℝ := -10 * x^2 + 2500 * x - 150000

/-- Profit function for price between 100 and 120 yuan -/
def profit_below (x : ℝ) : ℝ := -30 * x^2 + 6900 * x - 390000

/-- The maximum profit occurs at 115 yuan and equals 6750 yuan -/
theorem max_profit :
  ∃ (x : ℝ), x = 115 ∧ 
  profit_below x = 6750 ∧
  ∀ (y : ℝ), y > 100 → profit_above y ≤ profit_below x ∧ profit_below y ≤ profit_below x :=
sorry

end max_profit_l3125_312561


namespace average_books_is_three_l3125_312513

/-- Represents the distribution of books read by book club members -/
structure BookDistribution where
  one_book : Nat
  two_books : Nat
  three_books : Nat
  four_books : Nat
  six_books : Nat

/-- Calculates the average number of books read, rounded to the nearest whole number -/
def averageBooksRead (d : BookDistribution) : Nat :=
  let totalBooks := d.one_book * 1 + d.two_books * 2 + d.three_books * 3 + d.four_books * 4 + d.six_books * 6
  let totalMembers := d.one_book + d.two_books + d.three_books + d.four_books + d.six_books
  (totalBooks + totalMembers / 2) / totalMembers

/-- Theorem stating that the average number of books read is 3 -/
theorem average_books_is_three (d : BookDistribution) 
  (h1 : d.one_book = 4)
  (h2 : d.two_books = 3)
  (h3 : d.three_books = 6)
  (h4 : d.four_books = 2)
  (h5 : d.six_books = 3) : 
  averageBooksRead d = 3 := by
  sorry

end average_books_is_three_l3125_312513


namespace total_shaded_area_is_72_l3125_312542

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a parallelogram with base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := p.base * p.height

/-- Represents the overlap between shapes -/
structure Overlap where
  width : ℝ
  height : ℝ

/-- Calculates the area of overlap -/
def overlapArea (o : Overlap) : ℝ := o.width * o.height

/-- Theorem: The total shaded area of intersection between the given rectangle and parallelogram is 72 square units -/
theorem total_shaded_area_is_72 (r : Rectangle) (p : Parallelogram) (o : Overlap) : 
  r.width = 4 ∧ r.height = 12 ∧ p.base = 10 ∧ p.height = 4 ∧ o.width = 4 ∧ o.height = 4 →
  rectangleArea r + parallelogramArea p - overlapArea o = 72 := by
  sorry


end total_shaded_area_is_72_l3125_312542


namespace calculation_correction_l3125_312594

theorem calculation_correction (x : ℝ) (h : 63 / x = 9) : 36 - x = 29 := by
  sorry

end calculation_correction_l3125_312594


namespace crabapple_sequences_l3125_312523

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 11

/-- The number of times Mrs. Crabapple teaches per week -/
def classes_per_week : ℕ := 5

/-- The number of different sequences of crabapple recipients in one week -/
def num_sequences : ℕ := num_students ^ classes_per_week

theorem crabapple_sequences :
  num_sequences = 161051 :=
by sorry

end crabapple_sequences_l3125_312523


namespace sum_range_l3125_312552

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2*x - x^2 else |Real.log x|

theorem sum_range (a b c d : ℝ) :
  a < b ∧ b < c ∧ c < d ∧ f a = f b ∧ f b = f c ∧ f c = f d →
  1 < a + b + c + 2*d ∧ a + b + c + 2*d < 181/10 :=
sorry

end sum_range_l3125_312552


namespace one_tricycle_l3125_312511

/-- The number of cars in the driveway -/
def num_cars : ℕ := 2

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of bikes in the driveway -/
def num_bikes : ℕ := 2

/-- The number of wheels on each bike -/
def wheels_per_bike : ℕ := 2

/-- The number of trash cans in the driveway -/
def num_trash_cans : ℕ := 1

/-- The number of wheels on each trash can -/
def wheels_per_trash_can : ℕ := 2

/-- The number of roller skates (individual skates, not pairs) -/
def num_roller_skates : ℕ := 2

/-- The number of wheels on each roller skate -/
def wheels_per_roller_skate : ℕ := 4

/-- The total number of wheels in the driveway -/
def total_wheels : ℕ := 25

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

theorem one_tricycle :
  ∃ (num_tricycles : ℕ),
    num_tricycles * wheels_per_tricycle =
      total_wheels -
      (num_cars * wheels_per_car +
       num_bikes * wheels_per_bike +
       num_trash_cans * wheels_per_trash_can +
       num_roller_skates * wheels_per_roller_skate) ∧
    num_tricycles = 1 := by
  sorry

end one_tricycle_l3125_312511


namespace tan_theta_value_l3125_312590

theorem tan_theta_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π/2)
  (h3 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2*θ) = 3) :
  Real.tan θ = 2 - Real.sqrt 3 := by
sorry

end tan_theta_value_l3125_312590


namespace cord_length_proof_l3125_312593

/-- Given a cord divided into 19 equal parts, which when cut results in 20 pieces
    with the longest piece being 8 meters and the shortest being 2 meters,
    prove that the original length of the cord is 114 meters. -/
theorem cord_length_proof (n : ℕ) (longest shortest : ℝ) :
  n = 19 ∧
  longest = 8 ∧
  shortest = 2 →
  n * ((longest + shortest) / 2 + 1) = 114 :=
by sorry

end cord_length_proof_l3125_312593


namespace expression_equality_l3125_312524

theorem expression_equality : -Real.sqrt 4 + |(-Real.sqrt 2 - 1)| + (π - 2013)^0 - (1/5)^0 = Real.sqrt 2 - 1 := by
  sorry

end expression_equality_l3125_312524


namespace jane_daniel_difference_l3125_312544

/-- The width of the streets in Newville -/
def street_width : ℝ := 30

/-- The length of one side of a square block in Newville -/
def block_side : ℝ := 500

/-- The length of Daniel's path around one block -/
def daniel_lap : ℝ := 4 * block_side

/-- The length of Jane's path around one block -/
def jane_lap : ℝ := 4 * (block_side + street_width)

/-- The theorem stating the difference between Jane's and Daniel's lap distances -/
theorem jane_daniel_difference : jane_lap - daniel_lap = 120 := by
  sorry

end jane_daniel_difference_l3125_312544


namespace other_diagonal_length_l3125_312536

/-- A rhombus with known properties -/
structure Rhombus where
  /-- The length of one diagonal -/
  diagonal1 : ℝ
  /-- The area of one of the two equal triangles that make up the rhombus -/
  triangle_area : ℝ
  /-- Assumption that the diagonal1 is positive -/
  diagonal1_pos : 0 < diagonal1
  /-- Assumption that the triangle_area is positive -/
  triangle_area_pos : 0 < triangle_area

/-- The theorem stating the length of the other diagonal given specific conditions -/
theorem other_diagonal_length (r : Rhombus) (h1 : r.diagonal1 = 15) (h2 : r.triangle_area = 75) :
  ∃ diagonal2 : ℝ, diagonal2 = 20 ∧ r.diagonal1 * diagonal2 / 2 = 2 * r.triangle_area := by
  sorry

end other_diagonal_length_l3125_312536


namespace gcd_282_470_l3125_312570

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end gcd_282_470_l3125_312570


namespace marble_bag_total_l3125_312528

/-- Represents the total number of marbles in a bag with red, blue, and green marbles. -/
def total_marbles (red : ℕ) (blue : ℕ) (green : ℕ) : ℕ := red + blue + green

/-- Theorem: Given a bag of marbles with only red, blue, and green marbles,
    where the ratio of red to blue to green marbles is 2:3:4,
    and there are 36 blue marbles, the total number of marbles in the bag is 108. -/
theorem marble_bag_total :
  ∀ (red blue green : ℕ),
  red = 2 * n ∧ blue = 3 * n ∧ green = 4 * n →
  blue = 36 →
  total_marbles red blue green = 108 :=
by
  sorry

end marble_bag_total_l3125_312528


namespace new_person_weight_l3125_312563

theorem new_person_weight (n : Nat) (original_weight replaced_weight increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 50 ∧ 
  increase = 2.5 →
  (n : ℝ) * increase + replaced_weight = 70 := by
  sorry

end new_person_weight_l3125_312563


namespace ab_positive_sufficient_not_necessary_l3125_312588

theorem ab_positive_sufficient_not_necessary :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b > 0) ∧
  (∃ a b : ℝ, a * b > 0 ∧ ¬(a > 0 ∧ b > 0)) :=
by sorry

end ab_positive_sufficient_not_necessary_l3125_312588


namespace trigonometric_identities_l3125_312585

theorem trigonometric_identities (θ : Real) 
  (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : 
  Real.tan (2 * θ) = 4/3 ∧ 
  (Real.sin θ + Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -3 := by
  sorry

end trigonometric_identities_l3125_312585


namespace one_carton_per_case_l3125_312532

/-- Given that each carton contains b boxes, each box contains 200 paper clips,
    and 400 paper clips are contained in 2 cases, prove that there is 1 carton in a case. -/
theorem one_carton_per_case (b : ℕ) (h1 : b > 0) :
  (∃ c : ℕ, c > 0 ∧ c * b * 200 = 200) → c = 1 :=
by sorry

end one_carton_per_case_l3125_312532


namespace shaded_triangle_area_and_percentage_l3125_312555

/-- Given an equilateral triangle with side length 4 cm, prove the area and percentage of a shaded region -/
theorem shaded_triangle_area_and_percentage :
  let side_length : ℝ := 4
  let original_height : ℝ := side_length * (Real.sqrt 3) / 2
  let original_area : ℝ := side_length^2 * (Real.sqrt 3) / 4
  let shaded_base : ℝ := side_length * 3 / 4
  let shaded_height : ℝ := original_height / 2
  let shaded_area : ℝ := shaded_base * shaded_height / 2
  let percentage : ℝ := shaded_area / original_area * 100
  shaded_area = 3 * (Real.sqrt 3) / 2 ∧ percentage = 37.5 := by
  sorry


end shaded_triangle_area_and_percentage_l3125_312555


namespace freshman_class_size_l3125_312577

theorem freshman_class_size : ∃! n : ℕ, n < 500 ∧ n % 23 = 22 ∧ n % 21 = 14 ∧ n = 413 := by
  sorry

end freshman_class_size_l3125_312577


namespace surface_area_increase_percentage_l3125_312589

/-- The percentage increase in surface area when placing a hemispherical cap on a sphere -/
theorem surface_area_increase_percentage (R : ℝ) (R_pos : R > 0) : 
  let sphere_area := 4 * Real.pi * R^2
  let cap_radius := R * Real.sqrt 3 / 2
  let cap_area := 2 * Real.pi * cap_radius^2
  let covered_cap_height := R / 2
  let covered_cap_area := 2 * Real.pi * R * covered_cap_height
  let area_increase := cap_area - covered_cap_area
  area_increase / sphere_area * 100 = 12.5 :=
sorry

end surface_area_increase_percentage_l3125_312589


namespace anushas_share_multiple_l3125_312574

/-- Proves that the multiple of Anusha's share is 12 given the problem conditions -/
theorem anushas_share_multiple (anusha babu esha : ℕ) (m : ℕ) : 
  anusha = 84 →
  m * anusha = 8 * babu →
  8 * babu = 6 * esha →
  anusha + babu + esha = 378 →
  m = 12 := by
  sorry

end anushas_share_multiple_l3125_312574


namespace intersection_sum_l3125_312502

/-- Given two lines that intersect at (2,1), prove that a + b = 2 -/
theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 1 + a) → 
  (1 = (1/3) * 2 + b) → 
  a + b = 2 := by
  sorry

end intersection_sum_l3125_312502


namespace elmer_milton_ratio_l3125_312504

-- Define the daily food intake for each animal
def penelope_intake : ℚ := 20
def greta_intake : ℚ := penelope_intake / 10
def milton_intake : ℚ := greta_intake / 100
def elmer_intake : ℚ := penelope_intake + 60

-- Theorem statement
theorem elmer_milton_ratio : 
  elmer_intake / milton_intake = 4000 := by sorry

end elmer_milton_ratio_l3125_312504


namespace not_both_perfect_cubes_l3125_312507

theorem not_both_perfect_cubes (n : ℕ) : 
  ¬(∃ a b : ℕ, (n + 2 = a^3) ∧ (n^2 + n + 1 = b^3)) := by
  sorry

end not_both_perfect_cubes_l3125_312507


namespace sum_of_roots_tangent_equation_l3125_312538

theorem sum_of_roots_tangent_equation : 
  ∃ (x₁ x₂ : ℝ), 
    0 < x₁ ∧ x₁ < π ∧
    0 < x₂ ∧ x₂ < π ∧
    (Real.tan x₁)^2 - 5 * Real.tan x₁ + 6 = 0 ∧
    (Real.tan x₂)^2 - 5 * Real.tan x₂ + 6 = 0 ∧
    x₁ + x₂ = Real.arctan 3 + Real.arctan 2 :=
by sorry

end sum_of_roots_tangent_equation_l3125_312538


namespace equal_area_partition_pentagon_l3125_312548

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon (A B C D E : ℝ × ℝ) : Prop := True

-- Define the area of a triangle
def TriangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

-- State that a point is inside a pentagon
def InsidePentagon (M : ℝ × ℝ) (A B C D E : ℝ × ℝ) : Prop := sorry

-- The main theorem
theorem equal_area_partition_pentagon 
  (A B C D E : ℝ × ℝ) 
  (h_pentagon : Pentagon A B C D E)
  (h_convex : sorry) -- Additional hypothesis for convexity
  (h_equal_areas : TriangleArea A B C = TriangleArea B C D ∧ 
                   TriangleArea B C D = TriangleArea C D E ∧ 
                   TriangleArea C D E = TriangleArea D E A ∧ 
                   TriangleArea D E A = TriangleArea E A B) :
  ∃ M : ℝ × ℝ, 
    InsidePentagon M A B C D E ∧
    TriangleArea M A B = TriangleArea M B C ∧
    TriangleArea M B C = TriangleArea M C D ∧
    TriangleArea M C D = TriangleArea M D E ∧
    TriangleArea M D E = TriangleArea M E A :=
sorry

end equal_area_partition_pentagon_l3125_312548


namespace sum_inequality_l3125_312547

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a ≥ 12) : a + b + c ≥ 6 := by
  sorry

end sum_inequality_l3125_312547


namespace simplify_fraction_l3125_312525

theorem simplify_fraction (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  15 * x^2 * z^3 / (9 * x * y * z^2) = 20 := by
  sorry

end simplify_fraction_l3125_312525


namespace vectors_perpendicular_if_sum_norm_eq_diff_norm_l3125_312557

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vectors_perpendicular_if_sum_norm_eq_diff_norm 
  (a b : ℝ × ℝ) (h : ‖a + b‖ = ‖a - b‖) : 
  angle_between_vectors a b = π / 2 := by sorry

end vectors_perpendicular_if_sum_norm_eq_diff_norm_l3125_312557


namespace nelly_earnings_per_night_l3125_312519

/-- Calculates Nelly's earnings per night babysitting given the pizza party conditions -/
theorem nelly_earnings_per_night (total_people : ℕ) (pizza_cost : ℚ) (people_per_pizza : ℕ) (babysitting_nights : ℕ) : 
  total_people = 15 →
  pizza_cost = 12 →
  people_per_pizza = 3 →
  babysitting_nights = 15 →
  (total_people : ℚ) / (people_per_pizza : ℚ) * pizza_cost / (babysitting_nights : ℚ) = 4 := by
  sorry

#check nelly_earnings_per_night

end nelly_earnings_per_night_l3125_312519


namespace circle_radius_from_area_l3125_312510

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 250 * Real.pi) :
  A = Real.pi * r^2 → r = 5 * Real.sqrt 10 := by
  sorry

end circle_radius_from_area_l3125_312510


namespace crate_stacking_ways_l3125_312521

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to stack crates to achieve a specific height -/
def countStackingWays (dimensions : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of ways to stack 11 crates to 47ft -/
theorem crate_stacking_ways :
  let dimensions : CrateDimensions := { length := 3, width := 4, height := 5 }
  countStackingWays dimensions 11 47 = 2277 := by
  sorry

end crate_stacking_ways_l3125_312521


namespace arithmetic_sequence_sum_l3125_312562

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 10 and S_20 = 40, prove S_30 = 90 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 10 = 10) (h2 : a.S 20 = 40) : a.S 30 = 90 := by
  sorry

end arithmetic_sequence_sum_l3125_312562


namespace safety_rent_a_truck_cost_per_mile_l3125_312505

/-- The cost per mile for Safety Rent A Truck -/
def safety_cost_per_mile : ℝ := sorry

/-- The base cost for Safety Rent A Truck -/
def safety_base_cost : ℝ := 41.95

/-- The base cost for City Rentals -/
def city_base_cost : ℝ := 38.95

/-- The cost per mile for City Rentals -/
def city_cost_per_mile : ℝ := 0.31

/-- The number of miles for which the total costs are equal -/
def equal_cost_miles : ℝ := 150.0

theorem safety_rent_a_truck_cost_per_mile :
  safety_base_cost + equal_cost_miles * safety_cost_per_mile =
  city_base_cost + equal_cost_miles * city_cost_per_mile ∧
  safety_cost_per_mile = 0.29 := by sorry

end safety_rent_a_truck_cost_per_mile_l3125_312505


namespace last_digit_389_base5_is_4_l3125_312543

-- Define a function to convert a decimal number to base-5
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else loop (m / 5) ((m % 5) :: acc)
    loop n []

-- State the theorem
theorem last_digit_389_base5_is_4 :
  (toBase5 389).getLast? = some 4 :=
sorry

end last_digit_389_base5_is_4_l3125_312543


namespace triangle_properties_l3125_312535

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0)
  (h2 : t.c = 2)
  (h3 : t.a + t.b = t.a * t.b) : 
  t.C = Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end triangle_properties_l3125_312535


namespace min_sum_dimensions_l3125_312516

theorem min_sum_dimensions (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 → a * b * c = 3003 → a + b + c ≥ 57 := by
  sorry

end min_sum_dimensions_l3125_312516


namespace x_greater_than_half_l3125_312581

theorem x_greater_than_half (x : ℝ) (h1 : 1 / x^2 < 4) (h2 : 1 / x > -2) (h3 : x ≠ 0) : x > 1/2 := by
  sorry

end x_greater_than_half_l3125_312581


namespace car_cost_equation_l3125_312566

/-- Proves that the original cost of the car satisfies the given equation -/
theorem car_cost_equation (repair_cost selling_price profit_percent : ℝ) 
  (h1 : repair_cost = 15000)
  (h2 : selling_price = 64900)
  (h3 : profit_percent = 13.859649122807017) :
  ∃ C : ℝ, (1 + profit_percent / 100) * C = selling_price - repair_cost :=
sorry

end car_cost_equation_l3125_312566


namespace dice_probability_l3125_312508

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a 'low' number (1-8) -/
def prob_low : ℚ := 2/3

/-- The probability of rolling a 'mid' or 'high' number (9-12) -/
def prob_mid_high : ℚ := 1/3

/-- The number of ways to choose 2 dice out of 5 -/
def choose_two_from_five : ℕ := 10

/-- The probability of the desired outcome -/
theorem dice_probability : 
  (choose_two_from_five : ℚ) * prob_low^2 * prob_mid_high^3 = 40/243 := by sorry

end dice_probability_l3125_312508


namespace impossible_closed_line_l3125_312565

/-- Represents a prism with a given number of lateral edges and total edges. -/
structure Prism where
  lateral_edges : ℕ
  total_edges : ℕ

/-- Represents the possibility of forming a closed broken line from translated edges of a prism. -/
def can_form_closed_line (p : Prism) : Prop :=
  ∃ (arrangement : List ℝ), 
    arrangement.length = p.total_edges ∧ 
    arrangement.sum = 0 ∧
    (∀ i ∈ arrangement, i = 0 ∨ i = 1 ∨ i = -1)

/-- Theorem stating that it's impossible to form a closed broken line from the given prism's edges. -/
theorem impossible_closed_line (p : Prism) 
  (h1 : p.lateral_edges = 373) 
  (h2 : p.total_edges = 1119) : 
  ¬ can_form_closed_line p := by
  sorry

end impossible_closed_line_l3125_312565


namespace shirt_cost_l3125_312533

theorem shirt_cost (num_shirts : ℕ) (num_pants : ℕ) (pant_cost : ℕ) (total_cost : ℕ) : 
  num_shirts = 10 →
  num_pants = num_shirts / 2 →
  pant_cost = 8 →
  total_cost = 100 →
  num_shirts * (total_cost - num_pants * pant_cost) / num_shirts = 6 :=
by
  sorry

end shirt_cost_l3125_312533


namespace exactly_three_ways_l3125_312540

/-- The sum of consecutive integers from a to b, inclusive -/
def consecutiveSum (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- The predicate that checks if a pair (a, b) satisfies the conditions -/
def isValidPair (a b : ℕ) : Prop :=
  a < b ∧ consecutiveSum a b = 91

/-- The theorem stating that there are exactly 3 valid pairs -/
theorem exactly_three_ways :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 3 ∧ ∀ p, p ∈ s ↔ isValidPair p.1 p.2 :=
sorry

end exactly_three_ways_l3125_312540


namespace sequence_sum_formula_l3125_312584

def sequence_sum (n : ℕ) : ℕ → ℕ
| 0 => 5
| m + 1 => 2 * sequence_sum n m + (m + 1) + 5

theorem sequence_sum_formula (n : ℕ) :
  sequence_sum n n = 6 * 2^n - (n + 6) := by
  sorry

end sequence_sum_formula_l3125_312584


namespace infinite_decimal_sqrt_l3125_312545

theorem infinite_decimal_sqrt (x y : ℕ) : 
  x ∈ Finset.range 9 → y ∈ Finset.range 9 → 
  (Real.sqrt (x / 9 : ℝ) = y / 9) ↔ ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) := by
sorry

end infinite_decimal_sqrt_l3125_312545


namespace triangle_third_side_l3125_312571

theorem triangle_third_side (a b : ℝ) (h1 : a = 3.14) (h2 : b = 0.67) : 
  ∃ c : ℕ, c = 3 ∧ 
    a + b > c ∧
    a + c > b ∧
    b + c > a := by
  sorry

end triangle_third_side_l3125_312571


namespace solution_set_characterization_l3125_312568

def valid_digit (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

def base_10_value (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

def base_7_value (x y z : ℕ) : ℕ := 49 * x + 7 * y + z

def satisfies_equation (x y z : ℕ) : Prop :=
  base_10_value x y z = 2 * base_7_value x y z

def valid_triple (x y z : ℕ) : Prop :=
  valid_digit x ∧ valid_digit y ∧ valid_digit z ∧ satisfies_equation x y z

theorem solution_set_characterization :
  {t : ℕ × ℕ × ℕ | valid_triple t.1 t.2.1 t.2.2} =
  {(3,1,2), (5,2,2), (4,1,4), (6,2,4), (5,1,6)} := by sorry

end solution_set_characterization_l3125_312568


namespace a_minus_b_values_l3125_312522

theorem a_minus_b_values (a b : ℝ) (h1 : a < b) (h2 : |a| = 6) (h3 : |b| = 3) :
  a - b = -9 ∨ a - b = -3 := by
sorry

end a_minus_b_values_l3125_312522


namespace reflection_across_y_axis_l3125_312512

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem reflection_across_y_axis :
  let P : Point2D := { x := -3, y := 5 }
  reflectAcrossYAxis P = { x := 3, y := 5 } := by sorry

end reflection_across_y_axis_l3125_312512


namespace annual_savings_l3125_312517

/-- Represents the parking garage rental rates and conditions -/
structure ParkingGarage where
  regular_peak_weekly : ℕ
  regular_nonpeak_weekly : ℕ
  regular_peak_monthly : ℕ
  regular_nonpeak_monthly : ℕ
  large_peak_weekly : ℕ
  large_nonpeak_weekly : ℕ
  large_peak_monthly : ℕ
  large_nonpeak_monthly : ℕ
  holiday_surcharge : ℕ
  nonpeak_weeks : ℕ
  peak_holiday_weeks : ℕ
  total_weeks : ℕ

/-- Calculates the annual cost of renting a large space weekly -/
def weekly_cost (pg : ParkingGarage) : ℕ :=
  pg.large_nonpeak_weekly * pg.nonpeak_weeks +
  pg.large_peak_weekly * (pg.total_weeks - pg.nonpeak_weeks - pg.peak_holiday_weeks) +
  (pg.large_peak_weekly + pg.holiday_surcharge) * pg.peak_holiday_weeks

/-- Calculates the annual cost of renting a large space monthly -/
def monthly_cost (pg : ParkingGarage) : ℕ :=
  pg.large_nonpeak_monthly * (pg.nonpeak_weeks / 4) +
  pg.large_peak_monthly * ((pg.total_weeks - pg.nonpeak_weeks) / 4)

/-- Theorem: The annual savings from renting monthly instead of weekly is $124 -/
theorem annual_savings (pg : ParkingGarage) : weekly_cost pg - monthly_cost pg = 124 :=
  by
    have h1 : pg.regular_peak_weekly = 10 := by sorry
    have h2 : pg.regular_nonpeak_weekly = 8 := by sorry
    have h3 : pg.regular_peak_monthly = 40 := by sorry
    have h4 : pg.regular_nonpeak_monthly = 35 := by sorry
    have h5 : pg.large_peak_weekly = 12 := by sorry
    have h6 : pg.large_nonpeak_weekly = 10 := by sorry
    have h7 : pg.large_peak_monthly = 48 := by sorry
    have h8 : pg.large_nonpeak_monthly = 42 := by sorry
    have h9 : pg.holiday_surcharge = 2 := by sorry
    have h10 : pg.nonpeak_weeks = 16 := by sorry
    have h11 : pg.peak_holiday_weeks = 6 := by sorry
    have h12 : pg.total_weeks = 52 := by sorry
    sorry

end annual_savings_l3125_312517


namespace train_distance_problem_l3125_312506

/-- Proves that the distance between two stations is 450 km, given the conditions of the train problem. -/
theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : v2 > v1) (h4 : d > 0) :
  let t := d / v1
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 50 → d1 + d2 = 450 := by sorry

end train_distance_problem_l3125_312506


namespace front_view_correct_l3125_312515

def ColumnHeights := List Nat

def frontView (columns : List ColumnHeights) : List Nat :=
  columns.map (List.foldl max 0)

theorem front_view_correct (columns : List ColumnHeights) :
  frontView columns = [3, 4, 5, 2] :=
by
  -- The proof would go here
  sorry

#eval frontView [[3, 2], [1, 4, 2], [5], [2, 1]]

end front_view_correct_l3125_312515


namespace simplify_expression_l3125_312534

theorem simplify_expression (m n : ℝ) : 
  m - (m^2 * n + 3 * m - 4 * n) + (2 * n * m^2 - 3 * n) = m^2 * n - 2 * m + n := by
  sorry

end simplify_expression_l3125_312534


namespace a_power_value_l3125_312595

theorem a_power_value (a n : ℝ) (h : a^(2*n) = 3) : 2*a^(6*n) - 1 = 53 := by
  sorry

end a_power_value_l3125_312595


namespace specific_extended_parallelepiped_volume_l3125_312550

/-- The volume of the set of points that are inside or within one unit of a rectangular parallelepiped -/
def extended_parallelepiped_volume (l w h : ℝ) : ℝ :=
  (l + 2) * (w + 2) * (h + 2) - (l * w * h)

/-- The theorem stating the volume of the specific extended parallelepiped -/
theorem specific_extended_parallelepiped_volume :
  extended_parallelepiped_volume 5 6 7 = (1272 + 58 * Real.pi) / 3 := by
  sorry

end specific_extended_parallelepiped_volume_l3125_312550


namespace exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots_l3125_312592

/-- A cubic polynomial -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a*x^3 + b*x^2 + c*x + d

/-- The derivative of a cubic polynomial -/
def DerivativeCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ 3*a*x^2 + 2*b*x + c

/-- All roots of a function are positive -/
def AllRootsPositive (f : ℝ → ℝ) : Prop := ∀ x, f x = 0 → x > 0

/-- All roots of a function are negative -/
def AllRootsNegative (f : ℝ → ℝ) : Prop := ∀ x, f x = 0 → x < 0

/-- A function has at least one unique root -/
def HasUniqueRoot (f : ℝ → ℝ) : Prop := ∃ x, f x = 0 ∧ ∀ y, f y = 0 → y = x

theorem exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots :
  ∃ (a b c d : ℝ), 
    let P := CubicPolynomial a b c d
    let P' := DerivativeCubicPolynomial (3*a) (2*b) c
    AllRootsPositive P ∧
    AllRootsNegative P' ∧
    HasUniqueRoot P ∧
    HasUniqueRoot P' :=
sorry

end exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots_l3125_312592


namespace lloyds_hourly_rate_l3125_312530

-- Define Lloyd's regular work hours
def regular_hours : ℝ := 7.5

-- Define Lloyd's overtime multiplier
def overtime_multiplier : ℝ := 1.5

-- Define Lloyd's actual work hours on the given day
def actual_hours : ℝ := 10.5

-- Define Lloyd's total earnings for the day
def total_earnings : ℝ := 42

-- Theorem to prove Lloyd's hourly rate
theorem lloyds_hourly_rate :
  ∃ (rate : ℝ),
    rate * regular_hours + 
    (actual_hours - regular_hours) * rate * overtime_multiplier = total_earnings ∧
    rate = 3.5 := by
  sorry

end lloyds_hourly_rate_l3125_312530


namespace star_seven_three_l3125_312514

def star (a b : ℝ) : ℝ := 2*a + 5*b - a*b + 2

theorem star_seven_three : star 7 3 = 10 := by sorry

end star_seven_three_l3125_312514


namespace optimal_plan_is_best_l3125_312503

/-- Represents a bus purchasing plan -/
structure BusPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a bus plan is valid according to the given constraints -/
def isValidPlan (plan : BusPlan) : Prop :=
  plan.typeA + plan.typeB = 10 ∧
  100 * plan.typeA + 150 * plan.typeB ≤ 1200 ∧
  60 * plan.typeA + 100 * plan.typeB ≥ 680

/-- Calculates the total cost of a bus plan in million RMB -/
def totalCost (plan : BusPlan) : ℕ :=
  100 * plan.typeA + 150 * plan.typeB

/-- The optimal bus purchasing plan -/
def optimalPlan : BusPlan :=
  { typeA := 8, typeB := 2 }

/-- Theorem stating that the optimal plan is valid and minimizes the total cost -/
theorem optimal_plan_is_best :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
sorry

#check optimal_plan_is_best

end optimal_plan_is_best_l3125_312503


namespace geometry_propositions_l3125_312553

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) :=
sorry

end geometry_propositions_l3125_312553


namespace even_power_plus_one_all_digits_equal_l3125_312569

def is_all_digits_equal (n : ℕ) : Prop :=
  ∃ d : ℕ, ∀ k : ℕ, k < (Nat.log 10 n + 1) → (n / 10^k) % 10 = d

def solution_set : Set (ℕ × ℕ) :=
  {(2, 2), (2, 3), (2, 5), (6, 5)}

theorem even_power_plus_one_all_digits_equal :
  ∀ a b : ℕ,
    a ≥ 2 →
    b ≥ 2 →
    Even a →
    is_all_digits_equal (a^b + 1) →
    (a, b) ∈ solution_set :=
by sorry

end even_power_plus_one_all_digits_equal_l3125_312569


namespace intersection_line_ellipse_part1_intersection_line_ellipse_part2_l3125_312500

noncomputable section

-- Define the line and ellipse
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1
def ellipse (a : ℝ) (x y : ℝ) : Prop := 3 * x^2 + y^2 = a

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the area of a triangle given the coordinates of its vertices
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem intersection_line_ellipse_part1 (a : ℝ) (x1 y1 x2 y2 : ℝ) :
  line 1 x1 = y1 →
  line 1 x2 = y2 →
  ellipse a x1 y1 →
  ellipse a x2 y2 →
  distance x1 y1 x2 y2 = Real.sqrt 10 / 2 →
  a = 2 := by sorry

theorem intersection_line_ellipse_part2 (k a : ℝ) (x1 y1 x2 y2 : ℝ) :
  k ≠ 0 →
  line k x1 = y1 →
  line k x2 = y2 →
  ellipse a x1 y1 →
  ellipse a x2 y2 →
  x1 = -2 * x2 →
  ∃ (max_area : ℝ),
    (∀ (k' a' : ℝ) (x1' y1' x2' y2' : ℝ),
      k' ≠ 0 →
      line k' x1' = y1' →
      line k' x2' = y2' →
      ellipse a' x1' y1' →
      ellipse a' x2' y2' →
      x1' = -2 * x2' →
      triangle_area 0 0 x1' y1' x2' y2' ≤ max_area) ∧
    max_area = Real.sqrt 3 / 2 ∧
    a = 5 := by sorry

end intersection_line_ellipse_part1_intersection_line_ellipse_part2_l3125_312500


namespace restaurant_bill_proof_l3125_312559

theorem restaurant_bill_proof : 
  ∀ (total_friends : ℕ) (paying_friends : ℕ) (extra_payment : ℚ),
    total_friends = 12 →
    paying_friends = 10 →
    extra_payment = 3 →
    ∃ (bill : ℚ), 
      bill = paying_friends * (bill / total_friends + extra_payment) ∧
      bill = 180 := by
  sorry

end restaurant_bill_proof_l3125_312559


namespace equilateral_triangle_circles_l3125_312551

theorem equilateral_triangle_circles (rA rB rC : ℝ) : 
  rA < rB ∧ rB < rC →  -- radii form increasing sequence
  ∃ (d : ℝ), rB = rA + d ∧ rC = rA + 2*d →  -- arithmetic sequence
  6 - (rA + rB) = 3.5 →  -- shortest distance between circles A and B
  6 - (rA + rC) = 3 →  -- shortest distance between circles A and C
  (1/6) * (π * rA^2 + π * rB^2 + π * rC^2) = 29*π/24 := by
sorry

end equilateral_triangle_circles_l3125_312551


namespace exist_numbers_not_triangle_l3125_312579

/-- Theorem: There exist natural numbers a and b, both greater than 1000,
    such that for any perfect square c, the triple (a, b, c) does not
    satisfy the triangle inequality. -/
theorem exist_numbers_not_triangle : ∃ a b : ℕ,
  a > 1000 ∧ b > 1000 ∧
  ∀ c : ℕ, (∃ d : ℕ, c = d * d) →
    ¬(a + b > c ∧ b + c > a ∧ a + c > b) := by
  sorry

end exist_numbers_not_triangle_l3125_312579


namespace inequality_holds_iff_a_geq_neg_two_l3125_312531

theorem inequality_holds_iff_a_geq_neg_two :
  ∀ a : ℝ, (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by sorry

end inequality_holds_iff_a_geq_neg_two_l3125_312531


namespace sum_of_legs_special_triangle_l3125_312591

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- shorter leg
  b : ℕ  -- longer leg
  c : ℕ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2
  consecutive_even : b = a + 2

/-- The sum of legs of a right triangle with hypotenuse 50 and consecutive even legs is 70 -/
theorem sum_of_legs_special_triangle :
  ∀ (t : RightTriangle), t.c = 50 → t.a + t.b = 70 := by
  sorry

#check sum_of_legs_special_triangle

end sum_of_legs_special_triangle_l3125_312591


namespace expected_balls_original_positions_l3125_312560

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The probability that a ball is in its original position after Chris and Silva's actions -/
def prob_original_position : ℚ := 25 / 49

/-- The expected number of balls in their original positions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

/-- Theorem stating the expected number of balls in their original positions -/
theorem expected_balls_original_positions :
  expected_original_positions = 175 / 49 := by sorry

end expected_balls_original_positions_l3125_312560


namespace transform_point_l3125_312541

def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem transform_point (p : ℝ × ℝ) :
  reflectX (rotate180 p) = (p.1, -p.2) := by sorry

end transform_point_l3125_312541


namespace zoo_field_trip_l3125_312554

theorem zoo_field_trip (students : ℕ) (adults : ℕ) (vans : ℕ) : 
  students = 12 → adults = 3 → vans = 3 → (students + adults) / vans = 5 := by
  sorry

end zoo_field_trip_l3125_312554


namespace y_derivative_l3125_312537

noncomputable def y (x : ℝ) : ℝ := (1 - x^2) / Real.exp x

theorem y_derivative (x : ℝ) : 
  deriv y x = (x^2 - 2*x - 1) / Real.exp x :=
sorry

end y_derivative_l3125_312537


namespace triangular_number_gcd_bound_l3125_312586

/-- The nth triangular number -/
def T (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The statement to be proved -/
theorem triangular_number_gcd_bound :
  (∀ n : ℕ, n > 0 → Nat.gcd (8 * T n) (n + 1) ≤ 4) ∧
  (∃ n : ℕ, n > 0 ∧ Nat.gcd (8 * T n) (n + 1) = 4) := by
  sorry

end triangular_number_gcd_bound_l3125_312586


namespace vertex_on_x_axis_l3125_312576

/-- The parabola equation -/
def parabola (x d : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex -/
def vertex_y (d : ℝ) : ℝ := parabola vertex_x d

theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end vertex_on_x_axis_l3125_312576
