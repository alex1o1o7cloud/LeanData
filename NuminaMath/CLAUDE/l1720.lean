import Mathlib

namespace root_implies_product_bound_l1720_172012

theorem root_implies_product_bound (a b : ℝ) 
  (h : (a + b + a) * (a + b + b) = 9) : a * b ≤ 1 := by
  sorry

end root_implies_product_bound_l1720_172012


namespace midpoint_sum_scaled_triangle_l1720_172045

theorem midpoint_sum_scaled_triangle (a b c : ℝ) (h : a + b + c = 18) :
  let scaled_midpoint_sum := (2*a + 2*b) + (2*a + 2*c) + (2*b + 2*c)
  scaled_midpoint_sum = 36 := by
  sorry

end midpoint_sum_scaled_triangle_l1720_172045


namespace solution_set_equality_l1720_172065

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Define the set of x values satisfying both conditions
def S : Set ℝ := {x : ℝ | f x > 0 ∧ x < 3}

-- Theorem statement
theorem solution_set_equality : S = Set.Ioi (-1) ∪ Set.Ioo 1 3 :=
sorry

end solution_set_equality_l1720_172065


namespace maggie_bouncy_balls_l1720_172091

/-- The number of packs of green bouncy balls Maggie gave away -/
def green_packs_given_away : ℝ := 4.0

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs_bought : ℝ := 8.0

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℝ := 10.0

/-- The total number of bouncy balls Maggie kept -/
def balls_kept : ℕ := 80

theorem maggie_bouncy_balls :
  green_packs_given_away = 
    ((yellow_packs_bought + green_packs_bought) * balls_per_pack - balls_kept) / balls_per_pack :=
by sorry

end maggie_bouncy_balls_l1720_172091


namespace symmetric_points_sum_l1720_172080

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposites -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_about_x_axis (a, 4) (-2, b) → a + b = -6 := by
  sorry

end symmetric_points_sum_l1720_172080


namespace pencil_sale_problem_l1720_172085

theorem pencil_sale_problem (total_students : ℕ) (total_pencils : ℕ) 
  (h_total_students : total_students = 10)
  (h_total_pencils : total_pencils = 24)
  (h_first_two : 2 * 2 = 4)  -- First two students bought 2 pencils each
  (h_last_two : 2 * 1 = 2)   -- Last two students bought 1 pencil each
  : ∃ (middle_group : ℕ), 
    middle_group = 6 ∧ 
    middle_group * 3 + 4 + 2 = total_pencils ∧ 
    2 + middle_group + 2 = total_students :=
by sorry

end pencil_sale_problem_l1720_172085


namespace suv_wash_price_l1720_172051

/-- The price of a car wash in dollars -/
def car_price : ℕ := 5

/-- The price of a truck wash in dollars -/
def truck_price : ℕ := 6

/-- The total amount raised in dollars -/
def total_raised : ℕ := 100

/-- The number of SUVs washed -/
def num_suvs : ℕ := 5

/-- The number of trucks washed -/
def num_trucks : ℕ := 5

/-- The number of cars washed -/
def num_cars : ℕ := 7

/-- The price of an SUV wash in dollars -/
def suv_price : ℕ := 9

theorem suv_wash_price :
  car_price * num_cars + truck_price * num_trucks + suv_price * num_suvs = total_raised :=
by sorry

end suv_wash_price_l1720_172051


namespace shortest_paths_correct_l1720_172067

def shortest_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem shortest_paths_correct (m n : ℕ) : 
  shortest_paths m n = Nat.choose (m + n) m :=
by sorry

end shortest_paths_correct_l1720_172067


namespace extreme_value_inequality_l1720_172087

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - (1/2) * a * x^2 + (4-a) * x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 4 / x - a * x + (4-a)

theorem extreme_value_inequality (a : ℝ) (x₀ x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx₀ : x₀ > 0) 
  (hx₁ : x₁ > 0) 
  (hx₂ : x₂ > 0) 
  (h_order : x₁ < x₂) 
  (h_extreme : ∃ x, x > 0 ∧ ∀ y, y > 0 → f a x ≥ f a y) 
  (h_mean_value : f a x₁ - f a x₂ = f_deriv a x₀ * (x₁ - x₂)) :
  x₁ + x₂ > 2 * x₀ := by
  sorry

end extreme_value_inequality_l1720_172087


namespace square_perimeter_l1720_172078

/-- The perimeter of a square is equal to four times its side length. -/
theorem square_perimeter (side : ℝ) (h : side = 13) : 4 * side = 52 := by
  sorry

end square_perimeter_l1720_172078


namespace pet_store_siamese_cats_l1720_172089

/-- The number of siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 19

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 45

/-- The number of cats sold during the sale -/
def cats_sold : ℕ := 56

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

theorem pet_store_siamese_cats :
  initial_siamese_cats = 19 :=
by
  have h1 : initial_siamese_cats + initial_house_cats = initial_siamese_cats + 45 := by rfl
  have h2 : initial_siamese_cats + initial_house_cats - cats_sold = cats_remaining :=
    by sorry
  sorry

end pet_store_siamese_cats_l1720_172089


namespace equation_solution_l1720_172009

theorem equation_solution :
  ∃ x : ℚ, (1 / 3 + 1 / x = 2 / 3) ∧ (x = 3) :=
by
  sorry

end equation_solution_l1720_172009


namespace min_x_coordinate_midpoint_l1720_172017

/-- Given a line segment AB of length m on the right branch of the hyperbola x²/a² - y²/b² = 1,
    where m > 2b²/a, the minimum x-coordinate of the midpoint M of AB is a(m + 2a) / (2√(a² + b²)). -/
theorem min_x_coordinate_midpoint (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 2 * b^2 / a) :
  let min_x := a * (m + 2 * a) / (2 * Real.sqrt (a^2 + b^2))
  ∀ (x y z w : ℝ),
    x^2 / a^2 - y^2 / b^2 = 1 →
    z^2 / a^2 - w^2 / b^2 = 1 →
    (z - x)^2 + (w - y)^2 = m^2 →
    x > 0 →
    z > 0 →
    (x + z) / 2 ≥ min_x :=
by sorry

end min_x_coordinate_midpoint_l1720_172017


namespace correct_guess_probability_l1720_172084

/-- Represents a six-digit password with an unknown last digit -/
structure Password :=
  (first_five : Nat)
  (last_digit : Nat)

/-- The set of possible last digits -/
def possible_last_digits : Finset Nat := Finset.range 10

/-- The probability of guessing the correct password on the first try -/
def guess_probability (p : Password) : ℚ :=
  1 / (Finset.card possible_last_digits : ℚ)

theorem correct_guess_probability (p : Password) :
  p.last_digit ∈ possible_last_digits →
  guess_probability p = 1 / 10 := by
  sorry

end correct_guess_probability_l1720_172084


namespace melted_ice_cream_depth_l1720_172048

/-- Given a spherical scoop of ice cream with radius 3 inches that melts into a conical shape with radius 9 inches, prove that the height of the resulting cone is 4/3 inches, assuming constant density. -/
theorem melted_ice_cream_depth (sphere_radius : ℝ) (cone_radius : ℝ) (cone_height : ℝ) : 
  sphere_radius = 3 →
  cone_radius = 9 →
  (4 / 3) * Real.pi * sphere_radius^3 = (1 / 3) * Real.pi * cone_radius^2 * cone_height →
  cone_height = 4 / 3 := by
  sorry

#check melted_ice_cream_depth

end melted_ice_cream_depth_l1720_172048


namespace distribute_six_balls_three_boxes_l1720_172064

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 28 := by
  sorry

end distribute_six_balls_three_boxes_l1720_172064


namespace unique_satisfying_function_l1720_172007

/-- A function satisfying the given functional equation. -/
def SatisfyingFunction (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1)) - n

/-- The theorem stating that the only function satisfying the equation is f(p) = 1 - p. -/
theorem unique_satisfying_function :
  ∀ f : ℤ → ℤ, SatisfyingFunction f ↔ (∀ p : ℤ, f p = 1 - p) :=
sorry

end unique_satisfying_function_l1720_172007


namespace triangle_groups_count_l1720_172003

/-- The number of groups of 3 points from 12 points that can form a triangle -/
def triangle_groups : ℕ := 200

/-- The total number of points -/
def total_points : ℕ := 12

/-- Theorem stating that the number of groups of 3 points from 12 points 
    that can form a triangle is equal to 200 -/
theorem triangle_groups_count : 
  triangle_groups = 200 ∧ total_points = 12 := by sorry

end triangle_groups_count_l1720_172003


namespace inequality_condition_l1720_172060

theorem inequality_condition (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  |A - B + C| ≤ 2 * Real.sqrt (A * C) :=
by sorry

end inequality_condition_l1720_172060


namespace day_statistics_order_l1720_172053

/-- Represents the frequency distribution of days in a non-leap year -/
def day_frequency (n : ℕ) : ℕ :=
  if n ≤ 28 then 12
  else if n ≤ 30 then 11
  else if n = 31 then 6
  else 0

/-- The total number of days in a non-leap year -/
def total_days : ℕ := 365

/-- The median of modes for the day distribution -/
def median_of_modes : ℚ := 14.5

/-- The median of the day distribution -/
def median : ℕ := 13

/-- The mean of the day distribution -/
def mean : ℚ := 5707 / 365

theorem day_statistics_order :
  median_of_modes < median ∧ (median : ℚ) < mean :=
sorry

end day_statistics_order_l1720_172053


namespace center_digit_is_two_l1720_172016

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that returns the tens digit of a three-digit number --/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- The set of available digits --/
def digit_set : Finset ℕ := {2, 3, 4, 5, 6}

/-- A proposition stating that a number uses only digits from the digit set --/
def uses_digit_set (n : ℕ) : Prop :=
  (n / 100 ∈ digit_set) ∧ (tens_digit n ∈ digit_set) ∧ (n % 10 ∈ digit_set)

theorem center_digit_is_two :
  ∀ (a b : ℕ),
    a ≠ b
    → a ≥ 100 ∧ a < 1000
    → b ≥ 100 ∧ b < 1000
    → is_perfect_square a
    → is_perfect_square b
    → uses_digit_set a
    → uses_digit_set b
    → (Finset.card {a / 100, tens_digit a, a % 10, b / 100, tens_digit b, b % 10} = 5)
    → (tens_digit a = 2 ∨ tens_digit b = 2) :=
  sorry

end center_digit_is_two_l1720_172016


namespace parabola_solution_l1720_172083

/-- The parabola C: y^2 = 6x with focus F, and a point M(x,y) on C where |MF| = 5/2 and y > 0 -/
def parabola_problem (x y : ℝ) : Prop :=
  y^2 = 6*x ∧ y > 0 ∧ (x + 3/2)^2 + y^2 = (5/2)^2

/-- The coordinates of point M are (1, √6) -/
theorem parabola_solution :
  ∀ x y : ℝ, parabola_problem x y → x = 1 ∧ y = Real.sqrt 6 :=
by sorry

end parabola_solution_l1720_172083


namespace income_expenditure_ratio_l1720_172063

def income : ℕ := 19000
def savings : ℕ := 3800
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio :
  (income : ℚ) / (expenditure : ℚ) = 5 / 4 := by sorry

end income_expenditure_ratio_l1720_172063


namespace order_of_f_l1720_172035

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_increasing_nonneg : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- Theorem statement
theorem order_of_f : f (-2) < f 3 ∧ f 3 < f (-π) :=
sorry

end order_of_f_l1720_172035


namespace isosceles_triangle_side_lengths_l1720_172011

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle_side_lengths (x : ℝ) :
  (is_isosceles_triangle (x + 3) (2*x + 1) 11 ∧
   satisfies_triangle_inequality (x + 3) (2*x + 1) 11) →
  (x = 8 ∨ x = 5) :=
by sorry

end isosceles_triangle_side_lengths_l1720_172011


namespace chef_pies_l1720_172010

theorem chef_pies (apple_pies pecan_pies pumpkin_pies total_pies : ℕ) 
  (h1 : apple_pies = 2)
  (h2 : pecan_pies = 4)
  (h3 : total_pies = 13)
  (h4 : total_pies = apple_pies + pecan_pies + pumpkin_pies) :
  pumpkin_pies = 7 := by
  sorry

end chef_pies_l1720_172010


namespace tian_ji_winning_probability_l1720_172068

/-- Represents the horses of each competitor -/
inductive Horse : Type
  | top : Horse
  | middle : Horse
  | bottom : Horse

/-- Defines the ordering of horses based on their performance -/
def beats (h1 h2 : Horse) : Prop :=
  match h1, h2 with
  | Horse.top, Horse.middle => true
  | Horse.top, Horse.bottom => true
  | Horse.middle, Horse.bottom => true
  | _, _ => false

/-- King Qi's horses -/
def kingQi : Horse → Horse
  | Horse.top => Horse.top
  | Horse.middle => Horse.middle
  | Horse.bottom => Horse.bottom

/-- Tian Ji's horses -/
def tianJi : Horse → Horse
  | Horse.top => Horse.top
  | Horse.middle => Horse.middle
  | Horse.bottom => Horse.bottom

/-- The conditions of the horse performances -/
axiom horse_performance :
  (beats (tianJi Horse.top) (kingQi Horse.middle)) ∧
  (beats (kingQi Horse.top) (tianJi Horse.top)) ∧
  (beats (tianJi Horse.middle) (kingQi Horse.bottom)) ∧
  (beats (kingQi Horse.middle) (tianJi Horse.middle)) ∧
  (beats (kingQi Horse.bottom) (tianJi Horse.bottom))

/-- The probability of Tian Ji's horse winning -/
def winning_probability : ℚ := 1/3

/-- The main theorem to prove -/
theorem tian_ji_winning_probability :
  winning_probability = 1/3 := by sorry

end tian_ji_winning_probability_l1720_172068


namespace simplify_expression_l1720_172008

theorem simplify_expression : (-5)^2 - Real.sqrt 3 = 5 - Real.sqrt 3 := by
  sorry

end simplify_expression_l1720_172008


namespace smallest_number_divisible_l1720_172029

theorem smallest_number_divisible (n : ℕ) : n = 6297 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 18 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 70 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 100 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 84 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 3) = 18 * k₁ ∧ (n + 3) = 70 * k₂ ∧ (n + 3) = 100 * k₃ ∧ (n + 3) = 84 * k₄) :=
by sorry

#check smallest_number_divisible

end smallest_number_divisible_l1720_172029


namespace donut_selection_count_donut_selection_theorem_l1720_172040

theorem donut_selection_count : Nat → ℕ
  | n => 
    let total_donuts := 6
    let donut_types := 4
    let remaining_donuts := total_donuts - donut_types
    Nat.choose (remaining_donuts + donut_types - 1) (donut_types - 1)

theorem donut_selection_theorem : 
  donut_selection_count 6 = 10 := by
  sorry

end donut_selection_count_donut_selection_theorem_l1720_172040


namespace area_of_triangle_ABC_l1720_172015

-- Define the square WXYZ
def WXYZ : Real := 36

-- Define the side length of smaller squares
def small_square_side : Real := 2

-- Define the triangle ABC
structure Triangle :=
  (AB : Real)
  (AC : Real)
  (BC : Real)

-- Define the coincidence of point A with center O when folded
def coincides_with_center (t : Triangle) : Prop :=
  t.AB = t.AC ∧ t.AB = (WXYZ.sqrt / 2) + small_square_side

-- Define the theorem
theorem area_of_triangle_ABC :
  ∀ t : Triangle,
  coincides_with_center t →
  t.BC = WXYZ.sqrt - 2 * small_square_side →
  (1 / 2) * t.BC * ((WXYZ.sqrt / 2) + small_square_side) = 5 :=
sorry

end area_of_triangle_ABC_l1720_172015


namespace smallest_integers_difference_l1720_172027

theorem smallest_integers_difference : ∃ n₁ n₂ : ℕ,
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₁ % k = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₂ % k = 1) ∧
  n₁ > 1 ∧ n₂ > 1 ∧ n₂ > n₁ ∧
  (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 12 → m % k = 1) → m ≥ n₁) ∧
  n₂ - n₁ = 27720 :=
by sorry

end smallest_integers_difference_l1720_172027


namespace remaining_volume_is_five_sixths_l1720_172086

/-- The volume of a tetrahedron formed by planes passing through the midpoints
    of three edges sharing a vertex in a unit cube --/
def tetrahedron_volume : ℚ := 1 / 24

/-- The number of tetrahedra removed from the cube --/
def num_tetrahedra : ℕ := 8

/-- The volume of the remaining solid after removing tetrahedra from a unit cube --/
def remaining_volume : ℚ := 1 - num_tetrahedra * tetrahedron_volume

theorem remaining_volume_is_five_sixths :
  remaining_volume = 5 / 6 := by sorry

end remaining_volume_is_five_sixths_l1720_172086


namespace sum_of_digits_9cd_l1720_172073

def c : ℕ := 10^1984 + 6

def d : ℕ := 7 * (10^1984 - 1) / 9 + 4

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9cd : sum_of_digits (9 * c * d) = 33728 := by sorry

end sum_of_digits_9cd_l1720_172073


namespace exists_cycle_not_div_by_three_l1720_172062

/-- A graph is a structure with vertices and edges. -/
structure Graph (V : Type) :=
  (edges : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each consecutive pair is connected by an edge. -/
def is_path {V : Type} (G : Graph V) (path : List V) : Prop := sorry

/-- A cycle in a graph is a path that starts and ends at the same vertex. -/
def is_cycle {V : Type} (G : Graph V) (cycle : List V) : Prop := sorry

/-- The length of a path or cycle is the number of edges it contains. -/
def length {V : Type} (path : List V) : ℕ := path.length - 1

/-- Main theorem: In a graph where each vertex has degree at least 3, 
    there exists a cycle whose length is not divisible by 3. -/
theorem exists_cycle_not_div_by_three {V : Type} (G : Graph V) :
  (∀ v : V, degree G v ≥ 3) → 
  ∃ cycle : List V, is_cycle G cycle ∧ ¬(length cycle % 3 = 0) := by sorry

end exists_cycle_not_div_by_three_l1720_172062


namespace percent_relation_l1720_172006

theorem percent_relation (x y : ℝ) (h : 0.3 * (x - y) = 0.2 * (x + y)) :
  y = 0.2 * x := by
  sorry

end percent_relation_l1720_172006


namespace pm25_decrease_theorem_l1720_172059

/-- Calculates the PM2.5 concentration after two consecutive years of 10% decrease -/
def pm25_concentration (initial : ℝ) : ℝ :=
  initial * (1 - 0.1)^2

/-- Theorem stating that given an initial PM2.5 concentration of 50 micrograms per cubic meter
    two years ago, with a 10% decrease each year for two consecutive years,
    the resulting concentration is 40.5 micrograms per cubic meter -/
theorem pm25_decrease_theorem (initial : ℝ) (h : initial = 50) :
  pm25_concentration initial = 40.5 := by
  sorry

#eval pm25_concentration 50

end pm25_decrease_theorem_l1720_172059


namespace integer_tuple_solution_l1720_172079

theorem integer_tuple_solution : 
  ∀ (a b c : ℤ), (a - b)^3 * (a + b)^2 = c^2 + 2*(a - b) + 1 ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) := by
  sorry

end integer_tuple_solution_l1720_172079


namespace max_value_of_expression_l1720_172077

theorem max_value_of_expression (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_sum : a + b + c + d = 200) : 
  a * b + b * c + c * d + (1/2) * d * a ≤ 11250 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ 0 ≤ d₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = 200 ∧ 
    a₀ * b₀ + b₀ * c₀ + c₀ * d₀ + (1/2) * d₀ * a₀ = 11250 := by
  sorry

end max_value_of_expression_l1720_172077


namespace all_propositions_false_l1720_172095

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel (x y : Line ⊕ Plane) : Prop := sorry
def perpendicular (x y : Line ⊕ Plane) : Prop := sorry
def contains (p : Plane) (l : Line) : Prop := sorry
def intersects (p q : Plane) (l : Line) : Prop := sorry

-- Define the lines and planes
def m : Line := sorry
def n : Line := sorry
def a : Plane := sorry
def b : Plane := sorry

-- Define the propositions
def proposition1 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    parallel (Sum.inl m) (Sum.inr a) →
    parallel (Sum.inl n) (Sum.inr b) →
    parallel (Sum.inr a) (Sum.inr b) →
    parallel (Sum.inl m) (Sum.inl n)

def proposition2 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    parallel (Sum.inl m) (Sum.inl n) →
    contains a m →
    perpendicular (Sum.inl n) (Sum.inr b) →
    perpendicular (Sum.inr a) (Sum.inr b)

def proposition3 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    intersects a b m →
    parallel (Sum.inl m) (Sum.inl n) →
    (parallel (Sum.inl n) (Sum.inr a) ∧ parallel (Sum.inl n) (Sum.inr b))

def proposition4 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    perpendicular (Sum.inl m) (Sum.inl n) →
    intersects a b m →
    (perpendicular (Sum.inl n) (Sum.inr a) ∨ perpendicular (Sum.inl n) (Sum.inr b))

-- Theorem stating that all propositions are false
theorem all_propositions_false :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry

end all_propositions_false_l1720_172095


namespace worker_pay_is_40_l1720_172021

/-- Represents the plant supplier's sales and expenses --/
structure PlantSupplier where
  orchids : ℕ
  orchidPrice : ℕ
  chinesePlants : ℕ
  chinesePlantPrice : ℕ
  potCost : ℕ
  leftover : ℕ
  workers : ℕ

/-- Calculates the amount paid to each worker --/
def workerPay (ps : PlantSupplier) : ℕ :=
  let totalEarnings := ps.orchids * ps.orchidPrice + ps.chinesePlants * ps.chinesePlantPrice
  let totalSpent := ps.potCost + ps.leftover
  (totalEarnings - totalSpent) / ps.workers

/-- Theorem stating that each worker is paid $40 --/
theorem worker_pay_is_40 (ps : PlantSupplier) 
  (h1 : ps.orchids = 20)
  (h2 : ps.orchidPrice = 50)
  (h3 : ps.chinesePlants = 15)
  (h4 : ps.chinesePlantPrice = 25)
  (h5 : ps.potCost = 150)
  (h6 : ps.leftover = 1145)
  (h7 : ps.workers = 2) :
  workerPay ps = 40 := by
  sorry

end worker_pay_is_40_l1720_172021


namespace playstation_payment_l1720_172002

theorem playstation_payment (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_total : x₁ + x₂ + x₃ + x₄ + x₅ = 120)
  (h_x₁ : x₁ = (1/3) * (x₂ + x₃ + x₄ + x₅))
  (h_x₂ : x₂ = (1/4) * (x₁ + x₃ + x₄ + x₅))
  (h_x₃ : x₃ = (1/5) * (x₁ + x₂ + x₄ + x₅))
  (h_x₄ : x₄ = (1/6) * (x₁ + x₂ + x₃ + x₅)) :
  x₅ = 40 := by
sorry

end playstation_payment_l1720_172002


namespace negation_of_universal_proposition_l1720_172097

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) :=
by sorry

end negation_of_universal_proposition_l1720_172097


namespace vanilla_to_cream_cheese_ratio_l1720_172070

-- Define the ratios and quantities
def sugar_to_cream_cheese_ratio : ℚ := 1 / 4
def vanilla_to_eggs_ratio : ℚ := 1 / 2
def sugar_used : ℚ := 2
def eggs_used : ℚ := 8
def teaspoons_per_cup : ℚ := 48

-- Theorem to prove
theorem vanilla_to_cream_cheese_ratio :
  let cream_cheese := sugar_used / sugar_to_cream_cheese_ratio
  let vanilla := eggs_used * vanilla_to_eggs_ratio
  let cream_cheese_teaspoons := cream_cheese * teaspoons_per_cup
  vanilla / cream_cheese_teaspoons = 1 / 96 :=
by sorry

end vanilla_to_cream_cheese_ratio_l1720_172070


namespace equation_root_l1720_172058

theorem equation_root (a b c d : ℝ) (h1 : a + d = 2015) (h2 : b + c = 2015) (h3 : a ≠ c) :
  (∀ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d)) ↔ x = 1007.5 :=
by sorry

end equation_root_l1720_172058


namespace max_triangle_area_l1720_172039

/-- Given a triangle ABC where BC = 2 ∛3 and ∠BAC = π/3, the maximum possible area is 3 -/
theorem max_triangle_area (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt (Real.sqrt 3) * 2
  let BAC := π / 3
  let area := Real.sqrt 3 * BC^2 / 4
  BC = Real.sqrt (Real.sqrt 3) * 2 →
  BAC = π / 3 →
  area ≤ 3 ∧ ∃ (A' B' C' : ℝ × ℝ), 
    let BC' := Real.sqrt (Real.sqrt 3) * 2
    let BAC' := π / 3
    let area' := Real.sqrt 3 * BC'^2 / 4
    BC' = Real.sqrt (Real.sqrt 3) * 2 ∧
    BAC' = π / 3 ∧
    area' = 3 :=
by sorry

end max_triangle_area_l1720_172039


namespace polar_bear_salmon_consumption_l1720_172050

def daily_fish_consumption : ℝ := 0.6
def daily_trout_consumption : ℝ := 0.2

theorem polar_bear_salmon_consumption :
  daily_fish_consumption - daily_trout_consumption = 0.4 := by
  sorry

end polar_bear_salmon_consumption_l1720_172050


namespace sum_of_cubes_l1720_172038

theorem sum_of_cubes (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 270 → a + b + c = 7 := by
  sorry

end sum_of_cubes_l1720_172038


namespace score_difference_is_five_l1720_172043

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (0.20, 60),
  (0.25, 75),
  (0.15, 85),
  (0.30, 90),
  (0.10, 95)
]

-- Define the median score
def median_score : ℝ := 85

-- Define the function to calculate the mean score
def mean_score (distribution : List (ℝ × ℝ)) : ℝ :=
  (distribution.map (λ (p, s) => p * s)).sum

-- Theorem statement
theorem score_difference_is_five :
  median_score - mean_score score_distribution = 5 := by
  sorry


end score_difference_is_five_l1720_172043


namespace slope_of_line_intersecting_ellipse_l1720_172004

/-- Given an ellipse and a line that intersects it, this theorem proves
    that if (1,1) is the midpoint of the chord formed by the intersection,
    then the slope of the line is -1/4. -/
theorem slope_of_line_intersecting_ellipse 
  (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁^2/36 + y₁^2/9 = 1 →   -- Point (x₁, y₁) is on the ellipse
  x₂^2/36 + y₂^2/9 = 1 →   -- Point (x₂, y₂) is on the ellipse
  (x₁ + x₂)/2 = 1 →        -- x-coordinate of midpoint is 1
  (y₁ + y₂)/2 = 1 →        -- y-coordinate of midpoint is 1
  (y₂ - y₁)/(x₂ - x₁) = -1/4 :=  -- Slope of the line
by sorry

end slope_of_line_intersecting_ellipse_l1720_172004


namespace sum_of_integers_l1720_172055

theorem sum_of_integers (x y : ℕ+) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 := by
  sorry

end sum_of_integers_l1720_172055


namespace gcd_372_684_l1720_172019

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end gcd_372_684_l1720_172019


namespace line_symmetrical_to_itself_l1720_172031

/-- A line in the 2D plane represented by its slope-intercept form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The line of symmetry -/
def lineOfSymmetry : Line :=
  { slope := 1, intercept := -2 }

/-- The original line -/
def originalLine : Line :=
  { slope := 3, intercept := 3 }

/-- Find the symmetric point of a given point with respect to the line of symmetry -/
def symmetricPoint (p : Point) : Point :=
  { x := p.x, y := p.y }

theorem line_symmetrical_to_itself :
  ∀ (p : Point), pointOnLine p originalLine →
  pointOnLine (symmetricPoint p) originalLine :=
sorry

end line_symmetrical_to_itself_l1720_172031


namespace sum_congruence_l1720_172028

theorem sum_congruence : ∃ k : ℤ, (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) = 17 * k + 12 := by
  sorry

end sum_congruence_l1720_172028


namespace even_function_order_l1720_172061

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 6 * m * x + 2

-- State the theorem
theorem even_function_order (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
sorry


end even_function_order_l1720_172061


namespace sum_of_last_two_digits_l1720_172090

theorem sum_of_last_two_digits (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end sum_of_last_two_digits_l1720_172090


namespace class_average_l1720_172047

/-- Proves that the average score of a class is 45.6 given the specified conditions -/
theorem class_average (total_students : ℕ) (top_scorers : ℕ) (zero_scorers : ℕ) 
  (top_score : ℕ) (rest_average : ℕ) :
  total_students = 25 →
  top_scorers = 3 →
  zero_scorers = 3 →
  top_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - top_scorers - zero_scorers
  let total_score := top_scorers * top_score + zero_scorers * 0 + rest_students * rest_average
  (total_score : ℚ) / total_students = 45.6 := by
  sorry

end class_average_l1720_172047


namespace billboard_area_l1720_172094

/-- The area of a rectangular billboard with perimeter 44 feet and width 9 feet is 117 square feet. -/
theorem billboard_area (perimeter width : ℝ) (h1 : perimeter = 44) (h2 : width = 9) :
  let length := (perimeter - 2 * width) / 2
  width * length = 117 :=
by sorry

end billboard_area_l1720_172094


namespace coordinates_wrt_origin_l1720_172034

def Point := ℝ × ℝ

theorem coordinates_wrt_origin (p : Point) : p = p := by sorry

end coordinates_wrt_origin_l1720_172034


namespace sum_x_coordinates_of_Q3_l1720_172092

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon from the midpoints of another polygon's sides -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- The sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_of_Q3 (Q1 : Polygon) 
  (h1 : Q1.vertices.length = 45)
  (h2 : sumXCoordinates Q1 = 180) :
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumXCoordinates Q3 = 180 := by
  sorry

end sum_x_coordinates_of_Q3_l1720_172092


namespace line_intercepts_l1720_172022

/-- Given a line with equation x + 6y + 2 = 0, prove that its x-intercept is -2 and its y-intercept is -1/3 -/
theorem line_intercepts (x y : ℝ) :
  x + 6 * y + 2 = 0 →
  (x = -2 ∧ y = 0) ∨ (x = 0 ∧ y = -1/3) :=
by sorry

end line_intercepts_l1720_172022


namespace max_sum_prism_with_pyramid_l1720_172082

/-- Represents a triangular prism --/
structure TriangularPrism :=
  (faces : Nat)
  (edges : Nat)
  (vertices : Nat)

/-- Represents the result of adding a pyramid to a face of a prism --/
structure PrismWithPyramid :=
  (faces : Nat)
  (edges : Nat)
  (vertices : Nat)

/-- Calculates the sum of faces, edges, and vertices --/
def sumElements (shape : PrismWithPyramid) : Nat :=
  shape.faces + shape.edges + shape.vertices

/-- Adds a pyramid to a triangular face of the prism --/
def addPyramidToTriangularFace (prism : TriangularPrism) : PrismWithPyramid :=
  { faces := prism.faces - 1 + 3,
    edges := prism.edges + 3,
    vertices := prism.vertices + 1 }

/-- Adds a pyramid to a quadrilateral face of the prism --/
def addPyramidToQuadrilateralFace (prism : TriangularPrism) : PrismWithPyramid :=
  { faces := prism.faces - 1 + 4,
    edges := prism.edges + 4,
    vertices := prism.vertices + 1 }

/-- The main theorem to be proved --/
theorem max_sum_prism_with_pyramid :
  let prism := TriangularPrism.mk 5 9 6
  let triangularResult := addPyramidToTriangularFace prism
  let quadrilateralResult := addPyramidToQuadrilateralFace prism
  max (sumElements triangularResult) (sumElements quadrilateralResult) = 28 := by
  sorry

end max_sum_prism_with_pyramid_l1720_172082


namespace ice_cream_probability_l1720_172072

def probability_exactly_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem ice_cream_probability : 
  probability_exactly_k_successes 7 3 (3/4) = 945/16384 := by
  sorry

end ice_cream_probability_l1720_172072


namespace consecutive_page_numbers_sum_l1720_172023

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2156 → n + (n + 1) = 93 := by
  sorry

end consecutive_page_numbers_sum_l1720_172023


namespace isosceles_triangle_perimeter_l1720_172032

/-- An isosceles triangle with side lengths 5 and 8 has a perimeter of either 18 or 21. -/
theorem isosceles_triangle_perimeter : 
  ∀ a b c : ℝ, 
  (a = 5 ∧ b = 8) ∨ (a = 8 ∧ b = 5) → 
  (a = b ∨ a = c ∨ b = c) → 
  (a + b + c = 18 ∨ a + b + c = 21) := by
sorry

end isosceles_triangle_perimeter_l1720_172032


namespace orchard_fruit_sales_l1720_172041

theorem orchard_fruit_sales (total_fruit : ℕ) (frozen_fruit : ℕ) (fresh_fruit : ℕ) :
  total_fruit = 9792 →
  frozen_fruit = 3513 →
  fresh_fruit = total_fruit - frozen_fruit →
  fresh_fruit = 6279 := by
sorry

end orchard_fruit_sales_l1720_172041


namespace cubic_equation_solution_l1720_172024

theorem cubic_equation_solution : ∃ x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 ∧ x = 33 := by
  sorry

end cubic_equation_solution_l1720_172024


namespace expected_heads_after_turn_l1720_172044

/-- Represents the state of pennies on a table -/
structure PennyState where
  total : ℕ
  heads : ℕ
  tails : ℕ

/-- Represents the action of turning over pennies -/
def turn_pennies (state : PennyState) (num_turn : ℕ) : ℝ :=
  let p_heads := state.heads / state.total
  let p_tails := state.tails / state.total
  let expected_heads_turned := num_turn * p_heads
  let expected_tails_turned := num_turn * p_tails
  state.heads - expected_heads_turned + expected_tails_turned

/-- The main theorem to prove -/
theorem expected_heads_after_turn (initial_state : PennyState) 
  (h1 : initial_state.total = 100)
  (h2 : initial_state.heads = 30)
  (h3 : initial_state.tails = 70)
  (num_turn : ℕ)
  (h4 : num_turn = 40) :
  turn_pennies initial_state num_turn = 46 := by
  sorry


end expected_heads_after_turn_l1720_172044


namespace a_4_equals_zero_l1720_172030

def sequence_a (n : ℕ+) : ℤ := n.val^2 - 2*n.val - 8

theorem a_4_equals_zero : sequence_a 4 = 0 := by sorry

end a_4_equals_zero_l1720_172030


namespace rect_to_polar_equiv_l1720_172001

/-- Proves that the point (-1, √3) in rectangular coordinates 
    is equivalent to (2, 2π/3) in polar coordinates. -/
theorem rect_to_polar_equiv : 
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let r : ℝ := 2
  let θ : ℝ := 2 * Real.pi / 3
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) → 
  (x = r * Real.cos θ ∧ y = r * Real.sin θ) :=
by sorry

end rect_to_polar_equiv_l1720_172001


namespace abs_x_minus_5_lt_3_iff_2_lt_x_lt_8_l1720_172049

theorem abs_x_minus_5_lt_3_iff_2_lt_x_lt_8 :
  ∀ x : ℝ, |x - 5| < 3 ↔ 2 < x ∧ x < 8 := by
  sorry

end abs_x_minus_5_lt_3_iff_2_lt_x_lt_8_l1720_172049


namespace repeating_decimal_division_l1720_172076

theorem repeating_decimal_division (a b : ℚ) :
  a = 81 / 99 → b = 36 / 99 → a / b = 9 / 4 := by
  sorry

end repeating_decimal_division_l1720_172076


namespace triangle_third_side_l1720_172088

theorem triangle_third_side (a b c : ℝ) : 
  a = 3 → b = 10 → c > 0 → 
  a + b + c = 6 * (⌊(a + b + c) / 6⌋ : ℝ) →
  c = 11 := by sorry

end triangle_third_side_l1720_172088


namespace quadratic_equation_roots_l1720_172033

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ + 2*x₁ + 2*x₂ = 1) →
  k = -5 := by sorry

end quadratic_equation_roots_l1720_172033


namespace solve_gloria_pine_trees_l1720_172099

/-- The problem of determining the number of pine trees Gloria has -/
def GloriaPineTrees : Prop :=
  ∃ (num_pine_trees : ℕ),
    let cabin_cost : ℕ := 129000
    let cash : ℕ := 150
    let num_cypress : ℕ := 20
    let num_maple : ℕ := 24
    let cypress_price : ℕ := 100
    let maple_price : ℕ := 300
    let pine_price : ℕ := 200
    let leftover : ℕ := 350
    
    cabin_cost + leftover = 
      cash + num_cypress * cypress_price + num_maple * maple_price + num_pine_trees * pine_price ∧
    num_pine_trees = 600

theorem solve_gloria_pine_trees : GloriaPineTrees := by
  sorry

end solve_gloria_pine_trees_l1720_172099


namespace count_possible_roots_l1720_172014

/-- A polynomial with integer coefficients of the form 12x^5 + b₄x^4 + b₃x^3 + b₂x^2 + b₁x + 24 = 0 -/
def IntegerPolynomial (b₄ b₃ b₂ b₁ : ℤ) (x : ℚ) : ℚ :=
  12 * x^5 + b₄ * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 24

/-- The set of possible rational roots for the polynomial -/
def PossibleRoots : Finset ℚ :=
  {1, 2, 3, 4, 6, 8, 12, 24, 1/2, 1/3, 1/4, 1/6, 2/3, 3/2, 3/4, 4/3,
   -1, -2, -3, -4, -6, -8, -12, -24, -1/2, -1/3, -1/4, -1/6, -2/3, -3/2, -3/4, -4/3}

/-- Theorem stating that the number of possible rational roots is 32 -/
theorem count_possible_roots (b₄ b₃ b₂ b₁ : ℤ) :
  Finset.card PossibleRoots = 32 ∧
  ∀ q : ℚ, q ∉ PossibleRoots → IntegerPolynomial b₄ b₃ b₂ b₁ q ≠ 0 :=
sorry

end count_possible_roots_l1720_172014


namespace solution_satisfies_system_l1720_172054

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y = 3
def equation2 (x y : ℝ) : Prop := 2 * (x + y) - y = 5

-- Theorem stating that (2, 1) is the solution
theorem solution_satisfies_system :
  equation1 2 1 ∧ equation2 2 1 := by
  sorry

end solution_satisfies_system_l1720_172054


namespace train_length_calculation_l1720_172096

/-- Proves that a train with given speed, crossing a bridge of known length in a specific time, has a particular length. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (crossing_time : Real) (train_length : Real) : 
  train_speed = 36 → -- speed in km/hr
  bridge_length = 132 → -- bridge length in meters
  crossing_time = 24.198064154867613 → -- time to cross the bridge in seconds
  train_length = 109.98064154867613 → -- train length in meters
  (train_speed * 1000 / 3600) * crossing_time = bridge_length + train_length := by
  sorry

#check train_length_calculation

end train_length_calculation_l1720_172096


namespace highway_repair_time_l1720_172042

theorem highway_repair_time (x y : ℝ) : 
  (1 / x + 1 / y = 1 / 18) →  -- Combined work rate
  (2 * x / 3 + y / 3 = 40) →  -- Actual repair time
  (x = 45 ∧ y = 30) := by
  sorry

end highway_repair_time_l1720_172042


namespace original_car_cost_l1720_172057

/-- Proves that the original cost of a car is 42000 given the repair cost, selling price, and profit percentage. -/
theorem original_car_cost (repair_cost selling_price profit_percent : ℝ) : 
  repair_cost = 8000 →
  selling_price = 64900 →
  profit_percent = 29.8 →
  ∃ (original_cost : ℝ), 
    original_cost = 42000 ∧
    selling_price = (original_cost + repair_cost) * (1 + profit_percent / 100) :=
by
  sorry

end original_car_cost_l1720_172057


namespace matrix_equality_implies_ratio_l1720_172018

theorem matrix_equality_implies_ratio (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  A * B = B * A ∧ 4 * b ≠ c →
  (a - d) / (c - 4 * b) = 2 := by
  sorry

end matrix_equality_implies_ratio_l1720_172018


namespace total_soaking_time_l1720_172098

/-- Calculates the total soaking time for clothes with grass and marinara stains -/
theorem total_soaking_time
  (grass_stain_time : ℕ)
  (marinara_stain_time : ℕ)
  (grass_stain_count : ℕ)
  (marinara_stain_count : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : marinara_stain_time = 7)
  (h3 : grass_stain_count = 3)
  (h4 : marinara_stain_count = 1) :
  grass_stain_time * grass_stain_count + marinara_stain_time * marinara_stain_count = 19 :=
by
  sorry


end total_soaking_time_l1720_172098


namespace ethanol_percentage_in_fuel_A_l1720_172075

/-- Proves that the percentage of ethanol in fuel A is 12%, given the specified conditions. -/
theorem ethanol_percentage_in_fuel_A : ∀ (tank_capacity fuel_A_volume fuel_B_ethanol_percent total_ethanol : ℝ),
  tank_capacity = 204 →
  fuel_A_volume = 66 →
  fuel_B_ethanol_percent = 16 / 100 →
  total_ethanol = 30 →
  ∃ (fuel_A_ethanol_percent : ℝ),
    fuel_A_ethanol_percent * fuel_A_volume + 
    fuel_B_ethanol_percent * (tank_capacity - fuel_A_volume) = total_ethanol ∧
    fuel_A_ethanol_percent = 12 / 100 :=
by sorry

end ethanol_percentage_in_fuel_A_l1720_172075


namespace peggy_bought_three_folders_l1720_172005

/-- Represents the number of sheets in each folder -/
def sheets_per_folder : ℕ := 10

/-- Represents the number of stickers per sheet in the red folder -/
def red_stickers_per_sheet : ℕ := 3

/-- Represents the number of stickers per sheet in the green folder -/
def green_stickers_per_sheet : ℕ := 2

/-- Represents the number of stickers per sheet in the blue folder -/
def blue_stickers_per_sheet : ℕ := 1

/-- Represents the total number of stickers used -/
def total_stickers : ℕ := 60

/-- Theorem stating that Peggy bought 3 folders -/
theorem peggy_bought_three_folders :
  (sheets_per_folder * red_stickers_per_sheet) +
  (sheets_per_folder * green_stickers_per_sheet) +
  (sheets_per_folder * blue_stickers_per_sheet) = total_stickers :=
by sorry

end peggy_bought_three_folders_l1720_172005


namespace largest_value_at_negative_one_l1720_172036

/-- A monic cubic polynomial with non-negative real roots and f(0) = -64 -/
def MonicCubicPolynomial : Type := 
  {f : ℝ → ℝ // ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = (x - r₁) * (x - r₂) * (x - r₃)) ∧ 
                                  (r₁ ≥ 0 ∧ r₂ ≥ 0 ∧ r₃ ≥ 0) ∧
                                  (f 0 = -64)}

/-- The largest possible value of f(-1) for a MonicCubicPolynomial is -125 -/
theorem largest_value_at_negative_one (f : MonicCubicPolynomial) : 
  f.val (-1) ≤ -125 := by
  sorry

end largest_value_at_negative_one_l1720_172036


namespace hernandez_state_tax_l1720_172025

def calculate_state_tax (months_of_residency : ℕ) (taxable_income : ℝ) (tax_rate : ℝ) : ℝ :=
  let proportion_of_year := months_of_residency / 12
  let prorated_income := taxable_income * proportion_of_year
  prorated_income * tax_rate

theorem hernandez_state_tax :
  calculate_state_tax 9 42500 0.04 = 1275 := by
  sorry

end hernandez_state_tax_l1720_172025


namespace ratio_a_to_b_l1720_172052

/-- Given that 0.5% of a is 85 paise and 0.75% of b is 150 paise, 
    prove that the ratio of a to b is 17:20 -/
theorem ratio_a_to_b (a b : ℚ) 
  (ha : (5 / 1000) * a = 85 / 100) 
  (hb : (75 / 10000) * b = 150 / 100) : 
  a / b = 17 / 20 := by
  sorry

end ratio_a_to_b_l1720_172052


namespace fractional_equation_solution_l1720_172037

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 * x) / (x - 1) = 3 ∧ x = 3 :=
by sorry

end fractional_equation_solution_l1720_172037


namespace baseball_team_ratio_l1720_172071

def baseball_ratio (games_played : ℕ) (games_won : ℕ) : ℚ := 
  games_played / (games_played - games_won)

theorem baseball_team_ratio : 
  let games_played := 10
  let games_won := 5
  baseball_ratio games_played games_won = 2 := by
  sorry

end baseball_team_ratio_l1720_172071


namespace unique_number_between_cube_roots_l1720_172093

theorem unique_number_between_cube_roots : ∃! (n : ℕ), 
  n > 0 ∧ 
  24 ∣ n ∧ 
  (9 : ℝ) < (n : ℝ) ^ (1/3 : ℝ) ∧ 
  (n : ℝ) ^ (1/3 : ℝ) < (9.1 : ℝ) ∧
  n = 744 := by
  sorry

end unique_number_between_cube_roots_l1720_172093


namespace min_value_of_x_plus_nine_over_x_l1720_172069

theorem min_value_of_x_plus_nine_over_x (x : ℝ) (hx : x > 0) :
  x + 9 / x ≥ 6 ∧ (x + 9 / x = 6 ↔ x = 3) := by sorry

end min_value_of_x_plus_nine_over_x_l1720_172069


namespace rope_cutting_probability_l1720_172056

theorem rope_cutting_probability : 
  let rope_length : ℝ := 6
  let num_nodes : ℕ := 5
  let num_parts : ℕ := 6
  let min_segment_length : ℝ := 2

  let part_length : ℝ := rope_length / num_parts
  let favorable_cuts : ℕ := (num_nodes - 2)
  
  (favorable_cuts : ℝ) / num_nodes = 3 / 5 :=
by sorry

end rope_cutting_probability_l1720_172056


namespace line_equation_through_point_with_angle_l1720_172081

/-- The equation of a line passing through a given point with a given angle -/
theorem line_equation_through_point_with_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) 
  (h_x₀ : x₀ = Real.sqrt 3) 
  (h_y₀ : y₀ = -2 * Real.sqrt 3) 
  (h_θ : θ = 135 * π / 180) :
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ 
                 ∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 
                 y - y₀ = Real.tan θ * (x - x₀) :=
sorry

end line_equation_through_point_with_angle_l1720_172081


namespace evaluate_expression_l1720_172066

theorem evaluate_expression : 6 - 5 * (10 - (2 + 3)^2) * 2 = 306 := by
  sorry

end evaluate_expression_l1720_172066


namespace tangent_line_and_min_value_l1720_172000

/-- The function f(x) = -x^3 + 3x^2 + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

theorem tangent_line_and_min_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 22) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 22) →
  (∀ y : ℝ, (9 * 2 - y + 2 = 0) ↔ (y - f (-2) 2 = f' 2 * (y - 2))) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f 0 x ≤ f 0 y ∧ f 0 x = -7) :=
by sorry

end tangent_line_and_min_value_l1720_172000


namespace julia_total_kids_l1720_172074

/-- The number of kids Julia played with on each day of the week -/
structure WeeklyKids where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculate the total number of kids Julia played with throughout the week -/
def totalKids (w : WeeklyKids) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday + w.sunday

/-- The conditions given in the problem -/
def juliaWeek : WeeklyKids where
  monday := 15
  tuesday := 18
  wednesday := 25
  thursday := 30
  friday := 30 + (30 * 20 / 100)
  saturday := (30 + (30 * 20 / 100)) - ((30 + (30 * 20 / 100)) * 30 / 100)
  sunday := 15 * 2

/-- Theorem stating that the total number of kids Julia played with is 180 -/
theorem julia_total_kids : totalKids juliaWeek = 180 := by
  sorry

end julia_total_kids_l1720_172074


namespace laundry_detergent_price_l1720_172046

/-- Calculates the initial price of laundry detergent given grocery shopping conditions --/
theorem laundry_detergent_price
  (initial_amount : ℝ)
  (milk_price : ℝ)
  (bread_price : ℝ)
  (banana_price_per_pound : ℝ)
  (banana_quantity : ℝ)
  (detergent_coupon : ℝ)
  (amount_left : ℝ)
  (h1 : initial_amount = 20)
  (h2 : milk_price = 4)
  (h3 : bread_price = 3.5)
  (h4 : banana_price_per_pound = 0.75)
  (h5 : banana_quantity = 2)
  (h6 : detergent_coupon = 1.25)
  (h7 : amount_left = 4) :
  let discounted_milk_price := milk_price / 2
  let banana_total := banana_price_per_pound * banana_quantity
  let other_items_cost := discounted_milk_price + bread_price + banana_total
  let total_spent := initial_amount - amount_left
  let detergent_price_with_coupon := total_spent - other_items_cost
  let initial_detergent_price := detergent_price_with_coupon + detergent_coupon
  initial_detergent_price = 10.25 := by
sorry

end laundry_detergent_price_l1720_172046


namespace greatest_consecutive_odd_integers_sum_400_l1720_172013

/-- The sum of the first n odd integers -/
def sum_odd_integers (n : ℕ) : ℕ := n^2

/-- The problem statement -/
theorem greatest_consecutive_odd_integers_sum_400 :
  (∃ (n : ℕ), sum_odd_integers n = 400) ∧
  (∀ (m : ℕ), sum_odd_integers m = 400 → m ≤ 20) :=
by sorry

end greatest_consecutive_odd_integers_sum_400_l1720_172013


namespace derivative_exp_sin_derivative_frac_derivative_ln_derivative_product_derivative_cos_l1720_172026

variable (x : ℝ)

-- Function 1
theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x := by sorry

-- Function 2
theorem derivative_frac (x : ℝ) : 
  deriv (fun x => (x + 3) / (x + 2)) x = - 1 / ((x + 2) ^ 2) := by sorry

-- Function 3
theorem derivative_ln (x : ℝ) : 
  deriv (fun x => Real.log (2 * x + 3)) x = 2 / (2 * x + 3) := by sorry

-- Function 4
theorem derivative_product (x : ℝ) : 
  deriv (fun x => (x^2 + 2) * (2*x - 1)) x = 6 * x^2 - 2 * x + 4 := by sorry

-- Function 5
theorem derivative_cos (x : ℝ) : 
  deriv (fun x => Real.cos (2*x + Real.pi/3)) x = -2 * Real.sin (2*x + Real.pi/3) := by sorry

end derivative_exp_sin_derivative_frac_derivative_ln_derivative_product_derivative_cos_l1720_172026


namespace ratio_w_to_y_l1720_172020

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 4)
  (hy : y / z = 4 / 3)
  (hz : z / x = 1 / 8) :
  w / y = 15 / 2 := by
  sorry

end ratio_w_to_y_l1720_172020
