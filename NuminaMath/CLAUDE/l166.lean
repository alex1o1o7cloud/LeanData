import Mathlib

namespace solutions_cubic_equation_l166_16611

theorem solutions_cubic_equation :
  {x : ℝ | x^3 - 4*x = 0} = {0, -2, 2} := by sorry

end solutions_cubic_equation_l166_16611


namespace square_sum_existence_l166_16667

theorem square_sum_existence (k : ℤ) 
  (h1 : 2 * k + 1 > 17) 
  (h2 : ∃ m : ℤ, 6 * k + 1 = m^2) : 
  ∃ b c : ℤ, 
    b > 0 ∧ 
    c > 0 ∧ 
    b ≠ c ∧ 
    (∃ w : ℤ, (2 * k + 1 + b) = w^2) ∧ 
    (∃ x : ℤ, (2 * k + 1 + c) = x^2) ∧ 
    (∃ y : ℤ, (b + c) = y^2) ∧ 
    (∃ z : ℤ, (2 * k + 1 + b + c) = z^2) :=
sorry

end square_sum_existence_l166_16667


namespace seokgi_money_problem_l166_16624

theorem seokgi_money_problem (initial_money : ℕ) : 
  (initial_money / 2) / 2 = 1250 → initial_money = 5000 := by
  sorry

end seokgi_money_problem_l166_16624


namespace school_ratio_l166_16634

-- Define the school structure
structure School where
  b : ℕ  -- number of teachers
  c : ℕ  -- number of students
  k : ℕ  -- number of students each teacher teaches
  h : ℕ  -- number of teachers teaching any two different students

-- Define the theorem
theorem school_ratio (s : School) : 
  s.b / s.h = (s.c * (s.c - 1)) / (s.k * (s.k - 1)) := by
  sorry

end school_ratio_l166_16634


namespace distance_between_points_l166_16628

/-- The distance between points (2,5) and (7,1) is √41. -/
theorem distance_between_points : Real.sqrt 41 = Real.sqrt ((7 - 2)^2 + (1 - 5)^2) := by
  sorry

end distance_between_points_l166_16628


namespace groomer_problem_l166_16697

/-- The number of full-haired dogs a groomer has to dry --/
def num_full_haired_dogs : ℕ := by sorry

theorem groomer_problem :
  let time_short_haired : ℕ := 10  -- minutes to dry a short-haired dog
  let time_full_haired : ℕ := 2 * time_short_haired  -- minutes to dry a full-haired dog
  let num_short_haired : ℕ := 6  -- number of short-haired dogs
  let total_time : ℕ := 4 * 60  -- total time in minutes (4 hours)
  
  num_full_haired_dogs = 
    (total_time - num_short_haired * time_short_haired) / time_full_haired :=
by sorry

end groomer_problem_l166_16697


namespace rob_baseball_cards_l166_16664

theorem rob_baseball_cards (rob_total : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) :
  rob_doubles * 3 = rob_total →
  jess_doubles = rob_doubles * 5 →
  jess_doubles = 40 →
  rob_total = 24 := by
sorry

end rob_baseball_cards_l166_16664


namespace parabola_y_relationship_l166_16696

/-- A parabola defined by y = -x² + 6x + c -/
def parabola (x : ℝ) (c : ℝ) : ℝ := -x^2 + 6*x + c

/-- Three points on the parabola -/
structure PointsOnParabola (c : ℝ) where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  h₁ : parabola 1 c = y₁
  h₂ : parabola 3 c = y₂
  h₃ : parabola 4 c = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_y_relationship (c : ℝ) (p : PointsOnParabola c) :
  p.y₁ < p.y₃ ∧ p.y₃ < p.y₂ := by sorry

end parabola_y_relationship_l166_16696


namespace sin_cos_extrema_l166_16639

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∀ a b : ℝ, Real.sin a + Real.sin b = 1/3 → 
    Real.sin a - Real.cos b ^ 2 ≤ 4/9 ∧ 
    Real.sin a - Real.cos b ^ 2 ≥ -11/12) ∧
  (∃ c d : ℝ, Real.sin c + Real.sin d = 1/3 ∧ 
    Real.sin c - Real.cos d ^ 2 = 4/9) ∧
  (∃ e f : ℝ, Real.sin e + Real.sin f = 1/3 ∧ 
    Real.sin e - Real.cos f ^ 2 = -11/12) :=
by sorry

end sin_cos_extrema_l166_16639


namespace length_F_to_F_prime_l166_16637

/-- Triangle DEF with vertices D(-1, 3), E(5, -1), and F(-4, -2) is reflected over the y-axis.
    This theorem proves that the length of the segment from F to F' is 8. -/
theorem length_F_to_F_prime (D E F : ℝ × ℝ) : 
  D = (-1, 3) → E = (5, -1) → F = (-4, -2) → 
  let F' := (-(F.1), F.2)
  abs (F'.1 - F.1) = 8 := by
  sorry

end length_F_to_F_prime_l166_16637


namespace restaurant_bill_theorem_l166_16612

theorem restaurant_bill_theorem (total_people : ℕ) (total_bill : ℚ) (gratuity_rate : ℚ) :
  total_people = 6 →
  total_bill = 720 →
  gratuity_rate = 1/5 →
  (total_bill / (1 + gratuity_rate)) / total_people = 100 := by
sorry

end restaurant_bill_theorem_l166_16612


namespace canoe_kayak_revenue_l166_16661

/-- Represents the revenue calculation for a canoe and kayak rental business --/
theorem canoe_kayak_revenue
  (canoe_cost : ℕ)
  (kayak_cost : ℕ)
  (canoe_kayak_ratio : ℚ)
  (canoe_kayak_difference : ℕ)
  (h1 : canoe_cost = 12)
  (h2 : kayak_cost = 18)
  (h3 : canoe_kayak_ratio = 3 / 2)
  (h4 : canoe_kayak_difference = 7) :
  ∃ (num_canoes num_kayaks : ℕ),
    num_canoes = num_kayaks + canoe_kayak_difference ∧
    (num_canoes : ℚ) / num_kayaks = canoe_kayak_ratio ∧
    num_canoes * canoe_cost + num_kayaks * kayak_cost = 504 :=
by sorry

end canoe_kayak_revenue_l166_16661


namespace ice_cream_problem_l166_16638

/-- Ice cream purchase and profit maximization problem -/
theorem ice_cream_problem 
  (cost_equation1 : ℝ → ℝ → Prop) 
  (cost_equation2 : ℝ → ℝ → Prop)
  (total_budget : ℝ)
  (total_ice_creams : ℕ)
  (brand_a_constraint : ℕ → ℕ → Prop)
  (selling_price_a : ℝ)
  (selling_price_b : ℝ) :
  -- Part 1: Purchase prices
  ∃ (price_a price_b : ℝ),
    cost_equation1 price_a price_b ∧
    cost_equation2 price_a price_b ∧
    price_a = 12 ∧
    price_b = 15 ∧
  -- Part 2: Profit maximization
  ∃ (brand_a brand_b : ℕ),
    brand_a + brand_b = total_ice_creams ∧
    brand_a_constraint brand_a brand_b ∧
    price_a * brand_a + price_b * brand_b ≤ total_budget ∧
    brand_a = 20 ∧
    brand_b = 20 ∧
    ∀ (m n : ℕ), 
      m + n = total_ice_creams →
      brand_a_constraint m n →
      price_a * m + price_b * n ≤ total_budget →
      (selling_price_a - price_a) * brand_a + (selling_price_b - price_b) * brand_b ≥
      (selling_price_a - price_a) * m + (selling_price_b - price_b) * n :=
by sorry

end ice_cream_problem_l166_16638


namespace mixture_concentration_l166_16631

/-- Represents a vessel with spirit -/
structure Vessel where
  concentration : Rat
  volume : Rat

/-- Calculates the concentration of spirit in a mixture of vessels -/
def mixConcentration (vessels : List Vessel) : Rat :=
  let totalSpirit := vessels.map (λ v => v.concentration * v.volume) |>.sum
  let totalVolume := vessels.map (λ v => v.volume) |>.sum
  totalSpirit / totalVolume

/-- The main theorem stating that the mixture of given vessels results in 26% concentration -/
theorem mixture_concentration : 
  let vessels := [
    Vessel.mk (45/100) 4,
    Vessel.mk (30/100) 5,
    Vessel.mk (10/100) 6
  ]
  mixConcentration vessels = 26/100 := by
  sorry


end mixture_concentration_l166_16631


namespace rationalize_denominator_l166_16643

theorem rationalize_denominator : 
  (35 : ℝ) / Real.sqrt 15 = (7 / 3) * Real.sqrt 15 := by
  sorry

end rationalize_denominator_l166_16643


namespace gcf_of_4620_and_10780_l166_16699

theorem gcf_of_4620_and_10780 : Nat.gcd 4620 10780 = 1540 := by
  sorry

end gcf_of_4620_and_10780_l166_16699


namespace james_passenger_count_l166_16694

/-- Calculates the total number of passengers James has seen --/
def total_passengers (total_vehicles : ℕ) (trucks : ℕ) (buses : ℕ) (cars : ℕ) 
  (truck_passengers : ℕ) (bus_passengers : ℕ) (taxi_passengers : ℕ) 
  (motorbike_passengers : ℕ) (car_passengers : ℕ) : ℕ :=
  let taxis := 2 * buses
  let motorbikes := total_vehicles - trucks - buses - taxis - cars
  trucks * truck_passengers + 
  buses * bus_passengers + 
  taxis * taxi_passengers + 
  motorbikes * motorbike_passengers + 
  cars * car_passengers

theorem james_passenger_count : 
  total_passengers 52 12 2 30 2 15 2 1 3 = 156 := by
  sorry

end james_passenger_count_l166_16694


namespace car_subsidy_theorem_l166_16635

/-- Represents the sales and pricing data for a car dealership --/
structure CarSalesData where
  manual_nov : ℕ
  auto_nov : ℕ
  manual_dec : ℕ
  auto_dec : ℕ
  manual_price : ℕ
  auto_price : ℕ
  subsidy_rate : ℚ

/-- Calculates the total government subsidy based on car sales data --/
def total_subsidy (data : CarSalesData) : ℚ :=
  (data.manual_dec * data.manual_price + data.auto_dec * data.auto_price) * data.subsidy_rate

/-- Theorem stating the total government subsidy for the given scenario --/
theorem car_subsidy_theorem (data : CarSalesData) :
  data.manual_nov + data.auto_nov = 960 →
  data.manual_dec + data.auto_dec = 1228 →
  data.manual_dec = (13 * data.manual_nov) / 10 →
  data.auto_dec = (5 * data.auto_nov) / 4 →
  data.manual_price = 80000 →
  data.auto_price = 90000 →
  data.subsidy_rate = 1 / 20 →
  total_subsidy data = 516200000 / 1000 :=
by sorry

end car_subsidy_theorem_l166_16635


namespace dandelion_puffs_count_l166_16622

/-- The number of dandelion puffs Caleb originally picked -/
def original_puffs : ℕ := 40

/-- The number of puffs given to mom -/
def mom_puffs : ℕ := 3

/-- The number of puffs given to sister -/
def sister_puffs : ℕ := 3

/-- The number of puffs given to grandmother -/
def grandmother_puffs : ℕ := 5

/-- The number of puffs given to dog -/
def dog_puffs : ℕ := 2

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of puffs each friend received -/
def puffs_per_friend : ℕ := 9

theorem dandelion_puffs_count :
  original_puffs = mom_puffs + sister_puffs + grandmother_puffs + dog_puffs + num_friends * puffs_per_friend :=
by sorry

end dandelion_puffs_count_l166_16622


namespace imaginary_part_of_reciprocal_l166_16674

theorem imaginary_part_of_reciprocal (z : ℂ) : z = 1 / (2 - I) → z.im = 1 / 5 := by
  sorry

end imaginary_part_of_reciprocal_l166_16674


namespace second_term_of_geometric_series_l166_16613

/-- 
Given an infinite geometric series with common ratio 1/4 and sum 40,
prove that the second term of the sequence is 7.5.
-/
theorem second_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (1/4)^n = 40) → a * (1/4) = 7.5 := by
  sorry

end second_term_of_geometric_series_l166_16613


namespace paper_strip_dimensions_l166_16627

theorem paper_strip_dimensions (a b c : ℕ+) (h : 2 * a * b + 2 * a * c - a * a = 43) :
  a = 1 ∧ b + c = 22 := by
  sorry

end paper_strip_dimensions_l166_16627


namespace task_assignment_count_l166_16689

/-- The number of ways to assign tasks to volunteers -/
def task_assignments (num_volunteers : ℕ) (num_tasks : ℕ) : ℕ :=
  -- Number of ways to divide tasks into groups
  (num_tasks.choose (num_tasks - num_volunteers)) *
  -- Number of ways to permute volunteers
  (num_volunteers.factorial)

/-- Theorem: There are 36 ways to assign 4 tasks to 3 volunteers -/
theorem task_assignment_count :
  task_assignments 3 4 = 36 := by
sorry

end task_assignment_count_l166_16689


namespace camp_gender_ratio_l166_16610

theorem camp_gender_ratio (total : ℕ) (boys_added : ℕ) (girls_percent : ℝ) : 
  total = 100 → 
  boys_added = 100 → 
  girls_percent = 5 → 
  (total : ℝ) * girls_percent / 100 = (total - ((total + boys_added) * girls_percent / 100)) → 
  (100 : ℝ) * (total - ((total + boys_added) * girls_percent / 100)) / total = 90 :=
by sorry

end camp_gender_ratio_l166_16610


namespace max_boxes_theorem_l166_16651

def lifting_capacities : List Nat := [30, 45, 50, 60, 75, 100, 120]
def box_weights : List Nat := [15, 25, 35, 45, 55, 70, 80, 95, 110]

def max_boxes_lifted (capacities : List Nat) (weights : List Nat) : Nat :=
  sorry

theorem max_boxes_theorem :
  max_boxes_lifted lifting_capacities box_weights = 7 := by
  sorry

end max_boxes_theorem_l166_16651


namespace fenced_area_calculation_l166_16623

theorem fenced_area_calculation : 
  let yard_length : ℕ := 20
  let yard_width : ℕ := 18
  let large_cutout_side : ℕ := 4
  let small_cutout_side : ℕ := 2
  let yard_area := yard_length * yard_width
  let large_cutout_area := large_cutout_side * large_cutout_side
  let small_cutout_area := small_cutout_side * small_cutout_side
  yard_area - large_cutout_area - small_cutout_area = 340 := by
sorry

end fenced_area_calculation_l166_16623


namespace quadratic_inequality_l166_16653

theorem quadratic_inequality (x : ℝ) : -3*x^2 + 6*x + 9 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_l166_16653


namespace inscribed_sphere_in_cone_l166_16654

/-- Given a right cone with base radius 15 cm and height 30 cm, 
    and an inscribed sphere with radius r = b√d - b cm, 
    prove that b + d = 12.5 -/
theorem inscribed_sphere_in_cone (b d : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * (d.sqrt - 1)
  sphere_radius = (cone_height * cone_base_radius) / (cone_base_radius + (cone_base_radius^2 + cone_height^2).sqrt) →
  b + d = 12.5 := by sorry

end inscribed_sphere_in_cone_l166_16654


namespace calzone_knead_time_l166_16660

def calzone_time_problem (total_time onion_time knead_time : ℝ) : Prop :=
  let garlic_pepper_time := onion_time / 4
  let rest_time := 2 * knead_time
  let assemble_time := (knead_time + rest_time) / 10
  total_time = onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time

theorem calzone_knead_time :
  ∃ (knead_time : ℝ), 
    calzone_time_problem 124 20 knead_time ∧ 
    knead_time = 30 := by
  sorry

end calzone_knead_time_l166_16660


namespace validBinaryStrings_10_l166_16680

/-- A function that returns the number of binary strings of length n 
    that do not contain the substrings 101 or 010 -/
def validBinaryStrings (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | m + 3 => validBinaryStrings (m + 2) + validBinaryStrings (m + 1)

/-- Theorem stating that the number of binary strings of length 10 
    that do not contain the substrings 101 or 010 is 178 -/
theorem validBinaryStrings_10 : validBinaryStrings 10 = 178 := by
  sorry

end validBinaryStrings_10_l166_16680


namespace consecutive_four_product_plus_one_is_square_l166_16656

theorem consecutive_four_product_plus_one_is_square (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k ^ 2 := by
  sorry

end consecutive_four_product_plus_one_is_square_l166_16656


namespace max_value_sin_cos_l166_16668

theorem max_value_sin_cos (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  ∃ (M : ℝ), M = 4/9 ∧ ∀ (a b : ℝ), Real.sin a + Real.sin b = 1/3 →
  Real.sin b - Real.cos a ^ 2 ≤ M :=
by sorry

end max_value_sin_cos_l166_16668


namespace chrysler_leeward_floor_difference_l166_16650

theorem chrysler_leeward_floor_difference :
  ∀ (chrysler_floors leeward_floors : ℕ),
    chrysler_floors > leeward_floors →
    chrysler_floors + leeward_floors = 35 →
    chrysler_floors = 23 →
    chrysler_floors - leeward_floors = 11 := by
  sorry

end chrysler_leeward_floor_difference_l166_16650


namespace complex_product_equals_43_l166_16620

theorem complex_product_equals_43 (x : ℂ) (h : x = Complex.exp (2 * Real.pi * Complex.I / 7)) :
  (2*x + x^2) * (2*x^2 + x^4) * (2*x^3 + x^6) * (2*x^4 + x^8) * (2*x^5 + x^10) * (2*x^6 + x^12) = 43 := by
  sorry

end complex_product_equals_43_l166_16620


namespace price_decrease_l166_16600

theorem price_decrease (original_price : ℝ) : 
  (original_price * (1 - 0.24) = 836) → original_price = 1100 := by
  sorry

end price_decrease_l166_16600


namespace division_problem_l166_16614

theorem division_problem (N D Q R : ℕ) : 
  D = 5 → Q = 4 → R = 3 → N = D * Q + R → N = 23 := by
sorry

end division_problem_l166_16614


namespace tom_flashlight_batteries_l166_16616

/-- The number of batteries Tom used on his flashlights -/
def flashlight_batteries : ℕ := 19 - 15 - 2

/-- Proof that Tom used 4 batteries on his flashlights -/
theorem tom_flashlight_batteries :
  flashlight_batteries = 4 :=
by sorry

end tom_flashlight_batteries_l166_16616


namespace netflix_series_episodes_l166_16692

/-- A TV series with the given properties -/
structure TVSeries where
  seasons : ℕ
  episodes_per_day : ℕ
  days_to_complete : ℕ

/-- Calculate the number of episodes per season -/
def episodes_per_season (series : TVSeries) : ℕ :=
  (series.episodes_per_day * series.days_to_complete) / series.seasons

/-- Theorem stating that for the given TV series, each season has 20 episodes -/
theorem netflix_series_episodes (series : TVSeries) 
  (h1 : series.seasons = 3)
  (h2 : series.episodes_per_day = 2)
  (h3 : series.days_to_complete = 30) :
  episodes_per_season series = 20 := by
  sorry

#check netflix_series_episodes

end netflix_series_episodes_l166_16692


namespace a5_is_zero_in_825_factorial_base_l166_16640

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorialBaseCoeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem a5_is_zero_in_825_factorial_base : 
  factorialBaseCoeff 825 5 = 0 := by sorry

end a5_is_zero_in_825_factorial_base_l166_16640


namespace exists_intersecting_line_no_circle_through_origin_l166_16662

-- Define the set of circles C_k
def C_k (k : ℕ+) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (k - 1))^2 + (p.2 - 3*k)^2 = 2*k^4}

-- Statement 1: There exists a fixed line that intersects all circles
theorem exists_intersecting_line :
  ∃ (m b : ℝ), ∀ (k : ℕ+), ∃ (x y : ℝ), (y = m*x + b) ∧ (x, y) ∈ C_k k :=
sorry

-- Statement 2: No circle passes through the origin
theorem no_circle_through_origin :
  ∀ (k : ℕ+), (0, 0) ∉ C_k k :=
sorry

end exists_intersecting_line_no_circle_through_origin_l166_16662


namespace parallel_statements_l166_16626

-- Define the concept of parallel lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define the concept of parallel planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Define a line being parallel to a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_statements :
  -- Statement 1
  (∀ l1 l2 l3 : Line, parallel_lines l1 l3 → parallel_lines l2 l3 → parallel_lines l1 l2) ∧
  -- Statement 2
  (∀ p1 p2 p3 : Plane, parallel_planes p1 p3 → parallel_planes p2 p3 → parallel_planes p1 p2) ∧
  -- Statement 3 (negation)
  (∃ l1 l2 : Line, ∃ p : Plane, 
    parallel_lines l1 l2 ∧ line_parallel_to_plane l1 p ∧ ¬line_parallel_to_plane l2 p) ∧
  -- Statement 4 (negation)
  (∃ l : Line, ∃ p1 p2 : Plane,
    parallel_planes p1 p2 ∧ line_parallel_to_plane l p1 ∧ ¬line_parallel_to_plane l p2) :=
by sorry

end parallel_statements_l166_16626


namespace only_expr1_is_inequality_l166_16659

-- Define the type for mathematical expressions
inductive MathExpression
  | LessThan : ℝ → ℝ → MathExpression
  | LinearExpr : ℝ → ℝ → MathExpression
  | Equation : ℝ → ℝ → ℝ → ℝ → MathExpression
  | Monomial : ℝ → ℕ → MathExpression

-- Define what it means for an expression to be an inequality
def isInequality : MathExpression → Prop
  | MathExpression.LessThan _ _ => True
  | _ => False

-- Define the given expressions
def expr1 : MathExpression := MathExpression.LessThan 0 19
def expr2 : MathExpression := MathExpression.LinearExpr 1 (-2)
def expr3 : MathExpression := MathExpression.Equation 2 3 (-1) 0
def expr4 : MathExpression := MathExpression.Monomial 1 2

-- Theorem statement
theorem only_expr1_is_inequality :
  isInequality expr1 ∧
  ¬isInequality expr2 ∧
  ¬isInequality expr3 ∧
  ¬isInequality expr4 :=
by sorry

end only_expr1_is_inequality_l166_16659


namespace cat_meows_l166_16604

theorem cat_meows (cat1 : ℝ) (cat2 : ℝ) (cat3 : ℝ) : 
  cat2 = 2 * cat1 →
  cat3 = (2 * cat1) / 3 →
  5 * cat1 + 5 * cat2 + 5 * cat3 = 55 →
  cat1 = 3 := by
sorry

end cat_meows_l166_16604


namespace greatest_root_of_f_l166_16630

def f (x : ℝ) := 12 * x^4 - 8 * x^2 + 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = Real.sqrt 2 / 2 ∧ 
  f r = 0 ∧ 
  ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end greatest_root_of_f_l166_16630


namespace pete_miles_walked_l166_16605

/-- Represents a pedometer with a maximum step count --/
structure Pedometer :=
  (max_steps : ℕ)

/-- Calculates the total number of steps given the number of resets and final reading --/
def total_steps (p : Pedometer) (resets : ℕ) (final_reading : ℕ) : ℕ :=
  resets * (p.max_steps + 1) + final_reading

/-- Converts steps to miles --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℚ :=
  (steps : ℚ) / (steps_per_mile : ℚ)

/-- Theorem stating the approximate number of miles Pete walked --/
theorem pete_miles_walked :
  let p : Pedometer := ⟨99999⟩
  let resets : ℕ := 44
  let final_reading : ℕ := 50000
  let steps_per_mile : ℕ := 1800
  let total_steps := total_steps p resets final_reading
  let miles_walked := steps_to_miles total_steps steps_per_mile
  ∃ ε > 0, abs (miles_walked - 2472.22) < ε := by
  sorry

end pete_miles_walked_l166_16605


namespace sqrt_equation_solution_l166_16665

theorem sqrt_equation_solution :
  ∃ y : ℚ, (40 : ℚ) / 60 = Real.sqrt (y / 60) → y = 80 / 3 := by
  sorry

end sqrt_equation_solution_l166_16665


namespace triangle_sum_equality_l166_16648

theorem triangle_sum_equality (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = a^2)
  (eq2 : y^2 + y*z + z^2 = b^2)
  (eq3 : x^2 + x*z + z^2 = c^2) :
  let p := (a + b + c) / 2
  x*y + y*z + x*z = 4 * Real.sqrt ((p * (p - a) * (p - b) * (p - c)) / 3) := by
sorry

end triangle_sum_equality_l166_16648


namespace fixed_points_theorem_l166_16609

/-- A function f(x) = ax^2 + (b+1)x + (b-1) where a ≠ 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

/-- x0 is a fixed point of f if f(x0) = x0 -/
def is_fixed_point (a b x0 : ℝ) : Prop := f a b x0 = x0

/-- The function has two distinct fixed points -/
def has_two_distinct_fixed_points (a b : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point a b x1 ∧ is_fixed_point a b x2

/-- The fixed points are symmetric with respect to the line y = kx + 1/(2a^2 + 1) -/
def fixed_points_symmetric (a b : ℝ) : Prop :=
  ∃ x1 x2 k : ℝ, x1 ≠ x2 ∧ is_fixed_point a b x1 ∧ is_fixed_point a b x2 ∧
    (f a b x1 + f a b x2) / 2 = k * (x1 + x2) / 2 + 1 / (2 * a^2 + 1)

/-- Main theorem -/
theorem fixed_points_theorem (a b : ℝ) (ha : a ≠ 0) :
  (has_two_distinct_fixed_points a b ↔ 0 < a ∧ a < 1) ∧
  (0 < a ∧ a < 1 → ∃ b_min : ℝ, ∀ b : ℝ, fixed_points_symmetric a b → b ≥ b_min) :=
sorry

end fixed_points_theorem_l166_16609


namespace sound_pressure_relations_l166_16602

noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

theorem sound_pressure_relations
  (p₀ : ℝ) (hp₀ : p₀ > 0)
  (p₁ p₂ p₃ : ℝ)
  (hp₁ : 60 ≤ sound_pressure_level p₁ p₀ ∧ sound_pressure_level p₁ p₀ ≤ 90)
  (hp₂ : 50 ≤ sound_pressure_level p₂ p₀ ∧ sound_pressure_level p₂ p₀ ≤ 60)
  (hp₃ : sound_pressure_level p₃ p₀ = 40) :
  p₁ ≥ p₂ ∧ p₃ = 100 * p₀ ∧ p₁ ≤ 100 * p₂ :=
by sorry

end sound_pressure_relations_l166_16602


namespace curve_tangent_to_line_l166_16658

/-- The curve y = x^2 - x + a is tangent to the line y = x + 1 if and only if a = 2 -/
theorem curve_tangent_to_line (a : ℝ) : 
  (∃ x y : ℝ, y = x^2 - x + a ∧ y = x + 1 ∧ 2*x - 1 = 1) ↔ a = 2 := by
  sorry

end curve_tangent_to_line_l166_16658


namespace cubic_roots_sum_cubes_l166_16672

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 220 * p - 7 = 0) →
  (3 * q^3 - 4 * q^2 + 220 * q - 7 = 0) →
  (3 * r^3 - 4 * r^2 + 220 * r - 7 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 64.556 := by
  sorry

end cubic_roots_sum_cubes_l166_16672


namespace range_of_a_l166_16686

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 9 :=
by sorry

end range_of_a_l166_16686


namespace sum_of_circle_areas_l166_16657

/-- The sum of the areas of an infinite sequence of circles with decreasing radii -/
theorem sum_of_circle_areas : 
  ∀ (π : ℝ), π > 0 → 
  (∑' n, π * (3 / (3 ^ n : ℝ))^2) = 9 * π / 8 := by
  sorry

end sum_of_circle_areas_l166_16657


namespace book_arrangement_l166_16666

theorem book_arrangement (n m : ℕ) (h : n + m = 8) :
  Nat.choose 8 n = Nat.choose 8 m :=
sorry

end book_arrangement_l166_16666


namespace fourth_task_end_time_l166_16632

-- Define the start time of the first task
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes since midnight

-- Define the end time of the third task
def end_third_task : Nat := 11 * 60 + 30  -- 11:30 AM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem fourth_task_end_time :
  let total_time := end_third_task - start_time
  let task_duration := total_time / 3
  let fourth_task_end := end_third_task + task_duration
  fourth_task_end = 12 * 60 + 20  -- 12:20 PM in minutes since midnight
  := by sorry

end fourth_task_end_time_l166_16632


namespace sufficient_not_necessary_l166_16687

/-- The function f(x) = ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The condition a < -2 is sufficient but not necessary for f to have a zero in [-1, 2] -/
theorem sufficient_not_necessary (a : ℝ) :
  (a < -2 → ∃ x ∈ Set.Icc (-1) 2, f a x = 0) ∧
  ¬(∃ x ∈ Set.Icc (-1) 2, f a x = 0 → a < -2) :=
sorry

end sufficient_not_necessary_l166_16687


namespace intersection_to_left_focus_distance_l166_16642

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection point in the first quadrant
def intersection_point (x y : ℝ) : Prop :=
  ellipse x y ∧ parabola x y ∧ x > 0 ∧ y > 0

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem intersection_to_left_focus_distance :
  ∀ x y : ℝ, intersection_point x y →
  Real.sqrt ((x - left_focus.1)^2 + (y - left_focus.2)^2) = 5/3 := by
  sorry

end intersection_to_left_focus_distance_l166_16642


namespace min_value_of_P_l166_16646

/-- The polynomial function P(x,y) -/
def P (x y : ℝ) : ℝ := 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2

/-- Theorem stating that the minimal value of P(x,y) is 3 -/
theorem min_value_of_P :
  (∀ x y : ℝ, P x y ≥ 3) ∧ (∃ x y : ℝ, P x y = 3) := by
  sorry

end min_value_of_P_l166_16646


namespace unique_prime_with_six_divisors_l166_16690

/-- A function that counts the number of distinct divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (p : ℕ) : Prop := sorry

theorem unique_prime_with_six_divisors :
  ∀ p : ℕ, is_prime p → (count_divisors (p^2 + 11) = 6) → p = 3 := by sorry

end unique_prime_with_six_divisors_l166_16690


namespace quadratic_integer_roots_l166_16685

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n = 0) ↔ (n = 3 ∨ n = 4) := by
  sorry

end quadratic_integer_roots_l166_16685


namespace least_n_radios_l166_16676

/-- Represents the problem of finding the least number of radios bought by a dealer. -/
def RadioProblem (n d : ℕ) : Prop :=
  d > 0 ∧  -- d is a positive integer
  (2 * d + (n - 4) * (d + 10 * n)) = n * (d + 100)  -- profit equation

/-- The least possible value of n that satisfies the RadioProblem. -/
theorem least_n_radios : 
  ∀ n d, RadioProblem n d → n ≥ 14 :=
sorry

end least_n_radios_l166_16676


namespace vehicle_distance_after_three_minutes_l166_16617

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

theorem vehicle_distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_minutes : ℝ := 3
  let time_hours : ℝ := time_minutes / 60
  distance_between_vehicles truck_speed car_speed time_hours = 1 := by
  sorry

end vehicle_distance_after_three_minutes_l166_16617


namespace complex_coordinate_l166_16603

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) : 
  z = 4 - 2 * Complex.I := by sorry

end complex_coordinate_l166_16603


namespace melanie_dimes_l166_16647

theorem melanie_dimes (initial_dimes mother_dimes final_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : mother_dimes = 4)
  (h3 : final_dimes = 19) :
  final_dimes - (initial_dimes + mother_dimes) = 8 := by
sorry

end melanie_dimes_l166_16647


namespace park_layout_diameter_l166_16675

/-- The diameter of the outer boundary of a circular park layout -/
def outer_boundary_diameter (statue_diameter bench_width path_width : ℝ) : ℝ :=
  statue_diameter + 2 * (bench_width + path_width)

/-- Theorem: The diameter of the outer boundary of the jogging path is 46 feet -/
theorem park_layout_diameter :
  outer_boundary_diameter 12 10 7 = 46 := by
  sorry

end park_layout_diameter_l166_16675


namespace quadratic_root_transformation_l166_16679

theorem quadratic_root_transformation (p q r : ℝ) (u v : ℝ) : 
  (p * u^2 + q * u + r = 0) ∧ (p * v^2 + q * v + r = 0) →
  ((2*p*u + q)^2 - (q/(4*p)) * (2*p*u + q) + r = 0) ∧
  ((2*p*v + q)^2 - (q/(4*p)) * (2*p*v + q) + r = 0) := by
sorry

end quadratic_root_transformation_l166_16679


namespace roger_tray_collection_l166_16608

/-- The number of trips required to collect trays -/
def numTrips (capacity traysTable1 traysTable2 : ℕ) : ℕ :=
  (traysTable1 + traysTable2 + capacity - 1) / capacity

theorem roger_tray_collection (capacity traysTable1 traysTable2 : ℕ) 
  (h1 : capacity = 4) 
  (h2 : traysTable1 = 10) 
  (h3 : traysTable2 = 2) : 
  numTrips capacity traysTable1 traysTable2 = 3 := by
  sorry

end roger_tray_collection_l166_16608


namespace function_equivalence_l166_16619

-- Define the function f
noncomputable def f : ℝ → ℝ :=
  fun x => -x^2 + 1/x - 2

-- State the theorem
theorem function_equivalence (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) (hx3 : x ≠ -1) :
  f (x - 1/x) = x / (x^2 - 1) - x^2 - 1/x^2 :=
by
  sorry


end function_equivalence_l166_16619


namespace dave_tickets_l166_16618

theorem dave_tickets (tickets_used : ℕ) (tickets_left : ℕ) : 
  tickets_used = 6 → tickets_left = 7 → tickets_used + tickets_left = 13 := by
  sorry

end dave_tickets_l166_16618


namespace wood_per_table_is_12_l166_16688

/-- The number of pieces of wood required to make a table -/
def wood_per_table : ℕ := sorry

/-- The total number of pieces of wood available -/
def total_wood : ℕ := 672

/-- The number of pieces of wood required to make a chair -/
def wood_per_chair : ℕ := 8

/-- The number of tables that can be made -/
def num_tables : ℕ := 24

/-- The number of chairs that can be made -/
def num_chairs : ℕ := 48

theorem wood_per_table_is_12 :
  wood_per_table = 12 :=
by sorry

end wood_per_table_is_12_l166_16688


namespace unique_monic_polynomial_l166_16673

/-- A monic polynomial of degree 2 satisfying f(0) = 10 and f(1) = 14 -/
def f (x : ℝ) : ℝ := x^2 + 3*x + 10

/-- The theorem stating that f is the unique monic polynomial of degree 2 satisfying the given conditions -/
theorem unique_monic_polynomial :
  ∀ g : ℝ → ℝ, (∃ a b : ℝ, ∀ x, g x = x^2 + a*x + b) →
  g 0 = 10 → g 1 = 14 → g = f :=
by sorry

end unique_monic_polynomial_l166_16673


namespace inscribed_square_side_length_l166_16671

theorem inscribed_square_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b * c) / (a^2 + b^2)
  s = 525 / 96 := by sorry

end inscribed_square_side_length_l166_16671


namespace twentieth_term_of_sequence_l166_16633

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_sequence (a₁ a₁₃ a₂₀ : ℝ) :
  a₁ = 3 →
  a₁₃ = 27 →
  (∃ d : ℝ, ∀ n : ℕ, arithmetic_sequence a₁ d n = a₁ + (n - 1 : ℝ) * d) →
  a₂₀ = arithmetic_sequence a₁ ((a₁₃ - a₁) / 12) 20 →
  a₂₀ = 41 := by
sorry

end twentieth_term_of_sequence_l166_16633


namespace circle_tangent_to_line_l166_16607

/-- A circle with equation x^2 + y^2 = n is tangent to the line x + y + 1 = 0 if and only if n = 1/2 -/
theorem circle_tangent_to_line (n : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = n ∧ x + y + 1 = 0 ∧ 
    ∀ (x' y' : ℝ), x'^2 + y'^2 = n → x' + y' + 1 ≠ 0 ∨ (x' = x ∧ y' = y)) ↔ 
  n = 1/2 := by
sorry


end circle_tangent_to_line_l166_16607


namespace total_spending_over_four_years_l166_16652

/-- The annual toy spending of three friends over four years. -/
def annual_toy_spending (trevor_spending : ℕ) (reed_diff : ℕ) (quinn_ratio : ℕ) (years : ℕ) : ℕ :=
  let reed_spending := trevor_spending - reed_diff
  let quinn_spending := reed_spending / quinn_ratio
  (trevor_spending + reed_spending + quinn_spending) * years

/-- Theorem stating the total spending of three friends over four years. -/
theorem total_spending_over_four_years :
  annual_toy_spending 80 20 2 4 = 680 := by
  sorry

#eval annual_toy_spending 80 20 2 4

end total_spending_over_four_years_l166_16652


namespace third_month_sale_l166_16629

def average_sale : ℕ := 7500
def num_months : ℕ := 6
def sale_month1 : ℕ := 7435
def sale_month2 : ℕ := 7927
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7562
def sale_month6 : ℕ := 5991

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 7855 := by
sorry

end third_month_sale_l166_16629


namespace min_value_fraction_l166_16682

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x - 2*y + 3*z = 0) : y^2 / (x*z) ≥ 3 :=
by sorry

end min_value_fraction_l166_16682


namespace house_sale_profit_l166_16601

theorem house_sale_profit (market_value : ℝ) (over_market_percentage : ℝ) 
  (tax_rate : ℝ) (num_people : ℕ) : 
  market_value = 500000 ∧ 
  over_market_percentage = 0.2 ∧ 
  tax_rate = 0.1 ∧ 
  num_people = 4 → 
  (market_value * (1 + over_market_percentage) * (1 - tax_rate)) / num_people = 135000 := by
  sorry

end house_sale_profit_l166_16601


namespace integral_special_function_l166_16683

theorem integral_special_function : 
  ∫ x in (0 : ℝ)..(Real.pi / 2), (1 - 5 * x^2) * Real.sin x = 11 - 5 * Real.pi := by
  sorry

end integral_special_function_l166_16683


namespace calculate_expression_l166_16655

theorem calculate_expression : |-3| + 8 / (-2) + Real.sqrt 16 = 3 := by
  sorry

end calculate_expression_l166_16655


namespace yellow_ball_count_l166_16606

/-- Given a bag with red and yellow balls, this theorem proves the number of yellow balls
    when the number of red balls and the probability of drawing a red ball are known. -/
theorem yellow_ball_count (total : ℕ) (red : ℕ) (p : ℚ) : 
  red = 8 →
  p = 1/3 →
  p = red / total →
  total - red = 16 := by
  sorry

end yellow_ball_count_l166_16606


namespace line_vector_at_negative_two_l166_16669

-- Define the line parameterization
def line_param (t : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem line_vector_at_negative_two :
  (∃ line_param : ℝ → ℝ × ℝ,
    (line_param 1 = (2, 5)) ∧
    (line_param 4 = (5, -7))) →
  (∃ line_param : ℝ → ℝ × ℝ,
    (line_param 1 = (2, 5)) ∧
    (line_param 4 = (5, -7)) ∧
    (line_param (-2) = (-1, 17))) :=
by sorry

end line_vector_at_negative_two_l166_16669


namespace perpendicular_vectors_x_value_l166_16677

def vector_a : ℝ × ℝ := (3, -1)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = 3 := by
  sorry

end perpendicular_vectors_x_value_l166_16677


namespace cubic_function_symmetry_l166_16691

/-- Given a cubic function f(x) = ax³ + bx + 1 where ab ≠ 0,
    if f(2016) = k, then f(-2016) = 2 - k -/
theorem cubic_function_symmetry (a b k : ℝ) (h1 : a * b ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  f 2016 = k → f (-2016) = 2 - k := by
  sorry

end cubic_function_symmetry_l166_16691


namespace vector_expression_l166_16670

/-- Given vectors a, b, and c in ℝ², prove that c = 2a - b -/
theorem vector_expression (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (-2, 3)) 
  (hc : c = (4, 1)) : 
  c = (2 : ℝ) • a - b := by sorry

end vector_expression_l166_16670


namespace problem_statement_l166_16649

/-- Given real numbers x and y satisfying x + y/4 = 1, prove:
    1. If |7-y| < 2x+3, then -1 < x < 0
    2. If x > 0 and y > 0, then sqrt(xy) ≥ xy -/
theorem problem_statement (x y : ℝ) (h1 : x + y / 4 = 1) :
  (∀ h2 : |7 - y| < 2*x + 3, -1 < x ∧ x < 0) ∧
  (∀ h3 : x > 0, ∀ h4 : y > 0, Real.sqrt (x * y) ≥ x * y) := by
  sorry

end problem_statement_l166_16649


namespace quadratic_polynomial_remainders_l166_16625

theorem quadratic_polynomial_remainders (m n : ℚ) : 
  (∀ x, (x^2 + m*x + n) % (x - m) = m ∧ (x^2 + m*x + n) % (x - n) = n) ↔ 
  ((m = 0 ∧ n = 0) ∨ (m = 1/2 ∧ n = 0) ∨ (m = 1 ∧ n = -1)) :=
by sorry

end quadratic_polynomial_remainders_l166_16625


namespace min_even_integers_l166_16621

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 → 
  a + b + c + d = 50 → 
  a + b + c + d + e + f = 70 → 
  ∃ (evens : Finset ℤ), evens ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ evens, Even x) ∧ 
    evens.card = 2 ∧ 
    (∀ (other_evens : Finset ℤ), other_evens ⊆ {a, b, c, d, e, f} → 
      (∀ x ∈ other_evens, Even x) → other_evens.card ≥ 2) :=
by sorry

end min_even_integers_l166_16621


namespace sum_of_integers_l166_16645

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 4)
  (eq2 : q - r + s = 5)
  (eq3 : r - s + p = 7)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 19 := by
  sorry

end sum_of_integers_l166_16645


namespace inequality_proof_l166_16663

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end inequality_proof_l166_16663


namespace birthday_problem_l166_16681

theorem birthday_problem (n : ℕ) (m : ℕ) (h1 : n = 400) (h2 : m = 365) :
  ∃ (i j : ℕ), i ≠ j ∧ i ≤ n ∧ j ≤ n ∧ (i.mod m = j.mod m) :=
sorry

end birthday_problem_l166_16681


namespace inverse_power_of_three_l166_16644

theorem inverse_power_of_three : 3⁻¹ = (1 : ℚ) / 3 := by sorry

end inverse_power_of_three_l166_16644


namespace expression_equality_l166_16678

theorem expression_equality : 
  (2025^3 - 3 * 2025^2 * 2026 + 5 * 2025 * 2026^2 - 2026^3 + 4) / (2025 * 2026) = 
  4052 + 3 / 2025 := by
sorry

end expression_equality_l166_16678


namespace campsite_coordinates_l166_16693

/-- Calculates the coordinates of a point that divides a line segment in a given ratio -/
def divideLineSegment (x1 y1 x2 y2 m n : ℚ) : ℚ × ℚ :=
  ((m * x2 + n * x1) / (m + n), (m * y2 + n * y1) / (m + n))

/-- The campsite coordinates problem -/
theorem campsite_coordinates :
  let annaStart : ℚ × ℚ := (3, -5)
  let bobStart : ℚ × ℚ := (7, 4)
  let campsite := divideLineSegment annaStart.1 annaStart.2 bobStart.1 bobStart.2 2 1
  campsite = (17/3, 1) := by
  sorry


end campsite_coordinates_l166_16693


namespace analysis_time_l166_16695

/-- The number of bones in the human body -/
def num_bones : ℕ := 206

/-- The time in minutes spent analyzing each bone -/
def minutes_per_bone : ℕ := 45

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The total time in hours required to analyze all bones in the human body -/
theorem analysis_time : (num_bones * minutes_per_bone : ℚ) / minutes_per_hour = 154.5 := by
  sorry

end analysis_time_l166_16695


namespace sum_of_digits_of_greatest_prime_divisor_l166_16684

def number : ℕ := 32767

/-- The greatest prime divisor of a natural number -/
def greatest_prime_divisor (n : ℕ) : ℕ := sorry

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor number) = 14 := by sorry

end sum_of_digits_of_greatest_prime_divisor_l166_16684


namespace largest_valid_n_l166_16641

def engineers : Nat := 6
def technicians : Nat := 12
def workers : Nat := 18

def total_individuals : Nat := engineers + technicians + workers

def is_valid_n (n : Nat) : Prop :=
  n ∣ total_individuals ∧
  n ≤ Nat.lcm (Nat.lcm engineers technicians) workers ∧
  ¬((n + 1) ∣ total_individuals)

theorem largest_valid_n :
  ∃ (n : Nat), is_valid_n n ∧ ∀ (m : Nat), is_valid_n m → m ≤ n :=
by
  sorry

end largest_valid_n_l166_16641


namespace birth_year_property_l166_16615

def current_year : Nat := 2023
def birth_year : Nat := 1957

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.repr.data.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem birth_year_property : 
  current_year - birth_year = sum_of_digits birth_year := by
  sorry

end birth_year_property_l166_16615


namespace annual_interest_calculation_l166_16636

theorem annual_interest_calculation (principal : ℝ) (quarterly_rate : ℝ) :
  principal = 10000 →
  quarterly_rate = 0.05 →
  (principal * quarterly_rate * 4) = 2000 := by
  sorry

end annual_interest_calculation_l166_16636


namespace garden_furniture_cost_l166_16698

/-- The combined cost of a garden table and bench, given their price relationship -/
theorem garden_furniture_cost (bench_price : ℕ) (table_price : ℕ) : 
  bench_price = 150 → 
  table_price = 2 * bench_price → 
  bench_price + table_price = 450 := by
  sorry

end garden_furniture_cost_l166_16698
