import Mathlib

namespace NUMINAMATH_CALUDE_x_varies_as_three_sevenths_power_of_z_l3746_374624

/-- Given that x varies directly as the cube of y, and y varies directly as the seventh root of z,
    prove that x varies as the (3/7)th power of z. -/
theorem x_varies_as_three_sevenths_power_of_z 
  (x y z : ℝ) 
  (hxy : ∃ (k : ℝ), x = k * y^3) 
  (hyz : ∃ (j : ℝ), y = j * z^(1/7)) :
  ∃ (m : ℝ), x = m * z^(3/7) := by
sorry

end NUMINAMATH_CALUDE_x_varies_as_three_sevenths_power_of_z_l3746_374624


namespace NUMINAMATH_CALUDE_water_pouring_time_l3746_374604

/-- Proves that pouring 18 gallons at a rate of 1 gallon every 20 seconds takes 6 minutes -/
theorem water_pouring_time (tank_capacity : ℕ) (pour_rate : ℚ) (remaining : ℕ) (poured : ℕ) :
  tank_capacity = 50 →
  pour_rate = 1 / 20 →
  remaining = 32 →
  poured = 18 →
  (poured : ℚ) / pour_rate / 60 = 6 :=
by sorry

end NUMINAMATH_CALUDE_water_pouring_time_l3746_374604


namespace NUMINAMATH_CALUDE_bryan_pushups_l3746_374603

/-- The number of push-ups Bryan did in total -/
def total_pushups (sets : ℕ) (pushups_per_set : ℕ) (reduction : ℕ) : ℕ :=
  (sets - 1) * pushups_per_set + (pushups_per_set - reduction)

/-- Proof that Bryan did 40 push-ups in total -/
theorem bryan_pushups :
  total_pushups 3 15 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bryan_pushups_l3746_374603


namespace NUMINAMATH_CALUDE_total_present_age_is_72_l3746_374661

/-- Given three people p, q, and r, prove that their total present age is 72 years -/
theorem total_present_age_is_72 
  (p q r : ℕ) -- Present ages of p, q, and r
  (h1 : p - 12 = (q - 12) / 2) -- 12 years ago, p was half of q's age
  (h2 : r - 12 = (p - 12) + (q - 12) - 3) -- r was 3 years younger than the sum of p and q's ages 12 years ago
  (h3 : ∃ (x : ℕ), p = 3*x ∧ q = 4*x ∧ r = 5*x) -- The ratio of their present ages is 3 : 4 : 5
  : p + q + r = 72 := by
  sorry


end NUMINAMATH_CALUDE_total_present_age_is_72_l3746_374661


namespace NUMINAMATH_CALUDE_divisiblity_by_thirty_l3746_374607

theorem divisiblity_by_thirty (p : ℕ) (h_prime : Nat.Prime p) (h_geq_seven : p ≥ 7) :
  ∃ k : ℕ, p^2 - 1 = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisiblity_by_thirty_l3746_374607


namespace NUMINAMATH_CALUDE_corn_selling_price_l3746_374652

/-- Calculates the selling price per bag of corn to achieve a desired profit percentage --/
theorem corn_selling_price 
  (seed_cost fertilizer_cost labor_cost : ℕ) 
  (num_bags : ℕ) 
  (profit_percentage : ℚ) 
  (h1 : seed_cost = 50)
  (h2 : fertilizer_cost = 35)
  (h3 : labor_cost = 15)
  (h4 : num_bags = 10)
  (h5 : profit_percentage = 10 / 100) :
  (seed_cost + fertilizer_cost + labor_cost : ℚ) * (1 + profit_percentage) / num_bags = 11 := by
sorry

end NUMINAMATH_CALUDE_corn_selling_price_l3746_374652


namespace NUMINAMATH_CALUDE_prove_complex_circle_theorem_l3746_374665

def complex_circle_theorem (z : ℂ) : Prop :=
  Complex.abs (z - Complex.I) = Real.sqrt 5 →
  ∃ (center : ℂ) (radius : ℝ),
    center = Complex.mk 0 1 ∧
    radius = Real.sqrt 5 ∧
    Complex.abs (z - center) = radius

theorem prove_complex_circle_theorem :
  ∀ z : ℂ, complex_circle_theorem z :=
by
  sorry

end NUMINAMATH_CALUDE_prove_complex_circle_theorem_l3746_374665


namespace NUMINAMATH_CALUDE_function_extrema_sum_l3746_374672

/-- Given f(x) = 2x^3 - ax^2 + 1 where a > 0, if the sum of the maximum and minimum values 
    of f(x) on [-1, 1] is 1, then a = 1/2 -/
theorem function_extrema_sum (a : ℝ) (h1 : a > 0) : 
  let f := fun x => 2 * x^3 - a * x^2 + 1
  (∃ M m : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ M ∧ m ≤ f x) ∧ M + m = 1) → 
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_function_extrema_sum_l3746_374672


namespace NUMINAMATH_CALUDE_polynomial_property_l3746_374681

theorem polynomial_property (P : ℤ → ℤ) (h_poly : ∀ a b : ℤ, ∃ c : ℤ, P a - P b = c * (a - b)) :
  P 1 = 2019 →
  P 2019 = 1 →
  ∃ k : ℤ, P k = k →
  k = 1010 :=
sorry

end NUMINAMATH_CALUDE_polynomial_property_l3746_374681


namespace NUMINAMATH_CALUDE_smallest_divisor_exponent_l3746_374698

def polynomial (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_exponent :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (z : ℂ), polynomial z = 0 → z^k = 1) ∧
  (∀ (m : ℕ), m > 0 → m < k → ∃ (w : ℂ), polynomial w = 0 ∧ w^m ≠ 1) ∧
  k = 120 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_exponent_l3746_374698


namespace NUMINAMATH_CALUDE_intersection_A_B_l3746_374687

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 > 0}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | -3 ≤ x ∧ x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3746_374687


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3746_374659

def last_two_digits (n : ℕ) : ℕ × ℕ :=
  ((n / 10) % 10, n % 10)

theorem last_two_digits_product (n : ℕ) 
  (h1 : n % 4 = 0) 
  (h2 : (last_two_digits n).1 + (last_two_digits n).2 = 14) : 
  (last_two_digits n).1 * (last_two_digits n).2 = 48 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3746_374659


namespace NUMINAMATH_CALUDE_mike_dogs_count_l3746_374660

/-- Represents the number of dogs Mike has -/
def number_of_dogs : ℕ := 2

/-- Weight of a cup of dog food in pounds -/
def cup_weight : ℚ := 1/4

/-- Number of cups each dog eats per feeding -/
def cups_per_feeding : ℕ := 6

/-- Number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- Number of bags of dog food Mike buys per month -/
def bags_per_month : ℕ := 9

/-- Weight of each bag of dog food in pounds -/
def bag_weight : ℕ := 20

/-- Number of days in a month -/
def days_per_month : ℕ := 30

theorem mike_dogs_count :
  number_of_dogs = 
    (bags_per_month * bag_weight) / 
    (cups_per_feeding * feedings_per_day * cup_weight * days_per_month) := by
  sorry

end NUMINAMATH_CALUDE_mike_dogs_count_l3746_374660


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3746_374631

theorem binomial_expansion_example : (7 + 2)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3746_374631


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l3746_374623

theorem hyperbola_minimum_value (x y : ℝ) :
  x^2 / 4 - y^2 = 1 →
  3 * x^2 - 2 * x * y ≥ 6 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l3746_374623


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_product_absolute_value_equation_solution_product_holds_l3746_374609

theorem absolute_value_equation_solution_product : ℝ → Prop :=
  fun x ↦ (|2 * x - 14| - 5 = 1) → 
    ∃ y, (|2 * y - 14| - 5 = 1) ∧ x * y = 40 ∧ 
    ∀ z, (|2 * z - 14| - 5 = 1) → (z = x ∨ z = y)

-- Proof
theorem absolute_value_equation_solution_product_holds :
  ∃ a b : ℝ, absolute_value_equation_solution_product a ∧
             absolute_value_equation_solution_product b ∧
             a ≠ b :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_product_absolute_value_equation_solution_product_holds_l3746_374609


namespace NUMINAMATH_CALUDE_bisector_proof_l3746_374611

/-- A circle in the 2D plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the 2D plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The given circle -/
def givenCircle : Circle :=
  { equation := λ x y ↦ x^2 + y^2 - 2*x - 4*y + 1 = 0 }

/-- The line to be proved as the bisector -/
def bisectorLine : Line :=
  { equation := λ x y ↦ x - y + 1 = 0 }

/-- Definition of a line bisecting a circle -/
def bisects (l : Line) (c : Circle) : Prop :=
  ∃ (center_x center_y : ℝ),
    (∀ x y, c.equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = 4) ∧
    l.equation center_x center_y

/-- Theorem stating that the given line bisects the given circle -/
theorem bisector_proof : bisects bisectorLine givenCircle := by
  sorry

end NUMINAMATH_CALUDE_bisector_proof_l3746_374611


namespace NUMINAMATH_CALUDE_f_negative_m_value_l3746_374649

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + x + 9) / (x^2 + 3)

theorem f_negative_m_value (m : ℝ) (h : f m = 10) : f (-m) = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_m_value_l3746_374649


namespace NUMINAMATH_CALUDE_book_pages_theorem_l3746_374692

theorem book_pages_theorem :
  ∀ (book1 book2 book3 : ℕ),
    (2 * book1) / 3 - (book1 / 3) = 20 →
    (3 * book2) / 5 - (2 * book2) / 5 = 15 →
    (3 * book3) / 4 - (book3 / 4) = 30 →
    book1 = 60 ∧ book2 = 75 ∧ book3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l3746_374692


namespace NUMINAMATH_CALUDE_marble_jar_ratio_l3746_374663

/-- Proves that the ratio of marbles in the second jar to the first jar is 2:1 --/
theorem marble_jar_ratio :
  ∀ (jar1 jar2 jar3 : ℕ),
  jar1 = 80 →
  jar3 = jar1 / 4 →
  jar1 + jar2 + jar3 = 260 →
  jar2 = 2 * jar1 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_jar_ratio_l3746_374663


namespace NUMINAMATH_CALUDE_tan_is_odd_l3746_374683

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define the property of being an odd function
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem tan_is_odd : IsOdd Real.tan := by
  sorry

end NUMINAMATH_CALUDE_tan_is_odd_l3746_374683


namespace NUMINAMATH_CALUDE_remaining_area_after_triangles_cut_l3746_374673

theorem remaining_area_after_triangles_cut (grid_side : ℕ) (dark_rect_dim : ℕ × ℕ) (light_rect_dim : ℕ × ℕ) : 
  grid_side = 6 →
  dark_rect_dim = (1, 3) →
  light_rect_dim = (2, 3) →
  (grid_side^2 : ℝ) - (dark_rect_dim.1 * dark_rect_dim.2 + light_rect_dim.1 * light_rect_dim.2 : ℝ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_after_triangles_cut_l3746_374673


namespace NUMINAMATH_CALUDE_quadratic_solution_and_sum_l3746_374626

theorem quadratic_solution_and_sum (x : ℝ) : 
  x^2 + 14*x = 96 → 
  ∃ (a b : ℕ), 
    (x = Real.sqrt a - b) ∧ 
    (x > 0) ∧ 
    (a = 145) ∧ 
    (b = 7) ∧ 
    (a + b = 152) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_and_sum_l3746_374626


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a1_value_l3746_374627

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a1_value
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a 1)
  (h_geom_mean : a 2 ^ 2 = a 1 * a 4) :
  a 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a1_value_l3746_374627


namespace NUMINAMATH_CALUDE_red_peaches_count_l3746_374639

theorem red_peaches_count (total_peaches : ℕ) (num_baskets : ℕ) (green_peaches : ℕ) :
  total_peaches = 10 →
  num_baskets = 1 →
  green_peaches = 6 →
  total_peaches - green_peaches = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l3746_374639


namespace NUMINAMATH_CALUDE_ferry_tourist_sum_l3746_374641

/-- The number of trips made by the ferry -/
def num_trips : ℕ := 15

/-- The initial number of tourists -/
def initial_tourists : ℕ := 100

/-- The decrease in number of tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  (n : ℤ) * (2 * a + (n - 1) * d) / 2

theorem ferry_tourist_sum :
  arithmetic_sum num_trips initial_tourists (-tourist_decrease) = 1290 :=
sorry

end NUMINAMATH_CALUDE_ferry_tourist_sum_l3746_374641


namespace NUMINAMATH_CALUDE_juan_running_l3746_374618

/-- The distance traveled when moving at a constant speed for a given time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Juan's running problem -/
theorem juan_running :
  let speed : ℝ := 10  -- miles per hour
  let time : ℝ := 8    -- hours
  distance speed time = 80 := by
  sorry

end NUMINAMATH_CALUDE_juan_running_l3746_374618


namespace NUMINAMATH_CALUDE_constant_speed_journey_time_l3746_374634

/-- Given a constant speed journey, prove the total travel time -/
theorem constant_speed_journey_time 
  (total_distance : ℝ) 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (h1 : total_distance = 400) 
  (h2 : initial_distance = 100) 
  (h3 : initial_time = 1) 
  (h4 : initial_distance / initial_time = (total_distance - initial_distance) / (total_time - initial_time)) : 
  total_time = 4 :=
by
  sorry

#check constant_speed_journey_time

end NUMINAMATH_CALUDE_constant_speed_journey_time_l3746_374634


namespace NUMINAMATH_CALUDE_sum_in_base_6_l3746_374655

/-- Converts a number from base 6 to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem sum_in_base_6 :
  let a := toBase10 [4, 3, 2, 1]  -- 1234₆
  let b := toBase10 [4, 3, 2]     -- 234₆
  let c := toBase10 [4, 3]        -- 34₆
  toBase6 (a + b + c) = [0, 5, 5, 2] -- 2550₆
  := by sorry

end NUMINAMATH_CALUDE_sum_in_base_6_l3746_374655


namespace NUMINAMATH_CALUDE_smallest_perimeter_l3746_374642

/-- Triangle PQR with positive integer side lengths, PQ = PR, and J is the intersection of angle bisectors of ∠Q and ∠R with QJ = 10 -/
structure IsoscelesTriangle where
  PQ : ℕ+
  QR : ℕ+
  J : ℝ × ℝ
  QJ_length : ℝ
  qj_eq_10 : QJ_length = 10

/-- The smallest possible perimeter of triangle PQR is 96 -/
theorem smallest_perimeter (t : IsoscelesTriangle) : 
  ∃ (min_perimeter : ℕ), min_perimeter = 96 ∧ 
  ∀ (perimeter : ℕ), perimeter ≥ min_perimeter :=
by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l3746_374642


namespace NUMINAMATH_CALUDE_number_999_in_column_C_l3746_374619

/-- Represents the columns in which numbers are arranged --/
inductive Column
  | A | B | C | D | E | F | G

/-- Determines the column for a given positive integer greater than 1 --/
def column_for_number (n : ℕ) : Column :=
  sorry

/-- The main theorem stating that 999 is in column C --/
theorem number_999_in_column_C : column_for_number 999 = Column.C := by
  sorry

end NUMINAMATH_CALUDE_number_999_in_column_C_l3746_374619


namespace NUMINAMATH_CALUDE_second_number_value_l3746_374602

theorem second_number_value (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 3 / 4)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0) :
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3746_374602


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3746_374676

/-- Given an arithmetic sequence {a_n} where a₂ + a₄ = 8 and a₁ = 2, prove that a₅ = 6 -/
theorem arithmetic_sequence_fifth_term (a : ℕ → ℝ) 
  (h_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_sum : a 2 + a 4 = 8)
  (h_first : a 1 = 2) : 
  a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3746_374676


namespace NUMINAMATH_CALUDE_kolya_time_on_DE_l3746_374612

/-- Represents a point in the park --/
structure Point

/-- Represents a route in the park --/
structure Route (α : Type) where
  points : List α

/-- Represents a cyclist --/
structure Cyclist where
  name : String
  route : Route Point
  speed : ℝ

theorem kolya_time_on_DE (petya kolya : Cyclist) 
  (h1 : petya.name = "Petya")
  (h2 : kolya.name = "Kolya")
  (h3 : petya.route = Route.mk [Point.mk, Point.mk, Point.mk]) -- A-B-C
  (h4 : kolya.route = Route.mk [Point.mk, Point.mk, Point.mk, Point.mk, Point.mk]) -- A-D-E-F-C
  (h5 : kolya.speed = 1.2 * petya.speed)
  (h6 : ∃ (t : ℝ), t = 12 ∧ 
    t = (List.length petya.route.points - 1) / petya.speed ∧
    t = (List.length kolya.route.points - 1) / kolya.speed) :
  ∃ (t_DE : ℝ), t_DE = 1 ∧ 
    t_DE = (1 / kolya.speed) * 
      ((List.length kolya.route.points - 1) / (List.length petya.route.points - 1)) :=
sorry

end NUMINAMATH_CALUDE_kolya_time_on_DE_l3746_374612


namespace NUMINAMATH_CALUDE_f_plus_g_equals_one_l3746_374621

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_equals_one
  (h1 : is_even f)
  (h2 : is_odd g)
  (h3 : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_equals_one_l3746_374621


namespace NUMINAMATH_CALUDE_symmetry_line_l3746_374643

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a circle --/
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a point is on a line --/
def onLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two circles are symmetric with respect to a line --/
def symmetric (c1 c2 : Circle) (l : Line) : Prop :=
  ∀ p : ℝ × ℝ, onCircle p c1 → 
    ∃ q : ℝ × ℝ, onCircle q c2 ∧ onLine ((p.1 + q.1) / 2, (p.2 + q.2) / 2) l

/-- The main theorem --/
theorem symmetry_line : 
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (2, -2), radius := 3 }
  let l : Line := { a := 1, b := -1, c := -2 }
  symmetric c1 c2 l := by sorry

end NUMINAMATH_CALUDE_symmetry_line_l3746_374643


namespace NUMINAMATH_CALUDE_cricketer_score_l3746_374699

theorem cricketer_score : ∀ (total_score : ℝ),
  (12 * 4 + 2 * 6 : ℝ) + 0.55223880597014926 * total_score = total_score →
  total_score = 134 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_score_l3746_374699


namespace NUMINAMATH_CALUDE_circle_trajectory_and_tangent_line_l3746_374600

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the moving circle P
def circle_P (x y r : ℝ) : Prop := (x - 2)^2 + y^2 = r^2

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the tangent line l
def line_l (x y k : ℝ) : Prop := y = k * (x + 4)

-- Theorem statement
theorem circle_trajectory_and_tangent_line :
  ∀ (x y r k : ℝ),
  (∃ (x₁ y₁ : ℝ), circle_M x₁ y₁ ∧ circle_P (x₁ - 1) y₁ r) →
  (∃ (x₂ y₂ : ℝ), circle_N x₂ y₂ ∧ circle_P (x₂ + 1) y₂ (3 - r)) →
  curve_C x y →
  line_l x y k →
  (∃ (x₃ y₃ : ℝ), circle_M x₃ y₃ ∧ line_l x₃ y₃ k) →
  (∃ (x₄ y₄ : ℝ), circle_P x₄ y₄ 2 ∧ line_l x₄ y₄ k) →
  (∀ (x₅ y₅ : ℝ), curve_C x₅ y₅ → line_l x₅ y₅ k → 
    (x₅ - x)^2 + (y₅ - y)^2 ≤ (18/7)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_tangent_line_l3746_374600


namespace NUMINAMATH_CALUDE_test_questions_count_l3746_374670

theorem test_questions_count : 
  ∀ (total : ℕ), 
    (total % 4 = 0) →  -- The test has 4 sections with equal number of questions
    (20 : ℚ) / total > (60 : ℚ) / 100 → -- Correct answer percentage > 60%
    (20 : ℚ) / total < (70 : ℚ) / 100 → -- Correct answer percentage < 70%
    total = 32 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l3746_374670


namespace NUMINAMATH_CALUDE_stone_value_proof_l3746_374666

/-- Represents the worth of a precious stone based on its weight and a proportionality constant -/
def stone_worth (weight : ℝ) (k : ℝ) : ℝ := k * weight^2

/-- Calculates the total worth of two pieces of a stone -/
def pieces_worth (weight1 : ℝ) (weight2 : ℝ) (k : ℝ) : ℝ :=
  stone_worth weight1 k + stone_worth weight2 k

theorem stone_value_proof (k : ℝ) :
  let original_weight : ℝ := 35
  let smaller_piece : ℝ := 2 * (original_weight / 7)
  let larger_piece : ℝ := 5 * (original_weight / 7)
  let loss : ℝ := 5000
  stone_worth original_weight k - pieces_worth smaller_piece larger_piece k = loss →
  stone_worth original_weight k = 12250 := by
sorry

end NUMINAMATH_CALUDE_stone_value_proof_l3746_374666


namespace NUMINAMATH_CALUDE_injective_function_property_l3746_374605

theorem injective_function_property {A : Type*} (f : A → A) (h : Function.Injective f) :
  ∀ (x₁ x₂ : A), x₁ ≠ x₂ → f x₁ ≠ f x₂ := by
  sorry

end NUMINAMATH_CALUDE_injective_function_property_l3746_374605


namespace NUMINAMATH_CALUDE_ten_cuts_eleven_pieces_l3746_374622

/-- The number of pieces resulting from cutting a log -/
def num_pieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: 10 cuts on a log result in 11 pieces -/
theorem ten_cuts_eleven_pieces : num_pieces 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ten_cuts_eleven_pieces_l3746_374622


namespace NUMINAMATH_CALUDE_count_solution_pairs_l3746_374638

/-- The number of pairs of positive integers (x, y) satisfying 2x + 3y = 2007 -/
def solution_count : ℕ := 334

/-- The predicate that checks if a pair of natural numbers satisfies the equation -/
def satisfies_equation (x y : ℕ) : Prop :=
  2 * x + 3 * y = 2007

theorem count_solution_pairs :
  (∃! n : ℕ, n = solution_count ∧
    ∃ s : Finset (ℕ × ℕ),
      s.card = n ∧
      (∀ p : ℕ × ℕ, p ∈ s ↔ (satisfies_equation p.1 p.2 ∧ p.1 > 0 ∧ p.2 > 0))) :=
sorry

end NUMINAMATH_CALUDE_count_solution_pairs_l3746_374638


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l3746_374617

/-- Two lines are parallel if their slopes are equal and not equal to their y-intercept ratios -/
def are_parallel (a : ℝ) : Prop :=
  a ≠ 0 ∧ a ≠ -1 ∧ (1 / a = a / 1) ∧ (1 / a ≠ (-2*a - 2) / (-a - 1))

/-- Given two lines l₁: x + ay = 2a + 2 and l₂: ax + y = a + 1 are parallel, prove that a = 1 -/
theorem parallel_lines_imply_a_equals_one :
  ∀ a : ℝ, are_parallel a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l3746_374617


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_seven_l3746_374669

theorem sum_of_roots_equals_seven : ∃ (r₁ r₂ : ℝ), 
  r₁^2 - 7*r₁ + 10 = 0 ∧ 
  r₂^2 - 7*r₂ + 10 = 0 ∧ 
  r₁ + r₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_seven_l3746_374669


namespace NUMINAMATH_CALUDE_students_walking_home_l3746_374632

theorem students_walking_home (bus_fraction automobile_fraction bicycle_fraction skateboard_fraction : ℚ) :
  bus_fraction = 1/3 →
  automobile_fraction = 1/5 →
  bicycle_fraction = 1/10 →
  skateboard_fraction = 1/15 →
  1 - (bus_fraction + automobile_fraction + bicycle_fraction + skateboard_fraction) = 3/10 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l3746_374632


namespace NUMINAMATH_CALUDE_minimum_value_of_function_minimum_value_achieved_l3746_374636

theorem minimum_value_of_function (x : ℝ) (h : x > 1) :
  2 * x + 2 / (x - 1) ≥ 6 :=
sorry

theorem minimum_value_achieved (x : ℝ) (h : x > 1) :
  2 * x + 2 / (x - 1) = 6 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_minimum_value_achieved_l3746_374636


namespace NUMINAMATH_CALUDE_orthogonal_families_l3746_374662

/-- A family of curves in the x-y plane -/
structure Curve :=
  (equation : ℝ → ℝ → ℝ → Prop)

/-- The given family of curves x^2 + y^2 = 2ax -/
def given_family : Curve :=
  ⟨λ a x y ↦ x^2 + y^2 = 2*a*x⟩

/-- The orthogonal family of curves x^2 + y^2 = Cy -/
def orthogonal_family : Curve :=
  ⟨λ C x y ↦ x^2 + y^2 = C*y⟩

/-- Two curves are orthogonal if their tangent lines are perpendicular at each intersection point -/
def orthogonal (c1 c2 : Curve) : Prop :=
  ∀ a C x y, c1.equation a x y → c2.equation C x y →
    ∃ m1 m2 : ℝ, (m1 * m2 = -1) ∧
      (∀ h, h ≠ 0 → (c1.equation a (x + h) (y + m1*h) ↔ c1.equation a x y)) ∧
      (∀ h, h ≠ 0 → (c2.equation C (x + h) (y + m2*h) ↔ c2.equation C x y))

/-- The main theorem stating that the given family and the orthogonal family are indeed orthogonal -/
theorem orthogonal_families : orthogonal given_family orthogonal_family :=
sorry

end NUMINAMATH_CALUDE_orthogonal_families_l3746_374662


namespace NUMINAMATH_CALUDE_vector_scalar_product_l3746_374657

/-- Given two vectors in R², prove that their scalar product equals 14 -/
theorem vector_scalar_product (a b : ℝ × ℝ) : 
  a = (2, 3) → b = (-1, 2) → (a + 2 • b) • b = 14 := by
  sorry

end NUMINAMATH_CALUDE_vector_scalar_product_l3746_374657


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3746_374651

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3746_374651


namespace NUMINAMATH_CALUDE_wintersweet_bouquet_solution_l3746_374654

/-- Represents the number of branches in a bouquet --/
structure BouquetComposition where
  typeA : ℕ
  typeB : ℕ

/-- Represents the total number of branches available --/
structure TotalBranches where
  typeA : ℕ
  typeB : ℕ

/-- Represents the number of bouquets of each type --/
structure BouquetCounts where
  alpha : ℕ
  beta : ℕ

def totalBranches : TotalBranches := { typeA := 142, typeB := 104 }

def alphaBouquet : BouquetComposition := { typeA := 6, typeB := 4 }
def betaBouquet : BouquetComposition := { typeA := 5, typeB := 4 }

/-- The theorem states that given the total branches and bouquet compositions,
    the solution of 12 Alpha bouquets and 14 Beta bouquets is correct --/
theorem wintersweet_bouquet_solution :
  ∃ (solution : BouquetCounts),
    solution.alpha = 12 ∧
    solution.beta = 14 ∧
    solution.alpha * alphaBouquet.typeA + solution.beta * betaBouquet.typeA = totalBranches.typeA ∧
    solution.alpha * alphaBouquet.typeB + solution.beta * betaBouquet.typeB = totalBranches.typeB :=
by sorry

end NUMINAMATH_CALUDE_wintersweet_bouquet_solution_l3746_374654


namespace NUMINAMATH_CALUDE_bhupathi_amount_l3746_374620

theorem bhupathi_amount (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A + B = 1210) (h4 : (4/15) * A = (2/5) * B) : B = 484 := by
  sorry

end NUMINAMATH_CALUDE_bhupathi_amount_l3746_374620


namespace NUMINAMATH_CALUDE_other_coin_denomination_l3746_374645

/-- Given a total of 336 coins with a total value of 7100 paise,
    where 260 of the coins are 20 paise coins,
    prove that the denomination of the other type of coin is 25 paise. -/
theorem other_coin_denomination
  (total_coins : ℕ)
  (total_value : ℕ)
  (twenty_paise_coins : ℕ)
  (h_total_coins : total_coins = 336)
  (h_total_value : total_value = 7100)
  (h_twenty_paise_coins : twenty_paise_coins = 260) :
  let other_coins := total_coins - twenty_paise_coins
  let other_denomination := (total_value - 20 * twenty_paise_coins) / other_coins
  other_denomination = 25 :=
by sorry

end NUMINAMATH_CALUDE_other_coin_denomination_l3746_374645


namespace NUMINAMATH_CALUDE_coin_to_sphere_weight_change_l3746_374647

theorem coin_to_sphere_weight_change 
  (R₁ R₂ R₃ : ℝ) 
  (h_positive : 0 < R₁ ∧ 0 < R₂ ∧ 0 < R₃) 
  (h_balance : R₁^2 + R₂^2 = R₃^2) : 
  R₁^3 + R₂^3 < R₃^3 := by
sorry

end NUMINAMATH_CALUDE_coin_to_sphere_weight_change_l3746_374647


namespace NUMINAMATH_CALUDE_ceramic_firing_probabilities_l3746_374613

/-- Represents the probability of success for a craft in each firing process -/
structure CraftProbabilities where
  first : Float
  second : Float

/-- Calculates the probability of exactly one success out of three independent events -/
def probExactlyOne (p1 p2 p3 : Float) : Float :=
  p1 * (1 - p2) * (1 - p3) + (1 - p1) * p2 * (1 - p3) + (1 - p1) * (1 - p2) * p3

/-- Calculates the expected value of a binomial distribution -/
def binomialExpectedValue (n : Nat) (p : Float) : Float :=
  n.toFloat * p

/-- Theorem about ceramic firing probabilities -/
theorem ceramic_firing_probabilities
  (craftA craftB craftC : CraftProbabilities)
  (h1 : craftA.first = 0.5)
  (h2 : craftB.first = 0.6)
  (h3 : craftC.first = 0.4)
  (h4 : craftA.second = 0.6)
  (h5 : craftB.second = 0.5)
  (h6 : craftC.second = 0.75) :
  (probExactlyOne craftA.first craftB.first craftC.first = 0.38) ∧
  (binomialExpectedValue 3 (craftA.first * craftA.second) = 0.9) := by
  sorry


end NUMINAMATH_CALUDE_ceramic_firing_probabilities_l3746_374613


namespace NUMINAMATH_CALUDE_probability_of_triangle_in_15_gon_l3746_374635

/-- Definition of a regular 15-gon -/
def regular_15_gon : Set (ℝ × ℝ) := sorry

/-- Function to check if three segments can form a triangle with positive area -/
def can_form_triangle (s1 s2 s3 : ℝ × ℝ × ℝ × ℝ) : Prop := sorry

/-- Total number of ways to choose 3 distinct segments from a 15-gon -/
def total_choices : ℕ := Nat.choose (Nat.choose 15 2) 3

/-- Number of ways to choose 3 distinct segments that form a triangle -/
def valid_choices : ℕ := sorry

theorem probability_of_triangle_in_15_gon :
  (valid_choices : ℚ) / total_choices = 163 / 455 := by sorry

end NUMINAMATH_CALUDE_probability_of_triangle_in_15_gon_l3746_374635


namespace NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l3746_374693

theorem ab_positive_necessary_not_sufficient (a b : ℝ) :
  (∀ a b, b / a + a / b > 2 → a * b > 0) ∧
  (∃ a b, a * b > 0 ∧ ¬(b / a + a / b > 2)) :=
sorry

end NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l3746_374693


namespace NUMINAMATH_CALUDE_valid_outfit_count_l3746_374644

/-- The number of shirts available. -/
def num_shirts : ℕ := 7

/-- The number of pants available. -/
def num_pants : ℕ := 5

/-- The number of hats available. -/
def num_hats : ℕ := 7

/-- The number of colors available for pants. -/
def num_pants_colors : ℕ := 5

/-- The number of colors available for shirts and hats. -/
def num_shirt_hat_colors : ℕ := 7

/-- The number of valid outfit choices. -/
def num_valid_outfits : ℕ := num_shirts * num_pants * num_hats - num_pants_colors

theorem valid_outfit_count : num_valid_outfits = 240 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_count_l3746_374644


namespace NUMINAMATH_CALUDE_num_triangles_in_polygon_l3746_374667

/-- 
A polygon with n sides, where n is at least 3.
-/
structure Polygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- 
The number of triangles formed by non-intersecting diagonals in an n-gon.
-/
def num_triangles (p : Polygon) : ℕ := p.n - 2

/-- 
Theorem: The number of triangles formed by non-intersecting diagonals 
in an n-gon is equal to n-2.
-/
theorem num_triangles_in_polygon (p : Polygon) : 
  num_triangles p = p.n - 2 := by
  sorry

end NUMINAMATH_CALUDE_num_triangles_in_polygon_l3746_374667


namespace NUMINAMATH_CALUDE_dozen_chocolate_cost_l3746_374678

/-- The cost of a dozen chocolate bars given the relative prices of magazines and chocolates -/
theorem dozen_chocolate_cost (magazine_price : ℝ) (chocolate_bar_price : ℝ) : 
  magazine_price = 1 →
  4 * chocolate_bar_price = 8 * magazine_price →
  12 * chocolate_bar_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_dozen_chocolate_cost_l3746_374678


namespace NUMINAMATH_CALUDE_min_plates_for_five_colors_l3746_374684

/-- The minimum number of plates to pull out to guarantee a matching pair -/
def min_plates_for_match (num_colors : ℕ) : ℕ :=
  num_colors + 1

/-- Theorem stating that for 5 colors, the minimum number of plates to pull out for a match is 6 -/
theorem min_plates_for_five_colors :
  min_plates_for_match 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_plates_for_five_colors_l3746_374684


namespace NUMINAMATH_CALUDE_carpet_exchange_theorem_l3746_374679

theorem carpet_exchange_theorem (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  ∃ c : ℝ, c > 0 ∧ ((c > 1 ∧ a / c < 1) ∨ (c < 1 ∧ a / c > 1)) := by
  sorry

end NUMINAMATH_CALUDE_carpet_exchange_theorem_l3746_374679


namespace NUMINAMATH_CALUDE_bird_migration_difference_l3746_374668

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 38

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 47

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 94

/-- Theorem: The difference between the number of bird families that flew to Asia
    and the number of bird families that flew to Africa is 47 -/
theorem bird_migration_difference :
  asia_families - africa_families = 47 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l3746_374668


namespace NUMINAMATH_CALUDE_hayden_earnings_l3746_374685

/-- Calculates the total earnings for a limo driver based on given parameters. -/
def limo_driver_earnings (hourly_wage : ℕ) (hours_worked : ℕ) (ride_bonus : ℕ) (rides_given : ℕ) 
  (review_bonus : ℕ) (positive_reviews : ℕ) (gas_price : ℕ) (gas_used : ℕ) : ℕ :=
  hourly_wage * hours_worked + ride_bonus * rides_given + review_bonus * positive_reviews + gas_price * gas_used

/-- Proves that Hayden's earnings for the day equal $226 given the specified conditions. -/
theorem hayden_earnings : 
  limo_driver_earnings 15 8 5 3 20 2 3 17 = 226 := by
  sorry

end NUMINAMATH_CALUDE_hayden_earnings_l3746_374685


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_l3746_374633

def b (n : ℕ) : ℕ := n.factorial + n^2

theorem max_gcd_consecutive_b : ∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 2 ∧ 
  ∃ m : ℕ, Nat.gcd (b m) (b (m + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_l3746_374633


namespace NUMINAMATH_CALUDE_natasha_dimes_problem_l3746_374630

theorem natasha_dimes_problem :
  ∃! n : ℕ, 100 < n ∧ n < 200 ∧
    n % 3 = 2 ∧
    n % 4 = 2 ∧
    n % 5 = 2 ∧
    n % 7 = 2 ∧
    n = 182 := by
  sorry

end NUMINAMATH_CALUDE_natasha_dimes_problem_l3746_374630


namespace NUMINAMATH_CALUDE_vector_relations_l3746_374691

-- Define the vectors
def a : ℝ × ℝ := (3, -2)
def b (y : ℝ) : ℝ × ℝ := (-1, y)
def c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Define perpendicularity for 2D vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define parallelism for 2D vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_relations :
  (∀ y : ℝ, perpendicular a (b y) → y = 3/2) ∧
  (∀ x : ℝ, parallel a (c x) → x = 15/2) := by sorry

end NUMINAMATH_CALUDE_vector_relations_l3746_374691


namespace NUMINAMATH_CALUDE_range_of_trig_function_l3746_374680

theorem range_of_trig_function :
  ∀ x : ℝ, (3 / 8 : ℝ) ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧
            Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_trig_function_l3746_374680


namespace NUMINAMATH_CALUDE_morning_campers_count_l3746_374637

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 39

/-- The additional number of campers who went rowing in the morning compared to the afternoon -/
def additional_morning_campers : ℕ := 5

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := afternoon_campers + additional_morning_campers

theorem morning_campers_count : morning_campers = 44 := by
  sorry

end NUMINAMATH_CALUDE_morning_campers_count_l3746_374637


namespace NUMINAMATH_CALUDE_initial_cookies_l3746_374615

/-- The number of cookies remaining after the first day -/
def remaining_after_first_day (C : ℚ) : ℚ :=
  C * (1/4) * (4/5)

/-- The number of cookies remaining after the second day -/
def remaining_after_second_day (C : ℚ) : ℚ :=
  remaining_after_first_day C * (1/2)

/-- Theorem stating the initial number of cookies -/
theorem initial_cookies : ∃ C : ℚ, C > 0 ∧ remaining_after_second_day C = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_cookies_l3746_374615


namespace NUMINAMATH_CALUDE_josie_safari_count_l3746_374696

/-- The total number of animals Josie counted on safari -/
def total_animals (antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants : ℕ) : ℕ :=
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants

/-- Theorem stating the total number of animals Josie counted -/
theorem josie_safari_count : ∃ (antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants : ℕ),
  antelopes = 80 ∧
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards = rabbits / 2 ∧
  giraffes = antelopes + 15 ∧
  lions = leopards + giraffes ∧
  elephants = 3 * lions ∧
  total_animals antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants = 1308 :=
by
  sorry

end NUMINAMATH_CALUDE_josie_safari_count_l3746_374696


namespace NUMINAMATH_CALUDE_range_of_a_l3746_374689

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ a ∈ Set.Ioo (-1 : ℝ) 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3746_374689


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3746_374690

theorem max_value_on_circle (x y : ℝ) : 
  (x - 1)^2 + y^2 = 4 → 
  ∃ b : ℝ, (∀ x' y' : ℝ, (x' - 1)^2 + y'^2 = 4 → 2*x' + y'^2 ≤ b) ∧ 
           (∃ x'' y'' : ℝ, (x'' - 1)^2 + y''^2 = 4 ∧ 2*x'' + y''^2 = b) ∧
           b = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3746_374690


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l3746_374675

theorem largest_power_of_two_dividing_difference : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (17^4 - 13^4) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (17^4 - 13^4) → m ≤ k :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l3746_374675


namespace NUMINAMATH_CALUDE_simplify_power_l3746_374656

theorem simplify_power (y : ℝ) : (3 * y^4)^4 = 81 * y^16 := by sorry

end NUMINAMATH_CALUDE_simplify_power_l3746_374656


namespace NUMINAMATH_CALUDE_probability_of_scoring_five_l3746_374677

def num_balls : ℕ := 2
def num_draws : ℕ := 3
def red_ball_score : ℕ := 2
def black_ball_score : ℕ := 1
def target_score : ℕ := 5

def probability_of_drawing_red : ℚ := 1 / 2

theorem probability_of_scoring_five (n : ℕ) (k : ℕ) (p : ℚ) :
  n = num_draws →
  k = 2 →
  p = probability_of_drawing_red →
  (Nat.choose n k * p^k * (1 - p)^(n - k) : ℚ) = 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_scoring_five_l3746_374677


namespace NUMINAMATH_CALUDE_f_of_5_equals_20_l3746_374640

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem f_of_5_equals_20 : f 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_20_l3746_374640


namespace NUMINAMATH_CALUDE_village_population_l3746_374664

/-- Given that 40% of a village's population is 23040, prove that the total population is 57600. -/
theorem village_population (population : ℕ) (h : (40 : ℕ) * population = 100 * 23040) : population = 57600 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3746_374664


namespace NUMINAMATH_CALUDE_intersection_of_line_and_curve_l3746_374658

/-- Line l is defined by the equation 2x - y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- Curve C is defined by the equation y² = 2x -/
def curve_C (x y : ℝ) : Prop := y^2 = 2 * x

/-- The intersection points of line l and curve C -/
def intersection_points : Set (ℝ × ℝ) := {(2, 2), (1/2, -1)}

/-- Theorem stating that the intersection points of line l and curve C are (2, 2) and (1/2, -1) -/
theorem intersection_of_line_and_curve :
  ∀ p : ℝ × ℝ, (line_l p.1 p.2 ∧ curve_C p.1 p.2) ↔ p ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_line_and_curve_l3746_374658


namespace NUMINAMATH_CALUDE_equation_solution_l3746_374625

theorem equation_solution :
  ∃! x : ℚ, x ≠ -2 ∧ (5 * x^2 + 4 * x + 2) / (x + 2) = 5 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3746_374625


namespace NUMINAMATH_CALUDE_new_partner_associate_ratio_l3746_374650

/-- Given a firm with partners and associates, this theorem proves the new ratio
    after hiring additional associates. -/
theorem new_partner_associate_ratio
  (initial_partner_count : ℕ)
  (initial_associate_count : ℕ)
  (additional_associates : ℕ)
  (h1 : initial_partner_count = 18)
  (h2 : initial_associate_count = 567)
  (h3 : additional_associates = 45) :
  (initial_partner_count : ℚ) / (initial_associate_count + additional_associates : ℚ) = 1 / 34 := by
  sorry

#check new_partner_associate_ratio

end NUMINAMATH_CALUDE_new_partner_associate_ratio_l3746_374650


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l3746_374694

theorem sum_mod_thirteen : (9010 + 9011 + 9012 + 9013 + 9014) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l3746_374694


namespace NUMINAMATH_CALUDE_floor_abs_sum_equality_l3746_374646

theorem floor_abs_sum_equality : ⌊|(-3.7 : ℝ)|⌋ + |⌊(-3.7 : ℝ)⌋| = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_equality_l3746_374646


namespace NUMINAMATH_CALUDE_circle_area_l3746_374695

theorem circle_area (r : ℝ) (h : 6 * (1 / (2 * Real.pi * r)) = r) : π * r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l3746_374695


namespace NUMINAMATH_CALUDE_degree_of_specific_monomial_l3746_374628

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (m : ℕ) (n : ℕ) : ℕ := m + n

/-- The degree of the monomial (1/7)mn^2 is 3 -/
theorem degree_of_specific_monomial : degree_of_monomial 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_specific_monomial_l3746_374628


namespace NUMINAMATH_CALUDE_distance_between_centers_l3746_374608

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0
  xy_length : (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 80^2
  xz_length : (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 150^2
  yz_length : (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 170^2

-- Define the inscribed circle C₁
def InscribedCircle (T : Triangle X Y Z) (C : ℝ × ℝ) (r : ℝ) : Prop := sorry

-- Define MN perpendicular to XZ and tangent to C₁
def MN_Perpendicular_Tangent (T : Triangle X Y Z) (C₁ : ℝ × ℝ) (r₁ : ℝ) (M N : ℝ × ℝ) : Prop := sorry

-- Define AB perpendicular to XY and tangent to C₁
def AB_Perpendicular_Tangent (T : Triangle X Y Z) (C₁ : ℝ × ℝ) (r₁ : ℝ) (A B : ℝ × ℝ) : Prop := sorry

-- Define the inscribed circle C₂ of MZN
def InscribedCircle_MZN (T : Triangle X Y Z) (M N : ℝ × ℝ) (C₂ : ℝ × ℝ) (r₂ : ℝ) : Prop := sorry

-- Define the inscribed circle C₃ of YAB
def InscribedCircle_YAB (T : Triangle X Y Z) (A B : ℝ × ℝ) (C₃ : ℝ × ℝ) (r₃ : ℝ) : Prop := sorry

theorem distance_between_centers (X Y Z M N A B C₁ C₂ C₃ : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) 
  (h_triangle : Triangle X Y Z)
  (h_c₁ : InscribedCircle h_triangle C₁ r₁)
  (h_mn : MN_Perpendicular_Tangent h_triangle C₁ r₁ M N)
  (h_ab : AB_Perpendicular_Tangent h_triangle C₁ r₁ A B)
  (h_c₂ : InscribedCircle_MZN h_triangle M N C₂ r₂)
  (h_c₃ : InscribedCircle_YAB h_triangle A B C₃ r₃) :
  (C₂.1 - C₃.1)^2 + (C₂.2 - C₃.2)^2 = 9884.5 := by sorry

end NUMINAMATH_CALUDE_distance_between_centers_l3746_374608


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3746_374686

theorem geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence definition
  a 4 = 1.5 →                   -- 4th term is 1.5
  a 10 = 1.62 →                 -- 10th term is 1.62
  a 7 = Real.sqrt 2.43 :=        -- 7th term is √2.43
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l3746_374686


namespace NUMINAMATH_CALUDE_fraction_subtraction_proof_l3746_374616

theorem fraction_subtraction_proof :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_proof_l3746_374616


namespace NUMINAMATH_CALUDE_primes_rounding_to_40_l3746_374653

def roundToNearestTen (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem primes_rounding_to_40 :
  ∃! (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p ∧ roundToNearestTen p = 40) ∧ 
    (∀ p, Nat.Prime p → roundToNearestTen p = 40 → p ∈ S) ∧ 
    S.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_primes_rounding_to_40_l3746_374653


namespace NUMINAMATH_CALUDE_problem_statement_l3746_374688

theorem problem_statement (x y : ℝ) : (x + 1)^2 + |y - 2| = 0 → 2*x + 3*y = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3746_374688


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3746_374601

/-- An isosceles triangle with perimeter 3.74 and leg length 1.5 has a base length of 0.74 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let perimeter : ℝ := 3.74
    let leg : ℝ := 1.5
    (2 * leg + base = perimeter) → (base = 0.74)

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 0.74 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3746_374601


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3746_374671

/-- The equation of a hyperbola given specific conditions -/
theorem hyperbola_equation :
  ∀ (a b : ℝ) (P : ℝ × ℝ),
  a > 0 ∧ b > 0 →
  (∀ (x y : ℝ), y^2 = -8*x → (x + 2)^2 + y^2 = 4) →  -- Focus of parabola is (-2, 0)
  (P.1)^2 / a^2 - (P.2)^2 / b^2 = 1 →  -- P lies on the hyperbola
  P = (2 * Real.sqrt 3, 2) →
  (∀ (x y : ℝ), x^2 / 4 - y^2 / 2 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3746_374671


namespace NUMINAMATH_CALUDE_min_value_theorem_l3746_374674

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 3 → 1 / (a + 1) + 2 / b ≥ 1 / (x + 1) + 2 / y) →
  1 / (x + 1) + 2 / y = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3746_374674


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3746_374648

/-- The standard equation of a hyperbola with given foci and real axis length -/
theorem hyperbola_equation (x y : ℝ) : 
  let foci_distance : ℝ := 8
  let real_axis_length : ℝ := 4
  let a : ℝ := real_axis_length / 2
  let c : ℝ := foci_distance / 2
  let b_squared : ℝ := c^2 - a^2
  x^2 / a^2 - y^2 / b_squared = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3746_374648


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_isosceles_triangle_area_proof_l3746_374610

/-- The area of an isosceles triangle with two sides of length 13 and a base of 10 is 60 -/
theorem isosceles_triangle_area : ℝ → Prop :=
  fun area =>
    ∃ (x y z : ℝ),
      x = 13 ∧ y = 13 ∧ z = 10 ∧  -- Two sides are 13, base is 10
      x = y ∧                     -- Isosceles condition
      area = (z * (x ^ 2 - (z / 2) ^ 2).sqrt) / 2 ∧  -- Area formula
      area = 60

/-- Proof of the theorem -/
theorem isosceles_triangle_area_proof : isosceles_triangle_area 60 := by
  sorry

#check isosceles_triangle_area_proof

end NUMINAMATH_CALUDE_isosceles_triangle_area_isosceles_triangle_area_proof_l3746_374610


namespace NUMINAMATH_CALUDE_gabby_savings_l3746_374629

/-- Represents the cost of the makeup set in dollars -/
def makeup_cost : ℕ := 65

/-- Represents the amount Gabby's mom gives her in dollars -/
def mom_gift : ℕ := 20

/-- Represents the additional amount Gabby needs after receiving the gift in dollars -/
def additional_needed : ℕ := 10

/-- Represents Gabby's initial savings in dollars -/
def initial_savings : ℕ := 35

theorem gabby_savings :
  initial_savings + mom_gift + additional_needed = makeup_cost :=
by sorry

end NUMINAMATH_CALUDE_gabby_savings_l3746_374629


namespace NUMINAMATH_CALUDE_min_value_x_minus_3y_l3746_374697

theorem min_value_x_minus_3y (x y : ℝ) (hx : x > 1) (hy : y < 0) (h : 3 * y * (1 - x) = x + 8) :
  ∀ z, x - 3 * y ≥ z → z ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_minus_3y_l3746_374697


namespace NUMINAMATH_CALUDE_diagonals_properties_l3746_374682

/-- Number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem diagonals_properties :
  (∀ n : ℕ, n ≥ 4 → num_diagonals n = n * (n - 3) / 2) →
  num_diagonals 4 = 2 →
  num_diagonals 5 = 5 ∧
  num_diagonals 6 - num_diagonals 5 = 4 ∧
  ∀ n : ℕ, n ≥ 4 → num_diagonals (n + 1) - num_diagonals n = n - 1 :=
by sorry

end NUMINAMATH_CALUDE_diagonals_properties_l3746_374682


namespace NUMINAMATH_CALUDE_small_circle_radius_l3746_374606

/-- Given a large circle with radius 6 meters containing five congruent smaller circles
    arranged such that the diameter of the large circle equals the sum of the diameters
    of three smaller circles, the radius of each smaller circle is 2 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 6 → 2 * R = 3 * (2 * r) → r = 2 :=
by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3746_374606


namespace NUMINAMATH_CALUDE_ice_cream_bar_price_l3746_374614

theorem ice_cream_bar_price 
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℚ)
  (sundae_price : ℚ)
  (h1 : num_ice_cream_bars = 125)
  (h2 : num_sundaes = 125)
  (h3 : total_price = 225)
  (h4 : sundae_price = 6/5) :
  (total_price - num_sundaes * sundae_price) / num_ice_cream_bars = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_bar_price_l3746_374614
