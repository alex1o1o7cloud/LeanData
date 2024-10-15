import Mathlib

namespace NUMINAMATH_CALUDE_second_order_de_solution_l2819_281994

/-- Given a second-order linear homogeneous differential equation with constant coefficients:
    y'' - 5y' - 6y = 0, prove that y = C₁e^(6x) + C₂e^(-x) is the general solution. -/
theorem second_order_de_solution (y : ℝ → ℝ) (C₁ C₂ : ℝ) :
  (∀ x, (deriv^[2] y) x - 5 * (deriv y) x - 6 * y x = 0) ↔
  (∃ C₁ C₂, ∀ x, y x = C₁ * Real.exp (6 * x) + C₂ * Real.exp (-x)) :=
sorry


end NUMINAMATH_CALUDE_second_order_de_solution_l2819_281994


namespace NUMINAMATH_CALUDE_combination_sum_equals_466_l2819_281969

theorem combination_sum_equals_466 (n : ℕ) 
  (h1 : 38 ≥ n) 
  (h2 : 3 * n ≥ 38 - n) 
  (h3 : n + 21 ≥ 3 * n) : 
  Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n) = 466 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_466_l2819_281969


namespace NUMINAMATH_CALUDE_distinct_numbers_ratio_l2819_281943

theorem distinct_numbers_ratio (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (h4 : (b - a)^2 - 4*(b - c)*(c - a) = 0) : 
  (b - c) / (c - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_distinct_numbers_ratio_l2819_281943


namespace NUMINAMATH_CALUDE_pizza_delivery_time_per_stop_l2819_281971

theorem pizza_delivery_time_per_stop 
  (total_pizzas : ℕ) 
  (double_order_stops : ℕ) 
  (total_delivery_time : ℕ) 
  (h1 : total_pizzas = 12) 
  (h2 : double_order_stops = 2) 
  (h3 : total_delivery_time = 40) : 
  (total_delivery_time : ℚ) / (total_pizzas - double_order_stops : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_delivery_time_per_stop_l2819_281971


namespace NUMINAMATH_CALUDE_car_speed_time_relationship_l2819_281948

theorem car_speed_time_relationship 
  (distance : ℝ) 
  (speed_A time_A : ℝ) 
  (speed_B time_B : ℝ) 
  (h1 : distance > 0) 
  (h2 : speed_A > 0) 
  (h3 : speed_B = 3 * speed_A) 
  (h4 : distance = speed_A * time_A) 
  (h5 : distance = speed_B * time_B) : 
  time_B = time_A / 3 := by
sorry


end NUMINAMATH_CALUDE_car_speed_time_relationship_l2819_281948


namespace NUMINAMATH_CALUDE_total_books_is_14_l2819_281977

/-- The number of books a librarian takes away. -/
def librarian_books : ℕ := 2

/-- The number of books that can fit on a shelf. -/
def books_per_shelf : ℕ := 3

/-- The number of shelves Roger needs. -/
def shelves_needed : ℕ := 4

/-- The total number of books to put away. -/
def total_books : ℕ := librarian_books + books_per_shelf * shelves_needed

theorem total_books_is_14 : total_books = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_14_l2819_281977


namespace NUMINAMATH_CALUDE_adi_change_l2819_281929

/-- Calculate the change Adi will receive after purchasing items and paying with a $20 bill. -/
theorem adi_change (pencil_cost notebook_cost colored_pencils_cost paid : ℚ) : 
  pencil_cost = 35/100 →
  notebook_cost = 3/2 →
  colored_pencils_cost = 11/4 →
  paid = 20 →
  paid - (pencil_cost + notebook_cost + colored_pencils_cost) = 77/5 := by
  sorry

#eval (20 : ℚ) - (35/100 + 3/2 + 11/4)

end NUMINAMATH_CALUDE_adi_change_l2819_281929


namespace NUMINAMATH_CALUDE_min_value_expression_l2819_281932

theorem min_value_expression (x y z : ℝ) 
  (hx : -1 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 1) 
  (hz : -1 < z ∧ z < 1) : 
  1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2819_281932


namespace NUMINAMATH_CALUDE_swimming_ratio_proof_l2819_281944

/-- Given information about the swimming abilities of Yvonne, Joel, and their younger sister,
    prove that the ratio of laps swum by the younger sister to Yvonne is 1:2. -/
theorem swimming_ratio_proof (yvonne_laps joel_laps : ℕ) (joel_ratio : ℕ) :
  yvonne_laps = 10 →
  joel_laps = 15 →
  joel_ratio = 3 →
  (joel_laps / joel_ratio : ℚ) / yvonne_laps = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_ratio_proof_l2819_281944


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l2819_281980

theorem stratified_sampling_male_count :
  let total_students : ℕ := 980
  let male_students : ℕ := 560
  let sample_size : ℕ := 280
  let sample_ratio : ℚ := sample_size / total_students
  sample_ratio * male_students = 160 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l2819_281980


namespace NUMINAMATH_CALUDE_intersected_cubes_in_4x4x4_cube_l2819_281992

/-- Represents a cube composed of unit cubes -/
structure UnitCube where
  side : ℕ

/-- Represents a plane in 3D space -/
structure Plane

/-- Predicate to check if a plane is perpendicular to and bisects an internal diagonal of a cube -/
def is_perpendicular_bisector (c : UnitCube) (p : Plane) : Prop :=
  sorry

/-- Counts the number of unit cubes intersected by a plane in a given cube -/
def intersected_cubes (c : UnitCube) (p : Plane) : ℕ :=
  sorry

/-- Theorem stating that a plane perpendicular to and bisecting an internal diagonal
    of a 4x4x4 cube intersects exactly 40 unit cubes -/
theorem intersected_cubes_in_4x4x4_cube (c : UnitCube) (p : Plane) :
  c.side = 4 → is_perpendicular_bisector c p → intersected_cubes c p = 40 :=
by sorry

end NUMINAMATH_CALUDE_intersected_cubes_in_4x4x4_cube_l2819_281992


namespace NUMINAMATH_CALUDE_not_sixth_power_l2819_281917

theorem not_sixth_power (n : ℕ) : ¬ ∃ (k : ℤ), 6 * (n : ℤ)^3 + 3 = k^6 := by
  sorry

end NUMINAMATH_CALUDE_not_sixth_power_l2819_281917


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l2819_281934

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- An event when drawing balls from the bag -/
structure Event (bag : Bag) where
  pred : (ℕ × ℕ) → Prop

/-- The bag in our problem -/
def problem_bag : Bag := { red := 3, white := 3 }

/-- The event "At least 2 white balls" -/
def at_least_2_white (bag : Bag) : Event bag :=
  { pred := λ (r, w) => w ≥ 2 }

/-- The event "All red balls" -/
def all_red (bag : Bag) : Event bag :=
  { pred := λ (r, w) => r = 3 ∧ w = 0 }

/-- Two events are mutually exclusive -/
def mutually_exclusive (bag : Bag) (e1 e2 : Event bag) : Prop :=
  ∀ r w, (r + w = 3) → ¬(e1.pred (r, w) ∧ e2.pred (r, w))

/-- Two events are contradictory -/
def contradictory (bag : Bag) (e1 e2 : Event bag) : Prop :=
  ∀ r w, (r + w = 3) → (e1.pred (r, w) ↔ ¬e2.pred (r, w))

/-- The main theorem to prove -/
theorem events_mutually_exclusive_not_contradictory :
  mutually_exclusive problem_bag (at_least_2_white problem_bag) (all_red problem_bag) ∧
  ¬contradictory problem_bag (at_least_2_white problem_bag) (all_red problem_bag) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l2819_281934


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2819_281972

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | x^2 + 2*x - 3 > 0} →
  B = {-1, 0, 1, 2} →
  A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2819_281972


namespace NUMINAMATH_CALUDE_equation_solution_l2819_281996

theorem equation_solution :
  ∀ y : ℝ, (3 + 1.5 * y^2 = 0.5 * y^2 + 16) ↔ (y = Real.sqrt 13 ∨ y = -Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2819_281996


namespace NUMINAMATH_CALUDE_tv_purchase_months_l2819_281941

/-- Calculates the number of months required to purchase a TV given income and expenses -/
def monthsToTV (monthlyIncome : ℕ) (foodExpense : ℕ) (utilitiesExpense : ℕ) (otherExpenses : ℕ)
                (currentSavings : ℕ) (tvCost : ℕ) : ℕ :=
  let totalExpenses := foodExpense + utilitiesExpense + otherExpenses
  let disposableIncome := monthlyIncome - totalExpenses
  let amountNeeded := tvCost - currentSavings
  (amountNeeded + disposableIncome - 1) / disposableIncome

theorem tv_purchase_months :
  monthsToTV 30000 15000 5000 2500 10000 25000 = 2 :=
sorry

end NUMINAMATH_CALUDE_tv_purchase_months_l2819_281941


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2819_281931

theorem quadratic_equation_solution (x : ℝ) : 16 * x^2 = 81 ↔ x = 2.25 ∨ x = -2.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2819_281931


namespace NUMINAMATH_CALUDE_total_value_of_treats_l2819_281924

-- Define the given values
def hotel_price_per_night : ℝ := 4000
def hotel_nights : ℕ := 2
def hotel_discount : ℝ := 0.05
def car_price : ℝ := 30000
def car_tax : ℝ := 0.10
def house_multiplier : ℝ := 4
def house_tax : ℝ := 0.02
def yacht_multiplier : ℝ := 2
def yacht_discount : ℝ := 0.07
def gold_multiplier : ℝ := 1.5
def gold_tax : ℝ := 0.03

-- Define the calculated values
def hotel_total : ℝ := hotel_price_per_night * hotel_nights * (1 - hotel_discount)
def car_total : ℝ := car_price * (1 + car_tax)
def house_total : ℝ := car_price * house_multiplier * (1 + house_tax)
def yacht_total : ℝ := (hotel_price_per_night * hotel_nights + car_price) * yacht_multiplier * (1 - yacht_discount)
def gold_total : ℝ := (hotel_price_per_night * hotel_nights + car_price) * yacht_multiplier * gold_multiplier * (1 + gold_tax)

-- Theorem statement
theorem total_value_of_treats : 
  hotel_total + car_total + house_total + yacht_total + gold_total = 339100 := by
  sorry

end NUMINAMATH_CALUDE_total_value_of_treats_l2819_281924


namespace NUMINAMATH_CALUDE_equilateral_triangle_probability_l2819_281906

/-- Given a circle divided into 30 equal parts, the probability of randomly selecting
    3 different points that form an equilateral triangle is 1/406. -/
theorem equilateral_triangle_probability (n : ℕ) (h : n = 30) :
  let total_combinations := n.choose 3
  let equilateral_triangles := n / 3
  (equilateral_triangles : ℚ) / total_combinations = 1 / 406 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_probability_l2819_281906


namespace NUMINAMATH_CALUDE_tetrahedral_pyramid_marbles_hypertetrahedron_marbles_formula_l2819_281922

/-- The number of marbles in a d-dimensional hypertetrahedron with N layers -/
def hypertetrahedron_marbles (d : ℕ) (N : ℕ) : ℕ := Nat.choose (N + d - 1) d

/-- Theorem: The number of marbles in a tetrahedral pyramid with N layers is (N + 2) choose 3 -/
theorem tetrahedral_pyramid_marbles (N : ℕ) : 
  hypertetrahedron_marbles 3 N = Nat.choose (N + 2) 3 := by sorry

/-- Theorem: The number of marbles in a d-dimensional hypertetrahedron with N layers is (N + d - 1) choose d -/
theorem hypertetrahedron_marbles_formula (d : ℕ) (N : ℕ) : 
  hypertetrahedron_marbles d N = Nat.choose (N + d - 1) d := by sorry

end NUMINAMATH_CALUDE_tetrahedral_pyramid_marbles_hypertetrahedron_marbles_formula_l2819_281922


namespace NUMINAMATH_CALUDE_wilsons_theorem_l2819_281975

theorem wilsons_theorem (p : ℕ) (hp : p > 1) :
  (p.factorial - 1) % p = 0 ↔ Nat.Prime p := by sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l2819_281975


namespace NUMINAMATH_CALUDE_sqrt_22_greater_than_4_l2819_281995

theorem sqrt_22_greater_than_4 : Real.sqrt 22 > 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_22_greater_than_4_l2819_281995


namespace NUMINAMATH_CALUDE_tension_in_rope_l2819_281976

/-- A system of pulleys and masses as described in the problem -/
structure PulleySystem (m : ℝ) where
  /-- The acceleration due to gravity -/
  g : ℝ
  /-- The tension in the rope connecting the bodies m and 2m through the upper pulley -/
  tension : ℝ

/-- The theorem stating the tension in the rope connecting the bodies m and 2m -/
theorem tension_in_rope (m : ℝ) (sys : PulleySystem m) (hm : m > 0) :
  sys.tension = (10 / 3) * m * sys.g := by
  sorry


end NUMINAMATH_CALUDE_tension_in_rope_l2819_281976


namespace NUMINAMATH_CALUDE_furniture_purchase_proof_l2819_281956

/-- Calculates the number of furniture pieces purchased given the total payment, reimbursement, and cost per piece. -/
def furniture_pieces (total_payment : ℕ) (reimbursement : ℕ) (cost_per_piece : ℕ) : ℕ :=
  (total_payment - reimbursement) / cost_per_piece

/-- Proves that given the specific values in the problem, the number of furniture pieces is 150. -/
theorem furniture_purchase_proof :
  furniture_pieces 20700 600 134 = 150 := by
  sorry

#eval furniture_pieces 20700 600 134

end NUMINAMATH_CALUDE_furniture_purchase_proof_l2819_281956


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l2819_281942

/-- Given four square regions with specified perimeters and a relation between sides,
    prove that the ratio of areas of region III to region IV is 9/4. -/
theorem area_ratio_of_squares (perimeter_I perimeter_II perimeter_IV : ℝ) 
    (h1 : perimeter_I = 16)
    (h2 : perimeter_II = 20)
    (h3 : perimeter_IV = 32)
    (h4 : ∀ s : ℝ, s > 0 → perimeter_I = 4 * s → 3 * s = side_length_III) :
    (side_length_III ^ 2) / ((perimeter_IV / 4) ^ 2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l2819_281942


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2819_281916

/-- Given (1-2x)^7 = a + a₁x + a₂x² + ... + a₇x⁷, prove that a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 12 -/
theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2819_281916


namespace NUMINAMATH_CALUDE_marble_count_l2819_281970

/-- Given a bag of marbles with red, blue, and yellow marbles in the ratio 2:3:4,
    and 36 yellow marbles, prove that there are 81 marbles in total. -/
theorem marble_count (red blue yellow total : ℕ) : 
  red + blue + yellow = total →
  red = 2 * n ∧ blue = 3 * n ∧ yellow = 4 * n →
  yellow = 36 →
  total = 81 :=
by
  sorry

#check marble_count

end NUMINAMATH_CALUDE_marble_count_l2819_281970


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_ellipse_foci_l2819_281963

/-- Given an ellipse and a hyperbola E, if the hyperbola has the foci of the ellipse as its vertices,
    then the equation of the hyperbola E can be determined. -/
theorem hyperbola_equation_from_ellipse_foci (x y : ℝ) :
  (x^2 / 10 + y^2 / 5 = 1) →  -- Equation of the ellipse
  (∃ k : ℝ, 3*x + 4*y = k) →  -- Asymptote equation of hyperbola E
  (∃ a b : ℝ, a^2 = 5 ∧ x^2 / 10 + y^2 / 5 = 1 → (x = a ∨ x = -a) ∧ y = 0) →  -- Foci of ellipse as vertices of hyperbola
  (x^2 / 5 - 16*y^2 / 45 = 1)  -- Equation of hyperbola E
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_ellipse_foci_l2819_281963


namespace NUMINAMATH_CALUDE_Bob_is_shortest_l2819_281955

-- Define a type for the friends
inductive Friend
| Amy
| Bob
| Carla
| Dan
| Eric

-- Define a relation for "taller than"
def taller_than : Friend → Friend → Prop :=
  sorry

-- State the theorem
theorem Bob_is_shortest (h1 : taller_than Friend.Amy Friend.Carla)
                        (h2 : taller_than Friend.Eric Friend.Dan)
                        (h3 : taller_than Friend.Dan Friend.Bob)
                        (h4 : taller_than Friend.Carla Friend.Eric) :
  ∀ f : Friend, f ≠ Friend.Bob → taller_than f Friend.Bob :=
by sorry

end NUMINAMATH_CALUDE_Bob_is_shortest_l2819_281955


namespace NUMINAMATH_CALUDE_correct_ticket_count_l2819_281981

/-- The number of stations between Ernakulam and Chennai -/
def num_stations : ℕ := 50

/-- The number of different train routes -/
def num_routes : ℕ := 3

/-- The number of second class tickets needed for one route -/
def tickets_per_route : ℕ := num_stations * (num_stations - 1) / 2

/-- The total number of second class tickets needed for all routes -/
def total_tickets : ℕ := num_routes * tickets_per_route

theorem correct_ticket_count : total_tickets = 3675 := by
  sorry

end NUMINAMATH_CALUDE_correct_ticket_count_l2819_281981


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2819_281919

theorem fraction_sum_equality : (2 : ℚ) / 5 - 1 / 10 + 3 / 5 = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2819_281919


namespace NUMINAMATH_CALUDE_fraction_simplification_l2819_281947

theorem fraction_simplification (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^(2*b) * b^a) / (b^(2*a) * a^b) = (a/b)^b := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2819_281947


namespace NUMINAMATH_CALUDE_greatest_integer_of_2e_minus_5_l2819_281973

theorem greatest_integer_of_2e_minus_5 :
  ⌊2 * Real.exp 1 - 5⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_of_2e_minus_5_l2819_281973


namespace NUMINAMATH_CALUDE_john_index_cards_l2819_281988

/-- Given that John buys 2 packs for each student, has 6 classes, and each class has 30 students,
    prove that the total number of packs John bought is 360. -/
theorem john_index_cards (packs_per_student : ℕ) (num_classes : ℕ) (students_per_class : ℕ)
  (h1 : packs_per_student = 2)
  (h2 : num_classes = 6)
  (h3 : students_per_class = 30) :
  packs_per_student * num_classes * students_per_class = 360 := by
  sorry

end NUMINAMATH_CALUDE_john_index_cards_l2819_281988


namespace NUMINAMATH_CALUDE_abc_product_l2819_281935

theorem abc_product (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧
  (∃ k : ℕ, b + 8 = k * a) ∧
  (∃ m n : ℕ, b^2 - 1 = m * a ∧ b^2 - 1 = n * c) ∧
  b + c = a^2 - 1 →
  a * b * c = 2009 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l2819_281935


namespace NUMINAMATH_CALUDE_conference_seating_arrangements_l2819_281905

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def bench_arrangements (g1 g2 g3 g4 : ℕ) : ℕ :=
  factorial g1 * factorial g2 * factorial g3 * factorial g4 * factorial 4

def table_arrangements (g1 g2 g3 g4 : ℕ) : ℕ :=
  factorial g1 * factorial g2 * factorial g3 * factorial g4 * factorial 3

theorem conference_seating_arrangements :
  bench_arrangements 4 2 3 4 = 165888 ∧
  table_arrangements 4 2 3 4 = 41472 := by
  sorry

end NUMINAMATH_CALUDE_conference_seating_arrangements_l2819_281905


namespace NUMINAMATH_CALUDE_infinitely_many_special_numbers_l2819_281986

/-- The false derived function -/
noncomputable def false_derived (n : ℕ) : ℕ :=
  sorry

/-- The set of natural numbers n > 1 such that f(n) = f(n-1) + 1 -/
def special_set : Set ℕ :=
  {n : ℕ | n > 1 ∧ false_derived n = false_derived (n - 1) + 1}

/-- Theorem: There are infinitely many natural numbers n such that f(n) = f(n-1) + 1 -/
theorem infinitely_many_special_numbers : Set.Infinite special_set := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_special_numbers_l2819_281986


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2819_281949

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 - i) / (1 + i) = Complex.mk (5/2) (-7/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2819_281949


namespace NUMINAMATH_CALUDE_cottage_configuration_exists_l2819_281940

/-- A configuration of points on a circle -/
def Configuration := List ℕ

/-- The sum of elements in a list -/
def list_sum (l : List ℕ) : ℕ := l.foldl (·+·) 0

/-- Check if all elements in a list are unique -/
def all_unique (l : List ℕ) : Prop := l.Nodup

/-- Generate all distances between points in a circular configuration -/
def generate_distances (config : Configuration) : List ℕ :=
  let n := config.length
  let total := list_sum config
  List.range n >>= fun i =>
    List.range n >>= fun j =>
      if i < j then
        let dist := (list_sum (config.take j) - list_sum (config.take i) + total) % total
        [min dist (total - dist)]
      else
        []

/-- The main theorem statement -/
theorem cottage_configuration_exists : ∃ (config : Configuration),
  (config.length = 6) ∧
  (list_sum config = 27) ∧
  (all_unique (generate_distances config)) ∧
  (∀ d, d ∈ generate_distances config → d ≥ 1 ∧ d ≤ 26) :=
sorry

end NUMINAMATH_CALUDE_cottage_configuration_exists_l2819_281940


namespace NUMINAMATH_CALUDE_inverse_of_i_power_2023_l2819_281979

theorem inverse_of_i_power_2023 : ∃ z : ℂ, z = (Complex.I : ℂ) ^ 2023 ∧ z⁻¹ = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_i_power_2023_l2819_281979


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2819_281985

/-- Given a hyperbola with the following properties:
    1) Standard form equation: x²/a² - y²/b² = 1
    2) a > 0 and b > 0
    3) Focal length is 2√5
    4) One asymptote is perpendicular to the line 2x + y = 0
    Prove that the equation of the hyperbola is x²/4 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_focal : (2 * Real.sqrt 5 : ℝ) = 2 * Real.sqrt (a^2 + b^2))
  (h_asymptote : b / a = 1 / 2) :
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2819_281985


namespace NUMINAMATH_CALUDE_clothing_price_difference_l2819_281974

theorem clothing_price_difference :
  ∀ (x y : ℝ),
    9 * x + 10 * y = 1810 →
    11 * x + 8 * y = 1790 →
    x - y = -10 :=
by sorry

end NUMINAMATH_CALUDE_clothing_price_difference_l2819_281974


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_5_plus_2sqrt6_equality_condition_l2819_281945

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = x*y - 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_is_5_plus_2sqrt6 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  a + 2*b ≥ 5 + 2*Real.sqrt 6 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  a + 2*b = 5 + 2*Real.sqrt 6 ↔ b = 2 + Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_5_plus_2sqrt6_equality_condition_l2819_281945


namespace NUMINAMATH_CALUDE_min_area_theorem_l2819_281989

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Represents the configuration of shapes within the larger square -/
structure ShapeConfiguration where
  largeSquare : Square
  rectangle1 : Rectangle
  square1 : Square
  rectangleR : Rectangle

/-- The theorem statement -/
theorem min_area_theorem (config : ShapeConfiguration) : 
  config.rectangle1.width = 1 ∧ 
  config.rectangle1.height = 4 ∧
  config.square1.side = 1 ∧
  config.largeSquare.side ≥ 4 →
  config.largeSquare.side ^ 2 ≥ 16 ∧
  config.rectangleR.width * config.rectangleR.height = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_area_theorem_l2819_281989


namespace NUMINAMATH_CALUDE_complement_U_P_l2819_281946

-- Define the set U
def U : Set ℝ := {x | x^2 - 2*x < 3}

-- Define the set P
def P : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Theorem statement
theorem complement_U_P : 
  (U \ P) = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_U_P_l2819_281946


namespace NUMINAMATH_CALUDE_box_C_in_A_l2819_281953

/-- The number of Box B that can fill one Box A -/
def box_B_in_A : ℕ := 4

/-- The number of Box C that can fill one Box B -/
def box_C_in_B : ℕ := 6

/-- The theorem stating that 24 Box C are needed to fill Box A -/
theorem box_C_in_A : box_B_in_A * box_C_in_B = 24 := by
  sorry

end NUMINAMATH_CALUDE_box_C_in_A_l2819_281953


namespace NUMINAMATH_CALUDE_fraction_comparison_l2819_281915

def numerator (x : ℝ) : ℝ := 5 * x + 3

def denominator (x : ℝ) : ℝ := 8 - 3 * x

theorem fraction_comparison (x : ℝ) (h : -3 ≤ x ∧ x ≤ 3) :
  numerator x > denominator x ↔ 5/8 < x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2819_281915


namespace NUMINAMATH_CALUDE_loan_duration_to_c_l2819_281939

/-- Proves that the number of years A lent money to C is 4, given the specified conditions. -/
theorem loan_duration_to_c (principal_b principal_c total_interest : ℚ) 
  (duration_b : ℚ) (rate : ℚ) : 
  principal_b = 5000 →
  principal_c = 3000 →
  duration_b = 2 →
  rate = 7.000000000000001 / 100 →
  total_interest = 1540 →
  total_interest = principal_b * rate * duration_b + principal_c * rate * (4 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_to_c_l2819_281939


namespace NUMINAMATH_CALUDE_M_intersect_N_l2819_281928

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2*x + 1}

theorem M_intersect_N : M ∩ N = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2819_281928


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_433_over_18_l2819_281954

theorem sqrt_fraction_sum_equals_sqrt_433_over_18 :
  Real.sqrt (25 / 36 + 16 / 81 + 4 / 9) = Real.sqrt 433 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_433_over_18_l2819_281954


namespace NUMINAMATH_CALUDE_mariams_neighborhood_houses_l2819_281920

/-- The number of houses in Mariam's neighborhood -/
def total_houses (houses_one_side : ℕ) (multiplier : ℕ) : ℕ :=
  houses_one_side + houses_one_side * multiplier

/-- Theorem stating the total number of houses in Mariam's neighborhood -/
theorem mariams_neighborhood_houses : 
  total_houses 40 3 = 160 := by sorry

end NUMINAMATH_CALUDE_mariams_neighborhood_houses_l2819_281920


namespace NUMINAMATH_CALUDE_problem_solution_l2819_281959

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.exp x - t * (x + 1)

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := f t x + t / Real.exp x

theorem problem_solution :
  (∀ t : ℝ, (∀ x : ℝ, x > 0 → f t x ≥ 0) → t ≤ 1) ∧
  (∀ t : ℝ, t ≤ -1 →
    (∀ m : ℝ, (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ →
      y₁ = g t x₁ → y₂ = g t x₂ → (y₂ - y₁) / (x₂ - x₁) > m) → m < 3)) ∧
  (∀ n : ℕ, n > 0 →
    Real.log (1 + n) < (Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℝ))) ∧
    (Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℝ))) ≤ 1 + Real.log n) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2819_281959


namespace NUMINAMATH_CALUDE_regular_24gon_symmetry_sum_l2819_281961

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Additional properties of regular polygons can be added here if needed

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

/-- Theorem: For a regular 24-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 39 -/
theorem regular_24gon_symmetry_sum :
  ∀ (p : RegularPolygon 24),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 39 := by sorry

end NUMINAMATH_CALUDE_regular_24gon_symmetry_sum_l2819_281961


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2819_281901

def solution_set (x : ℝ) : Prop := x ≥ 3 ∨ x ≤ 1

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (f_even : ∀ x, f x = f (-x))
  (f_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (f_one_eq_zero : f 1 = 0) :
  ∀ x, f (x - 2) ≥ 0 ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2819_281901


namespace NUMINAMATH_CALUDE_mask_selection_probability_l2819_281923

theorem mask_selection_probability :
  let total_colors : ℕ := 5
  let selected_masks : ℕ := 3
  let favorable_outcomes : ℕ := (total_colors - 2).choose 1
  let total_outcomes : ℕ := total_colors.choose selected_masks
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_mask_selection_probability_l2819_281923


namespace NUMINAMATH_CALUDE_solve_for_x_l2819_281993

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2819_281993


namespace NUMINAMATH_CALUDE_hallie_monday_tips_l2819_281921

/-- Represents Hallie's work and earnings over three days --/
structure WaitressEarnings where
  hourly_rate : ℝ
  monday_hours : ℝ
  tuesday_hours : ℝ
  wednesday_hours : ℝ
  tuesday_tips : ℝ
  wednesday_tips : ℝ
  total_earnings : ℝ

/-- Calculates Hallie's tips on Monday given her work schedule and earnings --/
def monday_tips (e : WaitressEarnings) : ℝ :=
  e.total_earnings -
  (e.hourly_rate * (e.monday_hours + e.tuesday_hours + e.wednesday_hours)) -
  e.tuesday_tips - e.wednesday_tips

/-- Theorem stating that Hallie's tips on Monday were $18 --/
theorem hallie_monday_tips (e : WaitressEarnings)
  (h1 : e.hourly_rate = 10)
  (h2 : e.monday_hours = 7)
  (h3 : e.tuesday_hours = 5)
  (h4 : e.wednesday_hours = 7)
  (h5 : e.tuesday_tips = 12)
  (h6 : e.wednesday_tips = 20)
  (h7 : e.total_earnings = 240) :
  monday_tips e = 18 := by
  sorry

end NUMINAMATH_CALUDE_hallie_monday_tips_l2819_281921


namespace NUMINAMATH_CALUDE_flavoring_corn_syrup_ratio_comparison_l2819_281930

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_formulation : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_formulation : DrinkRatio :=
  { flavoring := 1.25, corn_syrup := 5, water := 75 }

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation -/
axiom sport_water_ratio : 
  sport_formulation.flavoring / sport_formulation.water = 
  (standard_formulation.flavoring / standard_formulation.water) / 2

/-- The theorem to be proved -/
theorem flavoring_corn_syrup_ratio_comparison : 
  (sport_formulation.flavoring / sport_formulation.corn_syrup) / 
  (standard_formulation.flavoring / standard_formulation.corn_syrup) = 3 := by
  sorry

end NUMINAMATH_CALUDE_flavoring_corn_syrup_ratio_comparison_l2819_281930


namespace NUMINAMATH_CALUDE_shoe_box_problem_l2819_281958

theorem shoe_box_problem (num_pairs : ℕ) (prob_match : ℚ) :
  num_pairs = 9 →
  prob_match = 1 / 17 →
  (num_pairs * 2 : ℕ) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l2819_281958


namespace NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l2819_281967

/-- Slope angle of a line with given parametric equations -/
theorem slope_angle_of_parametric_line :
  ∀ (t : ℝ),
  let x := -3 + t
  let y := 1 + Real.sqrt 3 * t
  let k := (y - 1) / (x + 3)  -- Slope calculation
  let α := Real.arctan k      -- Angle calculation
  α = π / 3 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l2819_281967


namespace NUMINAMATH_CALUDE_journey_distance_journey_distance_proof_l2819_281950

theorem journey_distance : ℝ → Prop :=
  fun d : ℝ =>
    let t := d / 40
    t + 1/4 = d / 35 →
    d = 70

-- The proof is omitted
theorem journey_distance_proof : journey_distance 70 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_journey_distance_proof_l2819_281950


namespace NUMINAMATH_CALUDE_counterexample_five_l2819_281998

theorem counterexample_five : 
  ∃ n : ℕ, ¬(3 ∣ n) ∧ ¬(Prime (n^2 - 1)) ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_counterexample_five_l2819_281998


namespace NUMINAMATH_CALUDE_derivative_sin_squared_minus_cos_squared_l2819_281965

theorem derivative_sin_squared_minus_cos_squared (x : ℝ) : 
  deriv (λ x => Real.sin x ^ 2 - Real.cos x ^ 2) x = 2 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_squared_minus_cos_squared_l2819_281965


namespace NUMINAMATH_CALUDE_max_sum_of_coefficients_l2819_281990

theorem max_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + b * A.2 = 1) ∧ 
    (a * B.1 + b * B.2 = 1) ∧ 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (A ≠ B)) →
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + b * A.2 = 1) ∧ 
    (a * B.1 + b * B.2 = 1) ∧ 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (abs (A.1 * B.2 - A.2 * B.1) = 1)) →
  a + b ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_coefficients_l2819_281990


namespace NUMINAMATH_CALUDE_bernardo_wins_with_92_l2819_281999

def game_sequence (M : ℕ) : ℕ → ℕ 
| 0 => M
| 1 => 3 * M
| 2 => 3 * M + 40
| 3 => 9 * M + 120
| 4 => 9 * M + 160
| 5 => 27 * M + 480
| _ => 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_wins_with_92 :
  ∃ (M : ℕ), 
    M ≥ 1 ∧ 
    M ≤ 1000 ∧ 
    game_sequence M 5 < 3000 ∧ 
    game_sequence M 5 + 40 ≥ 3000 ∧
    sum_of_digits M = 11 ∧
    (∀ (N : ℕ), N < M → 
      (game_sequence N 5 < 3000 → game_sequence N 5 + 40 < 3000) ∨ 
      game_sequence N 5 ≥ 3000) :=
by
  use 92
  sorry

#eval game_sequence 92 5  -- Should output 2964
#eval game_sequence 92 5 + 40  -- Should output 3004
#eval sum_of_digits 92  -- Should output 11

end NUMINAMATH_CALUDE_bernardo_wins_with_92_l2819_281999


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l2819_281910

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 12) 
  (h2 : current_speed = 5) : 
  speed_against_current + 2 * current_speed = 22 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_with_current_l2819_281910


namespace NUMINAMATH_CALUDE_staircase_covering_l2819_281913

/-- A staircase tile with dimensions 6 × 1 -/
structure StaircaseTile where
  length : Nat := 6
  width : Nat := 1

/-- Predicate to check if a field can be covered with staircase tiles -/
def canCoverField (m n : Nat) : Prop :=
  ∃ (a b c d : Nat), 
    ((m = 12 * a ∧ n ≥ b ∧ b ≥ 6) ∨ 
     (n = 12 * a ∧ m ≥ b ∧ b ≥ 6) ∨
     (m = 3 * c ∧ n = 4 * d ∧ c ≥ 2 ∧ d ≥ 3) ∨
     (n = 3 * c ∧ m = 4 * d ∧ c ≥ 2 ∧ d ≥ 3))

theorem staircase_covering (m n : Nat) (hm : m ≥ 6) (hn : n ≥ 6) :
  canCoverField m n ↔ 
    ∃ (tiles : List StaircaseTile), 
      (tiles.length * 6 = m * n) ∧ 
      (∀ t ∈ tiles, t.length = 6 ∧ t.width = 1) :=
by sorry

end NUMINAMATH_CALUDE_staircase_covering_l2819_281913


namespace NUMINAMATH_CALUDE_quadratic_roots_l2819_281900

theorem quadratic_roots (a c : ℝ) (h1 : a ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 - 2*a*x + c
  (f (-1) = 0) →
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧
    ∀ x : ℝ, (a * x^2 - 2*a*x + c = 0) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2819_281900


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2819_281978

theorem diophantine_equation_solution :
  ∃ (m n : ℕ), 26019 * m - 649 * n = 118 ∧ m = 2 ∧ n = 80 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2819_281978


namespace NUMINAMATH_CALUDE_inscribed_box_dimension_l2819_281912

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  x : ℝ
  y : ℝ
  z : ℝ
  sphere_radius : ℝ
  surface_area : ℝ
  edge_sum : ℝ
  sphere_constraint : x^2 + y^2 + z^2 = 4 * sphere_radius^2
  surface_area_constraint : 2*x*y + 2*y*z + 2*x*z = surface_area
  edge_sum_constraint : 4*(x + y + z) = edge_sum

/-- Theorem: For a rectangular box inscribed in a sphere of radius 10,
    with surface area 416 and sum of edge lengths 120,
    one of its dimensions is 10 -/
theorem inscribed_box_dimension (Q : InscribedBox)
    (h_radius : Q.sphere_radius = 10)
    (h_surface : Q.surface_area = 416)
    (h_edges : Q.edge_sum = 120) :
    Q.x = 10 ∨ Q.y = 10 ∨ Q.z = 10 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_dimension_l2819_281912


namespace NUMINAMATH_CALUDE_evaluate_expression_l2819_281962

theorem evaluate_expression : -1^2010 + (-1)^2011 + 1^2012 - 1^2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2819_281962


namespace NUMINAMATH_CALUDE_prime_sum_probability_l2819_281933

def first_twelve_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_valid_pair (p q : Nat) : Bool :=
  p ∈ first_twelve_primes ∧ q ∈ first_twelve_primes ∧ p ≠ q ∧
  Nat.Prime (p + q) ∧ p + q > 20

def count_valid_pairs : Nat :=
  (List.filter (fun (pair : Nat × Nat) => is_valid_pair pair.1 pair.2)
    (List.product first_twelve_primes first_twelve_primes)).length

def total_pairs : Nat := (first_twelve_primes.length * (first_twelve_primes.length - 1)) / 2

theorem prime_sum_probability :
  count_valid_pairs / total_pairs = 1 / 66 := by sorry

end NUMINAMATH_CALUDE_prime_sum_probability_l2819_281933


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l2819_281997

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > n → ¬(m + 12 ∣ m^3 + 144)) ∧ 
  (n + 12 ∣ n^3 + 144) ∧ 
  n = 132 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l2819_281997


namespace NUMINAMATH_CALUDE_min_value_expression_l2819_281964

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + c = 2 * b) (h4 : a ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - a)^2) / a^2 ≥ 7/2 ∧
  ∃ a b c : ℝ, a > b ∧ b > c ∧ a + c = 2 * b ∧ a ≠ 0 ∧
    ((a + b)^2 + (b - c)^2 + (c - a)^2) / a^2 = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2819_281964


namespace NUMINAMATH_CALUDE_movie_profit_calculation_l2819_281951

/-- Calculate profit for a movie given its earnings and costs -/
def movie_profit (
  opening_weekend : ℝ
  ) (
  domestic_multiplier : ℝ
  ) (
  international_multiplier : ℝ
  ) (
  domestic_tax_rate : ℝ
  ) (
  international_tax_rate : ℝ
  ) (
  royalty_rate : ℝ
  ) (
  production_cost : ℝ
  ) (
  marketing_cost : ℝ
  ) : ℝ :=
  let domestic_earnings := opening_weekend * domestic_multiplier
  let international_earnings := domestic_earnings * international_multiplier
  let domestic_after_tax := domestic_earnings * domestic_tax_rate
  let international_after_tax := international_earnings * international_tax_rate
  let total_after_tax := domestic_after_tax + international_after_tax
  let total_earnings := domestic_earnings + international_earnings
  let royalties := total_earnings * royalty_rate
  total_after_tax - royalties - production_cost - marketing_cost

/-- The profit calculation for the given movie is correct -/
theorem movie_profit_calculation :
  movie_profit 120 3.5 1.8 0.6 0.45 0.05 60 40 = 433.4 :=
by sorry

end NUMINAMATH_CALUDE_movie_profit_calculation_l2819_281951


namespace NUMINAMATH_CALUDE_smo_board_sum_l2819_281903

/-- Represents the state of the board at any given step -/
structure BoardState where
  numbers : List Nat

/-- Represents a single step in the process -/
def step (state : BoardState) : BoardState :=
  sorry

/-- The sum of all numbers on the board -/
def board_sum (state : BoardState) : Nat :=
  state.numbers.sum

theorem smo_board_sum (m : Nat) : 
  ∀ (final_state : BoardState),
    (∃ (initial_state : BoardState),
      initial_state.numbers = List.replicate (2^m) 1 ∧
      final_state = (step^[m * 2^(m-1)]) initial_state) →
    board_sum final_state ≥ 4^m :=
  sorry

end NUMINAMATH_CALUDE_smo_board_sum_l2819_281903


namespace NUMINAMATH_CALUDE_complex_angle_in_second_quadrant_l2819_281918

theorem complex_angle_in_second_quadrant 
  (z : ℂ) (θ : ℝ) 
  (h1 : z = Complex.exp (θ * Complex.I))
  (h2 : Real.cos θ < 0)
  (h3 : Real.sin θ > 0) : 
  π / 2 < θ ∧ θ < π :=
by sorry

end NUMINAMATH_CALUDE_complex_angle_in_second_quadrant_l2819_281918


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2819_281936

-- Define the inequality function
def f (x : ℝ) : ℝ := |x - 2| * (x - 1)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 2} = Set.Ioi (-Real.pi) ∩ Set.Iio 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2819_281936


namespace NUMINAMATH_CALUDE_chenny_candy_problem_l2819_281991

theorem chenny_candy_problem (initial_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) : 
  initial_candies = 10 →
  num_friends = 7 →
  candies_per_friend = 2 →
  num_friends * candies_per_friend - initial_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_chenny_candy_problem_l2819_281991


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l2819_281925

/-- Theorem: Doubling the radius of a right circular cylinder with volume 6 liters increases its volume by 18 liters -/
theorem cylinder_volume_increase (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 6 → π * (2*r)^2 * h - π * r^2 * h = 18 := by
  sorry

#check cylinder_volume_increase

end NUMINAMATH_CALUDE_cylinder_volume_increase_l2819_281925


namespace NUMINAMATH_CALUDE_crayon_cost_theorem_l2819_281909

/-- The number of crayons in half a dozen -/
def half_dozen : ℕ := 6

/-- The number of half dozens bought -/
def num_half_dozens : ℕ := 4

/-- The cost of each crayon in dollars -/
def cost_per_crayon : ℕ := 2

/-- The total number of crayons bought -/
def total_crayons : ℕ := num_half_dozens * half_dozen

/-- The total cost of the crayons in dollars -/
def total_cost : ℕ := total_crayons * cost_per_crayon

theorem crayon_cost_theorem : total_cost = 48 := by
  sorry

end NUMINAMATH_CALUDE_crayon_cost_theorem_l2819_281909


namespace NUMINAMATH_CALUDE_power_of_product_of_ten_l2819_281937

theorem power_of_product_of_ten : (2 * 10^3)^3 = 8 * 10^9 := by sorry

end NUMINAMATH_CALUDE_power_of_product_of_ten_l2819_281937


namespace NUMINAMATH_CALUDE_equation_solution_l2819_281914

theorem equation_solution (x : ℝ) (h1 : x ≠ 6) (h2 : x ≠ 3/4) :
  (x^2 - 10*x + 24)/(x - 6) + (4*x^2 + 20*x - 24)/(4*x - 3) + 2*x = 5 ↔ x = 1/4 := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2819_281914


namespace NUMINAMATH_CALUDE_parabola_vertex_l2819_281907

/-- The vertex of the parabola y = 3x^2 + 2 has coordinates (0, 2) -/
theorem parabola_vertex (x y : ℝ) : y = 3 * x^2 + 2 → (0, 2) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2819_281907


namespace NUMINAMATH_CALUDE_sum_of_integers_with_given_difference_and_product_l2819_281987

theorem sum_of_integers_with_given_difference_and_product :
  ∀ x y : ℕ+, 
    (x : ℝ) - (y : ℝ) = 10 →
    (x : ℝ) * (y : ℝ) = 56 →
    (x : ℝ) + (y : ℝ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_with_given_difference_and_product_l2819_281987


namespace NUMINAMATH_CALUDE_divisor_ratio_of_M_l2819_281982

def M : ℕ := 36 * 45 * 98 * 160

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- The ratio of sum of odd divisors to sum of even divisors -/
def divisor_ratio (n : ℕ) : ℚ :=
  (sum_odd_divisors n : ℚ) / (sum_even_divisors n : ℚ)

theorem divisor_ratio_of_M :
  divisor_ratio M = 1 / 510 := by sorry

end NUMINAMATH_CALUDE_divisor_ratio_of_M_l2819_281982


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2819_281904

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d) ≥ 18 ∧
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d) = 18 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2819_281904


namespace NUMINAMATH_CALUDE_greatest_integer_absolute_value_l2819_281984

theorem greatest_integer_absolute_value (y : ℤ) : (∀ z : ℤ, |3*z - 4| ≤ 21 → z ≤ y) ↔ y = 8 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_absolute_value_l2819_281984


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l2819_281952

/-- Given a school with students playing football and cricket, calculate the number of students playing both sports. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (football : ℕ) 
  (cricket : ℕ) 
  (neither : ℕ) 
  (h1 : total = 470) 
  (h2 : football = 325) 
  (h3 : cricket = 175) 
  (h4 : neither = 50) : 
  football + cricket - (total - neither) = 80 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l2819_281952


namespace NUMINAMATH_CALUDE_distance_to_center_l2819_281983

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 98}

-- Define the properties of the points
def PointProperties (A B C : ℝ × ℝ) : Prop :=
  A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 ∧  -- AB = 8
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 9 ∧   -- BC = 3
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0  -- Angle ABC is right

-- The theorem
theorem distance_to_center (A B C : ℝ × ℝ) :
  PointProperties A B C → B.1^2 + B.2^2 = 50 := by sorry

end NUMINAMATH_CALUDE_distance_to_center_l2819_281983


namespace NUMINAMATH_CALUDE_juice_ratio_is_three_to_one_l2819_281960

/-- Represents the ratio of water cans to concentrate cans -/
structure JuiceRatio where
  water : ℕ
  concentrate : ℕ

/-- Calculates the juice ratio given the problem parameters -/
def calculateJuiceRatio (servings : ℕ) (servingSize : ℕ) (concentrateCans : ℕ) (canSize : ℕ) : JuiceRatio :=
  let totalOunces := servings * servingSize
  let totalCans := totalOunces / canSize
  let waterCans := totalCans - concentrateCans
  { water := waterCans, concentrate := concentrateCans }

theorem juice_ratio_is_three_to_one :
  let ratio := calculateJuiceRatio 320 6 40 12
  ratio.water = 3 * ratio.concentrate := by sorry

end NUMINAMATH_CALUDE_juice_ratio_is_three_to_one_l2819_281960


namespace NUMINAMATH_CALUDE_min_value_of_f_l2819_281908

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), f x_min = Real.exp (-1) ∧ ∀ (x : ℝ), f x ≥ Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2819_281908


namespace NUMINAMATH_CALUDE_perpendicular_to_same_line_implies_parallel_l2819_281927

-- Define a structure for a line in a plane
structure Line where
  -- You can add more properties if needed
  mk :: (id : Nat)

-- Define perpendicularity between two lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry -- Definition of perpendicularity

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop :=
  sorry -- Definition of parallelism

-- Theorem statement
theorem perpendicular_to_same_line_implies_parallel 
  (l1 l2 l3 : Line) : 
  perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2 :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_perpendicular_to_same_line_implies_parallel_l2819_281927


namespace NUMINAMATH_CALUDE_distance_between_foci_l2819_281957

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (2, -3)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 4 * Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l2819_281957


namespace NUMINAMATH_CALUDE_packs_needed_for_360_days_l2819_281938

/-- The number of dog walks per day -/
def walks_per_day : ℕ := 2

/-- The number of wipes used per walk -/
def wipes_per_walk : ℕ := 1

/-- The number of wipes in a pack -/
def wipes_per_pack : ℕ := 120

/-- The number of days we need to cover -/
def days_to_cover : ℕ := 360

/-- The number of packs needed for the given number of days -/
def packs_needed : ℕ := 
  (days_to_cover * walks_per_day * wipes_per_walk + wipes_per_pack - 1) / wipes_per_pack

theorem packs_needed_for_360_days : packs_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_packs_needed_for_360_days_l2819_281938


namespace NUMINAMATH_CALUDE_software_hours_calculation_l2819_281902

def total_hours : ℝ := 68.33333333333333
def help_user_hours : ℝ := 17
def other_services_percentage : ℝ := 0.4

theorem software_hours_calculation :
  let other_services_hours := total_hours * other_services_percentage
  let software_hours := total_hours - help_user_hours - other_services_hours
  software_hours = 24 := by sorry

end NUMINAMATH_CALUDE_software_hours_calculation_l2819_281902


namespace NUMINAMATH_CALUDE_shooter_scores_equal_l2819_281968

/-- The expected value of a binomial distribution -/
def binomialExpectation (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The score of shooter A -/
def X₁ : ℝ := binomialExpectation 10 0.9

/-- The score of shooter Y (intermediate for shooter B) -/
def Y : ℝ := binomialExpectation 5 0.8

/-- The score of shooter B -/
def X₂ : ℝ := 2 * Y + 1

theorem shooter_scores_equal : X₁ = X₂ := by sorry

end NUMINAMATH_CALUDE_shooter_scores_equal_l2819_281968


namespace NUMINAMATH_CALUDE_min_vertical_distance_l2819_281911

/-- The absolute value function -/
def f (x : ℝ) : ℝ := |x|

/-- The quadratic function -/
def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The vertical distance between f and g -/
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

/-- Theorem stating the minimum vertical distance between f and g -/
theorem min_vertical_distance :
  ∃ (min_dist : ℝ), min_dist = 3/4 ∧ ∀ (x : ℝ), vertical_distance x ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l2819_281911


namespace NUMINAMATH_CALUDE_percentage_calculation_l2819_281966

theorem percentage_calculation (P : ℝ) : 
  (0.05 * (P / 100 * 1600) = 20) → P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2819_281966


namespace NUMINAMATH_CALUDE_faster_train_length_l2819_281926

/-- Given two trains moving in the same direction, this theorem calculates the length of the faster train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_speed = 180)
  (h2 : slower_speed = 90)
  (h3 : crossing_time = 15)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := (faster_speed - slower_speed) * (5/18)
  (relative_speed * crossing_time) = 375 := by
sorry

end NUMINAMATH_CALUDE_faster_train_length_l2819_281926
