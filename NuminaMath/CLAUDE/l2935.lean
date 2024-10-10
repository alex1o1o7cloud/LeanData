import Mathlib

namespace largest_product_of_three_primes_digit_sum_l2935_293567

def is_single_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    is_single_digit d ∧
    is_single_digit e ∧
    is_prime d ∧
    is_prime e ∧
    is_prime (10 * d + e) ∧
    n = d * e * (10 * d + e) ∧
    (∀ (m : ℕ), m = d' * e' * (10 * d' + e') →
      is_single_digit d' →
      is_single_digit e' →
      is_prime d' →
      is_prime e' →
      is_prime (10 * d' + e') →
      m ≤ n) ∧
    sum_of_digits n = 12 :=
by sorry

end largest_product_of_three_primes_digit_sum_l2935_293567


namespace inequality_system_solution_l2935_293541

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end inequality_system_solution_l2935_293541


namespace max_unique_sundaes_l2935_293503

/-- The number of ice cream flavors --/
def num_flavors : ℕ := 8

/-- The number of flavors that must be served together --/
def num_paired_flavors : ℕ := 2

/-- The number of distinct choices after pairing --/
def num_choices : ℕ := num_flavors - num_paired_flavors + 1

/-- The number of scoops in a sundae --/
def scoops_per_sundae : ℕ := 2

theorem max_unique_sundaes :
  (Nat.choose (num_choices - 1) (scoops_per_sundae - 1)) + 1 = 7 := by
  sorry

end max_unique_sundaes_l2935_293503


namespace min_n_is_15_l2935_293550

/-- A type representing the vertices of a regular 9-sided polygon -/
inductive Vertex : Type
  | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9

/-- A function type representing an assignment of integers to vertices -/
def Assignment := Vertex → Fin 9

/-- Predicate to check if an assignment is valid (each integer used once) -/
def is_valid_assignment (f : Assignment) : Prop :=
  ∀ i j : Vertex, i ≠ j → f i ≠ f j

/-- Function to get the next vertex in cyclic order -/
def next_vertex : Vertex → Vertex
  | Vertex.v1 => Vertex.v2
  | Vertex.v2 => Vertex.v3
  | Vertex.v3 => Vertex.v4
  | Vertex.v4 => Vertex.v5
  | Vertex.v5 => Vertex.v6
  | Vertex.v6 => Vertex.v7
  | Vertex.v7 => Vertex.v8
  | Vertex.v8 => Vertex.v9
  | Vertex.v9 => Vertex.v1

/-- Predicate to check if the sum of any three consecutive vertices does not exceed n -/
def satisfies_sum_condition (f : Assignment) (n : ℕ) : Prop :=
  ∀ v : Vertex, (f v).val + 1 + (f (next_vertex v)).val + 1 + (f (next_vertex (next_vertex v))).val + 1 ≤ n

/-- The main theorem: the minimum value of n is 15 -/
theorem min_n_is_15 :
  ∃ (f : Assignment), is_valid_assignment f ∧ satisfies_sum_condition f 15 ∧
  ∀ (m : ℕ), m < 15 → ¬∃ (g : Assignment), is_valid_assignment g ∧ satisfies_sum_condition g m :=
sorry

end min_n_is_15_l2935_293550


namespace max_excellent_boys_100_l2935_293543

/-- Represents a person with height and weight -/
structure Person where
  height : ℝ
  weight : ℝ

/-- Defines the "not worse than" relation between two people -/
def notWorseThan (a b : Person) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Defines an "excellent boy" as someone who is not worse than all others -/
def excellentBoy (p : Person) (group : Finset Person) : Prop :=
  ∀ q ∈ group, p ≠ q → notWorseThan p q

/-- The main theorem: The maximum number of excellent boys in a group of 100 is 100 -/
theorem max_excellent_boys_100 :
  ∃ (group : Finset Person), group.card = 100 ∧
  ∃ (excellent : Finset Person), excellent ⊆ group ∧ excellent.card = 100 ∧
  ∀ p ∈ excellent, excellentBoy p group :=
sorry

end max_excellent_boys_100_l2935_293543


namespace isosceles_triangle_determination_l2935_293505

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

theorem isosceles_triangle_determination
  (I M H : Point) :
  ∃! (t : Triangle), isIsosceles t ∧
    incenter t = I ∧
    centroid t = M ∧
    orthocenter t = H :=
  sorry

end isosceles_triangle_determination_l2935_293505


namespace square_circle_union_area_l2935_293509

/-- The area of the union of a square with side length 8 and a circle with radius 8
    centered at one of the square's vertices is 64 + 48π square units. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 8
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1/4 : ℝ) * circle_area
  square_area + circle_area - overlap_area = 64 + 48 * π :=
by sorry

end square_circle_union_area_l2935_293509


namespace perpendicular_chords_intersection_distance_l2935_293548

theorem perpendicular_chords_intersection_distance (d r : ℝ) (AB CD : ℝ) (h1 : d = 10) (h2 : r = d / 2) (h3 : AB = 9) (h4 : CD = 8) :
  let S := r^2 - (AB/2)^2
  let R := r^2 - (CD/2)^2
  (S + R).sqrt = (55 : ℝ).sqrt / 2 := by sorry

end perpendicular_chords_intersection_distance_l2935_293548


namespace pentagonal_prism_volume_l2935_293563

/-- The volume of a pentagonal prism with specific dimensions -/
theorem pentagonal_prism_volume : 
  let square_side : ℝ := 2
  let prism_height : ℝ := 2
  let triangle_leg : ℝ := 1
  let base_area : ℝ := square_side ^ 2 - (1 / 2 * triangle_leg * triangle_leg)
  let volume : ℝ := base_area * prism_height
  volume = 7 := by sorry

end pentagonal_prism_volume_l2935_293563


namespace intersection_in_fourth_quadrant_l2935_293591

/-- Two lines intersect in the fourth quadrant if and only if m > -2/3 -/
theorem intersection_in_fourth_quadrant (m : ℝ) :
  (∃ x y : ℝ, 3 * x + 2 * y - 2 * m - 1 = 0 ∧
               2 * x + 4 * y - m = 0 ∧
               x > 0 ∧ y < 0) ↔
  m > -2/3 := by sorry

end intersection_in_fourth_quadrant_l2935_293591


namespace cara_card_is_five_l2935_293515

def is_valid_sequence (a b c d : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ a + b + c + d = 20

def alan_statement (a : ℕ) : Prop :=
  ∃ b c d, is_valid_sequence a b c d ∧
  ∃ b' c' d', b' ≠ b ∧ is_valid_sequence a b' c' d'

def bella_statement (a b : ℕ) : Prop :=
  ∃ c d, is_valid_sequence a b c d ∧
  ∃ c' d', c' ≠ c ∧ is_valid_sequence a b c' d'

def cara_statement (a b c : ℕ) : Prop :=
  ∃ d, is_valid_sequence a b c d ∧
  ∃ d', d' ≠ d ∧ is_valid_sequence a b c d'

def david_statement (a b c d : ℕ) : Prop :=
  is_valid_sequence a b c d ∧
  ∃ a' b' c', a' ≠ a ∧ is_valid_sequence a' b' c' d

theorem cara_card_is_five :
  ∀ a b c d : ℕ,
    is_valid_sequence a b c d →
    alan_statement a →
    bella_statement a b →
    cara_statement a b c →
    david_statement a b c d →
    c = 5 := by
  sorry

end cara_card_is_five_l2935_293515


namespace equation_condition_l2935_293576

theorem equation_condition (x y z : ℕ) 
  (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  (10 * x + y) * (10 * x + z) = 100 * x^2 + 110 * x + y * z ↔ y + z = 11 := by
  sorry

end equation_condition_l2935_293576


namespace cookies_eaten_l2935_293585

def initial_cookies : ℕ := 32
def remaining_cookies : ℕ := 23

theorem cookies_eaten :
  initial_cookies - remaining_cookies = 9 :=
by sorry

end cookies_eaten_l2935_293585


namespace book_selection_problem_l2935_293590

theorem book_selection_problem (total_books : ℕ) (novels : ℕ) (to_choose : ℕ) :
  total_books = 15 →
  novels = 5 →
  to_choose = 3 →
  (Nat.choose total_books to_choose) - (Nat.choose (total_books - novels) to_choose) = 335 := by
  sorry

end book_selection_problem_l2935_293590


namespace zoo_visitors_l2935_293526

theorem zoo_visitors (total_people : ℕ) (adult_price kid_price total_sales : ℚ)
  (h1 : total_people = 254)
  (h2 : adult_price = 28)
  (h3 : kid_price = 12)
  (h4 : total_sales = 3864) :
  ∃ (adults : ℕ), adults = 51 ∧
    ∃ (kids : ℕ), adults + kids = total_people ∧
      adult_price * adults + kid_price * kids = total_sales :=
by sorry

end zoo_visitors_l2935_293526


namespace cinnamon_swirls_distribution_l2935_293573

theorem cinnamon_swirls_distribution (total_pieces : Real) (num_people : Real) (jane_pieces : Real) : 
  total_pieces = 12.0 → num_people = 3.0 → jane_pieces = total_pieces / num_people → jane_pieces = 4.0 := by
  sorry

end cinnamon_swirls_distribution_l2935_293573


namespace hyperbola_asymptotes_l2935_293551

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  focus_on_y_axis : Bool
  semi_minor_axis : ℝ
  eccentricity : ℝ

-- Define the asymptote equation type
structure AsymptoticEquation where
  slope : ℝ

-- Theorem statement
theorem hyperbola_asymptotes 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focus : h.focus_on_y_axis = true)
  (h_semi_minor : h.semi_minor_axis = 4 * Real.sqrt 2)
  (h_eccentricity : h.eccentricity = 3) :
  ∃ (eq : AsymptoticEquation), eq.slope = Real.sqrt 2 / 4 ∨ eq.slope = -(Real.sqrt 2 / 4) :=
sorry

end hyperbola_asymptotes_l2935_293551


namespace max_y_value_l2935_293557

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 18*x + 40*y) : 
  ∃ (max_y : ℝ), max_y = 20 + Real.sqrt 481 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 18*x' + 40*y' → y' ≤ max_y := by
  sorry

end max_y_value_l2935_293557


namespace base_k_conversion_uniqueness_l2935_293514

theorem base_k_conversion_uniqueness :
  ∃! (k : ℕ), k ≥ 4 ∧ 1 * k^2 + 3 * k + 2 = 30 := by sorry

end base_k_conversion_uniqueness_l2935_293514


namespace expression_evaluation_l2935_293545

theorem expression_evaluation :
  (∀ x : ℤ, x = -2 → (3*x + 1)*(2*x - 3) - (6*x - 5)*(x - 4) = -67) ∧
  (∀ x y : ℤ, x = 1 ∧ y = 2 → (2*x - y)*(x + y) - 2*x*(-2*x + 3*y) + 6*x*(-x - 5/2*y) = -44) :=
by sorry

end expression_evaluation_l2935_293545


namespace inequality_proof_l2935_293511

theorem inequality_proof (n : ℕ+) (k : ℝ) (hk : k > 0) :
  1 - 1/k ≤ n * (k^(1/n : ℝ) - 1) ∧ n * (k^(1/n : ℝ) - 1) ≤ k - 1 :=
by sorry

end inequality_proof_l2935_293511


namespace grid_arrangement_theorem_l2935_293588

/-- A type representing the grid arrangement of digits -/
def GridArrangement := Fin 8 → Fin 9

/-- Function to check if a three-digit number is a multiple of k -/
def isMultipleOfK (n : ℕ) (k : ℕ) : Prop :=
  n % k = 0

/-- Function to extract a three-digit number from the grid -/
def extractNumber (g : GridArrangement) (start : Fin 8) : ℕ :=
  100 * (g start).val + 10 * (g ((start + 2) % 8)).val + (g ((start + 4) % 8)).val

/-- Predicate to check if all four numbers in the grid are multiples of k -/
def allMultiplesOfK (g : GridArrangement) (k : ℕ) : Prop :=
  ∀ i : Fin 4, isMultipleOfK (extractNumber g (2 * i)) k

/-- Predicate to check if a grid arrangement is valid (uses all digits 1 to 8 once) -/
def isValidArrangement (g : GridArrangement) : Prop :=
  ∀ i j : Fin 8, i ≠ j → g i ≠ g j

/-- The main theorem stating for which values of k a valid arrangement exists -/
theorem grid_arrangement_theorem :
  ∀ k : ℕ, 2 ≤ k → k ≤ 6 →
    (∃ g : GridArrangement, isValidArrangement g ∧ allMultiplesOfK g k) ↔ (k = 2 ∨ k = 3) :=
sorry

end grid_arrangement_theorem_l2935_293588


namespace fraction_value_l2935_293532

/-- Represents the numerator of the fraction as a function of k -/
def numerator (k : ℕ) : ℕ := 10^k + 6 * (10^k - 1) / 9

/-- Represents the denominator of the fraction as a function of k -/
def denominator (k : ℕ) : ℕ := 60 * (10^k - 1) / 9 + 4

/-- The main theorem stating that the fraction is always 1/4 for any positive k -/
theorem fraction_value (k : ℕ) (h : k > 0) : 
  (numerator k : ℚ) / (denominator k : ℚ) = 1/4 := by
  sorry

end fraction_value_l2935_293532


namespace complex_magnitude_squared_l2935_293571

/-- For a complex number z = 2x + 3iy, |z|^2 = 4x^2 + 9y^2 -/
theorem complex_magnitude_squared (x y : ℝ) : 
  let z : ℂ := 2*x + 3*y*Complex.I
  Complex.normSq z = 4*x^2 + 9*y^2 := by
sorry

end complex_magnitude_squared_l2935_293571


namespace paint_usage_fraction_l2935_293587

theorem paint_usage_fraction (total_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1 / 9 →
  total_used = 104 →
  let remaining_paint := total_paint - first_week_fraction * total_paint
  let second_week_usage := total_used - first_week_fraction * total_paint
  second_week_usage / remaining_paint = 1 / 5 := by
  sorry

end paint_usage_fraction_l2935_293587


namespace repeated_number_divisible_by_91_l2935_293594

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundred_nonzero : hundreds ≠ 0
  digit_bounds : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Represents the six-digit number formed by repeating a three-digit number -/
def repeated_number (n : ThreeDigitNumber) : Nat :=
  1000000 * n.hundreds + 100000 * n.tens + 10000 * n.ones +
  1000 * n.hundreds + 100 * n.tens + 10 * n.ones

/-- Theorem stating that the repeated number is divisible by 91 -/
theorem repeated_number_divisible_by_91 (n : ThreeDigitNumber) :
  (repeated_number n) % 91 = 0 := by
  sorry

end repeated_number_divisible_by_91_l2935_293594


namespace flock_size_lcm_equals_min_ducks_l2935_293504

/-- Represents the flock size of ducks -/
def duck_flock_size : ℕ := 18

/-- Represents the flock size of seagulls -/
def seagull_flock_size : ℕ := 10

/-- Represents the smallest number of ducks observed -/
def min_ducks_observed : ℕ := 90

/-- Theorem stating that the least common multiple of the flock sizes
    is equal to the smallest number of ducks observed -/
theorem flock_size_lcm_equals_min_ducks :
  Nat.lcm duck_flock_size seagull_flock_size = min_ducks_observed := by
  sorry

end flock_size_lcm_equals_min_ducks_l2935_293504


namespace calculation_proof_l2935_293523

theorem calculation_proof : (1/4 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 + (1/8 : ℚ) = 9/8 := by
  sorry

end calculation_proof_l2935_293523


namespace basketball_score_increase_l2935_293530

theorem basketball_score_increase (junior_score : ℕ) (total_score : ℕ) 
  (h1 : junior_score = 260) 
  (h2 : total_score = 572) : 
  (((total_score - junior_score) : ℚ) / junior_score) * 100 = 20 := by
  sorry

end basketball_score_increase_l2935_293530


namespace no_integer_coefficients_l2935_293537

theorem no_integer_coefficients : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end no_integer_coefficients_l2935_293537


namespace orange_tree_problem_l2935_293501

theorem orange_tree_problem (trees : ℕ) (picked_fraction : ℚ) (remaining : ℕ) :
  trees = 8 →
  picked_fraction = 2 / 5 →
  remaining = 960 →
  ∃ (initial : ℕ), initial = 200 ∧ 
    trees * (initial - picked_fraction * initial) = remaining :=
by sorry

end orange_tree_problem_l2935_293501


namespace hiker_speed_l2935_293599

theorem hiker_speed (supplies_per_mile : Real) (first_pack : Real) (resupply_ratio : Real)
  (hours_per_day : Real) (num_days : Real) :
  supplies_per_mile = 0.5 →
  first_pack = 40 →
  resupply_ratio = 0.25 →
  hours_per_day = 8 →
  num_days = 5 →
  (first_pack + first_pack * resupply_ratio) / supplies_per_mile / (hours_per_day * num_days) = 2.5 := by
  sorry

end hiker_speed_l2935_293599


namespace biology_score_calculation_l2935_293561

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 62
def average_score : ℕ := 74
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_subjects_total := math_score + science_score + social_studies_score + english_score
  let all_subjects_total := average_score * total_subjects
  all_subjects_total - known_subjects_total = 85 := by
sorry

end biology_score_calculation_l2935_293561


namespace sum_of_possible_p_values_l2935_293566

theorem sum_of_possible_p_values : ∃ (S : Finset Nat), 
  (∀ p ∈ S, ∃ q : Nat, 
    Nat.Prime p ∧ 
    p > 0 ∧ 
    q > 0 ∧ 
    p ∣ (q - 1) ∧ 
    (p + q) ∣ (p^2 + 2020*q^2)) ∧
  (∀ p : Nat, 
    (∃ q : Nat, 
      Nat.Prime p ∧ 
      p > 0 ∧ 
      q > 0 ∧ 
      p ∣ (q - 1) ∧ 
      (p + q) ∣ (p^2 + 2020*q^2)) → 
    p ∈ S) ∧
  S.sum id = 35 := by
sorry


end sum_of_possible_p_values_l2935_293566


namespace cylinder_views_l2935_293595

-- Define the cylinder and its orientation
structure Cylinder where
  upright : Bool
  on_horizontal_plane : Bool

-- Define the possible view shapes
inductive ViewShape
  | Rectangle
  | Circle

-- Define the function to get the view of the cylinder
def get_cylinder_view (c : Cylinder) (view : String) : ViewShape :=
  match view with
  | "front" => ViewShape.Rectangle
  | "side" => ViewShape.Rectangle
  | "top" => ViewShape.Circle
  | _ => ViewShape.Rectangle  -- Default case, though not needed for our problem

-- Theorem statement
theorem cylinder_views (c : Cylinder) 
  (h1 : c.upright = true) 
  (h2 : c.on_horizontal_plane = true) : 
  (get_cylinder_view c "front" = ViewShape.Rectangle) ∧ 
  (get_cylinder_view c "side" = ViewShape.Rectangle) ∧ 
  (get_cylinder_view c "top" = ViewShape.Circle) := by
  sorry


end cylinder_views_l2935_293595


namespace largest_inexpressible_is_19_l2935_293534

/-- Represents the value of a coin in soldi -/
inductive Coin : Type
| five : Coin
| six : Coin

/-- Checks if a natural number can be expressed as a sum of multiples of 5 and 6 -/
def canExpress (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

/-- The largest value that cannot be expressed as a sum of multiples of 5 and 6 -/
def largestInexpressible : ℕ := 19

theorem largest_inexpressible_is_19 :
  largestInexpressible = 19 ∧
  ¬(canExpress largestInexpressible) ∧
  ∀ n : ℕ, n > largestInexpressible → n ≤ 50 → canExpress n :=
by sorry

end largest_inexpressible_is_19_l2935_293534


namespace sqrt_9800_simplification_l2935_293542

theorem sqrt_9800_simplification : Real.sqrt 9800 = 70 * Real.sqrt 2 := by
  sorry

end sqrt_9800_simplification_l2935_293542


namespace symmetry_condition_l2935_293516

/-- A curve in the xy-plane represented by the equation x^2 + y^2 + Dx + Ey + F = 0 -/
structure Curve (D E F : ℝ) where
  condition : D^2 + E^2 - 4*F > 0

/-- Predicate for a curve being symmetric about the line y = x -/
def is_symmetric_about_y_eq_x (c : Curve D E F) : Prop :=
  D = E

/-- Theorem stating the condition for symmetry about y = x -/
theorem symmetry_condition (D E F : ℝ) (c : Curve D E F) :
  is_symmetric_about_y_eq_x c ↔ D = E :=
sorry

end symmetry_condition_l2935_293516


namespace mower_next_tangent_east_l2935_293518

/-- Represents the cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a circular garden with a mower --/
structure CircularGarden where
  garden_radius : ℝ
  mower_radius : ℝ
  initial_direction : Direction
  roll_direction : Bool  -- true for counterclockwise, false for clockwise

/-- 
  Determines the next tangent point where the mower's marker aims north again
  given a circular garden configuration
--/
def next_north_tangent (garden : CircularGarden) : Direction :=
  sorry

/-- The main theorem to be proved --/
theorem mower_next_tangent_east :
  let garden := CircularGarden.mk 15 5 Direction.North true
  next_north_tangent garden = Direction.East :=
sorry

end mower_next_tangent_east_l2935_293518


namespace f_inequality_l2935_293538

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem f_inequality (a : ℝ) : 
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → ((x + 1) * Real.log x + 2 * a) / ((x + 1)^2) < Real.log x / (x - 1)) ↔ 
  a ≤ 2 := by sorry

end f_inequality_l2935_293538


namespace max_value_of_fraction_l2935_293500

theorem max_value_of_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -6 ≤ x' ∧ x' ≤ -3 → 3 ≤ y' ∧ y' ≤ 5 → (x' - y') / y' ≤ (x - y) / y) →
  (x - y) / y = -2 :=
sorry

end max_value_of_fraction_l2935_293500


namespace incenter_is_circumcenter_of_A₁B₁C₁_l2935_293578

-- Define a structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def is_acute_angled (t : Triangle) : Prop := sorry
def is_non_equilateral (t : Triangle) : Prop := sorry

-- Define the circumradius
def circumradius (t : Triangle) : ℝ := sorry

-- Define the heights of the triangle
def height_A (t : Triangle) : ℝ × ℝ := sorry
def height_B (t : Triangle) : ℝ × ℝ := sorry
def height_C (t : Triangle) : ℝ × ℝ := sorry

-- Define points A₁, B₁, C₁ on the heights
def A₁ (t : Triangle) : ℝ × ℝ := sorry
def B₁ (t : Triangle) : ℝ × ℝ := sorry
def C₁ (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- The main theorem
theorem incenter_is_circumcenter_of_A₁B₁C₁ (t : Triangle) 
  (h_acute : is_acute_angled t) 
  (h_non_equilateral : is_non_equilateral t) 
  (h_A₁ : A₁ t = height_A t + (0, circumradius t))
  (h_B₁ : B₁ t = height_B t + (0, circumradius t))
  (h_C₁ : C₁ t = height_C t + (0, circumradius t)) :
  incenter t = circumcenter { A := A₁ t, B := B₁ t, C := C₁ t } := by
  sorry

end incenter_is_circumcenter_of_A₁B₁C₁_l2935_293578


namespace total_weight_chromic_acid_sodium_hydroxide_l2935_293598

/-- The total weight of Chromic acid and Sodium hydroxide in a neutralization reaction -/
theorem total_weight_chromic_acid_sodium_hydroxide 
  (moles_chromic_acid : ℝ) 
  (moles_sodium_hydroxide : ℝ) 
  (molar_mass_chromic_acid : ℝ) 
  (molar_mass_sodium_hydroxide : ℝ) : 
  moles_chromic_acid = 17.3 →
  moles_sodium_hydroxide = 8.5 →
  molar_mass_chromic_acid = 118.02 →
  molar_mass_sodium_hydroxide = 40.00 →
  moles_chromic_acid * molar_mass_chromic_acid + 
  moles_sodium_hydroxide * molar_mass_sodium_hydroxide = 2381.746 := by
  sorry

#check total_weight_chromic_acid_sodium_hydroxide

end total_weight_chromic_acid_sodium_hydroxide_l2935_293598


namespace burger_filler_percentage_l2935_293560

/-- Given a burger with specified total weight and filler weights, 
    calculate the percentage that is not filler -/
theorem burger_filler_percentage 
  (total_weight : ℝ) 
  (vegetable_filler : ℝ) 
  (grain_filler : ℝ) 
  (h1 : total_weight = 180) 
  (h2 : vegetable_filler = 45) 
  (h3 : grain_filler = 15) : 
  (total_weight - (vegetable_filler + grain_filler)) / total_weight = 2/3 := by
sorry

#eval (180 - (45 + 15)) / 180

end burger_filler_percentage_l2935_293560


namespace integer_floor_equation_l2935_293577

theorem integer_floor_equation (m n : ℕ+) :
  (⌊(m : ℝ)^2 / n⌋ + ⌊(n : ℝ)^2 / m⌋ = ⌊(m : ℝ) / n + (n : ℝ) / m⌋ + m * n) ↔
  (∃ k : ℕ+, (m = k ∧ n = k^2 + 1) ∨ (m = k^2 + 1 ∧ n = k)) :=
sorry

end integer_floor_equation_l2935_293577


namespace no_time_left_after_student_council_l2935_293512

/-- Represents the journey to school with various stops -/
structure SchoolJourney where
  totalTimeAvailable : ℕ
  travelTimeWithTraffic : ℕ
  timeToLibrary : ℕ
  timeToReturnBooks : ℕ
  extraTimeForManyBooks : ℕ
  timeToStudentCouncil : ℕ
  timeToSubmitProject : ℕ
  timeToClassroom : ℕ

/-- Calculates the time left after leaving the student council room -/
def timeLeftAfterStudentCouncil (journey : SchoolJourney) : Int :=
  journey.totalTimeAvailable - (journey.travelTimeWithTraffic + journey.timeToLibrary +
  journey.timeToReturnBooks + journey.extraTimeForManyBooks + journey.timeToStudentCouncil +
  journey.timeToSubmitProject)

/-- Theorem stating that in the worst-case scenario, there's no time left after leaving the student council room -/
theorem no_time_left_after_student_council (journey : SchoolJourney)
  (h1 : journey.totalTimeAvailable = 30)
  (h2 : journey.travelTimeWithTraffic = 25)
  (h3 : journey.timeToLibrary = 3)
  (h4 : journey.timeToReturnBooks = 2)
  (h5 : journey.extraTimeForManyBooks = 2)
  (h6 : journey.timeToStudentCouncil = 5)
  (h7 : journey.timeToSubmitProject = 3)
  (h8 : journey.timeToClassroom = 6) :
  timeLeftAfterStudentCouncil journey ≤ 0 := by
  sorry

end no_time_left_after_student_council_l2935_293512


namespace book_cost_price_l2935_293533

/-- The cost price of a book, given that selling it at 9% profit instead of 9% loss brings Rs 9 more -/
theorem book_cost_price (price : ℝ) : 
  (price * 1.09 - price * 0.91 = 9) → price = 50 := by
  sorry

end book_cost_price_l2935_293533


namespace smallest_to_large_square_area_ratio_l2935_293527

/-- Represents a square with a given side length -/
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side_length ^ 2

theorem smallest_to_large_square_area_ratio :
  ∀ (large : Square),
  ∃ (middle smallest : Square),
  (middle.side_length = large.side_length / 2) ∧
  (smallest.side_length = middle.side_length / 2) →
  smallest.area / large.area = 1 / 16 :=
by
  sorry

#check smallest_to_large_square_area_ratio

end smallest_to_large_square_area_ratio_l2935_293527


namespace one_cow_one_bag_days_l2935_293546

/-- Given that 34 cows eat 34 bags of husk in 34 days, 
    prove that one cow will eat one bag of husk in 34 days. -/
theorem one_cow_one_bag_days : 
  ∀ (cows bags days : ℕ), 
  cows = 34 → bags = 34 → days = 34 →
  (cows * bags = cows * days) →
  1 * days = 34 := by
  sorry

end one_cow_one_bag_days_l2935_293546


namespace students_not_enrolled_l2935_293584

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 69) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 15 := by
  sorry

end students_not_enrolled_l2935_293584


namespace bug_position_after_2023_jumps_l2935_293589

/-- Represents the seven points on the circle -/
inductive CirclePoint
  | one | two | three | four | five | six | seven

/-- Determines if a CirclePoint is prime -/
def isPrime : CirclePoint → Bool
  | CirclePoint.two => true
  | CirclePoint.three => true
  | CirclePoint.five => true
  | CirclePoint.seven => true
  | _ => false

/-- Determines if a CirclePoint is composite -/
def isComposite : CirclePoint → Bool
  | CirclePoint.four => true
  | CirclePoint.six => true
  | _ => false

/-- Moves the bug according to the jumping rule -/
def move (p : CirclePoint) : CirclePoint :=
  match p with
  | CirclePoint.one => CirclePoint.two
  | CirclePoint.two => CirclePoint.three
  | CirclePoint.three => CirclePoint.four
  | CirclePoint.four => CirclePoint.seven
  | CirclePoint.five => CirclePoint.six
  | CirclePoint.six => CirclePoint.two
  | CirclePoint.seven => CirclePoint.one

/-- Performs n jumps starting from a given point -/
def jumpN (start : CirclePoint) (n : Nat) : CirclePoint :=
  match n with
  | 0 => start
  | n + 1 => move (jumpN start n)

theorem bug_position_after_2023_jumps :
  jumpN CirclePoint.seven 2023 = CirclePoint.two := by
  sorry


end bug_position_after_2023_jumps_l2935_293589


namespace least_perimeter_triangle_l2935_293535

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem least_perimeter_triangle :
  ∃ (c : ℕ), 
    is_triangle 24 51 c ∧ 
    (∀ (x : ℕ), is_triangle 24 51 x → triangle_perimeter 24 51 c ≤ triangle_perimeter 24 51 x) ∧
    triangle_perimeter 24 51 c = 103 := by
  sorry

end least_perimeter_triangle_l2935_293535


namespace proportion_solution_l2935_293544

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 9) → x = 1.35 := by
  sorry

end proportion_solution_l2935_293544


namespace basketball_scoring_l2935_293564

/-- Basketball game scoring problem -/
theorem basketball_scoring
  (alex_points : ℕ)
  (sam_points : ℕ)
  (jon_points : ℕ)
  (jack_points : ℕ)
  (tom_points : ℕ)
  (h1 : jon_points = 2 * sam_points + 3)
  (h2 : sam_points = alex_points / 2)
  (h3 : alex_points = jack_points - 7)
  (h4 : jack_points = jon_points + 5)
  (h5 : tom_points = jon_points + jack_points - 4)
  (h6 : alex_points = 18) :
  jon_points + jack_points + tom_points + sam_points + alex_points = 115 := by
sorry

end basketball_scoring_l2935_293564


namespace no_primes_in_range_l2935_293556

theorem no_primes_in_range (n : ℕ) (hn : n > 2) :
  ∀ k : ℕ, n! + 2 < k ∧ k < n! + n → ¬ Nat.Prime k := by
  sorry

end no_primes_in_range_l2935_293556


namespace valentines_day_problem_l2935_293597

theorem valentines_day_problem (boys girls : ℕ) : 
  boys * girls = boys + girls + 52 → boys * girls = 108 := by
  sorry

end valentines_day_problem_l2935_293597


namespace cubic_function_zeros_l2935_293579

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

theorem cubic_function_zeros (c : ℝ) :
  (∀ a : ℝ, (a < -3 ∨ (1 < a ∧ a < 3/2) ∨ a > 3/2) →
    ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      f a (c - a) x₁ = 0 ∧ f a (c - a) x₂ = 0 ∧ f a (c - a) x₃ = 0) →
  c = 1 :=
sorry

end cubic_function_zeros_l2935_293579


namespace vector_collinearity_l2935_293521

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 1]
def c : Fin 2 → ℝ := ![2, 1]

def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v = fun i => t * u i

theorem vector_collinearity (k : ℝ) :
  collinear (fun i => k * a i + b i) c → k = -1 := by
  sorry

end vector_collinearity_l2935_293521


namespace position_2007_l2935_293510

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DCBA
  | ADCB
  | BADC
  | CBAD

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ADCB
  | SquarePosition.ADCB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.ABCD

-- Define the function to get the position after n transformations
def positionAfterN (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.DCBA
  | 2 => SquarePosition.ADCB
  | _ => SquarePosition.BADC

-- Theorem statement
theorem position_2007 : positionAfterN 2007 = SquarePosition.ADCB := by
  sorry


end position_2007_l2935_293510


namespace sin_cos_inequality_l2935_293581

theorem sin_cos_inequality (x : ℝ) : 
  2 - Real.sqrt 2 ≤ Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2 
  ∧ Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2 ≤ 2 + Real.sqrt 2 := by
sorry

end sin_cos_inequality_l2935_293581


namespace intersection_count_l2935_293506

-- Define the equations
def eq1 (x y : ℝ) : Prop := (x + 2*y - 10) * (x - 4*y + 8) = 0
def eq2 (x y : ℝ) : Prop := (2*x - y - 1) * (5*x + 3*y - 15) = 0

-- Define a function to count distinct intersection points
noncomputable def count_intersections : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem intersection_count : count_intersections = 3 := by
  sorry

end intersection_count_l2935_293506


namespace target_shopping_total_l2935_293528

/-- The total amount spent by Christy and Tanya at Target -/
def total_spent (face_moisturizer_price : ℕ) (body_lotion_price : ℕ) 
  (face_moisturizer_count : ℕ) (body_lotion_count : ℕ) : ℕ :=
  let tanya_spent := face_moisturizer_price * face_moisturizer_count + 
                     body_lotion_price * body_lotion_count
  2 * tanya_spent

/-- Theorem stating the total amount spent by Christy and Tanya -/
theorem target_shopping_total : 
  total_spent 50 60 2 4 = 1020 := by
  sorry

#eval total_spent 50 60 2 4

end target_shopping_total_l2935_293528


namespace polynomial_evaluation_l2935_293555

theorem polynomial_evaluation :
  let a : ℚ := 7/3
  (4 * a^2 - 11 * a + 7) * (2 * a - 3) = 140/27 := by
  sorry

end polynomial_evaluation_l2935_293555


namespace pyramid_fifth_face_sum_l2935_293519

/-- Represents a labeling of a square-based pyramid -/
structure PyramidLabeling where
  vertices : Fin 5 → Nat
  sum_to_15 : (vertices 0) + (vertices 1) + (vertices 2) + (vertices 3) + (vertices 4) = 15
  all_different : ∀ i j, i ≠ j → vertices i ≠ vertices j

/-- Represents the sums of faces in the pyramid -/
structure FaceSums (l : PyramidLabeling) where
  sums : Fin 5 → Nat
  four_given_sums : {7, 8, 9, 10} ⊆ (Finset.image sums Finset.univ)

theorem pyramid_fifth_face_sum (l : PyramidLabeling) (s : FaceSums l) :
  ∃ i, s.sums i = 13 :=
sorry

end pyramid_fifth_face_sum_l2935_293519


namespace least_number_divisible_l2935_293525

theorem least_number_divisible (n : ℕ) : n = 857 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 54 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 7) = 24 * k₁ ∧ (n + 7) = 32 * k₂ ∧ (n + 7) = 36 * k₃ ∧ (n + 7) = 54 * k₄) :=
by sorry

end least_number_divisible_l2935_293525


namespace quadratic_residue_minus_one_l2935_293531

theorem quadratic_residue_minus_one (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  (∃ x : Nat, x^2 ≡ -1 [ZMOD p]) ↔ p ≡ 1 [ZMOD 4] := by
  sorry

end quadratic_residue_minus_one_l2935_293531


namespace solve_equation_l2935_293580

theorem solve_equation (y : ℚ) (h : 1/3 - 1/4 = 1/y) : y = 12 := by
  sorry

end solve_equation_l2935_293580


namespace plato_city_schools_l2935_293502

/-- The number of high schools in Plato City -/
def num_schools : ℕ := 21

/-- The total number of participants in the competition -/
def total_participants : ℕ := 3 * num_schools

/-- Charlie's rank in the competition -/
def charlie_rank : ℕ := (total_participants + 1) / 2

/-- Alice's rank in the competition -/
def alice_rank : ℕ := 45

/-- Bob's rank in the competition -/
def bob_rank : ℕ := 58

/-- Theorem stating that the number of schools satisfies all conditions -/
theorem plato_city_schools :
  num_schools = 21 ∧
  charlie_rank < alice_rank ∧
  charlie_rank < bob_rank ∧
  charlie_rank ≤ 45 ∧
  3 * num_schools ≥ bob_rank :=
sorry

end plato_city_schools_l2935_293502


namespace leah_lost_money_l2935_293568

def total_earned : ℚ := 28
def milkshake_fraction : ℚ := 1/7
def savings_fraction : ℚ := 1/2
def remaining_in_wallet : ℚ := 1

theorem leah_lost_money : 
  let milkshake_cost := total_earned * milkshake_fraction
  let after_milkshake := total_earned - milkshake_cost
  let savings := after_milkshake * savings_fraction
  let in_wallet := after_milkshake - savings
  in_wallet - remaining_in_wallet = 11 := by sorry

end leah_lost_money_l2935_293568


namespace linear_function_midpoint_property_quadratic_function_midpoint_property_l2935_293575

/-- Linear function property -/
theorem linear_function_midpoint_property (a b x₁ x₂ : ℝ) :
  let f := fun x => a * x + b
  f ((x₁ + x₂) / 2) = (f x₁ + f x₂) / 2 := by sorry

/-- Quadratic function property -/
theorem quadratic_function_midpoint_property (a b x₁ x₂ : ℝ) :
  let g := fun x => x^2 + a * x + b
  g ((x₁ + x₂) / 2) ≤ (g x₁ + g x₂) / 2 := by sorry

end linear_function_midpoint_property_quadratic_function_midpoint_property_l2935_293575


namespace police_emergency_number_has_large_prime_divisor_l2935_293539

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def is_police_emergency_number (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_has_large_prime_divisor (n : ℕ) :
  is_police_emergency_number n → ∃ p : ℕ, p > 7 ∧ Nat.Prime p ∧ p ∣ n :=
sorry

end police_emergency_number_has_large_prime_divisor_l2935_293539


namespace function_inequality_l2935_293507

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)
variable (hf' : ∀ x, deriv f x < f x)

-- Define the theorem
theorem function_inequality (a : ℝ) (ha : a > 0) :
  f a < Real.exp a * f 0 := by sorry

end function_inequality_l2935_293507


namespace square_garden_perimeter_l2935_293547

theorem square_garden_perimeter (q p : ℝ) : 
  q = 49 → -- Area of the garden is 49 square feet
  q = p + 21 → -- Given relationship between q and p
  (4 * Real.sqrt q) = 28 -- Perimeter of the garden is 28 feet
:= by sorry

end square_garden_perimeter_l2935_293547


namespace intersection_P_Q_l2935_293552

-- Define set P
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- Define set Q
def Q : Set ℝ := {y | ∃ x : ℝ, y = x}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = {y : ℝ | y ≥ 0} := by
  sorry

end intersection_P_Q_l2935_293552


namespace arithmetic_sequence_a12_l2935_293513

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 4 = -4 → a 8 = 4 → a 12 = 12 := by
  sorry

end arithmetic_sequence_a12_l2935_293513


namespace jane_egg_money_l2935_293536

/-- Calculates the money made from selling eggs over a period of weeks. -/
def money_from_eggs (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  (num_chickens * eggs_per_chicken * num_weeks : ℚ) / 12 * price_per_dozen

/-- Proves that Jane makes $20 in 2 weeks from selling eggs. -/
theorem jane_egg_money :
  money_from_eggs 10 6 2 2 = 20 := by
  sorry

end jane_egg_money_l2935_293536


namespace intersection_implies_b_range_l2935_293592

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
by sorry

end intersection_implies_b_range_l2935_293592


namespace ratio_equals_seven_l2935_293558

theorem ratio_equals_seven (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : 2 * x + 6 * y - 21 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 7 := by
  sorry

end ratio_equals_seven_l2935_293558


namespace custom_mul_properties_l2935_293553

/-- Custom multiplication operation -/
def custom_mul (m : ℚ) (x y : ℚ) : ℚ := (x * y) / (m * x + 2 * y)

/-- Theorem stating the properties of the custom multiplication -/
theorem custom_mul_properties :
  ∃ (m : ℚ), 
    (custom_mul m 1 2 = 2/5) ∧
    (m = 1) ∧
    (custom_mul m 2 6 = 6/7) := by sorry

end custom_mul_properties_l2935_293553


namespace smallest_number_divisible_l2935_293565

theorem smallest_number_divisible (n : ℕ) : n = 44398 ↔ 
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 12 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 30 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 48 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 74 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 100 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, (n + 2) = 12 * k₁ ∧ 
                         (n + 2) = 30 * k₂ ∧ 
                         (n + 2) = 48 * k₃ ∧ 
                         (n + 2) = 74 * k₄ ∧ 
                         (n + 2) = 100 * k₅) :=
by
  sorry

end smallest_number_divisible_l2935_293565


namespace sum_of_coefficients_is_37_l2935_293596

def polynomial (x : ℝ) : ℝ := -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2) - 2 * (x^6 - 5)

theorem sum_of_coefficients_is_37 : 
  (polynomial 1) = 37 := by sorry

end sum_of_coefficients_is_37_l2935_293596


namespace range_of_m_l2935_293554

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, (x + y - m = 0) → ((x - 1)^2 + y^2 = 1) → False

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, (x₁^2 - x₁ + m - 4 = 0) ∧ (x₂^2 - x₂ + m - 4 = 0) ∧ (x₁ * x₂ < 0)

-- Main theorem
theorem range_of_m : 
  ∀ m : ℝ, (∀ m' : ℝ, p m' ∨ q m') → ¬(p m) → (1 - Real.sqrt 2 ≤ m ∧ m ≤ 1 + Real.sqrt 2) :=
by sorry

end range_of_m_l2935_293554


namespace circle_and_line_intersection_l2935_293522

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 16

-- Define the line m that bisects the circle
def line_m (x y : ℝ) : Prop :=
  3*x - y = 0

-- Define the line l passing through D(0,-1) with slope k
def line_l (k x y : ℝ) : Prop :=
  y = k*x - 1

-- Theorem statement
theorem circle_and_line_intersection :
  -- Circle C passes through A(1,-1) and B(5,3)
  circle_C 1 (-1) ∧ circle_C 5 3 ∧
  -- Circle C is bisected by line m
  (∀ x y, circle_C x y → line_m x y → x = 1 ∧ y = 3) →
  -- Part 1: Prove the equation of circle C
  (∀ x y, circle_C x y ↔ (x - 1)^2 + (y - 3)^2 = 16) ∧
  -- Part 2: Prove the range of k for which line l intersects circle C at two distinct points
  (∀ k, (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂) ↔
        (k < -8/15 ∨ k > 0)) :=
by sorry

end circle_and_line_intersection_l2935_293522


namespace sqrt_product_equals_sqrt_of_product_l2935_293572

theorem sqrt_product_equals_sqrt_of_product : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equals_sqrt_of_product_l2935_293572


namespace reinforcement_calculation_l2935_293524

/-- Calculates the size of reinforcement given initial garrison size, provision days, and remaining days after reinforcement --/
def calculate_reinforcement (initial_garrison : ℕ) (initial_provision_days : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_provision_days
  let provisions_left := initial_garrison * (initial_provision_days - days_before_reinforcement)
  (provisions_left / remaining_days) - initial_garrison

theorem reinforcement_calculation (initial_garrison : ℕ) (initial_provision_days : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) :
  initial_garrison = 1850 →
  initial_provision_days = 28 →
  days_before_reinforcement = 12 →
  remaining_days = 10 →
  calculate_reinforcement initial_garrison initial_provision_days days_before_reinforcement remaining_days = 1110 :=
by sorry

end reinforcement_calculation_l2935_293524


namespace intersection_of_M_and_N_l2935_293517

def M : Set Int := {-1, 0, 1}
def N : Set Int := {-2, -1, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0} := by
  sorry

end intersection_of_M_and_N_l2935_293517


namespace f_composition_one_sixteenth_l2935_293574

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 3^x

theorem f_composition_one_sixteenth : f (f (1/16)) = 1/9 := by
  sorry

end f_composition_one_sixteenth_l2935_293574


namespace horner_method_op_count_for_f_l2935_293549

/-- Horner's method operation count for polynomial evaluation -/
def horner_op_count (coeffs : List ℝ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: tail => 2 * (tail.length)

/-- The polynomial f(x) = x^5 + 4x^4 + 3x^3 + 2x^2 + 1 -/
def f_coeffs : List ℝ := [1, 4, 3, 2, 0, 1]

theorem horner_method_op_count_for_f :
  horner_op_count f_coeffs = 8 := by sorry

end horner_method_op_count_for_f_l2935_293549


namespace sphere_cross_section_distance_l2935_293569

theorem sphere_cross_section_distance
  (V : ℝ) (A : ℝ) (d : ℝ)
  (hV : V = 4 * Real.sqrt 3 * Real.pi)
  (hA : A = Real.pi) :
  d = Real.sqrt 2 :=
sorry

end sphere_cross_section_distance_l2935_293569


namespace root_sum_squares_l2935_293562

theorem root_sum_squares (a b c d : ℝ) : 
  (a^4 - 12*a^3 + 47*a^2 - 60*a + 24 = 0) →
  (b^4 - 12*b^3 + 47*b^2 - 60*b + 24 = 0) →
  (c^4 - 12*c^3 + 47*c^2 - 60*c + 24 = 0) →
  (d^4 - 12*d^3 + 47*d^2 - 60*d + 24 = 0) →
  (a+b)^2 + (b+c)^2 + (c+d)^2 + (d+a)^2 = 147 :=
by sorry

end root_sum_squares_l2935_293562


namespace perpendicular_vectors_m_value_l2935_293520

/-- Given two vectors a and b in ℝ², prove that if a = (1, 2) and b = (m, 1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![m, 1]
  (∀ i, i < 2 → a i * b i = 0) → m = -2 := by
  sorry

end perpendicular_vectors_m_value_l2935_293520


namespace largest_x_value_l2935_293583

theorem largest_x_value (x : ℝ) : 
  (3 * x / 7 + 2 / (9 * x) = 1) → 
  x ≤ (63 + Real.sqrt 2457) / 54 ∧ 
  ∃ y : ℝ, (3 * y / 7 + 2 / (9 * y) = 1) ∧ y = (63 + Real.sqrt 2457) / 54 :=
by sorry

end largest_x_value_l2935_293583


namespace count_four_digit_numbers_l2935_293508

theorem count_four_digit_numbers : 
  (Finset.range 4001).card = (Finset.Icc 1000 5000).card := by sorry

end count_four_digit_numbers_l2935_293508


namespace sandwich_cost_l2935_293559

/-- The cost of Anna's sandwich given her breakfast and lunch expenses -/
theorem sandwich_cost (bagel_cost orange_juice_cost milk_cost lunch_difference : ℝ) : 
  bagel_cost = 0.95 →
  orange_juice_cost = 0.85 →
  milk_cost = 1.15 →
  lunch_difference = 4 →
  ∃ sandwich_cost : ℝ, 
    sandwich_cost + milk_cost = (bagel_cost + orange_juice_cost) + lunch_difference ∧
    sandwich_cost = 4.65 := by
  sorry

end sandwich_cost_l2935_293559


namespace coloring_four_cells_six_colors_l2935_293570

def ColoringMethods (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  let twoColorMethods := (Nat.choose n 2) * 2
  let threeColorMethods := (Nat.choose n 3) * (3 * 2^3 - Nat.choose 3 2 * 2)
  twoColorMethods + threeColorMethods

theorem coloring_four_cells_six_colors :
  ColoringMethods 6 4 3 = 390 :=
sorry

end coloring_four_cells_six_colors_l2935_293570


namespace bayonet_on_third_draw_l2935_293582

/-- Represents the number of screw base bulbs initially in the box -/
def screw_bulbs : ℕ := 3

/-- Represents the number of bayonet base bulbs initially in the box -/
def bayonet_bulbs : ℕ := 7

/-- Represents the total number of bulbs initially in the box -/
def total_bulbs : ℕ := screw_bulbs + bayonet_bulbs

/-- The probability of selecting a bayonet base bulb on the third draw,
    given that the first two draws were screw base bulbs -/
def prob_bayonet_third : ℚ := 7 / 120

theorem bayonet_on_third_draw :
  (screw_bulbs / total_bulbs) *
  ((screw_bulbs - 1) / (total_bulbs - 1)) *
  (bayonet_bulbs / (total_bulbs - 2)) = prob_bayonet_third := by
  sorry

end bayonet_on_third_draw_l2935_293582


namespace modulus_of_complex_l2935_293529

theorem modulus_of_complex (z : ℂ) (h : z = 4 + 3*I) : Complex.abs z = 5 := by
  sorry

end modulus_of_complex_l2935_293529


namespace range_of_a_l2935_293586

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 ≤ 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, ¬(p x a) → ¬(q x))
  (h3 : ∃ x, ¬(p x a) ∧ q x) :
  0 < a ∧ a ≤ 1 := by sorry

end range_of_a_l2935_293586


namespace triangle_inequality_expression_l2935_293593

theorem triangle_inequality_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (a + c) < 1 / 16 := by
  sorry

end triangle_inequality_expression_l2935_293593


namespace cube_root_sum_equals_two_l2935_293540

theorem cube_root_sum_equals_two (x : ℝ) (h1 : x > 0) 
  (h2 : (2 - x^3)^(1/3) + (2 + x^3)^(1/3) = 2) : x^6 = 100/27 := by
  sorry

end cube_root_sum_equals_two_l2935_293540
