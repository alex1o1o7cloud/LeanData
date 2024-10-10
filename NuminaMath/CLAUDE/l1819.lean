import Mathlib

namespace equation_with_integer_roots_l1819_181961

theorem equation_with_integer_roots :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (x y : ℤ), x ≠ y ∧
  (1 : ℚ) / (x + a) + (1 : ℚ) / (x + b) = (1 : ℚ) / c ∧
  (1 : ℚ) / (y + a) + (1 : ℚ) / (y + b) = (1 : ℚ) / c :=
by sorry

end equation_with_integer_roots_l1819_181961


namespace flight_cost_calculation_l1819_181965

def trip_expenses (initial_savings hotel_cost food_cost remaining_money : ℕ) : Prop :=
  ∃ flight_cost : ℕ, 
    initial_savings = hotel_cost + food_cost + flight_cost + remaining_money

theorem flight_cost_calculation (initial_savings hotel_cost food_cost remaining_money : ℕ) 
  (h : trip_expenses initial_savings hotel_cost food_cost remaining_money) :
  ∃ flight_cost : ℕ, flight_cost = 1200 :=
by
  sorry

end flight_cost_calculation_l1819_181965


namespace triangle_angle_b_l1819_181947

/-- In a triangle ABC, if side a = 1, side b = √3, and angle A = 30°, then angle B = 60° -/
theorem triangle_angle_b (a b : ℝ) (A B : ℝ) : 
  a = 1 → b = Real.sqrt 3 → A = π / 6 → B = π / 3 := by
  sorry

end triangle_angle_b_l1819_181947


namespace average_weight_section_B_l1819_181967

/-- Given a class with two sections A and B, prove the average weight of section B. -/
theorem average_weight_section_B 
  (students_A : ℕ) 
  (students_B : ℕ) 
  (avg_weight_A : ℝ) 
  (avg_weight_total : ℝ) 
  (h1 : students_A = 60)
  (h2 : students_B = 70)
  (h3 : avg_weight_A = 60)
  (h4 : avg_weight_total = 70.77) :
  ∃ avg_weight_B : ℝ, abs (avg_weight_B - 79.99) < 0.01 := by
  sorry

end average_weight_section_B_l1819_181967


namespace complex_magnitude_problem_l1819_181931

theorem complex_magnitude_problem (z : ℂ) (h : z = 3 + I) :
  Complex.abs (z^2 - 3*z) = Real.sqrt 10 := by sorry

end complex_magnitude_problem_l1819_181931


namespace bryden_received_value_l1819_181977

/-- The face value of a state quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 10

/-- The percentage of face value the collector offers -/
def collector_offer_percentage : ℕ := 1500

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_received : ℚ := (bryden_quarters : ℚ) * quarter_value * (collector_offer_percentage : ℚ) / 100

theorem bryden_received_value : bryden_received = 37.5 := by
  sorry

end bryden_received_value_l1819_181977


namespace f_is_odd_and_decreasing_l1819_181997

def f (x : ℝ) : ℝ := -x^3

theorem f_is_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end f_is_odd_and_decreasing_l1819_181997


namespace inequality_and_equality_condition_l1819_181984

theorem inequality_and_equality_condition (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) ≥ 1/2) ∧
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1/2 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end inequality_and_equality_condition_l1819_181984


namespace equal_ratios_sum_l1819_181909

theorem equal_ratios_sum (K L M : ℚ) : 
  (4 : ℚ) / 7 = K / 63 ∧ (4 : ℚ) / 7 = 84 / L ∧ (4 : ℚ) / 7 = M / 98 → 
  K + L + M = 239 := by
  sorry

end equal_ratios_sum_l1819_181909


namespace solution_set_of_quadratic_inequality_l1819_181991

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- State the theorem
theorem solution_set_of_quadratic_inequality :
  {x : ℝ | f x < 0} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end solution_set_of_quadratic_inequality_l1819_181991


namespace lattice_points_on_hyperbola_l1819_181904

theorem lattice_points_on_hyperbola : 
  ∃! (points : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 65) ∧ 
    points.card = 8 := by
  sorry

end lattice_points_on_hyperbola_l1819_181904


namespace base9_multiplication_addition_l1819_181920

/-- Converts a base-9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 9 + d) 0

/-- Converts a natural number to its base-9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- The main theorem to be proved -/
theorem base9_multiplication_addition :
  (base9ToNat [3, 2, 4] * base9ToNat [4, 6, 7]) + base9ToNat [1, 2, 3] =
  base9ToNat [2, 3, 4, 4, 2] := by
  sorry

#eval natToBase9 ((base9ToNat [3, 2, 4] * base9ToNat [4, 6, 7]) + base9ToNat [1, 2, 3])

end base9_multiplication_addition_l1819_181920


namespace burj_khalifa_height_is_830_l1819_181946

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := 324

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters -/
def height_difference : ℝ := 506

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := eiffel_tower_height + height_difference

theorem burj_khalifa_height_is_830 : burj_khalifa_height = 830 := by
  sorry

end burj_khalifa_height_is_830_l1819_181946


namespace triangle_angle_inequality_l1819_181988

theorem triangle_angle_inequality (a b c α β γ : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : α > 0 ∧ β > 0 ∧ γ > 0)
  (h3 : α + β + γ = π)
  (h4 : a + b > c ∧ b + c > a ∧ c + a > b) : 
  π / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < π / 2 := by
sorry

end triangle_angle_inequality_l1819_181988


namespace production_days_l1819_181985

theorem production_days (n : ℕ) (h1 : (50 * n + 90) / (n + 1) = 54) : n = 9 := by
  sorry

end production_days_l1819_181985


namespace netflix_shows_l1819_181924

/-- The number of shows watched per week by Gina and her sister on Netflix. -/
def total_shows (gina_minutes : ℕ) (show_length : ℕ) (gina_ratio : ℕ) : ℕ :=
  let gina_shows := gina_minutes / show_length
  let sister_shows := gina_shows / gina_ratio
  gina_shows + sister_shows

/-- Theorem stating the total number of shows watched per week given the conditions. -/
theorem netflix_shows : total_shows 900 50 3 = 24 := by
  sorry

end netflix_shows_l1819_181924


namespace B_and_C_complementary_l1819_181956

-- Define the sample space (outcomes of rolling a fair die)
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event B (up-facing side's number is no more than 3)
def B : Finset Nat := {1, 2, 3}

-- Define event C (up-facing side's number is at least 4)
def C : Finset Nat := {4, 5, 6}

-- Theorem stating that B and C are complementary
theorem B_and_C_complementary : B ∪ C = Ω ∧ B ∩ C = ∅ := by
  sorry


end B_and_C_complementary_l1819_181956


namespace floor_tile_count_l1819_181916

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  diagonalTiles : ℕ

/-- The conditions of our specific tiled floor. -/
def specialFloor : TiledFloor where
  width := 19
  length := 38
  diagonalTiles := 39

theorem floor_tile_count (floor : TiledFloor) 
  (h1 : floor.length = 2 * floor.width)
  (h2 : floor.diagonalTiles = 39) : 
  floor.width * floor.length = 722 := by
  sorry

#check floor_tile_count specialFloor

end floor_tile_count_l1819_181916


namespace sum_of_common_divisors_l1819_181982

def numbers : List Int := [36, 72, -24, 120, 96]

def is_common_divisor (d : Nat) : Bool :=
  numbers.all (fun n => n % d = 0)

def common_divisors : List Nat :=
  (List.range 37).filter is_common_divisor

theorem sum_of_common_divisors :
  common_divisors.sum = 16 := by
  sorry

end sum_of_common_divisors_l1819_181982


namespace books_sum_is_95_l1819_181934

/-- The total number of books Tim, Mike, Sarah, and Emily have together -/
def total_books (tim_books mike_books sarah_books emily_books : ℕ) : ℕ :=
  tim_books + mike_books + sarah_books + emily_books

/-- Theorem stating that the total number of books is 95 -/
theorem books_sum_is_95 :
  total_books 22 20 35 18 = 95 := by
  sorry

end books_sum_is_95_l1819_181934


namespace log_product_simplification_l1819_181962

theorem log_product_simplification : 
  Real.log 9 / Real.log 8 * (Real.log 32 / Real.log 27) = 10 / 9 := by
  sorry

end log_product_simplification_l1819_181962


namespace second_triangle_invalid_l1819_181926

-- Define the sides of the triangle
def a : ℝ := 15
def b : ℝ := 15
def c : ℝ := 30

-- Define the condition for a valid triangle (triangle inequality)
def is_valid_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

-- Theorem statement
theorem second_triangle_invalid :
  ¬(is_valid_triangle a b c) :=
sorry

end second_triangle_invalid_l1819_181926


namespace fixed_point_on_line_l1819_181941

theorem fixed_point_on_line (a : ℝ) : 
  let line := fun (x y : ℝ) => a * x + y + a + 1 = 0
  line (-1) (-1) := by
  sorry

end fixed_point_on_line_l1819_181941


namespace subset_from_intersection_l1819_181963

theorem subset_from_intersection (M N : Set α) : M ∩ N = M → M ⊆ N := by
  sorry

end subset_from_intersection_l1819_181963


namespace binomial_mode_is_four_l1819_181914

/-- The number of trials in the binomial distribution -/
def n : ℕ := 20

/-- The probability of success in each trial -/
def p : ℝ := 0.2

/-- The binomial probability mass function -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating that 4 is the mode of the binomial distribution B(20, 0.2) -/
theorem binomial_mode_is_four :
  ∀ k : ℕ, k ≠ 4 → binomialPMF 4 ≥ binomialPMF k :=
sorry

end binomial_mode_is_four_l1819_181914


namespace pqr_product_l1819_181910

theorem pqr_product (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h1 : p ∣ (q * r - 1)) (h2 : q ∣ (r * p - 1)) (h3 : r ∣ (p * q - 1)) :
  p * q * r = 30 := by
  sorry

end pqr_product_l1819_181910


namespace finite_valid_hexagon_angles_l1819_181964

/-- Represents a sequence of interior angles of a hexagon -/
structure HexagonAngles where
  x : ℕ
  d : ℕ

/-- Checks if a given HexagonAngles satisfies the required conditions -/
def isValidHexagonAngles (angles : HexagonAngles) : Prop :=
  angles.x > 30 ∧
  angles.x + 5 * angles.d < 150 ∧
  2 * angles.x + 5 * angles.d = 240

/-- The set of all valid HexagonAngles -/
def validHexagonAnglesSet : Set HexagonAngles :=
  {angles | isValidHexagonAngles angles}

theorem finite_valid_hexagon_angles : Set.Finite validHexagonAnglesSet := by
  sorry

end finite_valid_hexagon_angles_l1819_181964


namespace geometric_sequence_sum_l1819_181950

/-- A geometric sequence with specified properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 1 →
  a 2 + a 3 = 2 →
  a 6 + a 7 = 32 := by
  sorry

end geometric_sequence_sum_l1819_181950


namespace ages_sum_after_three_years_l1819_181998

/-- Given four persons a, b, c, and d with the following age relationships:
    - The sum of their present ages is S
    - a's age is twice b's age
    - c's age is half of a's age
    - d's age is the difference between a's and c's ages
    This theorem proves that the sum of their ages after 3 years is S + 12 -/
theorem ages_sum_after_three_years
  (S : ℝ) -- Sum of present ages
  (a b c d : ℝ) -- Present ages of individuals
  (h1 : a + b + c + d = S) -- Sum of present ages is S
  (h2 : a = 2 * b) -- a's age is twice b's age
  (h3 : c = a / 2) -- c's age is half of a's age
  (h4 : d = a - c) -- d's age is the difference between a's and c's ages
  : (a + 3) + (b + 3) + (c + 3) + (d + 3) = S + 12 := by
  sorry


end ages_sum_after_three_years_l1819_181998


namespace parabola_equation_from_hyperbola_l1819_181955

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Given hyperbola and conditions on a parabola, proves the equation of the parabola -/
theorem parabola_equation_from_hyperbola (h : Hyperbola) (p : Parabola) :
  h.equation = (fun x y => 16 * x^2 - 9 * y^2 = 144) →
  p.vertex = (0, 0) →
  (p.focus = (3, 0) ∨ p.focus = (-3, 0)) →
  (p.equation = (fun x y => y^2 = 24 * x) ∨ p.equation = (fun x y => y^2 = -24 * x)) :=
by sorry

end parabola_equation_from_hyperbola_l1819_181955


namespace specific_prism_properties_l1819_181938

/-- A right prism with a triangular base -/
structure TriangularPrism where
  base_side_a : ℝ
  base_side_b : ℝ
  base_side_c : ℝ
  section_cut_a : ℝ
  section_cut_b : ℝ
  section_cut_c : ℝ

/-- Calculate the volume of the bounded figure -/
def bounded_volume (prism : TriangularPrism) : ℝ :=
  sorry

/-- Calculate the total surface area of the bounded figure -/
def bounded_surface_area (prism : TriangularPrism) : ℝ :=
  sorry

/-- Theorem stating the volume and surface area of the specific prism -/
theorem specific_prism_properties :
  let prism : TriangularPrism := {
    base_side_a := 6,
    base_side_b := 8,
    base_side_c := 10,
    section_cut_a := 12,
    section_cut_b := 12,
    section_cut_c := 18
  }
  bounded_volume prism = 336 ∧ bounded_surface_area prism = 396 :=
by sorry

end specific_prism_properties_l1819_181938


namespace parallel_lines_imply_a_eq_neg_one_l1819_181948

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

theorem parallel_lines_imply_a_eq_neg_one (a : ℝ) :
  let l₁ : Line := ⟨a, 2, 6⟩
  let l₂ : Line := ⟨1, a - 1, a^2 - 1⟩
  parallel l₁ l₂ → a = -1 := by
  sorry

end parallel_lines_imply_a_eq_neg_one_l1819_181948


namespace lowest_sale_price_percentage_l1819_181905

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (additional_discount : ℝ) : 
  list_price = 80 ∧ 
  max_regular_discount = 0.5 ∧ 
  additional_discount = 0.2 → 
  (list_price * (1 - max_regular_discount) - list_price * additional_discount) / list_price = 0.3 := by
sorry

end lowest_sale_price_percentage_l1819_181905


namespace sequence_property_l1819_181944

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map a |> List.sum

theorem sequence_property (a : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → sequence_sum a n = 2 * a n - 4) →
  (∀ n : ℕ, n > 0 → a n = 2^(n+1)) :=
by
  sorry

end sequence_property_l1819_181944


namespace polynomial_simplification_l1819_181969

theorem polynomial_simplification (q : ℝ) :
  (5 * q^4 - 4 * q^3 + 7 * q - 8) + (3 - 5 * q^2 + q^3 - 2 * q) =
  5 * q^4 - 3 * q^3 - 5 * q^2 + 5 * q - 5 :=
by sorry

end polynomial_simplification_l1819_181969


namespace binary_product_in_base4_l1819_181913

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The first binary number: 1101₂ -/
def binary1 : List Bool := [true, true, false, true]

/-- The second binary number: 111₂ -/
def binary2 : List Bool := [true, true, true]

/-- Statement: The product of 1101₂ and 111₂ in base 4 is 311₄ -/
theorem binary_product_in_base4 :
  decimal_to_base4 (binary_to_decimal binary1 * binary_to_decimal binary2) = [3, 1, 1] := by
  sorry

end binary_product_in_base4_l1819_181913


namespace students_in_all_three_sections_l1819_181952

/-- Represents the number of students in each section and their intersections -/
structure ClubSections where
  totalStudents : ℕ
  music : ℕ
  drama : ℕ
  dance : ℕ
  atLeastTwo : ℕ
  allThree : ℕ

/-- The theorem stating the number of students in all three sections -/
theorem students_in_all_three_sections 
  (club : ClubSections) 
  (h1 : club.totalStudents = 30)
  (h2 : club.music = 15)
  (h3 : club.drama = 18)
  (h4 : club.dance = 12)
  (h5 : club.atLeastTwo = 14)
  (h6 : ∀ s : ℕ, s ≤ club.totalStudents → s ≥ club.music ∨ s ≥ club.drama ∨ s ≥ club.dance) :
  club.allThree = 6 := by
  sorry


end students_in_all_three_sections_l1819_181952


namespace inequality_proof_l1819_181906

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l1819_181906


namespace fibonacci_primitive_roots_l1819_181940

theorem fibonacci_primitive_roots (p : Nat) (g : Nat) (k : Nat) 
    (h1 : Nat.Prime p)
    (h2 : IsPrimitiveRoot g p)
    (h3 : g^2 % p = (g + 1) % p)
    (h4 : p = 4*k + 3) :
  IsPrimitiveRoot (g - 1) p ∧ 
  (g - 1)^(2*k + 3) % p = (g - 2) % p ∧
  IsPrimitiveRoot (g - 2) p :=
by sorry

end fibonacci_primitive_roots_l1819_181940


namespace investment_problem_l1819_181921

/-- Represents the investment scenario described in the problem -/
structure Investment where
  total : ℝ
  interest : ℝ
  known_rate : ℝ
  unknown_amount : ℝ

/-- The theorem statement representing the problem -/
theorem investment_problem (inv : Investment) 
  (h1 : inv.total = 15000)
  (h2 : inv.interest = 1023)
  (h3 : inv.known_rate = 0.075)
  (h4 : inv.unknown_amount = 8200)
  (h5 : inv.unknown_amount + (inv.total - inv.unknown_amount) * inv.known_rate = inv.interest) :
  inv.unknown_amount = 8200 := by
  sorry

#check investment_problem

end investment_problem_l1819_181921


namespace problem_solution_l1819_181992

theorem problem_solution (m n : ℝ) (hm : m^2 - 2*m = 1) (hn : n^2 - 2*n = 1) (hne : m ≠ n) :
  (m + n) - (m * n) = 3 := by
  sorry

end problem_solution_l1819_181992


namespace sixth_doll_size_l1819_181960

def doll_size (n : ℕ) : ℚ :=
  243 * (2/3)^(n-1)

theorem sixth_doll_size : doll_size 6 = 32 := by
  sorry

end sixth_doll_size_l1819_181960


namespace min_perimeter_non_congruent_isosceles_triangles_l1819_181911

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

theorem min_perimeter_non_congruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    t1.base * 9 = t2.base * 10 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      s1.base * 9 = s2.base * 10 →
      perimeter t1 ≤ perimeter s1 ∧
      perimeter t1 = 728 :=
by sorry

end min_perimeter_non_congruent_isosceles_triangles_l1819_181911


namespace problem_solution_l1819_181939

theorem problem_solution : 
  (-24 / (1/2 - 1/6 + 1/3) = -36) ∧ 
  (-1^3 - |(-9)| + 3 + 6 * (-1/3)^2 = -19/3) := by
  sorry

end problem_solution_l1819_181939


namespace factor_implies_d_value_l1819_181918

-- Define the polynomial g(x)
def g (d : ℝ) (x : ℝ) : ℝ := d * x^3 + 25 * x^2 - 5 * d * x + 45

-- State the theorem
theorem factor_implies_d_value :
  ∀ d : ℝ, (∀ x : ℝ, (x + 5) ∣ g d x) → d = 6.7 :=
by sorry

end factor_implies_d_value_l1819_181918


namespace f_increasing_on_2_3_l1819_181980

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_increasing_on_2_3 (heven : is_even f) (hperiodic : is_periodic f 2) 
  (hdecr : is_decreasing_on f (-1) 0) : is_increasing_on f 2 3 := by sorry

end f_increasing_on_2_3_l1819_181980


namespace angle_measure_l1819_181951

/-- Given two angles AOB and BOC, proves that angle AOC is either the sum or difference of these angles. -/
theorem angle_measure (α β : ℝ) (hα : α = 30) (hβ : β = 15) :
  ∃ γ : ℝ, (γ = α + β ∨ γ = α - β) ∧ (γ = 45 ∨ γ = 15) := by
  sorry

end angle_measure_l1819_181951


namespace increasing_f_implies_t_ge_5_l1819_181919

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

/-- The function f(x) defined as the dot product of (x^2, x+1) and (1-x, t) -/
def f (t : ℝ) (x : ℝ) : ℝ := dot_product (x^2, x+1) (1-x, t)

/-- A function is increasing on an interval if for any two points in the interval, 
    the function value at the larger point is greater than at the smaller point -/
def is_increasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → g x < g y

theorem increasing_f_implies_t_ge_5 :
  ∀ t : ℝ, is_increasing (f t) (-1) 1 → t ≥ 5 := by
  sorry

end increasing_f_implies_t_ge_5_l1819_181919


namespace jeans_card_collection_l1819_181945

theorem jeans_card_collection (num_groups : ℕ) (cards_per_group : ℕ) 
  (h1 : num_groups = 9) (h2 : cards_per_group = 8) :
  num_groups * cards_per_group = 72 := by
  sorry

end jeans_card_collection_l1819_181945


namespace significant_figures_and_precision_of_0_03020_l1819_181949

/-- Represents a decimal number with its string representation -/
structure DecimalNumber where
  representation : String
  deriving Repr

/-- Counts the number of significant figures in a decimal number -/
def countSignificantFigures (n : DecimalNumber) : Nat :=
  sorry

/-- Determines the precision of a decimal number -/
inductive Precision
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths
  deriving Repr

def getPrecision (n : DecimalNumber) : Precision :=
  sorry

theorem significant_figures_and_precision_of_0_03020 :
  let n : DecimalNumber := { representation := "0.03020" }
  countSignificantFigures n = 4 ∧ getPrecision n = Precision.TenThousandths :=
sorry

end significant_figures_and_precision_of_0_03020_l1819_181949


namespace least_n_square_and_cube_n_144_satisfies_least_n_is_144_l1819_181970

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem least_n_square_and_cube :
  ∀ n : ℕ, n > 0 →
    (is_perfect_square (9*n) ∧ is_perfect_cube (12*n)) →
    n ≥ 144 :=
by sorry

theorem n_144_satisfies :
  is_perfect_square (9*144) ∧ is_perfect_cube (12*144) :=
by sorry

theorem least_n_is_144 :
  ∀ n : ℕ, n > 0 →
    (is_perfect_square (9*n) ∧ is_perfect_cube (12*n)) →
    n = 144 :=
by sorry

end least_n_square_and_cube_n_144_satisfies_least_n_is_144_l1819_181970


namespace sum_of_squares_of_roots_l1819_181979

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p - 9 = 0) → 
  (3 * q^3 - 2 * q^2 + 6 * q - 9 = 0) → 
  (3 * r^3 - 2 * r^2 + 6 * r - 9 = 0) → 
  p^2 + q^2 + r^2 = -32/9 := by
sorry

end sum_of_squares_of_roots_l1819_181979


namespace total_apples_in_basket_l1819_181937

theorem total_apples_in_basket (red_apples green_apples : ℕ) 
  (h1 : red_apples = 7) 
  (h2 : green_apples = 2) : 
  red_apples + green_apples = 9 := by
  sorry

end total_apples_in_basket_l1819_181937


namespace negation_of_universal_statement_l1819_181993

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x - 1| ≠ 2) ↔ (∃ x ∈ S, |x - 1| = 2) := by
  sorry

end negation_of_universal_statement_l1819_181993


namespace custom_operation_result_l1819_181976

def custom_operation (A B : Set ℕ) : Set ℕ :=
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem custom_operation_result :
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {4, 5, 6}
  custom_operation A B = {1, 2, 3, 6} := by
  sorry

end custom_operation_result_l1819_181976


namespace max_consecutive_sum_less_than_1000_l1819_181935

theorem max_consecutive_sum_less_than_1000 :
  ∀ n : ℕ, n > 0 → (n * (n + 1) / 2 < 1000 ↔ n ≤ 44) :=
by sorry

end max_consecutive_sum_less_than_1000_l1819_181935


namespace crayons_per_friend_l1819_181953

theorem crayons_per_friend (total_crayons : ℕ) (num_friends : ℕ) (crayons_per_friend : ℕ) : 
  total_crayons = 210 → num_friends = 30 → crayons_per_friend = total_crayons / num_friends →
  crayons_per_friend = 7 := by
sorry

end crayons_per_friend_l1819_181953


namespace complex_number_in_third_quadrant_l1819_181915

def complex_number : ℂ := Complex.I * ((-2 : ℝ) + Complex.I)

theorem complex_number_in_third_quadrant :
  let z := complex_number
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l1819_181915


namespace race_outcomes_count_l1819_181996

/-- The number of participants in the race -/
def n : ℕ := 6

/-- The number of places we're considering -/
def k : ℕ := 4

/-- The number of different possible outcomes for the first four places in the race -/
def race_outcomes : ℕ := n * (n - 1) * (n - 2) * (n - 3)

theorem race_outcomes_count : race_outcomes = 360 := by
  sorry

end race_outcomes_count_l1819_181996


namespace solution_set_equivalent_to_inequality_l1819_181908

def solution_set : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

def inequality (x : ℝ) : Prop := -x^2 + 3*x - 2 ≥ 0

theorem solution_set_equivalent_to_inequality :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end solution_set_equivalent_to_inequality_l1819_181908


namespace valid_triples_l1819_181912

-- Define the type for our triples
def Triple := (Nat × Nat × Nat)

-- Define the conditions
def satisfiesConditions (t : Triple) : Prop :=
  let (a, b, c) := t
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧  -- positive integers
  (a ≤ b) ∧ (b ≤ c) ∧  -- ordered
  (Nat.gcd a (Nat.gcd b c) = 1) ∧  -- gcd(a,b,c) = 1
  ((a + b + c) ∣ (a^12 + b^12 + c^12)) ∧
  ((a + b + c) ∣ (a^23 + b^23 + c^23)) ∧
  ((a + b + c) ∣ (a^11004 + b^11004 + c^11004))

-- The theorem
theorem valid_triples :
  {t : Triple | satisfiesConditions t} = {(1,1,1), (1,1,4)} := by
  sorry

end valid_triples_l1819_181912


namespace not_in_sequence_l1819_181954

theorem not_in_sequence : ¬∃ (n : ℕ), 24 - 2 * n = 3 := by sorry

end not_in_sequence_l1819_181954


namespace expression_simplification_l1819_181942

theorem expression_simplification : 
  ((3 + 4 + 5 + 6) / 2) + ((3 * 6 + 9) / 3 + 1) = 19 := by
  sorry

end expression_simplification_l1819_181942


namespace fish_difference_l1819_181901

-- Define the sizes of the tanks
def first_tank_size : ℕ := 48
def second_tank_size : ℕ := first_tank_size / 2

-- Define the fish sizes
def first_tank_fish_size : ℕ := 3
def second_tank_fish_size : ℕ := 2

-- Calculate the number of fish in each tank
def fish_in_first_tank : ℕ := first_tank_size / first_tank_fish_size
def fish_in_second_tank : ℕ := second_tank_size / second_tank_fish_size

-- Calculate the number of fish in the first tank after one is eaten
def fish_in_first_tank_after_eating : ℕ := fish_in_first_tank - 1

-- Theorem to prove
theorem fish_difference :
  fish_in_first_tank_after_eating - fish_in_second_tank = 3 :=
by
  sorry

end fish_difference_l1819_181901


namespace walnut_trees_before_planting_l1819_181974

theorem walnut_trees_before_planting 
  (total_after : ℕ) 
  (planted : ℕ) 
  (h1 : total_after = 55) 
  (h2 : planted = 33) :
  total_after - planted = 22 := by
  sorry

end walnut_trees_before_planting_l1819_181974


namespace expression_factorization_l1819_181903

theorem expression_factorization (a : ℝ) : 
  (8 * a^3 + 105 * a^2 + 7) - (-9 * a^3 + 16 * a^2 - 14) = a^2 * (17 * a + 89) + 21 := by
  sorry

end expression_factorization_l1819_181903


namespace root_zero_implies_k_five_l1819_181986

theorem root_zero_implies_k_five (k : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ 8 * x^2 - (k - 1) * x - k + 5 = 0) ∧ 
  (8 * 0^2 - (k - 1) * 0 - k + 5 = 0) → 
  k = 5 := by sorry

end root_zero_implies_k_five_l1819_181986


namespace highway_speed_l1819_181966

/-- Prove that given the conditions, the average speed on the highway is 87 km/h -/
theorem highway_speed (total_distance : ℝ) (total_time : ℝ) (highway_time : ℝ) (city_time : ℝ) (city_speed : ℝ) :
  total_distance = 59 →
  total_time = 1 →
  highway_time = 1/3 →
  city_time = 2/3 →
  city_speed = 45 →
  (total_distance - city_speed * city_time) / highway_time = 87 := by
sorry

end highway_speed_l1819_181966


namespace farmer_tomatoes_l1819_181959

theorem farmer_tomatoes (initial_tomatoes : ℕ) (picked_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 17)
  (h2 : initial_tomatoes - picked_tomatoes = 8) :
  picked_tomatoes = 9 := by
  sorry

end farmer_tomatoes_l1819_181959


namespace expression_evaluation_l1819_181936

theorem expression_evaluation (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (a*b + b*c + c*d + d*a + a*c + b*d)⁻¹ *
  ((a*b)⁻¹ + (b*c)⁻¹ + (c*d)⁻¹ + (d*a)⁻¹ + (a*c)⁻¹ + (b*d)⁻¹) = (a*b*c*d)⁻¹ := by
  sorry

end expression_evaluation_l1819_181936


namespace function_bounded_l1819_181989

/-- The function f(x, y) = √(4 - x² - y²) is bounded between 0 and 2 -/
theorem function_bounded (x y : ℝ) (h : x^2 + y^2 ≤ 4) :
  0 ≤ Real.sqrt (4 - x^2 - y^2) ∧ Real.sqrt (4 - x^2 - y^2) ≤ 2 :=
by sorry

end function_bounded_l1819_181989


namespace linear_function_not_in_third_quadrant_l1819_181983

/-- A linear function y = (2k-1)x + k does not pass through the third quadrant
    if and only if 0 ≤ k < 1/2 -/
theorem linear_function_not_in_third_quadrant (k : ℝ) :
  (∀ x y : ℝ, y = (2*k - 1)*x + k → ¬(x < 0 ∧ y < 0)) ↔ (0 ≤ k ∧ k < 1/2) := by
  sorry

end linear_function_not_in_third_quadrant_l1819_181983


namespace fraction_to_decimal_l1819_181957

theorem fraction_to_decimal : (31 : ℚ) / (2 * 5^6) = 0.000992 := by sorry

end fraction_to_decimal_l1819_181957


namespace compute_expression_l1819_181917

theorem compute_expression : 15 * (1 / 17) * 34 - (1 / 2) = 59 / 2 := by
  sorry

end compute_expression_l1819_181917


namespace bug_path_distance_l1819_181923

theorem bug_path_distance (r : Real) (leg : Real) (h1 : r = 40) (h2 : leg = 50) :
  let diameter := 2 * r
  let other_leg := Real.sqrt (diameter ^ 2 - leg ^ 2)
  diameter + leg + other_leg = 192.45 := by
  sorry

end bug_path_distance_l1819_181923


namespace dice_probability_l1819_181958

/-- The number of dice --/
def n : ℕ := 8

/-- The number of sides on each die --/
def sides : ℕ := 8

/-- The number of favorable outcomes (dice showing a number less than 5) --/
def k : ℕ := 4

/-- The probability of a single die showing a number less than 5 --/
def p : ℚ := 1/2

/-- The probability of exactly k out of n dice showing a number less than 5 --/
def probability : ℚ := (n.choose k) * p^k * (1-p)^(n-k)

theorem dice_probability : probability = 35/128 := by sorry

end dice_probability_l1819_181958


namespace power_of_i_sum_l1819_181943

theorem power_of_i_sum : ∃ (i : ℂ), i^2 = -1 ∧ i^14760 + i^14761 + i^14762 + i^14763 = 0 := by
  sorry

end power_of_i_sum_l1819_181943


namespace contribution_problem_l1819_181902

/-- The contribution problem -/
theorem contribution_problem (total_sum : ℕ) : 
  (10 : ℕ) * 300 = total_sum ∧ 
  (15 : ℕ) * (300 - 100) = total_sum := by
  sorry

#check contribution_problem

end contribution_problem_l1819_181902


namespace fraction_evaluation_l1819_181927

theorem fraction_evaluation : (36 - 12) / (12 - 4) = 3 := by sorry

end fraction_evaluation_l1819_181927


namespace rectangle_area_breadth_ratio_l1819_181981

/-- Theorem: For a rectangular plot with breadth 14 meters and length 10 meters greater than its breadth, the ratio of its area to its breadth is 24:1. -/
theorem rectangle_area_breadth_ratio :
  ∀ (length breadth area : ℝ),
  breadth = 14 →
  length = breadth + 10 →
  area = length * breadth →
  area / breadth = 24 := by
sorry

end rectangle_area_breadth_ratio_l1819_181981


namespace scientific_notation_of_12400_l1819_181999

theorem scientific_notation_of_12400 :
  let num_athletes : ℕ := 12400
  1.24 * (10 : ℝ)^4 = num_athletes := by sorry

end scientific_notation_of_12400_l1819_181999


namespace annes_age_l1819_181925

theorem annes_age (maude emile anne : ℕ) 
  (h1 : anne = 2 * emile)
  (h2 : emile = 6 * maude)
  (h3 : maude = 8) :
  anne = 96 := by
  sorry

end annes_age_l1819_181925


namespace parabola_translation_l1819_181973

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) + v }

/-- The original parabola y = x^2 - 2 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 - 2 }

/-- The translated parabola -/
def translated_parabola : Parabola :=
  translate original_parabola 1 3

theorem parabola_translation :
  translated_parabola.f = fun x => (x - 1)^2 + 1 := by
  sorry


end parabola_translation_l1819_181973


namespace set_membership_properties_l1819_181990

theorem set_membership_properties (M P : Set α) (h_nonempty : M.Nonempty) 
  (h_not_subset : ¬(M ⊆ P)) : 
  (∃ x, x ∈ M ∧ x ∉ P) ∧ (∃ y, y ∈ M ∧ y ∈ P) := by
  sorry

end set_membership_properties_l1819_181990


namespace correct_proposition_l1819_181968

def p : Prop := ∀ x : ℝ, x^2 - x + 2 < 0

def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 ≥ 1

theorem correct_proposition : ¬p ∧ q := by
  sorry

end correct_proposition_l1819_181968


namespace parabola_vertex_c_value_l1819_181987

/-- Given a parabola of the form y = 2x^2 + c with vertex at (0,1), prove that c = 1 -/
theorem parabola_vertex_c_value (c : ℝ) : 
  (∀ x y : ℝ, y = 2 * x^2 + c) →   -- Parabola equation
  (0, 1) = (0, 2 * 0^2 + c) →      -- Vertex at (0,1)
  c = 1 := by sorry

end parabola_vertex_c_value_l1819_181987


namespace partial_fraction_decomposition_l1819_181932

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 → x ≠ 3 → x ≠ 4 →
  (x^3 - 4*x^2 + 5*x - 7) / ((x - 1)*(x - 2)*(x - 3)*(x - 4)) =
  5/6 / (x - 1) + (-5/2) / (x - 2) + 1/2 / (x - 3) + 13/6 / (x - 4) :=
by sorry

end partial_fraction_decomposition_l1819_181932


namespace zero_area_quadrilateral_l1819_181995

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a quadrilateral given its four vertices in 3D space -/
def quadrilateralArea (A B C D : Point3D) : ℝ :=
  sorry

/-- The main theorem stating that the area of the quadrilateral with given vertices is 0 -/
theorem zero_area_quadrilateral :
  let A : Point3D := ⟨2, 4, 6⟩
  let B : Point3D := ⟨7, 9, 11⟩
  let C : Point3D := ⟨1, 3, 5⟩
  let D : Point3D := ⟨6, 8, 10⟩
  quadrilateralArea A B C D = 0 := by
  sorry

end zero_area_quadrilateral_l1819_181995


namespace quadratic_root_implies_s_value_l1819_181972

theorem quadratic_root_implies_s_value (p s : ℝ) :
  (∃ (x : ℂ), 3 * x^2 + p * x + s = 0 ∧ x = 4 + 3*I) →
  s = 75 := by
sorry

end quadratic_root_implies_s_value_l1819_181972


namespace jessica_apple_pie_servings_l1819_181922

/-- Calculates the number of apples per serving in Jessica's apple pies. -/
def apples_per_serving (num_guests : ℕ) (num_pies : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℚ) : ℚ :=
  let total_apples := num_guests * apples_per_guest
  let total_servings := num_pies * servings_per_pie
  total_apples / total_servings

/-- Theorem stating that given Jessica's conditions, each serving requires 1.5 apples. -/
theorem jessica_apple_pie_servings :
  apples_per_serving 12 3 8 3 = 3/2 := by
  sorry

end jessica_apple_pie_servings_l1819_181922


namespace halfway_between_fractions_l1819_181933

theorem halfway_between_fractions : (2 / 9 + 5 / 12) / 2 = 23 / 72 := by
  sorry

end halfway_between_fractions_l1819_181933


namespace farm_problem_solution_l1819_181978

/-- Represents the farm ploughing problem -/
structure FarmProblem where
  planned_daily_area : ℕ  -- Planned area to plough per day
  actual_daily_area : ℕ   -- Actual area ploughed per day
  extra_days : ℕ          -- Extra days worked
  total_field_area : ℕ    -- Total area of the farm field

/-- Calculates the area left to plough -/
def area_left_to_plough (fp : FarmProblem) : ℕ :=
  let planned_days := fp.total_field_area / fp.planned_daily_area
  let actual_days := planned_days + fp.extra_days
  let ploughed_area := fp.actual_daily_area * actual_days
  fp.total_field_area - ploughed_area

/-- Theorem stating the correct result for the given problem -/
theorem farm_problem_solution :
  let fp : FarmProblem := {
    planned_daily_area := 340,
    actual_daily_area := 85,
    extra_days := 2,
    total_field_area := 280
  }
  area_left_to_plough fp = 25 := by
  sorry

end farm_problem_solution_l1819_181978


namespace a_always_gets_half_rule_independent_l1819_181930

/-- The game rules for counter division --/
inductive Rule
| R1  -- B takes the biggest and smallest heaps
| R2  -- B takes the two middling heaps
| R3  -- B chooses between R1 and R2

/-- The optimal number of counters A can obtain --/
def optimal_counters (N : ℕ) (r : Rule) : ℕ :=
  N / 2

/-- Theorem: A always gets ⌊N/2⌋ counters regardless of the rule --/
theorem a_always_gets_half (N : ℕ) (h : N ≥ 4) (r : Rule) :
  optimal_counters N r = N / 2 := by
  sorry

/-- Corollary: The result is independent of the chosen rule --/
theorem rule_independent (N : ℕ) (h : N ≥ 4) (r1 r2 : Rule) :
  optimal_counters N r1 = optimal_counters N r2 := by
  sorry

end a_always_gets_half_rule_independent_l1819_181930


namespace base_conversion_subtraction_l1819_181994

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem base_conversion_subtraction :
  base8ToBase10 52103 - base9ToBase10 1452 = 20471 := by sorry

end base_conversion_subtraction_l1819_181994


namespace meal_combinations_l1819_181907

/-- The number of dishes available in the restaurant -/
def num_dishes : ℕ := 15

/-- The number of ways one person can choose their meal -/
def individual_choices (n : ℕ) : ℕ := n + n * n

/-- The total number of meal combinations for two people -/
def total_combinations (n : ℕ) : ℕ := (individual_choices n) * (individual_choices n)

/-- Theorem stating the total number of meal combinations -/
theorem meal_combinations : total_combinations num_dishes = 57600 := by
  sorry

end meal_combinations_l1819_181907


namespace intersection_locus_is_circle_l1819_181928

/-- The locus of intersection points of two parametric lines forms a circle -/
theorem intersection_locus_is_circle :
  ∀ (x y u : ℝ),
  (u * x - 3 * y - 2 * u = 0) →
  (2 * x - 3 * u * y + u = 0) →
  ∃ (center_x center_y radius : ℝ),
  (x - center_x)^2 + (y - center_y)^2 = radius^2 := by
  sorry

end intersection_locus_is_circle_l1819_181928


namespace smallest_n_for_interval_condition_l1819_181900

theorem smallest_n_for_interval_condition : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 →
    ∃ (k : ℕ), (m : ℚ) / 1993 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m : ℚ) + 1) / 1994) ∧
  (∀ (n' : ℕ), 0 < n' ∧ n' < n →
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 ∧
      ∀ (k : ℕ), ((m : ℚ) / 1993 ≥ (k : ℚ) / n' ∨ (k : ℚ) / n' ≥ ((m : ℚ) + 1) / 1994)) ∧
  n = 3987 :=
sorry

end smallest_n_for_interval_condition_l1819_181900


namespace sqrt_15_times_sqrt_3_minus_4_between_2_and_3_l1819_181929

theorem sqrt_15_times_sqrt_3_minus_4_between_2_and_3 :
  2 < Real.sqrt 15 * Real.sqrt 3 - 4 ∧ Real.sqrt 15 * Real.sqrt 3 - 4 < 3 := by
  sorry

end sqrt_15_times_sqrt_3_minus_4_between_2_and_3_l1819_181929


namespace driver_net_pay_rate_l1819_181971

/-- Calculates the net rate of pay for a driver given specific conditions. -/
theorem driver_net_pay_rate 
  (travel_time : ℝ) 
  (speed : ℝ) 
  (fuel_efficiency : ℝ) 
  (pay_per_mile : ℝ) 
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_price = 2.50) :
  (pay_per_mile * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 :=
by sorry

end driver_net_pay_rate_l1819_181971


namespace optimal_greening_arrangement_l1819_181975

/-- Represents a construction team with daily greening area and cost -/
structure Team where
  daily_area : ℝ
  daily_cost : ℝ

/-- The optimal greening arrangement problem -/
def OptimalGreeningArrangement (total_area : ℝ) (max_days : ℕ) (team_a team_b : Team) : Prop :=
  -- Team A is 1.8 times more efficient than Team B
  team_a.daily_area = 1.8 * team_b.daily_area ∧
  -- Team A takes 4 days less than Team B for 450 m²
  (450 / team_a.daily_area) + 4 = 450 / team_b.daily_area ∧
  -- Optimal arrangement
  ∃ (days_a days_b : ℕ),
    -- Total area constraint
    team_a.daily_area * days_a + team_b.daily_area * days_b ≥ total_area ∧
    -- Time constraint
    days_a + days_b ≤ max_days ∧
    -- Optimal solution
    days_a = 30 ∧ days_b = 18 ∧
    -- Minimum cost
    team_a.daily_cost * days_a + team_b.daily_cost * days_b = 40.5

/-- Theorem stating the optimal greening arrangement -/
theorem optimal_greening_arrangement :
  OptimalGreeningArrangement 3600 48
    (Team.mk 90 1.05)
    (Team.mk 50 0.5) := by
  sorry

end optimal_greening_arrangement_l1819_181975
