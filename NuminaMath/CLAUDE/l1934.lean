import Mathlib

namespace NUMINAMATH_CALUDE_base_seven_234567_equals_41483_l1934_193465

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_234567_equals_41483 :
  base_seven_to_decimal [7, 6, 5, 4, 3, 2] = 41483 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_234567_equals_41483_l1934_193465


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1934_193448

theorem fraction_equation_solution (x : ℚ) : 
  (x + 11) / (x - 4) = (x - 3) / (x + 6) → x = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1934_193448


namespace NUMINAMATH_CALUDE_triangle_properties_l1934_193454

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (a + b) / Real.sin (A + B) = (a - c) / (Real.sin A - Real.sin B) →
  b = 3 →
  Real.cos A = Real.sqrt 6 / 3 →
  B = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1934_193454


namespace NUMINAMATH_CALUDE_total_fruits_is_112_l1934_193457

/-- The number of apples and pears satisfying the given conditions -/
def total_fruits (apples pears : ℕ) : Prop :=
  ∃ (bags : ℕ),
    (5 * bags + 4 = apples) ∧
    (3 * bags = pears - 12) ∧
    (7 * bags = apples) ∧
    (3 * bags + 12 = pears)

/-- Theorem stating that the total number of fruits is 112 -/
theorem total_fruits_is_112 :
  ∃ (apples pears : ℕ), total_fruits apples pears ∧ apples + pears = 112 :=
sorry

end NUMINAMATH_CALUDE_total_fruits_is_112_l1934_193457


namespace NUMINAMATH_CALUDE_max_value_squared_l1934_193498

theorem max_value_squared (a b x y : ℝ) : 
  a > 0 → b > 0 → a ≥ b → 
  0 ≤ x → x < a → 
  0 ≤ y → y < b → 
  a^2 + y^2 = b^2 + x^2 → 
  (a - x)^2 + (b + y)^2 = b^2 + x^2 →
  x = a - 2*b →
  y = b/2 →
  (∀ ρ : ℝ, (a/b)^2 ≤ ρ^2 → ρ^2 ≤ 4/9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_squared_l1934_193498


namespace NUMINAMATH_CALUDE_spatial_relationships_l1934_193436

-- Define the basic concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships
def intersect (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- State the propositions
def proposition_1 (l1 l2 l3 l4 : Line) : Prop :=
  intersect l1 l3 → intersect l2 l4 → skew l3 l4 → skew l1 l2

def proposition_2 (l1 l2 : Line) (p1 p2 : Plane) : Prop :=
  parallel_planes p1 p2 → parallel_lines l1 p1 → parallel_lines l2 p2 → parallel_lines l1 l2

def proposition_3 (l1 l2 : Line) (p : Plane) : Prop :=
  perpendicular_to_plane l1 p → perpendicular_to_plane l2 p → parallel_lines l1 l2

def proposition_4 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular_planes p1 p2 →
  line_in_plane l p1 →
  ¬perpendicular_to_plane l (line_of_intersection p1 p2) →
  ¬perpendicular_to_plane l p2

theorem spatial_relationships :
  (∀ l1 l2 l3 l4 : Line, ¬proposition_1 l1 l2 l3 l4) ∧
  (∀ l1 l2 : Line, ∀ p1 p2 : Plane, ¬proposition_2 l1 l2 p1 p2) ∧
  (∀ l1 l2 : Line, ∀ p : Plane, proposition_3 l1 l2 p) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_4 p1 p2 l) :=
sorry

end NUMINAMATH_CALUDE_spatial_relationships_l1934_193436


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1934_193406

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + a = 0 ∧ x = 2) → 
  (a = 2 ∧ ∃ y : ℝ, y^2 - 3*y + a = 0 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1934_193406


namespace NUMINAMATH_CALUDE_multiply_63_57_l1934_193487

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_multiply_63_57_l1934_193487


namespace NUMINAMATH_CALUDE_divisible_by_72_sum_of_digits_l1934_193460

theorem divisible_by_72_sum_of_digits (A B : ℕ) : 
  A < 10 → B < 10 → 
  (100000 * A + 44610 + B) % 72 = 0 → 
  A + B = 12 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_72_sum_of_digits_l1934_193460


namespace NUMINAMATH_CALUDE_impossible_equal_side_sums_l1934_193450

/-- Represents the pattern of squares in the problem -/
structure SquarePattern :=
  (vertices : Fin 24 → ℕ)
  (is_consecutive : ∀ i : Fin 23, vertices i.succ = vertices i + 1)
  (is_bijective : Function.Bijective vertices)

/-- Represents a side of a square in the pattern -/
inductive Side : Type
| Top : Side
| Right : Side
| Bottom : Side
| Left : Side

/-- Gets the vertices on a given side of a square -/
def side_vertices (square : Fin 4) (side : Side) : Fin 24 → Prop :=
  sorry

/-- The sum of numbers on a side of a square -/
def side_sum (p : SquarePattern) (square : Fin 4) (side : Side) : ℕ :=
  sorry

/-- The theorem stating the impossibility of the required arrangement -/
theorem impossible_equal_side_sums :
  ¬ ∃ (p : SquarePattern),
    ∀ (s1 s2 : Fin 4) (side1 side2 : Side),
      side_sum p s1 side1 = side_sum p s2 side2 :=
sorry

end NUMINAMATH_CALUDE_impossible_equal_side_sums_l1934_193450


namespace NUMINAMATH_CALUDE_distance_to_focus_l1934_193443

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the point P
structure Point (x y : ℝ) where
  on_parabola : parabola x y
  distance_to_y_axis : x = 4

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem distance_to_focus (P : Point x y) : 
  Real.sqrt ((x - 2)^2 + y^2) = 6 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l1934_193443


namespace NUMINAMATH_CALUDE_function_q_polynomial_l1934_193433

/-- Given a function q(x) satisfying the equation
    q(x) + (2x^6 + 4x^4 + 5x^2 + 7) = (3x^4 + 18x^3 + 15x^2 + 8x + 3),
    prove that q(x) = -2x^6 - x^4 + 18x^3 + 10x^2 + 8x - 4 -/
theorem function_q_polynomial (q : ℝ → ℝ) :
  (∀ x, q x + (2*x^6 + 4*x^4 + 5*x^2 + 7) = (3*x^4 + 18*x^3 + 15*x^2 + 8*x + 3)) →
  (∀ x, q x = -2*x^6 - x^4 + 18*x^3 + 10*x^2 + 8*x - 4) := by
  sorry

end NUMINAMATH_CALUDE_function_q_polynomial_l1934_193433


namespace NUMINAMATH_CALUDE_total_labor_tools_l1934_193428

/-- Given a school with 3 grades, where each grade receives n sets of labor tools,
    prove that the total number of sets needed is 3n. -/
theorem total_labor_tools (n : ℕ) : 3 * n = 3 * n := by sorry

end NUMINAMATH_CALUDE_total_labor_tools_l1934_193428


namespace NUMINAMATH_CALUDE_odd_function_monotonicity_l1934_193483

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (4^x + 1)

theorem odd_function_monotonicity (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = -1/2 ∧ ∀ x₁ x₂, x₁ < x₂ → f a x₁ > f a x₂) := by sorry

end NUMINAMATH_CALUDE_odd_function_monotonicity_l1934_193483


namespace NUMINAMATH_CALUDE_initial_investment_is_200_l1934_193459

/-- Represents the simple interest calculation -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that given the conditions, the initial investment is $200 -/
theorem initial_investment_is_200 
  (P : ℝ) 
  (h1 : simpleInterest P (1/15) 3 = 240) 
  (h2 : simpleInterest 150 (1/15) 6 = 210) : 
  P = 200 := by
  sorry

#check initial_investment_is_200

end NUMINAMATH_CALUDE_initial_investment_is_200_l1934_193459


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1934_193478

theorem necessary_not_sufficient :
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1934_193478


namespace NUMINAMATH_CALUDE_sum_of_self_opposite_and_self_reciprocal_l1934_193409

theorem sum_of_self_opposite_and_self_reciprocal (a b : ℝ) : 
  ((-a) = a) → ((1 / b) = b) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_self_opposite_and_self_reciprocal_l1934_193409


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l1934_193485

/-- A cylinder with a rectangular front view of area 6 has a lateral area of 6π -/
theorem cylinder_lateral_area (h : ℝ) (h_pos : h > 0) : 
  let d := 6 / h  -- diameter of the base
  let lateral_area := π * d * h
  lateral_area = 6 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l1934_193485


namespace NUMINAMATH_CALUDE_sales_tax_difference_example_l1934_193476

/-- The difference between two sales tax amounts -/
def sales_tax_difference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate1 - price * rate2

/-- Theorem: The difference between a 7.25% sales tax and a 7% sales tax on an item priced at $50 before tax is $0.125 -/
theorem sales_tax_difference_example : 
  sales_tax_difference 50 0.0725 0.07 = 0.125 := by
sorry

end NUMINAMATH_CALUDE_sales_tax_difference_example_l1934_193476


namespace NUMINAMATH_CALUDE_not_cube_sum_l1934_193453

theorem not_cube_sum (a b : ℤ) : ¬ ∃ (k : ℤ), a^3 + b^3 + 4 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_not_cube_sum_l1934_193453


namespace NUMINAMATH_CALUDE_extreme_value_implies_m_eq_two_l1934_193458

/-- The function f(x) = x³ - (3/2)x² + m has an extreme value of 3/2 in the interval (0, 2) -/
def has_extreme_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo 0 2, f x = 3/2 ∧ ∀ y ∈ Set.Ioo 0 2, f y ≤ f x

/-- The main theorem stating that if f(x) = x³ - (3/2)x² + m has an extreme value of 3/2 
    in the interval (0, 2), then m = 2 -/
theorem extreme_value_implies_m_eq_two :
  ∀ m : ℝ, has_extreme_value (fun x => x^3 - (3/2)*x^2 + m) m → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_m_eq_two_l1934_193458


namespace NUMINAMATH_CALUDE_next_simultaneous_activation_l1934_193401

/-- Represents the time interval in minutes for each location's signal -/
structure SignalIntervals :=
  (fire : ℕ)
  (police : ℕ)
  (hospital : ℕ)

/-- Calculates the time in minutes until the next simultaneous activation -/
def timeUntilNextSimultaneous (intervals : SignalIntervals) : ℕ :=
  Nat.lcm (Nat.lcm intervals.fire intervals.police) intervals.hospital

/-- Theorem stating that for the given intervals, the next simultaneous activation occurs after 180 minutes -/
theorem next_simultaneous_activation (intervals : SignalIntervals)
  (h1 : intervals.fire = 12)
  (h2 : intervals.police = 18)
  (h3 : intervals.hospital = 30) :
  timeUntilNextSimultaneous intervals = 180 := by
  sorry

#eval timeUntilNextSimultaneous ⟨12, 18, 30⟩

end NUMINAMATH_CALUDE_next_simultaneous_activation_l1934_193401


namespace NUMINAMATH_CALUDE_max_gcd_sum_1085_l1934_193484

theorem max_gcd_sum_1085 :
  ∃ (m n : ℕ+), m + n = 1085 ∧ 
  ∀ (a b : ℕ+), a + b = 1085 → Nat.gcd a b ≤ Nat.gcd m n ∧
  Nat.gcd m n = 217 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1085_l1934_193484


namespace NUMINAMATH_CALUDE_grocer_coffee_percentage_l1934_193486

/-- Calculates the percentage of decaffeinated coffee in a grocer's stock -/
theorem grocer_coffee_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60)
  : (initial_stock * initial_decaf_percent / 100 + additional_stock * additional_decaf_percent / 100) /
    (initial_stock + additional_stock) * 100 = 28 := by
  sorry

end NUMINAMATH_CALUDE_grocer_coffee_percentage_l1934_193486


namespace NUMINAMATH_CALUDE_total_flowers_is_105_l1934_193492

/-- The total number of hibiscus, chrysanthemums, and dandelions -/
def total_flowers (h c d : ℕ) : ℕ := h + c + d

/-- Theorem: The total number of flowers is 105 -/
theorem total_flowers_is_105 
  (h : ℕ) 
  (c : ℕ) 
  (d : ℕ) 
  (h_count : h = 34)
  (h_vs_c : h = c - 13)
  (c_vs_d : c = d + 23) : 
  total_flowers h c d = 105 := by
  sorry

#check total_flowers_is_105

end NUMINAMATH_CALUDE_total_flowers_is_105_l1934_193492


namespace NUMINAMATH_CALUDE_vacation_book_pairs_l1934_193456

/-- The number of ways to choose two books of different genres -/
def different_genre_pairs (mystery fantasy biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem stating that choosing two books of different genres from the given collection results in 33 pairs -/
theorem vacation_book_pairs :
  different_genre_pairs 3 4 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_vacation_book_pairs_l1934_193456


namespace NUMINAMATH_CALUDE_remainder_of_2745_base12_div_5_l1934_193467

/-- Converts a base 12 number to base 10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base 12 representation of 2745 --/
def number_base12 : List Nat := [2, 7, 4, 5]

theorem remainder_of_2745_base12_div_5 :
  (base12ToBase10 number_base12) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2745_base12_div_5_l1934_193467


namespace NUMINAMATH_CALUDE_cookingAndYogaCount_l1934_193475

/-- Represents a group of people participating in various curriculums -/
structure CurriculumGroup where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- The number of people who study both cooking and yoga -/
def bothCookingAndYoga (g : CurriculumGroup) : ℕ :=
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums

/-- Theorem stating the number of people who study both cooking and yoga -/
theorem cookingAndYogaCount (g : CurriculumGroup) 
  (h1 : g.yoga = 35)
  (h2 : g.cooking = 20)
  (h3 : g.weaving = 15)
  (h4 : g.cookingOnly = 7)
  (h5 : g.allCurriculums = 3)
  (h6 : g.cookingAndWeaving = 5) :
  bothCookingAndYoga g = 5 := by
  sorry

#eval bothCookingAndYoga { yoga := 35, cooking := 20, weaving := 15, cookingOnly := 7, allCurriculums := 3, cookingAndWeaving := 5 }

end NUMINAMATH_CALUDE_cookingAndYogaCount_l1934_193475


namespace NUMINAMATH_CALUDE_equation_solution_l1934_193471

theorem equation_solution : ∃ x : ℚ, 2 * x - 5 = 10 + 4 * x ∧ x = -7.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1934_193471


namespace NUMINAMATH_CALUDE_odd_number_probability_l1934_193414

def roll_count : ℕ := 10
def success_count : ℕ := 7
def die_sides : ℕ := 6

theorem odd_number_probability :
  (Nat.choose roll_count success_count * (3 / die_sides) ^ success_count * (3 / die_sides) ^ (roll_count - success_count) : ℚ) = 15 / 128 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_probability_l1934_193414


namespace NUMINAMATH_CALUDE_probability_divisor_of_8_l1934_193403

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The set of divisors of 8 -/
def divisors_of_8 : Finset ℕ := {1, 2, 4, 8}

/-- The probability of an event occurring when rolling a fair die -/
def probability (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  (event ∩ die).card / die.card

theorem probability_divisor_of_8 :
  probability divisors_of_8 fair_8_sided_die = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisor_of_8_l1934_193403


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1934_193441

theorem inequality_and_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ x : ℝ, 0 < x → x < 1 → (1 - x)^2 / x + x^2 / (1 - x) ≥ 1) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ (1 - x)^2 / x + x^2 / (1 - x) = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1934_193441


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1934_193427

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : a + b + c = 6)
  (h2 : 2 * (a * b + b * c + a * c) = 24) :
  Real.sqrt (a^2 + b^2 + c^2) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1934_193427


namespace NUMINAMATH_CALUDE_min_value_theorem_l1934_193447

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 2) :
  (2 / a + 4 / b) ≥ 14 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 2 ∧ 2 / a₀ + 4 / b₀ = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1934_193447


namespace NUMINAMATH_CALUDE_fraction_simplest_form_l1934_193482

theorem fraction_simplest_form (x y : ℝ) : 
  ¬∃ (a b : ℝ), (x - y) / (x^2 + y^2) = a / b ∧ (a ≠ x - y ∨ b ≠ x^2 + y^2) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplest_form_l1934_193482


namespace NUMINAMATH_CALUDE_abc_inequality_l1934_193442

theorem abc_inequality : ∀ (a b c : ℕ),
  a = 20^22 → b = 21^21 → c = 22^20 → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1934_193442


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1934_193437

/-- If three consecutive integers have a product of 504, their sum is 24. -/
theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 504) → (a + b + c = 24) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1934_193437


namespace NUMINAMATH_CALUDE_total_books_is_283_l1934_193491

/-- The number of books borrowed on a given day -/
def books_borrowed (day : Nat) : Nat :=
  match day with
  | 1 => 40  -- Monday
  | 2 => 42  -- Tuesday
  | 3 => 44  -- Wednesday
  | 4 => 46  -- Thursday
  | 5 => 64  -- Friday
  | _ => 0   -- Weekend (handled separately)

/-- The total number of books borrowed during weekdays -/
def weekday_total : Nat :=
  (List.range 5).map books_borrowed |>.sum

/-- The number of books borrowed during the weekend -/
def weekend_books : Nat :=
  (weekday_total / 10) * 2

/-- The total number of books borrowed over the week -/
def total_books : Nat :=
  weekday_total + weekend_books

theorem total_books_is_283 : total_books = 283 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_283_l1934_193491


namespace NUMINAMATH_CALUDE_prime_with_integer_roots_range_l1934_193425

theorem prime_with_integer_roots_range (p : ℕ) : 
  Nat.Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 500*p = 0 ∧ y^2 + p*y - 500*p = 0) → 
  1 < p ∧ p ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_prime_with_integer_roots_range_l1934_193425


namespace NUMINAMATH_CALUDE_mean_median_difference_l1934_193474

def class_size : ℕ := 40

def score_distribution : List (ℕ × ℚ) := [
  (60, 15/100),
  (75, 35/100),
  (82, 10/100),
  (88, 20/100),
  (92, 20/100)
]

def mean_score : ℚ :=
  (score_distribution.map (λ (score, percentage) => score * (percentage * class_size))).sum / class_size

def median_score : ℕ := 75

theorem mean_median_difference : 
  ⌊mean_score - median_score⌋ = 4 :=
sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1934_193474


namespace NUMINAMATH_CALUDE_count_integers_in_range_l1934_193494

theorem count_integers_in_range : 
  (Finset.filter (fun x => 30 < x^2 + 8*x + 16 ∧ x^2 + 8*x + 16 < 60) (Finset.range 100)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l1934_193494


namespace NUMINAMATH_CALUDE_circle_equation_l1934_193438

/-- A circle C with center on the line x+y=0 and tangent to lines x-y=0 and x-y-4=0 -/
structure TangentCircle where
  /-- The x-coordinate of the circle's center -/
  a : ℝ
  /-- The circle's center is on the line x+y=0 -/
  center_on_line : a + (-a) = 0
  /-- The circle is tangent to the line x-y=0 -/
  tangent_to_line1 : ∃ (x y : ℝ), x - y = 0 ∧ (x - a)^2 + (y + a)^2 = (x - (x - 1))^2 + (y - (y + 1))^2
  /-- The circle is tangent to the line x-y-4=0 -/
  tangent_to_line2 : ∃ (x y : ℝ), x - y - 4 = 0 ∧ (x - a)^2 + (y + a)^2 = (x - (x - 1))^2 + (y - (y + 1))^2

/-- The equation of the circle C is (x-1)^2+(y+1)^2=2 -/
theorem circle_equation (c : TangentCircle) : ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 2 ↔ (x - c.a)^2 + (y + c.a)^2 = (1 - c.a)^2 + (1 + c.a)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1934_193438


namespace NUMINAMATH_CALUDE_range_of_trig_function_l1934_193477

theorem range_of_trig_function :
  ∀ x : ℝ, -4 * Real.sqrt 3 / 9 ≤ 2 * Real.sin x ^ 2 * Real.cos x ∧
           2 * Real.sin x ^ 2 * Real.cos x ≤ 4 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_trig_function_l1934_193477


namespace NUMINAMATH_CALUDE_largest_common_divisor_408_340_l1934_193400

theorem largest_common_divisor_408_340 : ∃ (n : ℕ), n = 68 ∧ 
  n ∣ 408 ∧ n ∣ 340 ∧ ∀ (m : ℕ), m ∣ 408 ∧ m ∣ 340 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_408_340_l1934_193400


namespace NUMINAMATH_CALUDE_dragon_boat_purchase_equations_l1934_193444

/-- Represents the purchase of items during the Dragon Boat Festival -/
structure DragonBoatPurchase where
  lotus_pouches : ℕ
  color_ropes : ℕ
  total_items : ℕ
  total_cost : ℕ
  lotus_price : ℕ
  rope_price : ℕ

/-- Theorem stating that the given system of equations correctly represents the purchase -/
theorem dragon_boat_purchase_equations (p : DragonBoatPurchase)
  (h1 : p.total_items = 20)
  (h2 : p.total_cost = 72)
  (h3 : p.lotus_price = 4)
  (h4 : p.rope_price = 3) :
  p.lotus_pouches + p.color_ropes = p.total_items ∧
  p.lotus_price * p.lotus_pouches + p.rope_price * p.color_ropes = p.total_cost :=
by sorry

end NUMINAMATH_CALUDE_dragon_boat_purchase_equations_l1934_193444


namespace NUMINAMATH_CALUDE_square_plus_one_l1934_193424

theorem square_plus_one (a : ℝ) (h : a^2 + 2*a - 2 = 0) : (a + 1)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_l1934_193424


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l1934_193445

/-- Given a circle and a moving chord, prove the trajectory of the chord's midpoint -/
theorem midpoint_trajectory (x y : ℝ) :
  (∃ (a b : ℝ), a^2 + b^2 = 25 ∧ (x - a)^2 + (y - b)^2 = 4) →
  x^2 + y^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l1934_193445


namespace NUMINAMATH_CALUDE_samantha_paint_cans_l1934_193495

/-- Represents the number of classrooms that can be painted with a given number of paint cans -/
def classrooms_paintable (cans : ℕ) : ℕ := sorry

theorem samantha_paint_cans : 
  -- Initial condition: Samantha had enough paint for 50 classrooms
  classrooms_paintable 21 = 50 ∧ 
  -- After losing 5 cans, she had enough for 38 classrooms
  classrooms_paintable (21 - 5) = 38 → 
  -- Therefore, she originally had 21 cans
  21 = 21 :=
sorry

end NUMINAMATH_CALUDE_samantha_paint_cans_l1934_193495


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l1934_193481

/-- Given that M(4,4) is the midpoint of AB and A has coordinates (8,4),
    prove that the sum of the coordinates of B is 4. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (4, 4) →
  A = (8, 4) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B.1 + B.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_l1934_193481


namespace NUMINAMATH_CALUDE_triangle_area_l1934_193411

def a : Fin 2 → ℝ := ![4, -3]
def b : Fin 2 → ℝ := ![6, 1]

theorem triangle_area : 
  (1/2 : ℝ) * |a 0 * b 1 - a 1 * b 0| = 11 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1934_193411


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l1934_193431

/-- Represents the age ratio problem between a father and his son Ronit -/
theorem father_son_age_ratio :
  ∀ (ronit_age : ℕ) (father_age : ℕ),
  father_age = 4 * ronit_age →
  father_age + 8 = (5/2) * (ronit_age + 8) →
  (father_age + 16) = 2 * (ronit_age + 16) :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l1934_193431


namespace NUMINAMATH_CALUDE_min_distance_tangent_line_l1934_193470

/-- Circle M -/
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- Line l -/
def line_l (x y : ℝ) : Prop := 2*x + y + 2 = 0

/-- Point P on line l -/
def point_P (x y : ℝ) : Prop := line_l x y

/-- Tangent points A and B on circle M -/
def tangent_points (xA yA xB yB : ℝ) : Prop := 
  circle_M xA yA ∧ circle_M xB yB

/-- Line AB -/
def line_AB (x y : ℝ) : Prop := 2*x + y + 1 = 0

theorem min_distance_tangent_line : 
  ∃ (xP yP xA yA xB yB : ℝ),
    point_P xP yP ∧
    tangent_points xA yA xB yB ∧
    (∀ (x'P y'P x'A y'A x'B y'B : ℝ),
      point_P x'P y'P → 
      tangent_points x'A y'A x'B y'B →
      (xP - 1)^2 + (yP - 1)^2 ≤ (x'P - 1)^2 + (y'P - 1)^2) →
    line_AB xA yA ∧ line_AB xB yB :=
sorry

end NUMINAMATH_CALUDE_min_distance_tangent_line_l1934_193470


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1934_193449

/-- Prove that an arithmetic sequence starting with 13, ending with 73, 
    and having a common difference of 3 has 21 terms. -/
theorem arithmetic_sequence_terms (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 13 → aₙ = 73 → d = 3 → 
  aₙ = a₁ + (n - 1) * d → n = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1934_193449


namespace NUMINAMATH_CALUDE_max_value_condition_l1934_193455

noncomputable def f (x a : ℝ) : ℝ := -(Real.sin x + a/2)^2 + 3 + a^2/4

theorem max_value_condition (a : ℝ) :
  (∀ x, f x a ≤ 5) ∧ (∃ x, f x a = 5) ↔ a = 3 ∨ a = -3 := by sorry

end NUMINAMATH_CALUDE_max_value_condition_l1934_193455


namespace NUMINAMATH_CALUDE_min_value_theorem_l1934_193404

def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r q : ℝ, r > 0 ∧ q > 1 ∧ ∀ n : ℕ, a (n + 1) - a n = r * q^n

theorem min_value_theorem (a : ℕ → ℝ) (h1 : is_arithmetic_geometric a)
  (h2 : a 9 = a 8 + 2 * a 7) (p q : ℕ) (h3 : a p * a q = 8 * (a 1)^2) :
  (1 : ℝ) / p + 4 / q ≥ 9 / 5 ∧ ∃ p₀ q₀ : ℕ, (1 : ℝ) / p₀ + 4 / q₀ = 9 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1934_193404


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1934_193461

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1934_193461


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1934_193469

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 3 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 1 →
  x ≥ 8 + Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1934_193469


namespace NUMINAMATH_CALUDE_find_number_l1934_193420

theorem find_number : ∃ x : ℚ, (4 * x) / 7 + 12 = 36 ∧ x = 42 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1934_193420


namespace NUMINAMATH_CALUDE_milkman_profit_is_51_l1934_193493

/-- Represents the milkman's problem --/
structure MilkmanProblem where
  total_milk : ℝ
  total_water : ℝ
  first_mixture_milk : ℝ
  first_mixture_water : ℝ
  second_mixture_water : ℝ
  pure_milk_cost : ℝ
  first_mixture_price : ℝ
  second_mixture_price : ℝ

/-- Calculate the total profit for the milkman --/
def calculate_profit (p : MilkmanProblem) : ℝ :=
  let second_mixture_milk := p.total_milk - p.first_mixture_milk
  let first_mixture_volume := p.first_mixture_milk + p.first_mixture_water
  let second_mixture_volume := second_mixture_milk + p.second_mixture_water
  let total_cost := p.pure_milk_cost * p.total_milk
  let total_revenue := p.first_mixture_price * first_mixture_volume + 
                       p.second_mixture_price * second_mixture_volume
  total_revenue - total_cost

/-- Theorem stating that the milkman's profit is 51 --/
theorem milkman_profit_is_51 : 
  let p : MilkmanProblem := {
    total_milk := 50,
    total_water := 15,
    first_mixture_milk := 30,
    first_mixture_water := 8,
    second_mixture_water := 7,
    pure_milk_cost := 20,
    first_mixture_price := 17,
    second_mixture_price := 15
  }
  calculate_profit p = 51 := by sorry


end NUMINAMATH_CALUDE_milkman_profit_is_51_l1934_193493


namespace NUMINAMATH_CALUDE_searchlight_probability_l1934_193488

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 2

/-- The number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

/-- The number of degrees in a full circle -/
def degrees_in_circle : ℝ := 360

/-- The minimum number of seconds the man should stay in the dark -/
def min_dark_seconds : ℝ := 5

/-- The probability of a man staying in the dark for at least 5 seconds
    when a searchlight makes 2 revolutions per minute -/
theorem searchlight_probability : 
  (degrees_in_circle - (min_dark_seconds / (seconds_per_minute / revolutions_per_minute)) * degrees_in_circle) / degrees_in_circle = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_searchlight_probability_l1934_193488


namespace NUMINAMATH_CALUDE_expression_percentage_of_y_l1934_193466

theorem expression_percentage_of_y (y : ℝ) (z : ℂ) (h : y > 0) :
  ((6 * y + 3 * z * Complex.I) / 20 + (3 * y + 4 * z * Complex.I) / 10) / y = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_expression_percentage_of_y_l1934_193466


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1934_193479

theorem cricket_team_average_age :
  let team_size : ℕ := 11
  let captain_age : ℕ := 26
  let wicket_keeper_age : ℕ := captain_age + 3
  let average_age : ℚ := (team_size : ℚ)⁻¹ * (captain_age + wicket_keeper_age + (team_size - 2) * (average_age - 1))
  average_age = 23 := by sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l1934_193479


namespace NUMINAMATH_CALUDE_trig_sum_equals_one_l1934_193480

theorem trig_sum_equals_one : 4 * Real.cos (Real.pi / 3) + 8 * Real.sin (Real.pi / 6) - 5 * Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_one_l1934_193480


namespace NUMINAMATH_CALUDE_jake_arrives_later_l1934_193473

/-- Represents the building with elevators and stairs --/
structure Building where
  floors : ℕ
  steps_per_floor : ℕ
  elevator_b_time : ℕ

/-- Represents a person descending the building --/
structure Person where
  steps_per_second : ℕ

def time_to_descend (b : Building) (p : Person) : ℕ :=
  let total_steps := b.steps_per_floor * (b.floors - 1)
  (total_steps + p.steps_per_second - 1) / p.steps_per_second

theorem jake_arrives_later (b : Building) (jake : Person) :
  b.floors = 12 →
  b.steps_per_floor = 25 →
  b.elevator_b_time = 90 →
  jake.steps_per_second = 3 →
  time_to_descend b jake - b.elevator_b_time = 2 := by
  sorry

#eval time_to_descend { floors := 12, steps_per_floor := 25, elevator_b_time := 90 } { steps_per_second := 3 }

end NUMINAMATH_CALUDE_jake_arrives_later_l1934_193473


namespace NUMINAMATH_CALUDE_bridge_length_proof_l1934_193419

/-- Given a train with length 156 meters, traveling at 45 km/h,
    that crosses a bridge in 40 seconds, prove that the bridge length is 344 meters. -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 156 →
  train_speed_kmh = 45 →
  crossing_time = 40 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 344 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l1934_193419


namespace NUMINAMATH_CALUDE_star_example_l1934_193412

def star (x y : ℝ) : ℝ := 5 * x - 2 * y

theorem star_example : (star 3 4) + (star 2 2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l1934_193412


namespace NUMINAMATH_CALUDE_pauls_recycling_bags_l1934_193472

theorem pauls_recycling_bags (x : ℕ) : 
  (∃ (bags_on_sunday : ℕ), 
    bags_on_sunday = 3 ∧ 
    (∀ (cans_per_bag : ℕ), cans_per_bag = 8 → 
      (∀ (total_cans : ℕ), total_cans = 72 → 
        cans_per_bag * (x + bags_on_sunday) = total_cans))) → 
  x = 6 := by sorry

end NUMINAMATH_CALUDE_pauls_recycling_bags_l1934_193472


namespace NUMINAMATH_CALUDE_simplify_expression_l1934_193490

theorem simplify_expression (a b : ℝ) : 2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1934_193490


namespace NUMINAMATH_CALUDE_tomatoes_left_l1934_193426

theorem tomatoes_left (initial : ℕ) (picked_yesterday : ℕ) (picked_today : ℕ) 
  (h1 : initial = 171)
  (h2 : picked_yesterday = 134)
  (h3 : picked_today = 30) : 
  initial - picked_yesterday - picked_today = 7 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l1934_193426


namespace NUMINAMATH_CALUDE_sum_is_linear_l1934_193446

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies the described transformations to a parabola -/
def transform (p : Parabola) : ℝ → ℝ := 
  fun x => p.a * (x - 4)^2 + p.b * (x - 4) + p.c + 2

/-- Applies the described transformations to the reflection of a parabola -/
def transform_reflection (p : Parabola) : ℝ → ℝ := 
  fun x => -p.a * (x + 6)^2 - p.b * (x + 6) - p.c + 2

/-- The sum of the transformed parabola and its reflection -/
def sum_of_transformations (p : Parabola) : ℝ → ℝ :=
  fun x => transform p x + transform_reflection p x

theorem sum_is_linear (p : Parabola) : 
  ∀ x, sum_of_transformations p x = -20 * p.a * x + 52 * p.a - 10 * p.b + 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_is_linear_l1934_193446


namespace NUMINAMATH_CALUDE_bret_in_seat_three_l1934_193421

-- Define the type for seats
inductive Seat
| one
| two
| three
| four

-- Define the type for people
inductive Person
| Abby
| Bret
| Carl
| Dana

-- Define the seating arrangement as a function from Seat to Person
def SeatingArrangement := Seat → Person

-- Define what it means for two people to be adjacent
def adjacent (s : SeatingArrangement) (p1 p2 : Person) : Prop :=
  ∃ (seat1 seat2 : Seat), 
    (s seat1 = p1 ∧ s seat2 = p2) ∧ 
    (seat1 = Seat.one ∧ seat2 = Seat.two ∨
     seat1 = Seat.two ∧ seat2 = Seat.three ∨
     seat1 = Seat.three ∧ seat2 = Seat.four ∨
     seat2 = Seat.one ∧ seat1 = Seat.two ∨
     seat2 = Seat.two ∧ seat1 = Seat.three ∨
     seat2 = Seat.three ∧ seat1 = Seat.four)

-- Define what it means for one person to be between two others
def between (s : SeatingArrangement) (p1 p2 p3 : Person) : Prop :=
  (s Seat.one = p1 ∧ s Seat.two = p2 ∧ s Seat.three = p3) ∨
  (s Seat.two = p1 ∧ s Seat.three = p2 ∧ s Seat.four = p3) ∨
  (s Seat.four = p1 ∧ s Seat.three = p2 ∧ s Seat.two = p3) ∨
  (s Seat.three = p1 ∧ s Seat.two = p2 ∧ s Seat.one = p3)

theorem bret_in_seat_three :
  ∀ (s : SeatingArrangement),
    (s Seat.two = Person.Abby) →
    (¬ adjacent s Person.Bret Person.Dana) →
    (¬ between s Person.Carl Person.Abby Person.Dana) →
    (s Seat.three = Person.Bret) :=
by sorry

end NUMINAMATH_CALUDE_bret_in_seat_three_l1934_193421


namespace NUMINAMATH_CALUDE_complex_square_root_l1934_193408

theorem complex_square_root (a b : ℕ+) (h : (a - b * Complex.I) ^ 2 = 8 - 6 * Complex.I) :
  a - b * Complex.I = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l1934_193408


namespace NUMINAMATH_CALUDE_bianca_birthday_money_l1934_193463

/-- The amount of money Bianca received for her birthday -/
def birthday_money (num_friends : ℕ) (dollars_per_friend : ℕ) : ℕ :=
  num_friends * dollars_per_friend

/-- Theorem stating that Bianca received 30 dollars for her birthday -/
theorem bianca_birthday_money :
  birthday_money 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bianca_birthday_money_l1934_193463


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l1934_193429

/-- Given a right triangle with consecutive whole number leg lengths and hypotenuse 29,
    prove that the sum of the leg lengths is 41. -/
theorem right_triangle_leg_sum : 
  ∃ (a b : ℕ), 
    a + 1 = b ∧                   -- legs are consecutive whole numbers
    a^2 + b^2 = 29^2 ∧            -- Pythagorean theorem for hypotenuse 29
    a + b = 41 :=                 -- sum of leg lengths is 41
by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l1934_193429


namespace NUMINAMATH_CALUDE_all_statements_correct_l1934_193413

/-- The volume of a rectangle with sides a and b, considered as a 3D object of unit height -/
def volume (a b : ℝ) : ℝ := a * b

theorem all_statements_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (volume (2 * a) b = 2 * volume a b) ∧
  (volume a (3 * b) = 3 * volume a b) ∧
  (volume (2 * a) (3 * b) = 6 * volume a b) ∧
  (volume (a / 2) (2 * b) = volume a b) ∧
  (volume (3 * a) (b / 2) = (3 / 2) * volume a b) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_correct_l1934_193413


namespace NUMINAMATH_CALUDE_vacation_cost_equalization_l1934_193418

theorem vacation_cost_equalization (X Y : ℝ) (h1 : X > Y) (h2 : X > 0) (h3 : Y > 0) :
  (X - Y) / 2 = (X + Y) / 2 - Y := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_equalization_l1934_193418


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1934_193451

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Generates the nth element of a systematic sample -/
def SystematicSample.nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  s.start + (n - 1) * s.interval

/-- Theorem: In a systematic sample of size 4 from a population of 50,
    if students with ID numbers 6, 30, and 42 are included,
    then the fourth student in the sample must have ID number 18 -/
theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 50)
  (h_sample : s.sample_size = 4)
  (h_start : s.start = 6)
  (h_interval : s.interval = 12)
  (h_30 : s.nth_element 3 = 30)
  (h_42 : s.nth_element 4 = 42) :
  s.nth_element 2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1934_193451


namespace NUMINAMATH_CALUDE_nabla_problem_l1934_193417

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + b^(2*a)

-- Theorem statement
theorem nabla_problem : nabla (nabla 2 1) 2 = 259 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l1934_193417


namespace NUMINAMATH_CALUDE_cube_sum_not_2016_l1934_193422

theorem cube_sum_not_2016 (a b : ℤ) : a^3 + 5*b^3 ≠ 2016 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_not_2016_l1934_193422


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1934_193496

theorem greatest_two_digit_multiple_of_17 :
  ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ 17 ∣ m → m ≤ n) ∧
  17 ∣ n ∧ 10 ≤ n ∧ n ≤ 99 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1934_193496


namespace NUMINAMATH_CALUDE_problem_solution_l1934_193489

def p (a : ℝ) : Prop := (1 + a)^2 + (1 - a)^2 < 4

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 ≥ 0

theorem problem_solution (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) ↔ a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1934_193489


namespace NUMINAMATH_CALUDE_tetrahedron_medians_intersect_l1934_193407

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A₁ : Point3D
  A₂ : Point3D
  A₃ : Point3D
  A₄ : Point3D

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Median of a tetrahedron -/
def median (t : Tetrahedron) (v : Fin 4) : Line3D :=
  sorry  -- Definition of median based on tetrahedron and vertex index

/-- Intersection point of two lines -/
def intersectionPoint (l1 l2 : Line3D) : Option Point3D :=
  sorry  -- Definition of intersection point of two lines

/-- Theorem: All medians of a tetrahedron intersect at a single point -/
theorem tetrahedron_medians_intersect (t : Tetrahedron) :
  ∃ (c : Point3D), ∀ (i j : Fin 4),
    intersectionPoint (median t i) (median t j) = some c :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_medians_intersect_l1934_193407


namespace NUMINAMATH_CALUDE_probability_adjacent_knights_l1934_193499

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen for the quest -/
def chosen_knights : ℕ := 4

/-- The probability that at least two of the four chosen knights were sitting next to each other -/
def Q : ℚ := 389 / 437

/-- Theorem stating that Q is the correct probability -/
theorem probability_adjacent_knights : 
  Q = 1 - (total_knights - chosen_knights) * (total_knights - chosen_knights - 2) * 
        (total_knights - chosen_knights - 4) * (total_knights - chosen_knights - 6) / 
        ((total_knights - 1) * total_knights * (total_knights + 1) * (total_knights - chosen_knights + 3)) :=
by sorry

end NUMINAMATH_CALUDE_probability_adjacent_knights_l1934_193499


namespace NUMINAMATH_CALUDE_three_digit_equation_solutions_l1934_193415

theorem three_digit_equation_solutions :
  ∀ x y z : ℕ,
  (100 ≤ x ∧ x ≤ 999) ∧
  (100 ≤ y ∧ y ≤ 999) ∧
  (100 ≤ z ∧ z ≤ 999) ∧
  (17 * x + 15 * y - 28 * z = 61) ∧
  (19 * x - 25 * y + 12 * z = 31) →
  ((x = 265 ∧ y = 372 ∧ z = 358) ∨
   (x = 525 ∧ y = 740 ∧ z = 713)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_equation_solutions_l1934_193415


namespace NUMINAMATH_CALUDE_hannah_unique_number_l1934_193464

/-- Represents a student's counting sequence -/
structure StudentSequence where
  start : Nat
  step : Nat

/-- The set of all numbers from 1 to 1200 -/
def allNumbers : Set Nat := {n | 1 ≤ n ∧ n ≤ 1200}

/-- Generate a sequence for a student -/
def generateSequence (s : StudentSequence) : Set Nat :=
  {n ∈ allNumbers | ∃ k, n = s.start + k * s.step}

/-- Alice's sequence -/
def aliceSeq : Set Nat := allNumbers \ (generateSequence ⟨4, 4⟩)

/-- Barbara's sequence -/
def barbaraSeq : Set Nat := (allNumbers \ aliceSeq) \ (generateSequence ⟨5, 5⟩)

/-- Candice's sequence -/
def candiceSeq : Set Nat := (allNumbers \ (aliceSeq ∪ barbaraSeq)) \ (generateSequence ⟨6, 6⟩)

/-- Debbie, Eliza, and Fatima's combined sequence -/
def defSeq : Set Nat := 
  (allNumbers \ (aliceSeq ∪ barbaraSeq ∪ candiceSeq)) \ 
  (generateSequence ⟨7, 7⟩ ∪ generateSequence ⟨14, 7⟩ ∪ generateSequence ⟨21, 7⟩)

/-- George's sequence -/
def georgeSeq : Set Nat := allNumbers \ (aliceSeq ∪ barbaraSeq ∪ candiceSeq ∪ defSeq)

/-- Hannah's number -/
def hannahNumber : Nat := 1189

/-- Theorem: Hannah's number is the only number not spoken by any other student -/
theorem hannah_unique_number : 
  hannahNumber ∈ allNumbers ∧ 
  hannahNumber ∉ aliceSeq ∧ 
  hannahNumber ∉ barbaraSeq ∧ 
  hannahNumber ∉ candiceSeq ∧ 
  hannahNumber ∉ defSeq ∧ 
  hannahNumber ∉ georgeSeq ∧
  ∀ n ∈ allNumbers, n ≠ hannahNumber → 
    n ∈ aliceSeq ∨ n ∈ barbaraSeq ∨ n ∈ candiceSeq ∨ n ∈ defSeq ∨ n ∈ georgeSeq := by
  sorry

end NUMINAMATH_CALUDE_hannah_unique_number_l1934_193464


namespace NUMINAMATH_CALUDE_solution_set_l1934_193439

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f 0 = 2)
variable (h2 : ∀ x : ℝ, f x + (deriv f) x > 1)

-- Define the theorem
theorem solution_set (f : ℝ → ℝ) (h1 : f 0 = 2) (h2 : ∀ x : ℝ, f x + (deriv f) x > 1) :
  {x : ℝ | Real.exp x * f x > Real.exp x + 1} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l1934_193439


namespace NUMINAMATH_CALUDE_container_capacity_l1934_193497

theorem container_capacity : ∀ (C : ℝ), 
  (0.30 * C + 27 = 0.75 * C) → C = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l1934_193497


namespace NUMINAMATH_CALUDE_armchair_price_l1934_193405

/-- Calculates the price of each armchair in a living room set purchase --/
theorem armchair_price (sofa_price : ℕ) (num_armchairs : ℕ) (coffee_table_price : ℕ) (total_invoice : ℕ) :
  sofa_price = 1250 →
  num_armchairs = 2 →
  coffee_table_price = 330 →
  total_invoice = 2430 →
  (total_invoice - sofa_price - coffee_table_price) / num_armchairs = 425 := by
  sorry

end NUMINAMATH_CALUDE_armchair_price_l1934_193405


namespace NUMINAMATH_CALUDE_initial_state_is_losing_l1934_193416

/-- Represents a game state with two piles of matches -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat)

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (move : Nat) (fromPile : Bool) : Prop :=
  if fromPile then
    move > 0 ∧ move ≤ state.pile1 ∧ state.pile2 % move = 0
  else
    move > 0 ∧ move ≤ state.pile2 ∧ state.pile1 % move = 0

/-- Defines a losing position in the game -/
def isLosingPosition (state : GameState) : Prop :=
  ∃ (k m n : Nat),
    state.pile1 = 2^k * (2*m + 1) ∧
    state.pile2 = 2^k * (2*n + 1)

/-- The main theorem stating that the initial position (100, 252) is a losing position -/
theorem initial_state_is_losing :
  isLosingPosition (GameState.mk 100 252) :=
sorry

#check initial_state_is_losing

end NUMINAMATH_CALUDE_initial_state_is_losing_l1934_193416


namespace NUMINAMATH_CALUDE_triangle_theorem_l1934_193434

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - t.a^2 = (4 * Real.sqrt 2 / 3) * t.b * t.c)
  (h2 : 3 * t.c / t.a = Real.sqrt 2 * Real.sin t.B / Real.sin t.A)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 2) :
  Real.sin t.A = 1/3 ∧ 
  t.c = 2 * Real.sqrt 2 ∧ 
  Real.sin (2 * t.C - π/6) = (10 * Real.sqrt 6 - 23) / 54 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1934_193434


namespace NUMINAMATH_CALUDE_tricycle_count_proof_l1934_193462

/-- Represents the number of wheels on a vehicle -/
def wheels : Nat → Nat
  | 0 => 2  -- bicycle
  | 1 => 3  -- tricycle
  | 2 => 2  -- scooter
  | _ => 0  -- undefined for other values

/-- Represents the count of each type of vehicle -/
structure VehicleCounts where
  bicycles : Nat
  tricycles : Nat
  scooters : Nat

theorem tricycle_count_proof (counts : VehicleCounts) : 
  counts.bicycles + counts.tricycles + counts.scooters = 10 →
  wheels 0 * counts.bicycles + wheels 1 * counts.tricycles + wheels 2 * counts.scooters = 29 →
  counts.tricycles = 9 := by
  sorry

#check tricycle_count_proof

end NUMINAMATH_CALUDE_tricycle_count_proof_l1934_193462


namespace NUMINAMATH_CALUDE_smallest_m_inequality_l1934_193452

theorem smallest_m_inequality (a b c : ℝ) :
  ∃ (M : ℝ), M = (9 * Real.sqrt 2) / 32 ∧
  |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2 ∧
  ∀ (N : ℝ), (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N * (x^2 + y^2 + z^2)^2) → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_inequality_l1934_193452


namespace NUMINAMATH_CALUDE_product_in_fourth_quadrant_l1934_193435

def z₁ : ℂ := 3 + Complex.I
def z₂ : ℂ := 1 - Complex.I

theorem product_in_fourth_quadrant :
  let z := z₁ * z₂
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_product_in_fourth_quadrant_l1934_193435


namespace NUMINAMATH_CALUDE_angle_between_planes_exists_l1934_193430

-- Define the basic structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  normal : Point3D
  d : ℝ

-- Define the projection axis
def ProjectionAxis : Point3D → Point3D → Prop :=
  sorry

-- Define a point on the projection axis
def PointOnProjectionAxis (p : Point3D) : Prop :=
  ∃ (q : Point3D), ProjectionAxis p q

-- Define a plane passing through a point
def PlaneThroughPoint (plane : Plane) (point : Point3D) : Prop :=
  plane.normal.x * point.x + plane.normal.y * point.y + plane.normal.z * point.z + plane.d = 0

-- Define the angle between two planes
def AngleBetweenPlanes (plane1 plane2 : Plane) : ℝ :=
  sorry

-- Theorem statement
theorem angle_between_planes_exists :
  ∀ (p : Point3D) (plane1 plane2 : Plane),
    PointOnProjectionAxis p →
    PlaneThroughPoint plane1 p →
    ∃ (angle : ℝ), AngleBetweenPlanes plane1 plane2 = angle :=
  sorry

end NUMINAMATH_CALUDE_angle_between_planes_exists_l1934_193430


namespace NUMINAMATH_CALUDE_ellipse_chords_and_bisector_l1934_193440

/-- Given an ellipse x²/2 + y² = 1, this theorem proves:
    1. The trajectory of midpoint of parallel chords with slope 2
    2. The trajectory of midpoint of chord defined by line passing through A(2,1)
    3. The line passing through P(1/2, 1/2) and bisected by P -/
theorem ellipse_chords_and_bisector 
  (x y : ℝ) (h : x^2/2 + y^2 = 1) :
  (∃ t : ℝ, x + 4*y = t) ∧ 
  (∃ s : ℝ, x^2 + 2*y^2 - 2*x - 2*y = s) ∧
  (2*x + 4*y - 3 = 0 → 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      x₁^2/2 + y₁^2 = 1 ∧ 
      x₂^2/2 + y₂^2 = 1 ∧ 
      (x₁ + x₂)/2 = 1/2 ∧ 
      (y₁ + y₂)/2 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chords_and_bisector_l1934_193440


namespace NUMINAMATH_CALUDE_sugar_distribution_l1934_193432

/-- The number of sugar boxes -/
def num_boxes : ℕ := 21

/-- The weight of sugar per box in kilograms -/
def sugar_per_box : ℚ := 6

/-- The amount of sugar distributed to each neighbor in kilograms -/
def sugar_per_neighbor : ℚ := 32 / 41

/-- The maximum number of neighbors who can receive sugar -/
def max_neighbors : ℕ := 161

theorem sugar_distribution :
  ⌊(num_boxes * sugar_per_box) / sugar_per_neighbor⌋ = max_neighbors := by
  sorry

end NUMINAMATH_CALUDE_sugar_distribution_l1934_193432


namespace NUMINAMATH_CALUDE_square_area_error_l1934_193468

theorem square_area_error (side_error : Real) (area_error : Real) : 
  side_error = 0.19 → area_error = 0.4161 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l1934_193468


namespace NUMINAMATH_CALUDE_black_shirts_per_pack_l1934_193423

/-- Given:
  * 3 packs of black shirts and 3 packs of yellow shirts were bought
  * Yellow shirts come in packs of 2
  * Total number of shirts is 21
Prove that the number of black shirts in each pack is 5 -/
theorem black_shirts_per_pack (black_packs yellow_packs : ℕ) 
  (yellow_per_pack total_shirts : ℕ) (black_per_pack : ℕ) :
  black_packs = 3 →
  yellow_packs = 3 →
  yellow_per_pack = 2 →
  total_shirts = 21 →
  black_packs * black_per_pack + yellow_packs * yellow_per_pack = total_shirts →
  black_per_pack = 5 := by
  sorry

#check black_shirts_per_pack

end NUMINAMATH_CALUDE_black_shirts_per_pack_l1934_193423


namespace NUMINAMATH_CALUDE_max_students_in_class_l1934_193410

theorem max_students_in_class (x : ℕ) : 
  x > 0 ∧ 
  2 ∣ x ∧ 
  4 ∣ x ∧ 
  7 ∣ x ∧ 
  x - (x / 2 + x / 4 + x / 7) < 6 →
  x ≤ 28 :=
by sorry

end NUMINAMATH_CALUDE_max_students_in_class_l1934_193410


namespace NUMINAMATH_CALUDE_total_feet_count_l1934_193402

theorem total_feet_count (total_heads : ℕ) (hen_count : ℕ) (hen_feet cow_feet : ℕ) : 
  total_heads = 48 → 
  hen_count = 26 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  (hen_count * hen_feet) + ((total_heads - hen_count) * cow_feet) = 140 := by
sorry

end NUMINAMATH_CALUDE_total_feet_count_l1934_193402
