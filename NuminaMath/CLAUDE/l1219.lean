import Mathlib

namespace NUMINAMATH_CALUDE_apples_per_pie_l1219_121904

/-- Given a box of apples, calculate the weight of apples needed per pie -/
theorem apples_per_pie (total_weight : ℝ) (num_pies : ℕ) : 
  total_weight = 120 → num_pies = 15 → (total_weight / 2) / num_pies = 4 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l1219_121904


namespace NUMINAMATH_CALUDE_sandwich_cost_l1219_121955

theorem sandwich_cost (sandwich_cost juice_cost milk_cost : ℝ) :
  juice_cost = 2 * sandwich_cost →
  milk_cost = 0.75 * (sandwich_cost + juice_cost) →
  sandwich_cost + juice_cost + milk_cost = 21 →
  sandwich_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_sandwich_cost_l1219_121955


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l1219_121930

theorem cats_sold_during_sale
  (initial_siamese : ℕ)
  (initial_house : ℕ)
  (cats_remaining : ℕ)
  (h1 : initial_siamese = 12)
  (h2 : initial_house = 20)
  (h3 : cats_remaining = 12) :
  initial_siamese + initial_house - cats_remaining = 20 :=
by sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l1219_121930


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1219_121935

theorem nested_fraction_equality : 
  2 + 1 / (2 + 1 / (2 + 1 / 2)) = 29 / 12 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1219_121935


namespace NUMINAMATH_CALUDE_system_solutions_l1219_121902

/-- The system of equations -/
def system (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₃ + x₄ + x₅)^5 = 3*x₁ ∧
  (x₄ + x₅ + x₁)^5 = 3*x₂ ∧
  (x₅ + x₁ + x₂)^5 = 3*x₃ ∧
  (x₁ + x₂ + x₃)^5 = 3*x₄ ∧
  (x₂ + x₃ + x₄)^5 = 3*x₅

/-- The solutions to the system of equations -/
def solutions : Set (ℝ × ℝ × ℝ × ℝ × ℝ) :=
  {(0, 0, 0, 0, 0), (1/3, 1/3, 1/3, 1/3, 1/3), (-1/3, -1/3, -1/3, -1/3, -1/3)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ, system x₁ x₂ x₃ x₄ x₅ ↔ (x₁, x₂, x₃, x₄, x₅) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l1219_121902


namespace NUMINAMATH_CALUDE_typing_area_percentage_l1219_121946

/-- Calculates the percentage of a rectangular sheet used for typing, given the sheet dimensions and margins. -/
theorem typing_area_percentage (sheet_width sheet_length side_margin top_bottom_margin : ℝ) :
  sheet_width = 20 ∧ 
  sheet_length = 30 ∧ 
  side_margin = 2 ∧ 
  top_bottom_margin = 3 →
  (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin) / (sheet_width * sheet_length) * 100 = 64 := by
  sorry

#check typing_area_percentage

end NUMINAMATH_CALUDE_typing_area_percentage_l1219_121946


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1219_121912

theorem inequality_and_equality_condition 
  (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) 
  (hab : a + b = 1) : 
  (1 / (a / x + b / y) ≤ a * x + b * y) ∧ 
  (1 / (a / x + b / y) = a * x + b * y ↔ a * y = b * x) :=
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1219_121912


namespace NUMINAMATH_CALUDE_room_dimension_increase_l1219_121948

/-- Given the cost of painting a room and the cost of painting an enlarged version of the same room,
    calculate the factor by which the room's dimensions were increased. -/
theorem room_dimension_increase (original_cost enlarged_cost : ℝ) 
    (h1 : original_cost = 350)
    (h2 : enlarged_cost = 3150) :
    ∃ (n : ℝ), n = 3 ∧ enlarged_cost = n^2 * original_cost := by
  sorry

end NUMINAMATH_CALUDE_room_dimension_increase_l1219_121948


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1219_121996

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 4 * x^2 + (k - 1) * x + 9 = (a * x + b)^2) → 
  (k = 13 ∨ k = -11) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1219_121996


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1219_121914

theorem quadratic_root_relation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (2 * x₁^2 - (2*m + 1)*x₁ + m^2 - 9*m + 39 = 0) ∧ 
    (2 * x₂^2 - (2*m + 1)*x₂ + m^2 - 9*m + 39 = 0) ∧ 
    (x₂ = 2 * x₁)) → 
  (m = 10 ∨ m = 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1219_121914


namespace NUMINAMATH_CALUDE_find_number_l1219_121968

theorem find_number : ∃ x : ℤ, x - 263 + 419 = 725 ∧ x = 569 := by sorry

end NUMINAMATH_CALUDE_find_number_l1219_121968


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1219_121901

theorem inequality_solution_range (b : ℝ) : 
  (∃ x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
   (∀ z : ℤ, z < 0 → (z - b > 0 ↔ z = x ∨ z = y))) → 
  -3 ≤ b ∧ b < -2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1219_121901


namespace NUMINAMATH_CALUDE_machine_profit_percentage_l1219_121937

/-- Calculates the profit percentage given the purchase price, repair cost, transportation charges, and selling price of a machine. -/
def profit_percentage (purchase_price repair_cost transport_charges selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost + transport_charges
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percentage for the given machine transaction is 50%. -/
theorem machine_profit_percentage :
  profit_percentage 13000 5000 1000 28500 = 50 := by
  sorry

end NUMINAMATH_CALUDE_machine_profit_percentage_l1219_121937


namespace NUMINAMATH_CALUDE_sqrt_negative_a_squared_plus_one_undefined_l1219_121960

theorem sqrt_negative_a_squared_plus_one_undefined (a : ℝ) : ¬ ∃ (x : ℝ), x^2 = -a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_negative_a_squared_plus_one_undefined_l1219_121960


namespace NUMINAMATH_CALUDE_movie_ticket_change_l1219_121919

/-- Calculates the change received by two sisters buying movie tickets -/
theorem movie_ticket_change (full_price : ℚ) (discount_percent : ℚ) (brought_money : ℚ) : 
  full_price = 8 →
  discount_percent = 25 / 100 →
  brought_money = 25 →
  let discounted_price := full_price * (1 - discount_percent)
  let total_cost := full_price + discounted_price
  brought_money - total_cost = 11 := by
sorry

end NUMINAMATH_CALUDE_movie_ticket_change_l1219_121919


namespace NUMINAMATH_CALUDE_percentage_and_reduction_l1219_121927

-- Define the relationship between two numbers
def is_five_percent_more (a b : ℝ) : Prop := a = b * 1.05

-- Define the reduction of 10 kilograms by 10%
def reduced_by_ten_percent (x : ℝ) : ℝ := x * 0.9

theorem percentage_and_reduction :
  (∀ a b : ℝ, is_five_percent_more a b → a = b * 1.05) ∧
  (reduced_by_ten_percent 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_percentage_and_reduction_l1219_121927


namespace NUMINAMATH_CALUDE_smallest_a_in_special_progression_l1219_121979

theorem smallest_a_in_special_progression (a b c : ℤ) : 
  a < b → b < c → 
  (2 * b = a + c) →  -- arithmetic progression condition
  (c * c = a * b) →  -- geometric progression condition
  (∀ a' b' c' : ℤ, a' < b' → b' < c' → (2 * b' = a' + c') → (c' * c' = a' * b') → a ≤ a') →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_in_special_progression_l1219_121979


namespace NUMINAMATH_CALUDE_count_negative_numbers_l1219_121995

theorem count_negative_numbers : ∃ (S : Finset ℝ), 
  S = {8, 0, |(-2)|, -5, -2/3, (-1)^2} ∧ 
  (S.filter (λ x => x < 0)).card = 2 := by
sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l1219_121995


namespace NUMINAMATH_CALUDE_tuesday_distance_l1219_121918

/-- Proves that the distance driven on Tuesday is 18 miles -/
theorem tuesday_distance (monday_distance : ℝ) (wednesday_distance : ℝ) (average_distance : ℝ) (num_days : ℕ) :
  monday_distance = 12 →
  wednesday_distance = 21 →
  average_distance = 17 →
  num_days = 3 →
  (monday_distance + wednesday_distance + (num_days * average_distance - monday_distance - wednesday_distance)) / num_days = average_distance →
  num_days * average_distance - monday_distance - wednesday_distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_distance_l1219_121918


namespace NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l1219_121933

theorem yellow_jelly_bean_probability :
  let red : ℕ := 4
  let green : ℕ := 8
  let yellow : ℕ := 9
  let blue : ℕ := 5
  let total : ℕ := red + green + yellow + blue
  (yellow : ℚ) / total = 9 / 26 := by sorry

end NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l1219_121933


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_ge_negation_of_quadratic_inequality_l1219_121994

theorem negation_of_existence (P : ℕ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_inequality_ge (a b : ℝ) :
  (¬ (a ≥ b)) ↔ (a < b) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℕ, x^2 + 2*x ≥ 3) ↔ (∀ x : ℕ, x^2 + 2*x < 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_ge_negation_of_quadratic_inequality_l1219_121994


namespace NUMINAMATH_CALUDE_non_negativity_and_extrema_l1219_121931

theorem non_negativity_and_extrema :
  (∀ x y : ℝ, (x - 1)^2 ≥ 0 ∧ x^2 + 1 > 0 ∧ |3*x + 2*y| ≥ 0) ∧
  (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 ∧ (∃ x₀ : ℝ, x₀^2 - 2*x₀ + 1 = 0)) ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 + x*y →
    (x - 3*y)^2 + 4*(y + x)*(x - y) ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_non_negativity_and_extrema_l1219_121931


namespace NUMINAMATH_CALUDE_albert_horses_l1219_121957

/-- Proves that Albert bought 4 horses given the conditions of the problem -/
theorem albert_horses : 
  ∀ (n : ℕ) (cow_price : ℕ),
  2000 * n + 9 * cow_price = 13400 →
  200 * n + 18 / 10 * cow_price = 1880 →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_albert_horses_l1219_121957


namespace NUMINAMATH_CALUDE_representatives_count_l1219_121920

/-- The number of ways to select 3 representatives from 4 boys and 4 girls,
    with at least two girls among them. -/
def select_representatives : ℕ :=
  Nat.choose 4 3 + Nat.choose 4 2 * Nat.choose 4 1

/-- Theorem stating that the number of ways to select the representatives is 28. -/
theorem representatives_count : select_representatives = 28 := by
  sorry

end NUMINAMATH_CALUDE_representatives_count_l1219_121920


namespace NUMINAMATH_CALUDE_am_length_l1219_121963

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
def problem_setup (c : Circle) (l1 l2 : Line) (M A B C : ℝ × ℝ) : Prop :=
  ∃ (BC BM : ℝ),
    -- l1 is tangent to c at A
    (∃ (t : ℝ), l1.point1 = c.center + t • (A - c.center) ∧ 
                l1.point2 = c.center + (t + 1) • (A - c.center) ∧
                ‖A - c.center‖ = c.radius) ∧
    -- l2 intersects c at B and C
    (∃ (t1 t2 : ℝ), l2.point1 + t1 • (l2.point2 - l2.point1) = B ∧
                    l2.point1 + t2 • (l2.point2 - l2.point1) = C ∧
                    ‖B - c.center‖ = c.radius ∧
                    ‖C - c.center‖ = c.radius) ∧
    -- BC = 7
    BC = 7 ∧
    -- BM = 9
    BM = 9

-- Theorem statement
theorem am_length (c : Circle) (l1 l2 : Line) (M A B C : ℝ × ℝ) 
  (h : problem_setup c l1 l2 M A B C) :
  ‖A - M‖ = 12 ∨ ‖A - M‖ = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_am_length_l1219_121963


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1219_121997

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (a b : Line) 
  (α β : Plane) 
  (h_diff_lines : a ≠ b) 
  (h_diff_planes : α ≠ β) 
  (h1 : perp_line a b) 
  (h2 : perp_line_plane a α) 
  (h3 : perp_line_plane b β) : 
  perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1219_121997


namespace NUMINAMATH_CALUDE_problem_solution_l1219_121975

theorem problem_solution (x y z p q r : ℝ) 
  (h1 : x * y / (x + y) = p)
  (h2 : x * z / (x + z) = q)
  (h3 : y * z / (y + z) = r)
  (h4 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h5 : x ≠ -y ∧ x ≠ -z ∧ y ≠ -z)
  (h6 : p = 3 * q)
  (h7 : p = 2 * r) :
  x = 3 * p / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1219_121975


namespace NUMINAMATH_CALUDE_intersection_trajectory_l1219_121977

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the endpoints of the major axis
def majorAxisEndpoints (A₁ A₂ : ℝ × ℝ) : Prop :=
  A₁ = (-3, 0) ∧ A₂ = (3, 0)

-- Define a chord perpendicular to the major axis
def perpendicularChord (P₁ P₂ : ℝ × ℝ) : Prop :=
  ellipse P₁.1 P₁.2 ∧ ellipse P₂.1 P₂.2 ∧ P₁.1 = P₂.1 ∧ P₁.2 = -P₂.2

-- Define the intersection point of A₁P₁ and A₂P₂
def intersectionPoint (Q : ℝ × ℝ) (A₁ A₂ P₁ P₂ : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ,
    Q = (1 - t₁) • A₁ + t₁ • P₁ ∧
    Q = (1 - t₂) • A₂ + t₂ • P₂

-- The theorem to be proved
theorem intersection_trajectory
  (A₁ A₂ P₁ P₂ Q : ℝ × ℝ)
  (h₁ : majorAxisEndpoints A₁ A₂)
  (h₂ : perpendicularChord P₁ P₂)
  (h₃ : intersectionPoint Q A₁ A₂ P₁ P₂) :
  Q.1^2 / 9 - Q.2^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_trajectory_l1219_121977


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1219_121947

def num_meats : ℕ := 10
def num_cheeses : ℕ := 12
def num_condiments : ℕ := 5

theorem sandwich_combinations :
  (num_meats) * (num_cheeses.choose 2) * (num_condiments) = 3300 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1219_121947


namespace NUMINAMATH_CALUDE_card_73_is_8_l1219_121945

def card_sequence : List String := [
  "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
  "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"
]

def cycle_length : Nat := card_sequence.length

theorem card_73_is_8 : 
  card_sequence[(73 - 1) % cycle_length] = "8" := by
  sorry

end NUMINAMATH_CALUDE_card_73_is_8_l1219_121945


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1219_121911

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 3*x - m*x + m - 1
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 3*x₁ - x₁*x₂ + 3*x₂ = 12 →
  x₁ = 0 ∧ x₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1219_121911


namespace NUMINAMATH_CALUDE_unique_solution_l1219_121932

/-- Represents the number of children in each family and the house number -/
structure FamilyData where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  N : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (fd : FamilyData) : Prop :=
  fd.a > fd.b ∧ fd.b > fd.c ∧ fd.c > fd.d ∧
  fd.a + fd.b + fd.c + fd.d < 18 ∧
  fd.a * fd.b * fd.c * fd.d = fd.N

/-- The theorem statement -/
theorem unique_solution :
  ∃! fd : FamilyData, satisfiesConditions fd ∧ fd.N = 120 ∧
    fd.a = 5 ∧ fd.b = 4 ∧ fd.c = 3 ∧ fd.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1219_121932


namespace NUMINAMATH_CALUDE_rectangular_prism_cut_out_l1219_121954

theorem rectangular_prism_cut_out (x y : ℤ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → 
  (0 < x) → 
  (x < 4) → 
  (0 < y) → 
  (y < 15) → 
  (x = 3 ∧ y = 12) := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_cut_out_l1219_121954


namespace NUMINAMATH_CALUDE_complex_point_location_l1219_121952

theorem complex_point_location (z : ℂ) (h : z = 1 + I) :
  let w := 2 / z + z^2
  0 < w.re ∧ 0 < w.im :=
by sorry

end NUMINAMATH_CALUDE_complex_point_location_l1219_121952


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l1219_121984

/-- For a rectangular plot with given conditions, prove the ratio of area to breadth -/
theorem rectangular_plot_ratio (b l : ℝ) (h1 : b = 5) (h2 : l - b = 10) : 
  (l * b) / b = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l1219_121984


namespace NUMINAMATH_CALUDE_sport_formulation_comparison_l1219_121907

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport : DrinkRatio :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

theorem sport_formulation_comparison : 
  (sport.flavoring / sport.water) / (standard.flavoring / standard.water) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_comparison_l1219_121907


namespace NUMINAMATH_CALUDE_intersection_and_complement_intersection_l1219_121900

def I : Set ℕ := Set.univ

def A : Set ℕ := {x | ∃ n : ℕ, x = 3 * n ∧ n % 2 = 0}

def B : Set ℕ := {y | 24 % y = 0}

theorem intersection_and_complement_intersection :
  (A ∩ B = {6, 12, 24}) ∧
  ((I \ A) ∩ B = {1, 2, 3, 4, 8}) := by sorry

end NUMINAMATH_CALUDE_intersection_and_complement_intersection_l1219_121900


namespace NUMINAMATH_CALUDE_programmer_is_odd_one_out_l1219_121986

-- Define the set of professions
inductive Profession
| Dentist
| ElementarySchoolTeacher
| Programmer

-- Define a predicate for having special pension benefits
def has_special_pension_benefits (p : Profession) : Prop :=
  match p with
  | Profession.Dentist => true
  | Profession.ElementarySchoolTeacher => true
  | Profession.Programmer => false

-- Define the odd one out
def is_odd_one_out (p : Profession) : Prop :=
  ¬(has_special_pension_benefits p) ∧
  ∀ q : Profession, q ≠ p → has_special_pension_benefits q

-- Theorem statement
theorem programmer_is_odd_one_out :
  is_odd_one_out Profession.Programmer :=
sorry

end NUMINAMATH_CALUDE_programmer_is_odd_one_out_l1219_121986


namespace NUMINAMATH_CALUDE_zero_point_of_f_l1219_121958

def f (x : ℝ) : ℝ := x + 1

theorem zero_point_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_of_f_l1219_121958


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1219_121951

theorem cos_alpha_value (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) :
  Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1219_121951


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l1219_121967

theorem power_tower_mod_1000 : 7^(7^(7^7)) ≡ 343 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l1219_121967


namespace NUMINAMATH_CALUDE_non_pen_pencil_sales_l1219_121916

/-- The percentage of June sales for pens -/
def pen_sales : ℝ := 42

/-- The percentage of June sales for pencils -/
def pencil_sales : ℝ := 27

/-- The total percentage of all sales -/
def total_sales : ℝ := 100

/-- Theorem: The combined percentage of June sales that were not pens or pencils is 31% -/
theorem non_pen_pencil_sales : 
  total_sales - (pen_sales + pencil_sales) = 31 := by sorry

end NUMINAMATH_CALUDE_non_pen_pencil_sales_l1219_121916


namespace NUMINAMATH_CALUDE_johns_number_is_55_l1219_121985

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_reversal (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem johns_number_is_55 :
  ∃! n : ℕ, is_three_digit n ∧
    321 ≤ digit_reversal (2 * n + 13) ∧
    digit_reversal (2 * n + 13) ≤ 325 ∧
    n = 55 :=
sorry

end NUMINAMATH_CALUDE_johns_number_is_55_l1219_121985


namespace NUMINAMATH_CALUDE_womens_average_age_l1219_121915

theorem womens_average_age 
  (n : ℕ) 
  (initial_men : ℕ) 
  (replaced_men_ages : ℕ × ℕ) 
  (age_increase : ℚ) :
  initial_men = 8 →
  replaced_men_ages = (20, 10) →
  age_increase = 2 →
  ∃ (total_age : ℚ),
    (total_age / initial_men + age_increase) * initial_men = 
      total_age - (replaced_men_ages.1 + replaced_men_ages.2) + 46 →
    46 / 2 = 23 :=
by sorry

end NUMINAMATH_CALUDE_womens_average_age_l1219_121915


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l1219_121923

/-- The minimum number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 2 →
  large_side = 16 →
  (large_side / small_side) ^ 2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l1219_121923


namespace NUMINAMATH_CALUDE_f_2008_l1219_121910

-- Define a real-valued function with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the condition f(9) = 18
axiom f_9 : f 9 = 18

-- Define the inverse relationship for f(x+1)
axiom inverse_shift : Function.LeftInverse (fun x => f⁻¹ (x + 1)) (fun x => f (x + 1))

-- State the theorem
theorem f_2008 : f 2008 = -1981 := by sorry

end NUMINAMATH_CALUDE_f_2008_l1219_121910


namespace NUMINAMATH_CALUDE_final_price_after_discounts_arun_paid_price_l1219_121917

/-- Calculates the final price of an article after applying two consecutive discounts -/
theorem final_price_after_discounts (original_price : ℝ) 
  (standard_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  let price_after_standard := original_price * (1 - standard_discount)
  let final_price := price_after_standard * (1 - additional_discount)
  final_price

/-- Proves that the final price of an article originally priced at 2000, 
    after a 30% standard discount and a 20% additional discount, is 1120 -/
theorem arun_paid_price : 
  final_price_after_discounts 2000 0.3 0.2 = 1120 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_arun_paid_price_l1219_121917


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_11_gon_l1219_121969

/-- The number of sides in our regular polygon -/
def n : ℕ := 11

/-- The total number of diagonals in an n-sided regular polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in an n-sided regular polygon -/
def shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in an n-sided regular polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  shortest_diagonals n / total_diagonals n

/-- Theorem: The probability of selecting a shortest diagonal in a regular 11-sided polygon is 1/4 -/
theorem prob_shortest_diagonal_11_gon :
  prob_shortest_diagonal n = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_11_gon_l1219_121969


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1219_121905

-- Define the right triangle
def rightTriangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Define the inscribed rectangle
def inscribedRectangle (x : ℝ) (a b c : ℝ) : Prop :=
  rightTriangle a b c ∧ x > 0 ∧ 2*x > 0 ∧ x ≤ a ∧ 2*x ≤ b

-- Theorem statement
theorem inscribed_rectangle_area (x : ℝ) :
  rightTriangle 24 (60 - 24) 60 →
  inscribedRectangle x 24 (60 - 24) 60 →
  x * (2*x) = 1440 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1219_121905


namespace NUMINAMATH_CALUDE_slope_of_right_triangle_l1219_121978

/-- Given a right triangle ABC in the x-y plane where ∠B = 90°, AC = 100, and AB = 80,
    the slope of AC is 4/3 -/
theorem slope_of_right_triangle (A B C : ℝ × ℝ) : 
  (B.2 - A.2) ^ 2 + (B.1 - A.1) ^ 2 = 80 ^ 2 →
  (C.2 - A.2) ^ 2 + (C.1 - A.1) ^ 2 = 100 ^ 2 →
  (C.2 - B.2) ^ 2 + (C.1 - B.1) ^ 2 = (C.2 - A.2) ^ 2 + (C.1 - A.1) ^ 2 - (B.2 - A.2) ^ 2 - (B.1 - A.1) ^ 2 →
  (C.2 - A.2) / (C.1 - A.1) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_right_triangle_l1219_121978


namespace NUMINAMATH_CALUDE_profit_to_cost_ratio_l1219_121983

theorem profit_to_cost_ratio (sale_price cost_price : ℚ) : 
  sale_price > 0 ∧ cost_price > 0 ∧ sale_price / cost_price = 6 / 2 → 
  (sale_price - cost_price) / cost_price = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_profit_to_cost_ratio_l1219_121983


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1219_121949

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 : ℂ) / (2 + Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1219_121949


namespace NUMINAMATH_CALUDE_distinct_permutations_eq_twelve_l1219_121906

/-- The number of distinct permutations of the multiset {2, 3, 3, 9} -/
def distinct_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

/-- Theorem stating that the number of distinct permutations of the multiset {2, 3, 3, 9} is 12 -/
theorem distinct_permutations_eq_twelve : distinct_permutations = 12 := by
  sorry

end NUMINAMATH_CALUDE_distinct_permutations_eq_twelve_l1219_121906


namespace NUMINAMATH_CALUDE_green_hat_cost_l1219_121990

/-- Proves that the cost of each green hat is $7 given the conditions of the problem -/
theorem green_hat_cost (total_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  blue_hat_cost = 6 →
  total_price = 550 →
  green_hats = 40 →
  (total_hats - green_hats) * blue_hat_cost + green_hats * 7 = total_price :=
by sorry

end NUMINAMATH_CALUDE_green_hat_cost_l1219_121990


namespace NUMINAMATH_CALUDE_frustum_volume_l1219_121921

/-- The volume of a frustum formed by cutting a triangular pyramid parallel to its base --/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ) :
  base_edge = 18 →
  altitude = 9 →
  small_base_edge = 9 →
  small_altitude = 3 →
  ∃ (v : ℝ), v = 212.625 * Real.sqrt 3 ∧ v = 
    ((1/3 * (Real.sqrt 3 / 4) * base_edge^2 * altitude) - 
     (1/3 * (Real.sqrt 3 / 4) * small_base_edge^2 * small_altitude)) :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l1219_121921


namespace NUMINAMATH_CALUDE_unique_perpendicular_projection_l1219_121973

-- Define the types for projections and points
def Projection : Type := ℝ → ℝ → ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the given projections and intersection points
variable (g' g'' d'' : Projection)
variable (A' A'' : Point)

-- Define the perpendicularity condition
def perpendicular (l1 l2 : Projection) : Prop := sorry

-- Define the intersection condition
def intersect (l1 l2 : Projection) (p : Point) : Prop := sorry

-- Theorem statement
theorem unique_perpendicular_projection :
  ∃! d' : Projection,
    intersect g' d' A' ∧
    intersect g'' d'' A'' ∧
    perpendicular g' d' ∧
    perpendicular g'' d'' :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_projection_l1219_121973


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1219_121974

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 2 = 0) → (x₂^2 + 5*x₂ - 2 = 0) → (x₁ + x₂ = -5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1219_121974


namespace NUMINAMATH_CALUDE_fraction_of_decimals_equals_300_l1219_121936

theorem fraction_of_decimals_equals_300 : (0.3 ^ 4) / (0.03 ^ 3) = 300 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_decimals_equals_300_l1219_121936


namespace NUMINAMATH_CALUDE_total_bankers_discount_l1219_121999

/-- Represents a bill with its amount, true discount, and interest rate -/
structure Bill where
  amount : ℝ
  trueDiscount : ℝ
  interestRate : ℝ

/-- Calculates the banker's discount for a given bill -/
def bankerDiscount (bill : Bill) : ℝ :=
  (bill.amount - bill.trueDiscount) * bill.interestRate

/-- The four bills given in the problem -/
def bills : List Bill := [
  { amount := 2260, trueDiscount := 360, interestRate := 0.08 },
  { amount := 3280, trueDiscount := 520, interestRate := 0.10 },
  { amount := 4510, trueDiscount := 710, interestRate := 0.12 },
  { amount := 6240, trueDiscount := 980, interestRate := 0.15 }
]

/-- Theorem: The total banker's discount for the given bills is 1673 -/
theorem total_bankers_discount :
  (bills.map bankerDiscount).sum = 1673 := by
  sorry

end NUMINAMATH_CALUDE_total_bankers_discount_l1219_121999


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l1219_121982

theorem common_root_quadratic_equations (p : ℝ) :
  (∃ x : ℝ, x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ↔
  (p = -3 ∨ p = 9) ∧
  ((p = -3 → ∃ x : ℝ, x = -1 ∧ x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ∧
   (p = 9 → ∃ x : ℝ, x = 3 ∧ x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0)) :=
by sorry

#check common_root_quadratic_equations

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l1219_121982


namespace NUMINAMATH_CALUDE_squares_different_areas_l1219_121988

-- Define what a square is
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define properties of squares
def Square.isEquiangular (s : Square) : Prop := true
def Square.isRectangle (s : Square) : Prop := true
def Square.isRegularPolygon (s : Square) : Prop := true
def Square.isSimilarTo (s1 s2 : Square) : Prop := true

-- Define the area of a square
def Square.area (s : Square) : ℝ := s.side * s.side

-- Theorem: There exist squares with different areas
theorem squares_different_areas :
  ∃ (s1 s2 : Square), 
    Square.isEquiangular s1 ∧ 
    Square.isEquiangular s2 ∧
    Square.isRectangle s1 ∧ 
    Square.isRectangle s2 ∧
    Square.isRegularPolygon s1 ∧ 
    Square.isRegularPolygon s2 ∧
    Square.isSimilarTo s1 s2 ∧
    Square.area s1 ≠ Square.area s2 :=
by
  sorry

end NUMINAMATH_CALUDE_squares_different_areas_l1219_121988


namespace NUMINAMATH_CALUDE_expression_evaluation_l1219_121939

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  3 * x^2 * y - (5 * x * y^2 + 2 * (x^2 * y - 1/2) + x^2 * y) + 6 * x * y^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1219_121939


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l1219_121966

theorem largest_stamps_per_page : Nat.gcd (Nat.gcd 1020 1275) 1350 = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l1219_121966


namespace NUMINAMATH_CALUDE_abc_value_l1219_121972

theorem abc_value (a b c : ℂ) 
  (eq1 : a * b + 4 * b = -16)
  (eq2 : b * c + 4 * c = -16)
  (eq3 : c * a + 4 * a = -16) :
  a * b * c = 64 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1219_121972


namespace NUMINAMATH_CALUDE_min_sum_of_coefficients_l1219_121991

theorem min_sum_of_coefficients (a b : ℕ+) (h : 2 * a * 2 + b * 1 = 13) : 
  ∃ (m n : ℕ+), 2 * m * 2 + n * 1 = 13 ∧ m + n ≤ a + b ∧ m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_coefficients_l1219_121991


namespace NUMINAMATH_CALUDE_jake_has_seven_peaches_l1219_121970

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 19
def steven_apples : ℕ := 14

-- Define Jake's peaches and apples in relation to Steven's
def jake_peaches : ℕ := steven_peaches - 12
def jake_apples : ℕ := steven_apples + 79

-- Theorem to prove
theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_seven_peaches_l1219_121970


namespace NUMINAMATH_CALUDE_largest_squared_fraction_l1219_121998

theorem largest_squared_fraction : 
  let a := (8/9 : ℚ)^2
  let b := (2/3 : ℚ)^2
  let c := (3/4 : ℚ)^2
  let d := (5/8 : ℚ)^2
  let e := (7/12 : ℚ)^2
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end NUMINAMATH_CALUDE_largest_squared_fraction_l1219_121998


namespace NUMINAMATH_CALUDE_g_expression_l1219_121924

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g using the given condition
def g (x : ℝ) : ℝ := f (x - 3)

-- Theorem statement
theorem g_expression : ∀ x : ℝ, g x = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l1219_121924


namespace NUMINAMATH_CALUDE_log_inequality_conditions_l1219_121944

/-- The set of positive real numbers excluding 1 -/
def S : Set ℝ := {x : ℝ | x > 0 ∧ x ≠ 1}

/-- The theorem stating the conditions for the logarithmic inequality -/
theorem log_inequality_conditions (a b : ℝ) :
  a > 0 → b > 0 → a ≠ 1 →
  (Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)) ↔
  ((b = 1 ∧ a ∈ S) ∨
   (a > b ∧ b > 1) ∨
   (b > 1 ∧ 1 > a) ∨
   (a < b ∧ b < 1) ∨
   (b < 1 ∧ 1 < a)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_conditions_l1219_121944


namespace NUMINAMATH_CALUDE_lemonade_stand_problem_l1219_121929

/-- Represents the lemonade stand problem -/
theorem lemonade_stand_problem 
  (total_days : ℕ) 
  (hot_days : ℕ) 
  (cups_per_day : ℕ) 
  (total_profit : ℚ) 
  (cost_per_cup : ℚ) 
  (hot_day_price_increase : ℚ) :
  total_days = 10 →
  hot_days = 4 →
  cups_per_day = 32 →
  total_profit = 350 →
  cost_per_cup = 3/4 →
  hot_day_price_increase = 1/4 →
  ∃ (regular_price : ℚ),
    regular_price > 0 ∧
    (total_days - hot_days) * cups_per_day * regular_price +
    hot_days * cups_per_day * (regular_price * (1 + hot_day_price_increase)) -
    total_days * cups_per_day * cost_per_cup = total_profit ∧
    regular_price * (1 + hot_day_price_increase) = 15/8 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_stand_problem_l1219_121929


namespace NUMINAMATH_CALUDE_system_and_expression_proof_l1219_121934

theorem system_and_expression_proof :
  -- Part 1: System of equations
  (∃ x y : ℚ, 2 * x - y = -4 ∧ 4 * x - 5 * y = -23 ∧ x = 1/2 ∧ y = 5) ∧
  -- Part 2: Expression evaluation
  (let x : ℚ := 2
   let y : ℚ := -1
   (x - 3 * y)^2 - (2 * x + y) * (y - 2 * x) = 40) := by
sorry

end NUMINAMATH_CALUDE_system_and_expression_proof_l1219_121934


namespace NUMINAMATH_CALUDE_enclosed_area_theorem_l1219_121953

/-- The common area enclosed by 4 equilateral triangles with side length 1, 
    each sharing a side with one of the 4 sides of a unit square. -/
def commonAreaEnclosedByTriangles : ℝ := -1

/-- The side length of the square -/
def squareSideLength : ℝ := 1

/-- The side length of each equilateral triangle -/
def triangleSideLength : ℝ := 1

/-- The number of equilateral triangles -/
def numberOfTriangles : ℕ := 4

theorem enclosed_area_theorem :
  commonAreaEnclosedByTriangles = -1 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_theorem_l1219_121953


namespace NUMINAMATH_CALUDE_negation_of_implication_l1219_121926

theorem negation_of_implication (x : ℝ) : 
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1219_121926


namespace NUMINAMATH_CALUDE_tv_screen_diagonal_l1219_121913

theorem tv_screen_diagonal (d : ℝ) : d > 0 → d^2 = 17^2 + 76 → d = Real.sqrt 365 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_diagonal_l1219_121913


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1219_121987

/-- Given a complex number z satisfying (2+i)z = 1+3i, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (z : ℂ) (h : (2 + Complex.I) * z = 1 + 3 * Complex.I) :
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1219_121987


namespace NUMINAMATH_CALUDE_equation_proof_l1219_121981

theorem equation_proof : Real.sqrt (5 + Real.sqrt (3 + Real.sqrt 14)) = (2 + Real.sqrt 14) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1219_121981


namespace NUMINAMATH_CALUDE_water_force_on_trapezoidal_dam_water_force_on_trapezoidal_dam_proof_l1219_121940

/-- The force exerted by water on a dam with an isosceles trapezoidal cross-section --/
theorem water_force_on_trapezoidal_dam
  (ρ : Real) -- density of water
  (g : Real) -- acceleration due to gravity
  (a : Real) -- top base of trapezoid
  (b : Real) -- bottom base of trapezoid
  (h : Real) -- height of trapezoid
  (hρ : ρ = 1000) -- density of water in kg/m³
  (hg : g = 10) -- acceleration due to gravity in m/s²
  (ha : a = 6.3) -- top base in meters
  (hb : b = 10.2) -- bottom base in meters
  (hh : h = 4.0) -- height in meters
  : Real :=
  -- The force F in Newtons
  608000

/-- Proof of the theorem --/
theorem water_force_on_trapezoidal_dam_proof
  (ρ g a b h : Real)
  (hρ : ρ = 1000)
  (hg : g = 10)
  (ha : a = 6.3)
  (hb : b = 10.2)
  (hh : h = 4.0)
  : water_force_on_trapezoidal_dam ρ g a b h hρ hg ha hb hh = 608000 := by
  sorry

end NUMINAMATH_CALUDE_water_force_on_trapezoidal_dam_water_force_on_trapezoidal_dam_proof_l1219_121940


namespace NUMINAMATH_CALUDE_min_value_perpendicular_vectors_l1219_121993

theorem min_value_perpendicular_vectors (x y : ℝ) :
  let m : ℝ × ℝ := (x - 1, 1)
  let n : ℝ × ℝ := (1, y)
  (m.1 * n.1 + m.2 * n.2 = 0) →
  (∀ a b : ℝ, m = (a - 1, 1) ∧ n = (1, b) → 2^a + 2^b ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, m = (a - 1, 1) ∧ n = (1, b) ∧ 2^a + 2^b = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_perpendicular_vectors_l1219_121993


namespace NUMINAMATH_CALUDE_intersection_M_N_l1219_121980

-- Define set M
def M : Set ℝ := {x | x < 2016}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.log (x - x^2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1219_121980


namespace NUMINAMATH_CALUDE_hexagon_areas_equal_l1219_121950

/-- Given a triangle T with sides of lengths r, g, and b, and area S,
    the area of both hexagons formed by extending the sides of T is equal to
    S * (4 + ((r^2 + g^2 + b^2)(r + g + b)) / (r * g * b)) -/
theorem hexagon_areas_equal (r g b S : ℝ) (hr : r > 0) (hg : g > 0) (hb : b > 0) (hS : S > 0) :
  let hexagon_area := S * (4 + ((r^2 + g^2 + b^2) * (r + g + b)) / (r * g * b))
  ∀ (area1 area2 : ℝ), area1 = hexagon_area ∧ area2 = hexagon_area → area1 = area2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_areas_equal_l1219_121950


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1219_121992

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose n k

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 70

/-- The probability of a triangle having at least one side that is a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1219_121992


namespace NUMINAMATH_CALUDE_parabola_decreasing_implies_m_bound_l1219_121941

/-- If the function y = -x^2 - 4mx + 1 is decreasing on the interval [2, +∞), then m ≥ -1. -/
theorem parabola_decreasing_implies_m_bound (m : ℝ) : 
  (∀ x ≥ 2, ∀ y > x, -y^2 - 4*m*y + 1 < -x^2 - 4*m*x + 1) → 
  m ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_decreasing_implies_m_bound_l1219_121941


namespace NUMINAMATH_CALUDE_total_matches_is_120_l1219_121976

/-- Represents the number of factions in the game -/
def num_factions : ℕ := 3

/-- Represents the number of players in each team -/
def team_size : ℕ := 4

/-- Represents the total number of players -/
def total_players : ℕ := 8

/-- Calculates the number of ways to form a team of given size from available factions -/
def ways_to_form_team (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Calculates the total number of distinct matches possible -/
def total_distinct_matches : ℕ :=
  let ways_one_team := ways_to_form_team team_size num_factions
  ways_one_team + Nat.choose ways_one_team 2

/-- Theorem stating that the total number of distinct matches is 120 -/
theorem total_matches_is_120 : total_distinct_matches = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_matches_is_120_l1219_121976


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l1219_121922

/-- A calendrical system where leap years occur every five years -/
structure CalendarSystem where
  leap_year_interval : ℕ
  leap_year_interval_eq : leap_year_interval = 5

/-- The number of years in the period we're considering -/
def period : ℕ := 200

/-- The maximum number of leap years in the given period -/
def max_leap_years (c : CalendarSystem) : ℕ := period / c.leap_year_interval

/-- Theorem: The maximum number of leap years in a 200-year period is 40 -/
theorem max_leap_years_in_period (c : CalendarSystem) : max_leap_years c = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l1219_121922


namespace NUMINAMATH_CALUDE_juice_cost_is_50_l1219_121961

/-- The cost of a candy bar in cents -/
def candy_cost : ℕ := 25

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The total cost in cents for the purchase -/
def total_cost : ℕ := 11 * 25

/-- The number of candy bars purchased -/
def num_candy : ℕ := 3

/-- The number of chocolate pieces purchased -/
def num_chocolate : ℕ := 2

/-- The number of juice packs purchased -/
def num_juice : ℕ := 1

theorem juice_cost_is_50 :
  ∃ (juice_cost : ℕ),
    juice_cost = 50 ∧
    total_cost = num_candy * candy_cost + num_chocolate * chocolate_cost + num_juice * juice_cost :=
by sorry

end NUMINAMATH_CALUDE_juice_cost_is_50_l1219_121961


namespace NUMINAMATH_CALUDE_ace_probabilities_l1219_121908

/-- The number of cards in a standard deck without jokers -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The probability of drawing an Ace twice without replacement from a standard deck -/
def prob_two_aces : ℚ := 1 / 221

/-- The conditional probability of drawing an Ace on the second draw, given that the first card drawn is an Ace -/
def prob_second_ace_given_first : ℚ := 1 / 17

/-- Theorem stating the probabilities for drawing Aces from a standard deck -/
theorem ace_probabilities :
  (prob_two_aces = (num_aces : ℚ) / deck_size * (num_aces - 1) / (deck_size - 1)) ∧
  (prob_second_ace_given_first = (num_aces - 1 : ℚ) / (deck_size - 1)) :=
sorry

end NUMINAMATH_CALUDE_ace_probabilities_l1219_121908


namespace NUMINAMATH_CALUDE_boa_constrictor_length_l1219_121938

/-- The length of a boa constrictor given the length of a garden snake and their relative sizes -/
theorem boa_constrictor_length 
  (garden_snake_length : ℝ) 
  (relative_size : ℝ) 
  (h1 : garden_snake_length = 10.0)
  (h2 : relative_size = 7.0) : 
  garden_snake_length / relative_size = 10.0 / 7.0 := by sorry

end NUMINAMATH_CALUDE_boa_constrictor_length_l1219_121938


namespace NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l1219_121925

/-- The perimeter of a rectangular garden with a width of 12 meters and an area equal to a 16x12 meter playground is 56 meters. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun garden_width garden_length playground_length playground_width =>
    garden_width = 12 ∧
    garden_length * garden_width = playground_length * playground_width ∧
    playground_length = 16 ∧
    playground_width = 12 →
    2 * (garden_length + garden_width) = 56

-- The proof is omitted
theorem garden_perimeter_proof : garden_perimeter 12 16 16 12 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l1219_121925


namespace NUMINAMATH_CALUDE_sum_of_xy_on_circle_l1219_121971

theorem sum_of_xy_on_circle (x y : ℝ) (h : x^2 + y^2 = 16*x - 12*y + 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_on_circle_l1219_121971


namespace NUMINAMATH_CALUDE_scientific_notation_of_14nm_l1219_121903

theorem scientific_notation_of_14nm (nm14 : ℝ) (h : nm14 = 0.000000014) :
  ∃ (a b : ℝ), a = 1.4 ∧ b = -8 ∧ nm14 = a * (10 : ℝ) ^ b :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_14nm_l1219_121903


namespace NUMINAMATH_CALUDE_complex_point_on_line_l1219_121909

theorem complex_point_on_line (a : ℝ) : 
  (∃ (z : ℂ), z = (a - 1 : ℝ) + 3*I ∧ z.im = z.re + 2) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l1219_121909


namespace NUMINAMATH_CALUDE_race_result_l1219_121965

-- Define the type for athlete positions
inductive Position
| First
| Second
| Third
| Fourth

-- Define a function to represent the statements of athletes
def athleteStatement (pos : Position) : Prop :=
  match pos with
  | Position.First => pos = Position.First
  | Position.Second => pos ≠ Position.First
  | Position.Third => pos = Position.First
  | Position.Fourth => pos = Position.Fourth

-- Define the theorem
theorem race_result :
  ∃ (p₁ p₂ p₃ p₄ : Position),
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    (athleteStatement p₁ ∧ athleteStatement p₂ ∧ athleteStatement p₃ ∧ ¬athleteStatement p₄) ∧
    p₃ = Position.First :=
by sorry


end NUMINAMATH_CALUDE_race_result_l1219_121965


namespace NUMINAMATH_CALUDE_ramanujan_identities_l1219_121962

/-- The function f₂ₙ as defined in Ramanujan's identities -/
def f (n : ℕ) (a b c d : ℝ) : ℝ :=
  (b + c + d)^(2*n) + (a + b + c)^(2*n) + (a - d)^(2*n) - 
  (a + c + d)^(2*n) - (a + b + d)^(2*n) - (b - c)^(2*n)

/-- Ramanujan's identities -/
theorem ramanujan_identities (a b c d : ℝ) (h : a * d = b * c) : 
  f 1 a b c d = 0 ∧ f 2 a b c d = 0 ∧ 64 * (f 3 a b c d) * (f 5 a b c d) = 45 * (f 4 a b c d)^2 := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_identities_l1219_121962


namespace NUMINAMATH_CALUDE_largest_value_in_special_sequence_l1219_121942

/-- A sequence of 8 increasing real numbers -/
def IncreasingSequence (a : Fin 8 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

/-- Checks if a sequence of 4 numbers is an arithmetic progression with a given common difference -/
def IsArithmeticProgression (a : Fin 4 → ℝ) (d : ℝ) : Prop :=
  ∀ i : Fin 3, a (i + 1) - a i = d

/-- Checks if a sequence of 4 numbers is a geometric progression -/
def IsGeometricProgression (a : Fin 4 → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Fin 3, a (i + 1) / a i = r

/-- The main theorem -/
theorem largest_value_in_special_sequence (a : Fin 8 → ℝ)
  (h_increasing : IncreasingSequence a)
  (h_arithmetic1 : ∃ i : Fin 5, IsArithmeticProgression (fun j => a (i + j)) 4)
  (h_arithmetic2 : ∃ i : Fin 5, IsArithmeticProgression (fun j => a (i + j)) 36)
  (h_geometric : ∃ i : Fin 5, IsGeometricProgression (fun j => a (i + j))) :
  a 7 = 126 ∨ a 7 = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_value_in_special_sequence_l1219_121942


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1219_121959

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1219_121959


namespace NUMINAMATH_CALUDE_myrtle_eggs_theorem_l1219_121989

/-- The number of eggs Myrtle has after all collections and drops -/
def eggs_remaining (num_hens : ℕ) (days_gone : ℕ) (neighbor_took : ℕ) (dropped_eggs : List ℕ)
  (daily_eggs : List ℕ) : ℕ :=
  let total_eggs := (List.sum daily_eggs) * days_gone
  let remaining_after_neighbor := total_eggs - neighbor_took
  remaining_after_neighbor - (List.sum dropped_eggs)

/-- Theorem stating the number of eggs Myrtle has after all collections and drops -/
theorem myrtle_eggs_theorem : 
  eggs_remaining 5 12 32 [3, 5, 2] [3, 4, 2, 5, 3] = 162 := by
  sorry

#eval eggs_remaining 5 12 32 [3, 5, 2] [3, 4, 2, 5, 3]

end NUMINAMATH_CALUDE_myrtle_eggs_theorem_l1219_121989


namespace NUMINAMATH_CALUDE_f_leq_g_l1219_121928

/-- Given functions f and g, prove that f(x) ≤ g(x) for all x > 0 when a ≥ 1 -/
theorem f_leq_g (x a : ℝ) (hx : x > 0) (ha : a ≥ 1) :
  Real.log x + 2 * x ≤ a * (x^2 + x) := by
  sorry

end NUMINAMATH_CALUDE_f_leq_g_l1219_121928


namespace NUMINAMATH_CALUDE_pencil_cost_calculation_l1219_121943

/-- The original cost of a pencil before discount -/
def original_cost : ℝ := 4.00

/-- The discount applied to the pencil -/
def discount : ℝ := 0.63

/-- The final price of the pencil after discount -/
def final_price : ℝ := 3.37

/-- Theorem stating that the original cost minus the discount equals the final price -/
theorem pencil_cost_calculation : original_cost - discount = final_price := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_calculation_l1219_121943


namespace NUMINAMATH_CALUDE_solution_characterization_l1219_121964

/-- The set of all solutions to the equation ab + bc + ca = 2(a + b + c) in natural numbers -/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (4, 1, 2), (4, 2, 1)}

/-- The equation ab + bc + ca = 2(a + b + c) -/
def SatisfiesEquation (t : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t
  a * b + b * c + c * a = 2 * (a + b + c)

theorem solution_characterization :
  ∀ t : ℕ × ℕ × ℕ, t ∈ SolutionSet ↔ SatisfiesEquation t :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l1219_121964


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_k_range_l1219_121956

theorem hyperbola_eccentricity_k_range :
  ∀ (k : ℝ) (e : ℝ),
    (∀ (x y : ℝ), x^2 / 4 + y^2 / k = 1) →
    (1 < e ∧ e < 3) →
    (e = Real.sqrt (1 - k / 4)) →
    (-32 < k ∧ k < 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_k_range_l1219_121956
