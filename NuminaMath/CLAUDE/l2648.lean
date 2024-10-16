import Mathlib

namespace NUMINAMATH_CALUDE_exists_triangle_with_different_colors_l2648_264888

/-- A color type representing the three possible colors of vertices -/
inductive Color
  | A
  | B
  | C

/-- A graph representing the triangulation -/
structure Graph (α : Type) where
  V : Set α
  E : Set (α × α)

/-- A coloring function that assigns a color to each vertex -/
def Coloring (α : Type) := α → Color

/-- A predicate to check if three vertices form a triangle in the graph -/
def IsTriangle {α : Type} (G : Graph α) (a b c : α) : Prop :=
  a ∈ G.V ∧ b ∈ G.V ∧ c ∈ G.V ∧
  (a, b) ∈ G.E ∧ (b, c) ∈ G.E ∧ (c, a) ∈ G.E

/-- The main theorem statement -/
theorem exists_triangle_with_different_colors {α : Type} (G : Graph α) (f : Coloring α)
  (hA : ∃ a ∈ G.V, f a = Color.A)
  (hB : ∃ b ∈ G.V, f b = Color.B)
  (hC : ∃ c ∈ G.V, f c = Color.C) :
  ∃ x y z : α, IsTriangle G x y z ∧ f x ≠ f y ∧ f y ≠ f z ∧ f z ≠ f x :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_with_different_colors_l2648_264888


namespace NUMINAMATH_CALUDE_x_power_minus_reciprocal_l2648_264808

theorem x_power_minus_reciprocal (φ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : x - 1 / x = (2 * Complex.I * Real.sin φ))
  (h4 : Odd n) :
  x^n - 1 / x^n = 2 * Complex.I^n * (Real.sin φ)^n :=
by sorry

end NUMINAMATH_CALUDE_x_power_minus_reciprocal_l2648_264808


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2648_264834

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (1 - i * z + 3 * i = -1 + i * z + 3 * i) ∧ (z = -i) :=
by
  sorry


end NUMINAMATH_CALUDE_complex_equation_solution_l2648_264834


namespace NUMINAMATH_CALUDE_complex_set_property_l2648_264878

def is_closed_under_multiplication (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_property (a b c d : ℂ) :
  let S : Set ℂ := {a, b, c, d}
  is_closed_under_multiplication S →
  a = 1 →
  b^2 = 1 →
  c^2 = b →
  b + c + d = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_set_property_l2648_264878


namespace NUMINAMATH_CALUDE_fourth_root_equality_exp_power_equality_cube_root_equality_sqrt_product_inequality_l2648_264868

-- Define π as a real number greater than 3
variable (π : ℝ) [Fact (π > 3)]

-- Theorem for option A
theorem fourth_root_equality : ∀ π : ℝ, π > 3 → (((3 - π) ^ 4) ^ (1/4 : ℝ)) = π - 3 := by sorry

-- Theorem for option B
theorem exp_power_equality : ∀ x : ℝ, Real.exp (2 * x) = (Real.exp x) ^ 2 := by sorry

-- Theorem for option C
theorem cube_root_equality : ∀ a b : ℝ, ((a - b) ^ 3) ^ (1/3 : ℝ) = a - b := by sorry

-- Theorem for option D (showing it's not always true)
theorem sqrt_product_inequality : ∃ a b : ℝ, (a * b) ^ (1/2 : ℝ) ≠ (a ^ (1/2 : ℝ)) * (b ^ (1/2 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_fourth_root_equality_exp_power_equality_cube_root_equality_sqrt_product_inequality_l2648_264868


namespace NUMINAMATH_CALUDE_three_valid_pairs_l2648_264832

/-- The number of ordered pairs (a, b) satisfying the floor painting conditions -/
def num_valid_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let b := p.2
    b > a ∧ (a - 4) * (b - 4) = a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 3 valid pairs -/
theorem three_valid_pairs : num_valid_pairs = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_valid_pairs_l2648_264832


namespace NUMINAMATH_CALUDE_gift_card_value_l2648_264869

theorem gift_card_value (original_value : ℝ) : 
  (3 / 8 : ℝ) * original_value = 75 → original_value = 200 := by
  sorry

end NUMINAMATH_CALUDE_gift_card_value_l2648_264869


namespace NUMINAMATH_CALUDE_min_value_theorem_l2648_264863

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- State the theorem
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (m^2 + 2) / m + (n^2 + 1) / n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2648_264863


namespace NUMINAMATH_CALUDE_x_minus_y_values_l2648_264824

theorem x_minus_y_values (x y : ℝ) (h : y = Real.sqrt (x^2 - 9) - Real.sqrt (9 - x^2) + 4) :
  x - y = -1 ∨ x - y = -7 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l2648_264824


namespace NUMINAMATH_CALUDE_sector_central_angle_l2648_264814

/-- A circular sector with perimeter 8 and area 4 has a central angle of 2 radians -/
theorem sector_central_angle (r : ℝ) (l : ℝ) (θ : ℝ) : 
  r > 0 → 
  2 * r + l = 8 →  -- perimeter equation
  1 / 2 * l * r = 4 →  -- area equation
  θ = l / r →  -- definition of central angle in radians
  θ = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2648_264814


namespace NUMINAMATH_CALUDE_intersection_parallel_line_equation_specific_line_equation_l2648_264818

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line_equation (a b c d e f g h i : ℝ) :
  (∃ x y, a * x + b * y = c ∧ d * x + e * y = f) →  -- Intersection point exists
  (∀ x y, (a * x + b * y = c ∧ d * x + e * y = f) → g * x + h * y + i = 0) →  -- Line passes through intersection
  (∃ k, ∀ x y, g * x + h * y + i = k * (g * x + h * y + 0)) →  -- Parallel to g * x + h * y + 0 = 0
  ∃ k, ∀ x y, g * x + h * y + i = k * (g * x + h * y - 27) :=
by sorry

/-- The specific case for the given problem -/
theorem specific_line_equation :
  (∃ x y, x + y = 9 ∧ 2 * x - y = 18) →
  (∀ x y, (x + y = 9 ∧ 2 * x - y = 18) → 3 * x - 2 * y + i = 0) →
  (∃ k, ∀ x y, 3 * x - 2 * y + i = k * (3 * x - 2 * y + 8)) →
  i = -27 :=
by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_equation_specific_line_equation_l2648_264818


namespace NUMINAMATH_CALUDE_vacation_homework_pages_l2648_264872

/-- Represents the number of days Garin divided her homework for -/
def days : ℕ := 24

/-- Represents the number of pages Garin can solve per day -/
def pages_per_day : ℕ := 19

/-- Calculates the total number of pages in Garin's vacation homework -/
def total_pages : ℕ := days * pages_per_day

/-- Proves that the total number of pages in Garin's vacation homework is 456 -/
theorem vacation_homework_pages : total_pages = 456 := by
  sorry

end NUMINAMATH_CALUDE_vacation_homework_pages_l2648_264872


namespace NUMINAMATH_CALUDE_pheasants_and_rabbits_l2648_264829

theorem pheasants_and_rabbits (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 35)
  (h2 : total_legs = 94) :
  ∃ (pheasants rabbits : ℕ),
    pheasants + rabbits = total_heads ∧
    2 * pheasants + 4 * rabbits = total_legs ∧
    pheasants = 23 ∧
    rabbits = 12 := by
  sorry

end NUMINAMATH_CALUDE_pheasants_and_rabbits_l2648_264829


namespace NUMINAMATH_CALUDE_cos_180_deg_l2648_264885

/-- The cosine of an angle in degrees -/
noncomputable def cos_deg (θ : ℝ) : ℝ := 
  (Complex.exp (θ * Complex.I * Real.pi / 180)).re

/-- Theorem: The cosine of 180 degrees is -1 -/
theorem cos_180_deg : cos_deg 180 = -1 := by sorry

end NUMINAMATH_CALUDE_cos_180_deg_l2648_264885


namespace NUMINAMATH_CALUDE_sales_ratio_l2648_264831

/-- Proves that the ratio of sales on a tough week to sales on a good week is 1:2 -/
theorem sales_ratio (tough_week_sales : ℝ) (total_sales : ℝ) : 
  tough_week_sales = 800 →
  total_sales = 10400 →
  ∃ (good_week_sales : ℝ),
    5 * good_week_sales + 3 * tough_week_sales = total_sales ∧
    tough_week_sales / good_week_sales = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sales_ratio_l2648_264831


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l2648_264820

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Theorem: 15! ends with 3 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l2648_264820


namespace NUMINAMATH_CALUDE_max_value_inequality_l2648_264876

theorem max_value_inequality (A : ℝ) (h : A > 0) :
  let M := max (2 + A / 2) (2 * Real.sqrt A)
  ∀ x y : ℝ, x > 0 → y > 0 →
    1 / x + 1 / y + A / (x + y) ≥ M / Real.sqrt (x * y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2648_264876


namespace NUMINAMATH_CALUDE_pie_distribution_probability_l2648_264895

/-- Represents the total number of pies -/
def total_pies : ℕ := 6

/-- Represents the number of growth pies -/
def growth_pies : ℕ := 2

/-- Represents the number of shrink pies -/
def shrink_pies : ℕ := 4

/-- Represents the number of pies given to Mary -/
def pies_given : ℕ := 3

/-- The probability that one of the girls does not have a single growth pie -/
def prob_no_growth_pie : ℚ := 7/10

theorem pie_distribution_probability :
  prob_no_growth_pie = 1 - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given) :=
by sorry

end NUMINAMATH_CALUDE_pie_distribution_probability_l2648_264895


namespace NUMINAMATH_CALUDE_race_distance_l2648_264815

theorem race_distance (d : ℝ) 
  (h1 : ∃ x y : ℝ, x > y ∧ d / x = (d - 25) / y)
  (h2 : ∃ y z : ℝ, y > z ∧ d / y = (d - 15) / z)
  (h3 : ∃ x z : ℝ, x > z ∧ d / x = (d - 35) / z)
  : d = 75 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l2648_264815


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l2648_264853

/-- The area of a square with adjacent vertices at (1,3) and (4,7) is 25 square units. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, 7)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l2648_264853


namespace NUMINAMATH_CALUDE_set_equality_l2648_264873

theorem set_equality : {p : ℝ × ℝ | p.1 + p.2 = 5 ∧ 2 * p.1 - p.2 = 1} = {(2, 3)} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2648_264873


namespace NUMINAMATH_CALUDE_bob_weight_is_165_l2648_264883

def jim_weight : ℝ := sorry
def bob_weight : ℝ := sorry

axiom combined_weight : jim_weight + bob_weight = 220
axiom weight_relation : bob_weight - 2 * jim_weight = bob_weight / 3

theorem bob_weight_is_165 : bob_weight = 165 := by sorry

end NUMINAMATH_CALUDE_bob_weight_is_165_l2648_264883


namespace NUMINAMATH_CALUDE_f_negation_l2648_264822

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the theorem
theorem f_negation (a b : ℝ) :
  f a b 2011 = 10 → f a b (-2011) = -14 := by
  sorry

end NUMINAMATH_CALUDE_f_negation_l2648_264822


namespace NUMINAMATH_CALUDE_distance_sum_squares_l2648_264844

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x - m * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - m + 3 = 0

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, -3)

-- Theorem statement
theorem distance_sum_squares (m : ℝ) (P : ℝ × ℝ) :
  l₁ m P.1 P.2 ∧ l₂ m P.1 P.2 →
  l₁ m A.1 A.2 ∧ l₂ m B.1 B.2 →
  (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_squares_l2648_264844


namespace NUMINAMATH_CALUDE_project_hours_difference_l2648_264866

theorem project_hours_difference (total_hours : ℕ) 
  (h1 : total_hours = 216) 
  (kate_hours : ℕ) 
  (pat_hours : ℕ) 
  (mark_hours : ℕ) 
  (h2 : pat_hours = 2 * kate_hours) 
  (h3 : pat_hours * 3 = mark_hours) 
  (h4 : kate_hours + pat_hours + mark_hours = total_hours) : 
  mark_hours - kate_hours = 120 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2648_264866


namespace NUMINAMATH_CALUDE_pages_per_day_l2648_264884

theorem pages_per_day (pages_per_book : ℕ) (days_per_book : ℕ) (h1 : pages_per_book = 249) (h2 : days_per_book = 3) :
  pages_per_book / days_per_book = 83 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l2648_264884


namespace NUMINAMATH_CALUDE_f_deriv_l2648_264846

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (2 * x - 3 + Real.sqrt (4 * x^2 - 12 * x + 10)) -
  Real.sqrt (4 * x^2 - 12 * x + 10) * Real.arctan (2 * x - 3)

theorem f_deriv :
  ∀ x : ℝ, DifferentiableAt ℝ f x →
    deriv f x = - Real.arctan (2 * x - 3) / Real.sqrt (4 * x^2 - 12 * x + 10) :=
by sorry

end NUMINAMATH_CALUDE_f_deriv_l2648_264846


namespace NUMINAMATH_CALUDE_max_value_f_on_interval_l2648_264890

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x ≤ f c ∧
  f c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_on_interval_l2648_264890


namespace NUMINAMATH_CALUDE_frannie_jumped_less_jump_difference_l2648_264881

/-- The number of times Frannie jumped -/
def frannies_jumps : ℕ := 53

/-- The number of times Meg jumped -/
def megs_jumps : ℕ := 71

/-- Frannie jumped fewer times than Meg -/
theorem frannie_jumped_less : frannies_jumps < megs_jumps := by sorry

/-- The difference in jumps between Meg and Frannie is 18 -/
theorem jump_difference : megs_jumps - frannies_jumps = 18 := by sorry

end NUMINAMATH_CALUDE_frannie_jumped_less_jump_difference_l2648_264881


namespace NUMINAMATH_CALUDE_restaurant_gratuity_l2648_264819

/-- Calculate the gratuity for a restaurant bill -/
theorem restaurant_gratuity (price1 price2 price3 : ℕ) (tip_percentage : ℚ) : 
  price1 = 10 → price2 = 13 → price3 = 17 → tip_percentage = 1/10 →
  (price1 + price2 + price3 : ℚ) * tip_percentage = 4 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_gratuity_l2648_264819


namespace NUMINAMATH_CALUDE_part_one_part_two_l2648_264861

-- Define the sets A, B, and U
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 3}
def U : Set ℝ := {x | x ≤ 4}

-- Part 1
theorem part_one (m : ℝ) (h : m = -1) :
  (Uᶜ ∩ A)ᶜ ∪ B m = {x | x < 2 ∨ x = 4} ∧
  A ∩ (Uᶜ ∩ B m)ᶜ = {x | 2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two :
  {m : ℝ | A ∪ B m = A} = {m | -1/2 ≤ m ∧ m ≤ 1} ∪ {m | 4 ≤ m} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2648_264861


namespace NUMINAMATH_CALUDE_min_seats_occupied_min_occupied_seats_is_fifty_l2648_264810

/-- Represents the number of seats in a row -/
def total_seats : Nat := 200

/-- Represents the size of each group (one person + three empty seats) -/
def group_size : Nat := 4

/-- The minimum number of occupied seats required -/
def min_occupied_seats : Nat := total_seats / group_size

/-- Theorem stating the minimum number of occupied seats -/
theorem min_seats_occupied (n : Nat) : 
  n ≥ min_occupied_seats → 
  ∀ (new_seat : Nat), new_seat > n ∧ new_seat ≤ total_seats → 
  ∃ (occupied_seat : Nat), occupied_seat ≤ n ∧ (new_seat = occupied_seat + 1 ∨ new_seat = occupied_seat - 1) :=
sorry

/-- Theorem proving the minimum number of occupied seats is indeed 50 -/
theorem min_occupied_seats_is_fifty : min_occupied_seats = 50 :=
sorry

end NUMINAMATH_CALUDE_min_seats_occupied_min_occupied_seats_is_fifty_l2648_264810


namespace NUMINAMATH_CALUDE_fraction_of_A_grades_l2648_264847

/-- Represents the distribution of grades in a math course. -/
structure GradeDistribution where
  totalStudents : ℕ
  fractionB : ℚ
  fractionC : ℚ
  numberD : ℕ

/-- Theorem stating that the fraction of A grades is 1/5 given the specified conditions. -/
theorem fraction_of_A_grades
  (dist : GradeDistribution)
  (h1 : dist.totalStudents = 800)
  (h2 : dist.fractionB = 1/4)
  (h3 : dist.fractionC = 1/2)
  (h4 : dist.numberD = 40) :
  (dist.totalStudents - (dist.fractionB * dist.totalStudents + dist.fractionC * dist.totalStudents + dist.numberD)) / dist.totalStudents = 1/5 :=
sorry

end NUMINAMATH_CALUDE_fraction_of_A_grades_l2648_264847


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l2648_264891

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

def prob_heads (n k : ℕ) : ℚ :=
  (binomial n k : ℚ) * (1 / 2) ^ n

theorem coin_flip_probability_difference : 
  |prob_heads 5 2 - prob_heads 5 4| = 5 / 32 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l2648_264891


namespace NUMINAMATH_CALUDE_sanity_question_suffices_l2648_264842

-- Define the types of beings in Transylvania
inductive Being
| Human
| Vampire

-- Define the possible responses to the question
inductive Response
| Yes
| No

-- Define the function that represents how a being responds to the question "Are you sane?"
def respond_to_sanity_question (b : Being) : Response :=
  match b with
  | Being.Human => Response.Yes
  | Being.Vampire => Response.No

-- Define the function that determines the being type based on the response
def determine_being (r : Response) : Being :=
  match r with
  | Response.Yes => Being.Human
  | Response.No => Being.Vampire

-- Theorem: Asking "Are you sane?" is sufficient to determine if a Transylvanian is a human or a vampire
theorem sanity_question_suffices :
  ∀ (b : Being), determine_being (respond_to_sanity_question b) = b :=
by sorry


end NUMINAMATH_CALUDE_sanity_question_suffices_l2648_264842


namespace NUMINAMATH_CALUDE_racket_deal_cost_l2648_264800

/-- Calculates the total cost of two rackets given a store's deal and the full price of each racket. -/
def totalCostTwoRackets (fullPrice : ℕ) : ℕ :=
  fullPrice + (fullPrice - fullPrice / 2)

/-- Theorem stating that the total cost of two rackets is $90 given the specific conditions. -/
theorem racket_deal_cost :
  totalCostTwoRackets 60 = 90 := by
  sorry

#eval totalCostTwoRackets 60

end NUMINAMATH_CALUDE_racket_deal_cost_l2648_264800


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2648_264823

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 5; 1, 3]

theorem matrix_inverse_proof :
  A⁻¹ = !![3, -5; -1, 2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2648_264823


namespace NUMINAMATH_CALUDE_f_minimum_value_F_monotonicity_l2648_264864

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x

def F (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x + 1

theorem f_minimum_value (x : ℝ) (hx : x > 0) :
  ∃ (min : ℝ), min = -1 / Real.exp 1 ∧ f x ≥ min := by sorry

theorem F_monotonicity (a : ℝ) (x : ℝ) (hx : x > 0) :
  (a ≥ 0 → StrictMono (F a)) ∧
  (a < 0 → 
    (∀ y z, 0 < y ∧ y < z ∧ z < Real.sqrt (-1 / (2 * a)) → F a y < F a z) ∧
    (∀ y z, Real.sqrt (-1 / (2 * a)) < y ∧ y < z → F a y > F a z)) := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_F_monotonicity_l2648_264864


namespace NUMINAMATH_CALUDE_toothpick_structure_count_l2648_264860

/-- Calculates the number of toothpicks in a rectangular grid --/
def rectangle_toothpicks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Calculates the number of toothpicks in a right-angled triangle --/
def triangle_toothpicks (base : ℕ) : ℕ :=
  base + (Int.sqrt (2 * base * base)).toNat + 1

/-- The total number of toothpicks in the structure --/
def total_toothpicks (length width : ℕ) : ℕ :=
  rectangle_toothpicks length width + triangle_toothpicks width

theorem toothpick_structure_count :
  total_toothpicks 40 20 = 1709 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_structure_count_l2648_264860


namespace NUMINAMATH_CALUDE_total_airflow_is_483000_l2648_264897

/-- Calculates the total airflow generated by three fans in one week -/
def totalWeeklyAirflow (fanA_flow : ℝ) (fanA_time : ℝ) (fanB_flow : ℝ) (fanB_time : ℝ) 
                        (fanC_flow : ℝ) (fanC_time : ℝ) : ℝ :=
  7 * (fanA_flow * fanA_time * 60 + fanB_flow * fanB_time * 60 + fanC_flow * fanC_time * 60)

/-- Theorem stating that the total airflow generated by the three fans in one week is 483,000 liters -/
theorem total_airflow_is_483000 : 
  totalWeeklyAirflow 10 10 15 20 25 30 = 483000 := by
  sorry

#eval totalWeeklyAirflow 10 10 15 20 25 30

end NUMINAMATH_CALUDE_total_airflow_is_483000_l2648_264897


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2648_264886

/-- 
Given two lines in the form of linear equations:
  3y - 2x - 6 = 0 and 4y + bx - 5 = 0
If these lines are perpendicular, then b = 6.
-/
theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y, 3 * y - 2 * x - 6 = 0 → 
           4 * y + b * x - 5 = 0 → 
           (2 / 3) * (-b / 4) = -1) → 
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2648_264886


namespace NUMINAMATH_CALUDE_special_triples_count_l2648_264858

/-- Represents a graph with a specific number of vertices and edges per vertex -/
structure Graph where
  numVertices : ℕ
  edgesPerVertex : ℕ

/-- Calculates the number of triples in a graph where each pair of vertices is either all connected or all disconnected -/
def countSpecialTriples (g : Graph) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem special_triples_count (g : Graph) (h1 : g.numVertices = 30) (h2 : g.edgesPerVertex = 6) :
  countSpecialTriples g = 1990 := by
  sorry

end NUMINAMATH_CALUDE_special_triples_count_l2648_264858


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l2648_264821

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 4 * a^2 + 7 * a + 3 = 2) :
  ∃ (m : ℝ), m = 3 * a + 2 ∧ ∀ (x : ℝ), (4 * x^2 + 7 * x + 3 = 2) → m ≤ 3 * x + 2 ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l2648_264821


namespace NUMINAMATH_CALUDE_pool_ground_area_l2648_264856

theorem pool_ground_area (length width : ℝ) (h1 : length = 5) (h2 : width = 4) :
  length * width = 20 := by
  sorry

end NUMINAMATH_CALUDE_pool_ground_area_l2648_264856


namespace NUMINAMATH_CALUDE_area_equality_function_unique_l2648_264851

/-- A function satisfying the given area equality property -/
def AreaEqualityFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ * f x₂ = (x₂ - x₁) * (f x₁ + f x₂)

theorem area_equality_function_unique
  (f : ℝ → ℝ)
  (h₁ : ∀ x, x > 0 → f x > 0)
  (h₂ : AreaEqualityFunction f)
  (h₃ : f 1 = 4) :
  (∀ x, x > 0 → f x = 4 / x) ∧ f 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_area_equality_function_unique_l2648_264851


namespace NUMINAMATH_CALUDE_two_solutions_equation_l2648_264839

/-- The value of a 2x2 matrix [[a, c], [d, b]] is defined as ab - cd -/
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

/-- The equation 2x^2 - x = 3 has exactly two real solutions -/
theorem two_solutions_equation :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ s ↔ matrix_value (2*x) x 1 x = 3 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_equation_l2648_264839


namespace NUMINAMATH_CALUDE_speed_calculation_l2648_264809

/-- Given a distance of 240 km and a travel time of 6 hours, prove that the speed is 40 km/hr. -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 240) (h2 : time = 6) :
  distance / time = 40 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l2648_264809


namespace NUMINAMATH_CALUDE_tobias_lawn_mowing_charge_tobias_lawn_mowing_charge_proof_l2648_264830

/-- Tobias' lawn mowing problem -/
theorem tobias_lawn_mowing_charge : ℕ → Prop :=
  fun x =>
    let shoe_cost : ℕ := 95
    let saving_months : ℕ := 3
    let monthly_allowance : ℕ := 5
    let shovel_charge : ℕ := 7
    let remaining_money : ℕ := 15
    let lawns_mowed : ℕ := 4
    let driveways_shoveled : ℕ := 5
    
    (saving_months * monthly_allowance + lawns_mowed * x + driveways_shoveled * shovel_charge
      = shoe_cost + remaining_money) →
    x = 15

/-- The proof of Tobias' lawn mowing charge -/
theorem tobias_lawn_mowing_charge_proof : tobias_lawn_mowing_charge 15 := by
  sorry

end NUMINAMATH_CALUDE_tobias_lawn_mowing_charge_tobias_lawn_mowing_charge_proof_l2648_264830


namespace NUMINAMATH_CALUDE_total_weekly_revenue_l2648_264898

def normal_price : ℝ := 5

def monday_sales : ℕ := 9
def tuesday_sales : ℕ := 12
def wednesday_sales : ℕ := 18
def thursday_sales : ℕ := 14
def friday_sales : ℕ := 16
def saturday_sales : ℕ := 20
def sunday_sales : ℕ := 11

def wednesday_discount : ℝ := 0.1
def friday_discount : ℝ := 0.05

def daily_revenue (sales : ℕ) (discount : ℝ) : ℝ :=
  (sales : ℝ) * normal_price * (1 - discount)

theorem total_weekly_revenue :
  daily_revenue monday_sales 0 +
  daily_revenue tuesday_sales 0 +
  daily_revenue wednesday_sales wednesday_discount +
  daily_revenue thursday_sales 0 +
  daily_revenue friday_sales friday_discount +
  daily_revenue saturday_sales 0 +
  daily_revenue sunday_sales 0 = 487 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_revenue_l2648_264898


namespace NUMINAMATH_CALUDE_inequality_selection_l2648_264870

/-- Given positive real numbers a, b, c, and a function f with minimum value 4,
    prove that a + b + c = 4 and find the minimum value of (1/4)a² + (1/9)b² + c² -/
theorem inequality_selection (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : ∀ x, |x + a| + |x - b| + c ≥ 4)
  (h5 : ∃ x, |x + a| + |x - b| + c = 4) :
  (a + b + c = 4) ∧
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 →
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 4 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_selection_l2648_264870


namespace NUMINAMATH_CALUDE_simon_blueberry_theorem_l2648_264850

/-- The number of blueberries Simon picked from his own bushes -/
def own_blueberries : ℕ := 100

/-- The number of blueberries needed for each pie -/
def blueberries_per_pie : ℕ := 100

/-- The number of pies Simon can make -/
def number_of_pies : ℕ := 3

/-- The number of blueberries Simon picked from nearby bushes -/
def nearby_blueberries : ℕ := number_of_pies * blueberries_per_pie - own_blueberries

theorem simon_blueberry_theorem : nearby_blueberries = 200 := by
  sorry

end NUMINAMATH_CALUDE_simon_blueberry_theorem_l2648_264850


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l2648_264825

/-- Proves that given an interest rate of 5% per annum for 2 years, 
    if the difference between compound interest and simple interest is 18, 
    then the principal amount is 7200. -/
theorem interest_difference_theorem (P : ℝ) : 
  P * (1 + 0.05)^2 - P - (P * 0.05 * 2) = 18 → P = 7200 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l2648_264825


namespace NUMINAMATH_CALUDE_junsu_is_winner_l2648_264812

-- Define the participants
inductive Participant
| Younghee
| Jimin
| Junsu

-- Define the amount of water drunk by each participant
def water_drunk : Participant → Float
  | Participant.Younghee => 1.4
  | Participant.Jimin => 1.8
  | Participant.Junsu => 2.1

-- Define the winner as the participant who drank the most water
def is_winner (p : Participant) : Prop :=
  ∀ q : Participant, water_drunk p ≥ water_drunk q

-- Theorem stating that Junsu is the winner
theorem junsu_is_winner : is_winner Participant.Junsu := by
  sorry

end NUMINAMATH_CALUDE_junsu_is_winner_l2648_264812


namespace NUMINAMATH_CALUDE_largest_expression_l2648_264833

theorem largest_expression (x : ℝ) : 
  (x + 1/4) * (x - 1/4) ≥ (x + 1) * (x - 1) ∧
  (x + 1/4) * (x - 1/4) ≥ (x + 1/2) * (x - 1/2) ∧
  (x + 1/4) * (x - 1/4) ≥ (x + 1/3) * (x - 1/3) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l2648_264833


namespace NUMINAMATH_CALUDE_flowers_per_pot_l2648_264826

theorem flowers_per_pot (num_gardens : ℕ) (pots_per_garden : ℕ) (total_flowers : ℕ) : 
  num_gardens = 10 →
  pots_per_garden = 544 →
  total_flowers = 174080 →
  total_flowers / (num_gardens * pots_per_garden) = 32 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_pot_l2648_264826


namespace NUMINAMATH_CALUDE_fraction_simplification_l2648_264804

theorem fraction_simplification (x : ℝ) (h : x = 5) :
  (x^6 - 25*x^3 + 144) / (x^3 - 12) = 114 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2648_264804


namespace NUMINAMATH_CALUDE_marble_distribution_l2648_264880

/-- Represents a distribution of marbles into bags -/
def Distribution := List Nat

/-- Checks if a distribution is valid for a given number of children -/
def isValidDistribution (d : Distribution) (numChildren : Nat) : Prop :=
  d.sum = 77 ∧ d.length ≥ numChildren ∧ (77 % numChildren = 0)

/-- The minimum number of bags needed -/
def minBags : Nat := 17

theorem marble_distribution :
  (∀ d : Distribution, d.length < minBags → ¬(isValidDistribution d 7 ∧ isValidDistribution d 11)) ∧
  (∃ d : Distribution, d.length = minBags ∧ isValidDistribution d 7 ∧ isValidDistribution d 11) :=
sorry

#check marble_distribution

end NUMINAMATH_CALUDE_marble_distribution_l2648_264880


namespace NUMINAMATH_CALUDE_expression_simplification_l2648_264835

theorem expression_simplification (x : ℝ) 
  (h1 : x * (x^2 - 4) = 0) 
  (h2 : x ≠ 0) 
  (h3 : x ≠ 2) : 
  (x - 3) / (3 * x^2 - 6 * x) / (x + 2 - 5 / (x - 2)) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2648_264835


namespace NUMINAMATH_CALUDE_line_parameterization_l2648_264806

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 5 * x - 7

/-- The parametric form of the line -/
def parametric_form (s m t x y : ℝ) : Prop :=
  x = s + 2 * t ∧ y = 3 + m * t

/-- The theorem stating that s = 2 and m = 10 for the given line and parametric form -/
theorem line_parameterization :
  ∃ (s m : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ parametric_form s m t x y) ∧ s = 2 ∧ m = 10 := by
  sorry


end NUMINAMATH_CALUDE_line_parameterization_l2648_264806


namespace NUMINAMATH_CALUDE_product_of_powers_equals_1260_l2648_264802

theorem product_of_powers_equals_1260 (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z = 1260) : 
  3*w + 4*x + 2*y + 2*z = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_1260_l2648_264802


namespace NUMINAMATH_CALUDE_parabola_vertex_l2648_264894

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -(x - 2)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 3)

/-- Theorem: The vertex of the parabola y = -(x - 2)^2 + 3 is (2, 3) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2648_264894


namespace NUMINAMATH_CALUDE_inequality_proof_l2648_264867

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (1 - a) ^ a > (1 - b) ^ b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2648_264867


namespace NUMINAMATH_CALUDE_cricket_bat_price_l2648_264803

/-- The final price of a cricket bat after two sales with given profits -/
def final_price (initial_cost : ℝ) (profit1 : ℝ) (profit2 : ℝ) : ℝ :=
  initial_cost * (1 + profit1) * (1 + profit2)

/-- Theorem stating the final price of the cricket bat -/
theorem cricket_bat_price :
  final_price 148 0.20 0.25 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l2648_264803


namespace NUMINAMATH_CALUDE_truncated_pyramid_lateral_area_l2648_264855

/-- Represents a regular quadrangular pyramid -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Represents a truncated regular quadrangular pyramid -/
structure TruncatedRegularQuadPyramid where
  base_side : ℝ
  height : ℝ
  cut_height : ℝ

/-- Calculates the lateral surface area of a truncated regular quadrangular pyramid -/
def lateral_surface_area (t : TruncatedRegularQuadPyramid) : ℝ :=
  sorry

theorem truncated_pyramid_lateral_area :
  let p : RegularQuadPyramid := { base_side := 6, height := 4 }
  let t : TruncatedRegularQuadPyramid := { base_side := 6, height := 4, cut_height := 1 }
  lateral_surface_area t = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_lateral_area_l2648_264855


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2648_264852

theorem fraction_equation_solution (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 6 →
    (P / (x + 5) + Q / (x * (x - 6)) : ℝ) = (x^2 - 4*x + 20) / (x^3 + x^2 - 30*x)) →
  (Q : ℚ) / P = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2648_264852


namespace NUMINAMATH_CALUDE_rectangle_equal_diagonals_l2648_264848

/-- A rectangle is a quadrilateral with four right angles. -/
structure Rectangle where
  vertices : Fin 4 → ℝ × ℝ
  is_rectangle : sorry

/-- The diagonal of a quadrilateral is the line segment connecting opposite vertices. -/
def diagonal (q : Rectangle) (i : Fin 2) : ℝ × ℝ := sorry

/-- The length of a line segment in ℝ² -/
def length (p q : ℝ × ℝ) : ℝ := sorry

/-- If a quadrilateral is a rectangle, then its diagonals are equal. -/
theorem rectangle_equal_diagonals (r : Rectangle) :
  length (diagonal r 0) = length (diagonal r 1) := by sorry

end NUMINAMATH_CALUDE_rectangle_equal_diagonals_l2648_264848


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l2648_264807

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15/16) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l2648_264807


namespace NUMINAMATH_CALUDE_fish_catch_difference_l2648_264859

/-- Given the number of fish caught by various birds and a fisherman, prove the difference in catch between the fisherman and pelican. -/
theorem fish_catch_difference (pelican kingfisher osprey fisherman : ℕ) : 
  pelican = 13 →
  kingfisher = pelican + 7 →
  osprey = 2 * kingfisher →
  fisherman = 4 * (pelican + kingfisher + osprey) →
  fisherman - pelican = 279 := by
  sorry

end NUMINAMATH_CALUDE_fish_catch_difference_l2648_264859


namespace NUMINAMATH_CALUDE_max_product_difference_l2648_264857

theorem max_product_difference (x y : ℕ) : 
  x > 0 → y > 0 → x + 2 * y = 2008 → (∀ a b : ℕ, a > 0 → b > 0 → a + 2 * b = 2008 → x * y ≥ a * b) → x - y = 502 := by
sorry

end NUMINAMATH_CALUDE_max_product_difference_l2648_264857


namespace NUMINAMATH_CALUDE_french_speakers_l2648_264896

theorem french_speakers (total : ℕ) (latin : ℕ) (neither : ℕ) (both : ℕ) : 
  total = 25 → latin = 13 → neither = 6 → both = 9 → 
  ∃ french : ℕ, french = 15 ∧ 
  (total - neither = latin + french - both) := by
sorry

end NUMINAMATH_CALUDE_french_speakers_l2648_264896


namespace NUMINAMATH_CALUDE_roots_not_in_interval_l2648_264865

theorem roots_not_in_interval (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2*a) → x ∉ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_not_in_interval_l2648_264865


namespace NUMINAMATH_CALUDE_optimal_profit_l2648_264875

/-- Represents the profit optimization problem for a shopping mall --/
structure ShoppingMall where
  total_boxes : ℕ
  profit_A : ℝ
  profit_B : ℝ
  profit_diff : ℝ
  price_change : ℝ
  box_change : ℝ

/-- Calculates the optimal price reduction and maximum profit --/
def optimize_profit (mall : ShoppingMall) : ℝ × ℝ :=
  sorry

/-- Theorem stating the optimal price reduction and maximum profit --/
theorem optimal_profit (mall : ShoppingMall) 
  (h1 : mall.total_boxes = 600)
  (h2 : mall.profit_A = 40000)
  (h3 : mall.profit_B = 160000)
  (h4 : mall.profit_diff = 200)
  (h5 : mall.price_change = 5)
  (h6 : mall.box_change = 2) :
  optimize_profit mall = (75, 204500) :=
sorry

end NUMINAMATH_CALUDE_optimal_profit_l2648_264875


namespace NUMINAMATH_CALUDE_range_of_m_l2648_264882

-- Define the quadratic function
def f (m x : ℝ) := m * x^2 - m * x - 1

-- Define the solution set
def solution_set (m : ℝ) := {x : ℝ | f m x ≥ 0}

-- State the theorem
theorem range_of_m : 
  (∀ m : ℝ, solution_set m = ∅) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2648_264882


namespace NUMINAMATH_CALUDE_lottery_tickets_theorem_lottery_tickets_minimality_l2648_264889

/-- The probability of winning with a single lottery ticket -/
def p : ℝ := 0.01

/-- The desired probability of winning at least once -/
def desired_prob : ℝ := 0.95

/-- The number of tickets needed to achieve the desired probability -/
def n : ℕ := 300

/-- Theorem stating that n tickets are sufficient to achieve the desired probability -/
theorem lottery_tickets_theorem :
  1 - (1 - p) ^ n ≥ desired_prob :=
sorry

/-- Theorem stating that n-1 tickets are not sufficient to achieve the desired probability -/
theorem lottery_tickets_minimality :
  1 - (1 - p) ^ (n - 1) < desired_prob :=
sorry

end NUMINAMATH_CALUDE_lottery_tickets_theorem_lottery_tickets_minimality_l2648_264889


namespace NUMINAMATH_CALUDE_bicycle_sprocket_rotation_l2648_264801

theorem bicycle_sprocket_rotation (large_teeth small_teeth : ℕ) (large_revolution : ℝ) :
  large_teeth = 48 →
  small_teeth = 20 →
  large_revolution = 1 →
  (large_teeth : ℝ) / small_teeth * (2 * Real.pi * large_revolution) = 4.8 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_sprocket_rotation_l2648_264801


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2648_264871

def A : Set (ℝ × ℝ) := {p | 3 * p.1 - p.2 = 7}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 + p.2 = 3}

theorem intersection_of_A_and_B : A ∩ B = {(2, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2648_264871


namespace NUMINAMATH_CALUDE_gym_weights_problem_l2648_264877

/-- Given the conditions of the gym weights problem, prove that each green weight is 3 pounds. -/
theorem gym_weights_problem (blue_weight : ℕ) (num_blue : ℕ) (num_green : ℕ) (bar_weight : ℕ) (total_weight : ℕ) :
  blue_weight = 2 →
  num_blue = 4 →
  num_green = 5 →
  bar_weight = 2 →
  total_weight = 25 →
  ∃ (green_weight : ℕ), green_weight = 3 ∧ total_weight = blue_weight * num_blue + green_weight * num_green + bar_weight :=
by sorry

end NUMINAMATH_CALUDE_gym_weights_problem_l2648_264877


namespace NUMINAMATH_CALUDE_function_equality_l2648_264899

/-- Given f(x) = 3x - 5, prove that 2 * [f(1)] - 16 = f(7) -/
theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x - 5) :
  2 * (f 1) - 16 = f 7 := by sorry

end NUMINAMATH_CALUDE_function_equality_l2648_264899


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2648_264841

def z : ℂ := (3 - Complex.I) * (2 - Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2648_264841


namespace NUMINAMATH_CALUDE_last_digit_of_189_in_ternary_l2648_264854

theorem last_digit_of_189_in_ternary (n : Nat) : n = 189 → n % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_189_in_ternary_l2648_264854


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l2648_264840

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 15 45 = 59 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l2648_264840


namespace NUMINAMATH_CALUDE_negation_equivalence_l2648_264827

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x < 2 ∧ x^2 - 2*x < 0)) ↔ (∀ x : ℝ, x < 2 → x^2 - 2*x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2648_264827


namespace NUMINAMATH_CALUDE_alice_departure_time_l2648_264838

/-- Proof that Alice must leave 30 minutes after Bob to arrive in city B just before him. -/
theorem alice_departure_time (distance : ℝ) (bob_speed : ℝ) (alice_speed : ℝ) 
  (h1 : distance = 220)
  (h2 : bob_speed = 40)
  (h3 : alice_speed = 44) :
  (distance / bob_speed - distance / alice_speed) * 60 = 30 := by
  sorry

#check alice_departure_time

end NUMINAMATH_CALUDE_alice_departure_time_l2648_264838


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l2648_264862

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l2648_264862


namespace NUMINAMATH_CALUDE_smallest_multiple_of_1_to_10_l2648_264849

theorem smallest_multiple_of_1_to_10 : ∃ n : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_1_to_10_l2648_264849


namespace NUMINAMATH_CALUDE_matchboxes_per_box_l2648_264837

/-- Proves that the number of matchboxes in each box is 20, given the total number of boxes,
    sticks per matchbox, and total number of sticks. -/
theorem matchboxes_per_box 
  (total_boxes : ℕ) 
  (sticks_per_matchbox : ℕ) 
  (total_sticks : ℕ) 
  (h1 : total_boxes = 4)
  (h2 : sticks_per_matchbox = 300)
  (h3 : total_sticks = 24000) :
  total_sticks / sticks_per_matchbox / total_boxes = 20 := by
  sorry

#eval 24000 / 300 / 4  -- Should output 20

end NUMINAMATH_CALUDE_matchboxes_per_box_l2648_264837


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2648_264892

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := 2 + Complex.I + (1 - Complex.I) * x
  (∃ (y : ℝ), z = Complex.I * y) → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2648_264892


namespace NUMINAMATH_CALUDE_percentage_calculation_approximation_l2648_264893

theorem percentage_calculation_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs ((0.47 * 1442 - 0.36 * 1412) + 63 - 232.42) < ε := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_approximation_l2648_264893


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2648_264828

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2648_264828


namespace NUMINAMATH_CALUDE_f_negative_one_value_l2648_264805

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_one_value : 
  (∀ x, f (x / (1 + x)) = x) → f (-1) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_one_value_l2648_264805


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2648_264845

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_max_value 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -40)
  (h2 : f a b c (-1) = -8)
  (h3 : f a b c (-3) = 8)
  (h4 : -b / (2 * a) = -4)
  (h5 : ∃ x₁ x₂, x₁ = -1 ∧ x₂ = -7 ∧ f a b c x₁ = -8 ∧ f a b c x₂ = -8)
  (h6 : a + b + c = -40) :
  ∃ x_max, ∀ x, f a b c x ≤ f a b c x_max ∧ f a b c x_max = 10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2648_264845


namespace NUMINAMATH_CALUDE_product_evaluation_l2648_264887

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2648_264887


namespace NUMINAMATH_CALUDE_perp_line_plane_relation_l2648_264813

-- Define the concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the perpendicularity relations
def perp_to_countless_lines (L : Line) (α : Plane) : Prop := sorry
def perp_to_plane (L : Line) (α : Plane) : Prop := sorry

-- State the theorem
theorem perp_line_plane_relation (L : Line) (α : Plane) :
  (perp_to_plane L α → perp_to_countless_lines L α) ∧
  ∃ L α, perp_to_countless_lines L α ∧ ¬perp_to_plane L α :=
sorry

end NUMINAMATH_CALUDE_perp_line_plane_relation_l2648_264813


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_min_sum_achieved_l2648_264874

/-- Given two natural numbers a and b satisfying 1a + 4b = 30,
    their sum is minimized when a = b = 6 -/
theorem min_sum_with_constraint (a b : ℕ) (h : a + 4 * b = 30) :
  a + b ≥ 12 := by
sorry

/-- The minimum sum of 12 is achieved when a = b = 6 -/
theorem min_sum_achieved : ∃ (a b : ℕ), a + 4 * b = 30 ∧ a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_min_sum_achieved_l2648_264874


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l2648_264843

theorem square_minus_product_equals_one (a : ℝ) (h : a = -4) : a^2 - (a+1)*(a-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l2648_264843


namespace NUMINAMATH_CALUDE_ellipse_inscribed_circle_max_area_l2648_264879

/-- The ellipse with equation x²/4 + y²/3 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (1, 0)

/-- A line passing through F₂ -/
def line_through_F₂ (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

/-- The area of the inscribed circle in triangle F₁MN -/
def inscribed_circle_area (m : ℝ) : ℝ := sorry

theorem ellipse_inscribed_circle_max_area :
  ∃ (max_area : ℝ),
    (∀ m : ℝ, inscribed_circle_area m ≤ max_area) ∧
    (max_area = 9 * Real.pi / 16) ∧
    (∀ m : ℝ, inscribed_circle_area m = max_area ↔ m = 0) :=
  sorry

end NUMINAMATH_CALUDE_ellipse_inscribed_circle_max_area_l2648_264879


namespace NUMINAMATH_CALUDE_inequality_range_l2648_264816

theorem inequality_range : 
  {a : ℝ | ∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a} = {a : ℝ | -1 ≤ a ∧ a ≤ 4} := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2648_264816


namespace NUMINAMATH_CALUDE_overlap_area_is_zero_l2648_264836

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculate the area of overlap between two triangles -/
def areaOfOverlap (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the area of overlap is zero -/
theorem overlap_area_is_zero :
  let t1 := Triangle.mk (Point.mk 0 0) (Point.mk 2 2) (Point.mk 2 0)
  let t2 := Triangle.mk (Point.mk 0 2) (Point.mk 2 2) (Point.mk 0 0)
  areaOfOverlap t1 t2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_overlap_area_is_zero_l2648_264836


namespace NUMINAMATH_CALUDE_batsman_average_after_15th_innings_l2648_264817

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored : ℚ) / (stats.innings + 1 : ℚ)

/-- Theorem: Batsman's average after 15th innings -/
theorem batsman_average_after_15th_innings
  (stats : BatsmanStats)
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 85 = stats.average + 3)
  : newAverage stats 85 = 43 := by
  sorry

#check batsman_average_after_15th_innings

end NUMINAMATH_CALUDE_batsman_average_after_15th_innings_l2648_264817


namespace NUMINAMATH_CALUDE_triangle_area_l2648_264811

theorem triangle_area (a b c A B C : Real) : 
  a + b = 3 →
  c = Real.sqrt 3 →
  Real.sin (2 * C - Real.pi / 6) = 1 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2648_264811
