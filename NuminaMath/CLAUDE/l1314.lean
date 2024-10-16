import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_minimum_l1314_131432

theorem quadratic_minimum (a b : ℝ) (x₀ : ℝ) (h : a > 0) :
  (a * x₀ = b) ↔ ∀ x : ℝ, (1/2) * a * x^2 - b * x ≥ (1/2) * a * x₀^2 - b * x₀ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1314_131432


namespace NUMINAMATH_CALUDE_johns_family_ages_l1314_131492

/-- Given information about John's family ages, prove John's and his sibling's ages -/
theorem johns_family_ages :
  ∀ (john_age dad_age sibling_age : ℕ),
  john_age + 30 = dad_age →
  john_age + dad_age = 90 →
  sibling_age = john_age + 5 →
  john_age = 30 ∧ sibling_age = 35 := by
  sorry

end NUMINAMATH_CALUDE_johns_family_ages_l1314_131492


namespace NUMINAMATH_CALUDE_circle_radius_increase_l1314_131411

theorem circle_radius_increase (r n : ℝ) : 
  r > 0 → r > n → π * (r + n)^2 = 3 * π * r^2 → r = n * (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l1314_131411


namespace NUMINAMATH_CALUDE_not_all_cells_marked_l1314_131485

/-- Represents a cell in the grid --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- The grid of cells --/
def Grid := List Cell

/-- Checks if two cells are neighbors --/
def isNeighbor (c1 c2 : Cell) : Bool :=
  (c1.x = c2.x ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

/-- Counts the number of marked neighbors for a cell --/
def countMarkedNeighbors (cell : Cell) (markedCells : List Cell) : Nat :=
  (markedCells.filter (isNeighbor cell)).length

/-- Spreads the marking to cells with at least two marked neighbors --/
def spread (grid : Grid) (markedCells : List Cell) : List Cell :=
  markedCells ++ (grid.filter (fun c => countMarkedNeighbors c markedCells ≥ 2))

/-- Creates a 10x10 grid --/
def createGrid : Grid :=
  List.range 10 >>= fun x => List.range 10 >>= fun y => [Cell.mk x y]

/-- The main theorem --/
theorem not_all_cells_marked (initialMarked : List Cell) 
  (h : initialMarked.length = 9) : 
  ∃ (finalMarked : List Cell), finalMarked = spread (createGrid) initialMarked ∧ 
  finalMarked.length < 100 := by
  sorry

end NUMINAMATH_CALUDE_not_all_cells_marked_l1314_131485


namespace NUMINAMATH_CALUDE_problem_solution_l1314_131472

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∀ x, x ∈ (A ∩ B (-4)) ↔ (1/2 ≤ x ∧ x < 2)) ∧
  (∀ x, x ∈ (A ∪ B (-4)) ↔ (-2 < x ∧ x ≤ 3)) ∧
  (∀ a, (Aᶜ ∩ B a = B a) ↔ a ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1314_131472


namespace NUMINAMATH_CALUDE_polygon_sides_l1314_131404

theorem polygon_sides (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1314_131404


namespace NUMINAMATH_CALUDE_cl2_moles_in_reaction_l1314_131469

/-- Represents the stoichiometric coefficients of the reaction CH4 + 2Cl2 → CHCl3 + 4HCl -/
structure ReactionCoefficients where
  ch4 : ℕ
  cl2 : ℕ
  chcl3 : ℕ
  hcl : ℕ

/-- The balanced equation coefficients for the reaction -/
def balancedEquation : ReactionCoefficients :=
  { ch4 := 1, cl2 := 2, chcl3 := 1, hcl := 4 }

/-- Calculates the moles of Cl2 combined given the moles of CH4 and HCl -/
def molesOfCl2Combined (molesCH4 : ℕ) (molesHCl : ℕ) : ℕ :=
  (balancedEquation.cl2 * molesHCl) / balancedEquation.hcl

theorem cl2_moles_in_reaction (molesCH4 : ℕ) (molesHCl : ℕ) :
  molesCH4 = balancedEquation.ch4 ∧ molesHCl = balancedEquation.hcl →
  molesOfCl2Combined molesCH4 molesHCl = balancedEquation.cl2 :=
by
  sorry

end NUMINAMATH_CALUDE_cl2_moles_in_reaction_l1314_131469


namespace NUMINAMATH_CALUDE_rectangular_plot_roots_l1314_131487

theorem rectangular_plot_roots (length width r s : ℝ) : 
  length^2 - 3*length + 2 = 0 →
  width^2 - 3*width + 2 = 0 →
  (1/length)^2 - r*(1/length) + s = 0 →
  (1/width)^2 - r*(1/width) + s = 0 →
  r*s = 0.75 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_roots_l1314_131487


namespace NUMINAMATH_CALUDE_equation_equality_l1314_131416

theorem equation_equality : (3 * 6 * 9) / 3 = (2 * 6 * 9) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l1314_131416


namespace NUMINAMATH_CALUDE_withdraw_representation_l1314_131498

-- Define a type for monetary transactions
inductive Transaction
  | deposit (amount : ℤ)
  | withdraw (amount : ℤ)

-- Define a function to represent transactions
def represent : Transaction → ℤ
  | Transaction.deposit amount => amount
  | Transaction.withdraw amount => -amount

-- State the theorem
theorem withdraw_representation :
  represent (Transaction.deposit 30000) = 30000 →
  represent (Transaction.withdraw 40000) = -40000 := by
  sorry

end NUMINAMATH_CALUDE_withdraw_representation_l1314_131498


namespace NUMINAMATH_CALUDE_condition_relationship_l1314_131418

theorem condition_relationship (x : ℝ) : 
  (∀ x, x = Real.sqrt (x + 2) → x^2 = x + 2) ∧ 
  (∃ x, x^2 = x + 2 ∧ x ≠ Real.sqrt (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l1314_131418


namespace NUMINAMATH_CALUDE_square_root_special_form_l1314_131467

theorem square_root_special_form :
  ∀ n : ℕ, 10 ≤ n ∧ n < 100 →
    (∃ a b : ℕ, n = 10 * a + b ∧ Real.sqrt n = a + Real.sqrt b) ↔
    (n = 64 ∨ n = 81) := by
  sorry

end NUMINAMATH_CALUDE_square_root_special_form_l1314_131467


namespace NUMINAMATH_CALUDE_midpoint_distance_after_move_l1314_131483

/-- Given two points P(p,q) and Q(r,s) on a Cartesian plane with midpoint N(x,y),
    prove that after moving P 3 units right and 5 units up, and Q 5 units left and 3 units down,
    the distance between N and the new midpoint N' is √2. -/
theorem midpoint_distance_after_move (p q r s x y : ℝ) :
  x = (p + r) / 2 →
  y = (q + s) / 2 →
  let x' := (p + 3 + r - 5) / 2
  let y' := (q + 5 + s - 3) / 2
  Real.sqrt ((x - x')^2 + (y - y')^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_after_move_l1314_131483


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1314_131479

theorem quadratic_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ = -a ∧ x₁ * x₂ = b) ∧
               (2*x₁ + 2*x₂ = -b ∧ 4*x₁*x₂ = c)) →
  a / c = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1314_131479


namespace NUMINAMATH_CALUDE_parcel_delivery_growth_l1314_131466

/-- Represents the equation for parcel delivery growth over three months -/
theorem parcel_delivery_growth 
  (initial_delivery : ℕ) 
  (total_delivery : ℕ) 
  (growth_rate : ℝ) : 
  initial_delivery = 20000 → 
  total_delivery = 72800 → 
  2 + 2 * (1 + growth_rate) + 2 * (1 + growth_rate)^2 = 7.28 := by
  sorry

#check parcel_delivery_growth

end NUMINAMATH_CALUDE_parcel_delivery_growth_l1314_131466


namespace NUMINAMATH_CALUDE_intersection_implies_k_range_l1314_131457

/-- The line equation kx - y - k - 1 = 0 intersects the line segment MN,
    where M(2,1) and N(3,2) are the endpoints of the segment. -/
def intersects_segment (k : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    k * (2 + t) - (1 + t) - k - 1 = 0

/-- The theorem states that if the line intersects the segment MN,
    then k is in the range [3/2, 2]. -/
theorem intersection_implies_k_range :
  ∀ k : ℝ, intersects_segment k → 3/2 ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_k_range_l1314_131457


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1314_131496

/-- For an infinite geometric series with first term a and common ratio r,
    if the sum of the series starting from the fourth term is 1/27 times
    the sum of the original series, then r = 1/3. -/
theorem geometric_series_ratio (a r : ℝ) (h : |r| < 1) :
  (a * r^3 / (1 - r)) = (1 / 27) * (a / (1 - r)) →
  r = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1314_131496


namespace NUMINAMATH_CALUDE_locus_of_q_l1314_131429

/-- The ellipse in the problem -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The hyperbola that is the locus of Q -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | (q.1^2 / a^2) - (q.2^2 / b^2) = 1}

/-- P and P' form a vertical chord of the ellipse -/
def VerticalChord (a b : ℝ) (p p' : ℝ × ℝ) : Prop :=
  p ∈ Ellipse a b ∧ p' ∈ Ellipse a b ∧ p.1 = p'.1

/-- Q is the intersection of A'P and AP' -/
def IntersectionPoint (a : ℝ) (p p' q : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ,
    q.1 = t * (p.1 + a) + (1 - t) * (-a) ∧
    q.2 = t * p.2 ∧
    q.1 = s * (p'.1 - a) + (1 - s) * a ∧
    q.2 = s * p'.2

/-- The main theorem -/
theorem locus_of_q (a b : ℝ) (p p' q : ℝ × ℝ) 
    (h_ab : a > 0 ∧ b > 0)
    (h_ellipse : p ∈ Ellipse a b ∧ p' ∈ Ellipse a b)
    (h_vertical : VerticalChord a b p p')
    (h_intersect : IntersectionPoint a p p' q) :
  q ∈ Hyperbola a b := by
  sorry

end NUMINAMATH_CALUDE_locus_of_q_l1314_131429


namespace NUMINAMATH_CALUDE_min_students_for_given_data_l1314_131437

/-- Represents the number of students receiving A's on each day of the week -/
structure GradeData where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- The minimum number of students in the class given the grade data -/
def minStudents (data : GradeData) : Nat :=
  max (data.monday + data.tuesday)
    (max (data.tuesday + data.wednesday)
      (max (data.wednesday + data.thursday)
        (data.thursday + data.friday)))

/-- Theorem stating the minimum number of students given the specific grade data -/
theorem min_students_for_given_data :
  let data : GradeData := {
    monday := 5,
    tuesday := 8,
    wednesday := 6,
    thursday := 4,
    friday := 9
  }
  minStudents data = 14 := by sorry

end NUMINAMATH_CALUDE_min_students_for_given_data_l1314_131437


namespace NUMINAMATH_CALUDE_functional_equation_unique_solution_l1314_131489

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)

/-- The main theorem stating that the only function satisfying the equation is f(x) = x - 1 -/
theorem functional_equation_unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → ∀ x : ℝ, f x = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_unique_solution_l1314_131489


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1314_131412

theorem arithmetic_sequence_nth_term (a₁ a₂ aₙ n : ℤ) : 
  a₁ = 11 → a₂ = 8 → aₙ = -49 → 
  (∀ k : ℕ, k > 0 → a₁ + (k - 1) * (a₂ - a₁) = aₙ ↔ k = n) →
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1314_131412


namespace NUMINAMATH_CALUDE_problem_solution_l1314_131406

def vector := ℝ × ℝ

noncomputable def problem (x : ℝ) : Prop :=
  let a : vector := (1, Real.sin x)
  let b : vector := (Real.sin x, -1)
  let c : vector := (1, Real.cos x)
  0 < x ∧ x < Real.pi ∧
  ¬ (∃ (k : ℝ), (1 + Real.sin x, Real.sin x - 1) = (k * c.1, k * c.2)) ∧
  x = Real.pi / 2 ∧
  ∃ (A B C : ℝ), 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    B = Real.pi / 2 ∧
    2 * (Real.sin B)^2 + 2 * (Real.sin C)^2 - 2 * (Real.sin A)^2 = Real.sin B * Real.sin C

theorem problem_solution (x : ℝ) (h : problem x) :
  ∃ (A B C : ℝ), Real.sin (C - Real.pi / 3) = (1 - 3 * Real.sqrt 5) / 8 := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l1314_131406


namespace NUMINAMATH_CALUDE_sequence_max_value_l1314_131454

theorem sequence_max_value (n : ℤ) : -2 * n^2 + 29 * n + 3 ≤ 108 := by
  sorry

end NUMINAMATH_CALUDE_sequence_max_value_l1314_131454


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l1314_131424

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | (x^2 + a*x + b)*(x - 1) = 0}

-- Define the theorem
theorem sum_of_a_and_b_is_one 
  (B C : Set ℝ) 
  (a b : ℝ) 
  (h1 : A a b ∩ B = {1, 2})
  (h2 : A a b ∩ (C ∪ B) = {3}) :
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l1314_131424


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1314_131407

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x < 23 ∧ (1054 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1054 + y) % 23 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1314_131407


namespace NUMINAMATH_CALUDE_exponent_for_28_decimal_places_l1314_131460

def base : ℝ := 10^4 * 3.456789

theorem exponent_for_28_decimal_places :
  ∀ n : ℕ, (∃ m : ℕ, base^n * 10^28 = m ∧ m < base^n * 10^29) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_exponent_for_28_decimal_places_l1314_131460


namespace NUMINAMATH_CALUDE_product_odd_implies_sum_odd_l1314_131423

theorem product_odd_implies_sum_odd (a b c : ℤ) : 
  Odd (a * b * c) → Odd (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_product_odd_implies_sum_odd_l1314_131423


namespace NUMINAMATH_CALUDE_child_ticket_price_l1314_131462

theorem child_ticket_price 
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (adult_price : ℕ)
  (child_tickets : ℕ)
  (h1 : total_tickets = 130)
  (h2 : total_receipts = 840)
  (h3 : adult_price = 12)
  (h4 : child_tickets = 90) :
  (total_receipts - (total_tickets - child_tickets) * adult_price) / child_tickets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_price_l1314_131462


namespace NUMINAMATH_CALUDE_melissa_games_played_l1314_131409

def points_per_game : ℕ := 12
def total_points : ℕ := 36

theorem melissa_games_played : 
  total_points / points_per_game = 3 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l1314_131409


namespace NUMINAMATH_CALUDE_emerson_first_part_distance_l1314_131434

/-- Emerson's rowing trip distances -/
structure RowingTrip where
  total : ℕ
  second : ℕ
  third : ℕ

/-- The distance covered in the first part of the rowing trip -/
def firstPartDistance (trip : RowingTrip) : ℕ :=
  trip.total - (trip.second + trip.third)

/-- Theorem: The first part distance of Emerson's specific trip is 6 miles -/
theorem emerson_first_part_distance :
  firstPartDistance ⟨39, 15, 18⟩ = 6 := by
  sorry

end NUMINAMATH_CALUDE_emerson_first_part_distance_l1314_131434


namespace NUMINAMATH_CALUDE_mary_clothing_expense_l1314_131408

-- Define the costs of the shirt and jacket
def shirt_cost : Real := 13.04
def jacket_cost : Real := 12.27

-- Define the total cost
def total_cost : Real := shirt_cost + jacket_cost

-- Theorem statement
theorem mary_clothing_expense : total_cost = 25.31 := by
  sorry

end NUMINAMATH_CALUDE_mary_clothing_expense_l1314_131408


namespace NUMINAMATH_CALUDE_base_3_312_property_l1314_131451

def base_3_representation (n : ℕ) : List ℕ :=
  sorry

def count_digit (l : List ℕ) (d : ℕ) : ℕ :=
  sorry

theorem base_3_312_property :
  let base_3_312 := base_3_representation 312
  let x := count_digit base_3_312 0
  let y := count_digit base_3_312 1
  let z := count_digit base_3_312 2
  z - y + x = 2 := by sorry

end NUMINAMATH_CALUDE_base_3_312_property_l1314_131451


namespace NUMINAMATH_CALUDE_max_value_n_is_3210_l1314_131443

/-- S(a) represents the sum of the digits of a natural number a -/
def S (a : ℕ) : ℕ := sorry

/-- allDigitsDifferent n is true if all digits of n are different -/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- maxValueN is the maximum value of n satisfying the given conditions -/
def maxValueN : ℕ := 3210

theorem max_value_n_is_3210 :
  ∀ n : ℕ, allDigitsDifferent n → S (3 * n) = 3 * S n → n ≤ maxValueN := by
  sorry

end NUMINAMATH_CALUDE_max_value_n_is_3210_l1314_131443


namespace NUMINAMATH_CALUDE_distance_travelled_l1314_131430

-- Define the velocity function
def v (t : ℝ) : ℝ := 2 * t - 3

-- Define the theorem
theorem distance_travelled (t₀ t₁ : ℝ) (h : 0 ≤ t₀ ∧ t₁ = 5) :
  ∫ t in t₀..t₁, |v t| = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_distance_travelled_l1314_131430


namespace NUMINAMATH_CALUDE_like_terms_exponent_value_l1314_131402

theorem like_terms_exponent_value (x y : ℝ) (m n : ℕ) :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^(m+3) * y^6 = b * x^5 * y^(2*n)) →
  m^n = 8 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_value_l1314_131402


namespace NUMINAMATH_CALUDE_sprint_competition_races_l1314_131400

/-- Calculates the number of races required to determine a champion in a sprint competition. -/
def races_to_champion (total_sprinters : ℕ) (lanes : ℕ) (eliminated_per_race : ℕ) (advance_interval : ℕ) : ℕ :=
  let regular_races := 32
  let special_races := 16
  regular_races + special_races

/-- Theorem stating that 48 races are required for the given sprint competition setup. -/
theorem sprint_competition_races :
  races_to_champion 300 8 6 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sprint_competition_races_l1314_131400


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_solution_l1314_131417

theorem smallest_four_digit_congruence_solution :
  let x : ℕ := 1011
  (∀ y : ℕ, y < x → y < 1000 ∨ ¬(5 * y ≡ 25 [ZMOD 20] ∧ 
                                 3 * y + 10 ≡ 19 [ZMOD 7] ∧ 
                                 y + 3 ≡ 2 * y [ZMOD 12])) ∧
  (5 * x ≡ 25 [ZMOD 20] ∧ 
   3 * x + 10 ≡ 19 [ZMOD 7] ∧ 
   x + 3 ≡ 2 * x [ZMOD 12]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_solution_l1314_131417


namespace NUMINAMATH_CALUDE_bobs_first_lap_time_l1314_131494

/-- Proves that the time for the first lap is 70 seconds given the conditions of Bob's run --/
theorem bobs_first_lap_time (track_length : ℝ) (num_laps : ℕ) (time_second_lap : ℝ) (time_third_lap : ℝ) (average_speed : ℝ) :
  track_length = 400 →
  num_laps = 3 →
  time_second_lap = 85 →
  time_third_lap = 85 →
  average_speed = 5 →
  (track_length * num_laps) / average_speed - (time_second_lap + time_third_lap) = 70 :=
by sorry

end NUMINAMATH_CALUDE_bobs_first_lap_time_l1314_131494


namespace NUMINAMATH_CALUDE_alpha_third_range_l1314_131440

open Real Set

theorem alpha_third_range (α : ℝ) (h1 : sin α > 0) (h2 : cos α < 0) (h3 : sin (α/3) > cos (α/3)) :
  ∃ k : ℤ, α/3 ∈ (Set.Ioo (2*k*π + π/4) (2*k*π + π/3)) ∪ (Set.Ioo (2*k*π + 5*π/6) (2*k*π + π)) :=
sorry

end NUMINAMATH_CALUDE_alpha_third_range_l1314_131440


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_equals_7_plus_2sqrt6_l1314_131490

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 3/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_equals_7_plus_2sqrt6 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  a + 2*b = 7 + 2*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_equals_7_plus_2sqrt6_l1314_131490


namespace NUMINAMATH_CALUDE_factor_expression_l1314_131480

theorem factor_expression (x : ℝ) : 75 * x^11 + 135 * x^22 = 15 * x^11 * (5 + 9 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1314_131480


namespace NUMINAMATH_CALUDE_parabola_equation_parabola_final_equation_l1314_131478

/-- A parabola with axis of symmetry parallel to the y-axis -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  eq_def : ∀ x, eq x = a * (x - 1) * (x - 4)

/-- The parabola passes through points (1,0) and (4,0) -/
def passes_through_points (p : Parabola) : Prop :=
  p.eq 1 = 0 ∧ p.eq 4 = 0

/-- The line y = 2x -/
def line (x : ℝ) : ℝ := 2 * x

/-- The parabola is tangent to the line y = 2x -/
def is_tangent (p : Parabola) : Prop :=
  ∃ x : ℝ, p.eq x = line x ∧ 
  ∀ y : ℝ, y ≠ x → p.eq y ≠ line y

/-- The main theorem -/
theorem parabola_equation (p : Parabola) 
  (h1 : passes_through_points p) 
  (h2 : is_tangent p) : 
  p.a = -2/9 ∨ p.a = -2 := by
  sorry

/-- The final result -/
theorem parabola_final_equation (p : Parabola) 
  (h1 : passes_through_points p) 
  (h2 : is_tangent p) : 
  (∀ x, p.eq x = -2/9 * (x - 1) * (x - 4)) ∨ 
  (∀ x, p.eq x = -2 * (x - 1) * (x - 4)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_parabola_final_equation_l1314_131478


namespace NUMINAMATH_CALUDE_toms_gas_expense_l1314_131495

/-- Proves that given the conditions of Tom's lawn mowing business,
    his monthly gas expense is $17. -/
theorem toms_gas_expense (lawns_mowed : ℕ) (price_per_lawn : ℕ) (extra_income : ℕ) (profit : ℕ) 
    (h1 : lawns_mowed = 3)
    (h2 : price_per_lawn = 12)
    (h3 : extra_income = 10)
    (h4 : profit = 29) :
  lawns_mowed * price_per_lawn + extra_income - profit = 17 := by
  sorry

end NUMINAMATH_CALUDE_toms_gas_expense_l1314_131495


namespace NUMINAMATH_CALUDE_pet_store_puppies_l1314_131477

theorem pet_store_puppies 
  (bought : ℝ) 
  (puppies_per_cage : ℝ) 
  (cages_used : ℝ) 
  (h1 : bought = 3.0)
  (h2 : puppies_per_cage = 5.0)
  (h3 : cages_used = 4.2) :
  cages_used * puppies_per_cage - bought = 18.0 := by
sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l1314_131477


namespace NUMINAMATH_CALUDE_baseball_weight_l1314_131499

theorem baseball_weight (total_weight : ℝ) (soccer_ball_weight : ℝ) (baseball_count : ℕ) (soccer_ball_count : ℕ) :
  total_weight = 10.98 →
  soccer_ball_weight = 0.8 →
  baseball_count = 7 →
  soccer_ball_count = 9 →
  (soccer_ball_count * soccer_ball_weight + baseball_count * ((total_weight - soccer_ball_count * soccer_ball_weight) / baseball_count) = total_weight) ∧
  ((total_weight - soccer_ball_count * soccer_ball_weight) / baseball_count = 0.54) :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_weight_l1314_131499


namespace NUMINAMATH_CALUDE_count_integer_segments_specific_triangle_l1314_131459

/-- Represents a right triangle ABC with integer leg lengths -/
structure RightTriangle where
  ab : ℕ  -- Length of leg AB
  bc : ℕ  -- Length of leg BC

/-- Calculates the number of distinct integer lengths of line segments 
    that can be drawn from vertex B to a point on hypotenuse AC -/
def count_integer_segments (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem count_integer_segments_specific_triangle : 
  let t : RightTriangle := { ab := 20, bc := 21 }
  count_integer_segments t = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_segments_specific_triangle_l1314_131459


namespace NUMINAMATH_CALUDE_point_movement_l1314_131456

/-- 
Given a point P on a number line that is moved 4 units to the right and then 7 units to the left,
if its final position is 9, then its original position was 12.
-/
theorem point_movement (P : ℝ) : 
  (P + 4 - 7 = 9) → P = 12 := by
sorry

end NUMINAMATH_CALUDE_point_movement_l1314_131456


namespace NUMINAMATH_CALUDE_integer_root_of_special_polynomial_l1314_131428

/-- Given a polynomial with integer coefficients of the form
    x^4 + b_3*x^3 + b_2*x^2 + b_1*x + 50,
    if s is an integer root of this polynomial and s^3 divides 50,
    then s = 1 or s = -1 -/
theorem integer_root_of_special_polynomial (b₃ b₂ b₁ s : ℤ) :
  (s^4 + b₃*s^3 + b₂*s^2 + b₁*s + 50 = 0) →
  (s^3 ∣ 50) →
  (s = 1 ∨ s = -1) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_special_polynomial_l1314_131428


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l1314_131450

/-- The number of cards Nell initially had -/
def initial_cards : ℕ := 304

/-- The number of cards Nell has left -/
def remaining_cards : ℕ := 276

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem cards_given_to_jeff : cards_given = 28 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l1314_131450


namespace NUMINAMATH_CALUDE_possible_x_values_l1314_131438

def A (x y : ℕ+) : ℕ := x^2 + y^2 + 2*x - 2*y + 2

def B (x : ℕ+) : ℤ := x^2 - 5*x + 5

theorem possible_x_values :
  ∀ x y : ℕ+, (B x)^(A x y) = 1 → x ∈ ({1, 2, 3, 4} : Set ℕ+) :=
sorry

end NUMINAMATH_CALUDE_possible_x_values_l1314_131438


namespace NUMINAMATH_CALUDE_xy_max_and_x2_4y2_min_l1314_131461

theorem xy_max_and_x2_4y2_min (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + 2*y = 3) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x*y ≥ a*b) ∧
  x*y = 9/8 ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x^2 + 4*y^2 ≤ a^2 + 4*b^2) ∧
  x^2 + 4*y^2 = 9/2 :=
sorry

end NUMINAMATH_CALUDE_xy_max_and_x2_4y2_min_l1314_131461


namespace NUMINAMATH_CALUDE_expression_simplification_l1314_131468

theorem expression_simplification (x y : ℝ) (h : x^2 ≠ y^2) :
  ((x^2 + y^2) / (x^2 - y^2)) + ((x^2 - y^2) / (x^2 + y^2)) = 2*(x^4 + y^4) / (x^4 - y^4) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1314_131468


namespace NUMINAMATH_CALUDE_second_term_of_sequence_l1314_131470

theorem second_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = n * (2 * n + 1)) → 
  a 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_second_term_of_sequence_l1314_131470


namespace NUMINAMATH_CALUDE_count_switchable_positions_l1314_131422

/-- Represents the number of revolutions a clock hand makes in one hour -/
def revolutions_per_hour (is_minute_hand : Bool) : ℚ :=
  if is_minute_hand then 1 else 1/12

/-- Represents a valid clock position -/
def is_valid_position (hour_pos : ℚ) (minute_pos : ℚ) : Prop :=
  0 ≤ hour_pos ∧ hour_pos < 1 ∧ 0 ≤ minute_pos ∧ minute_pos < 1

/-- Represents a clock position that remains valid when hands are switched -/
def is_switchable_position (t : ℚ) : Prop :=
  is_valid_position (t * revolutions_per_hour false) (t * revolutions_per_hour true) ∧
  is_valid_position (t * revolutions_per_hour true) (t * revolutions_per_hour false)

/-- The main theorem stating the number of switchable positions -/
theorem count_switchable_positions :
  (∃ (S : Finset ℚ), (∀ t ∈ S, is_switchable_position t) ∧ S.card = 143) :=
sorry

end NUMINAMATH_CALUDE_count_switchable_positions_l1314_131422


namespace NUMINAMATH_CALUDE_test_marks_theorem_l1314_131431

/-- Represents a test section with a number of questions and a success rate -/
structure TestSection where
  questions : ℕ
  successRate : ℚ
  
/-- Calculates the total marks for a given test -/
def calculateTotalMarks (sections : List TestSection) : ℚ :=
  let correctAnswers := sections.map (fun s => (s.questions : ℚ) * s.successRate)
  let totalCorrect := correctAnswers.sum
  let totalQuestions := (sections.map (fun s => s.questions)).sum
  let incorrectAnswers := totalQuestions - totalCorrect.floor
  totalCorrect.floor - 0.25 * incorrectAnswers

/-- The theorem states that given the specific test conditions, the total marks obtained is 115 -/
theorem test_marks_theorem :
  let sections := [
    { questions := 50, successRate := 85/100 },
    { questions := 60, successRate := 70/100 },
    { questions := 40, successRate := 95/100 }
  ]
  calculateTotalMarks sections = 115 := by
  sorry

end NUMINAMATH_CALUDE_test_marks_theorem_l1314_131431


namespace NUMINAMATH_CALUDE_min_weighings_for_ten_coins_l1314_131486

/-- Represents a weighing on a balance scale -/
inductive Weighing
  | Equal : Weighing
  | LeftLighter : Weighing
  | RightLighter : Weighing

/-- Represents the state of knowledge about the coins -/
structure CoinState where
  total : Nat
  genuine : Nat
  counterfeit : Nat

/-- A function that performs a weighing and updates the coin state -/
def performWeighing (state : CoinState) (w : Weighing) : CoinState :=
  sorry

/-- The minimum number of weighings required to find the counterfeit coin -/
def minWeighings (state : CoinState) : Nat :=
  sorry

/-- Theorem stating that the minimum number of weighings for 10 coins with 1 counterfeit is 3 -/
theorem min_weighings_for_ten_coins :
  let initialState : CoinState := ⟨10, 9, 1⟩
  minWeighings initialState = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_weighings_for_ten_coins_l1314_131486


namespace NUMINAMATH_CALUDE_hyperbola_I_equation_hyperbola_II_equation_l1314_131465

-- Part I
def hyperbola_I (x y : ℝ) : Prop :=
  let c : ℝ := 8  -- half of focal distance
  let e : ℝ := 4/3  -- eccentricity
  let a : ℝ := c/e
  let b : ℝ := Real.sqrt (c^2 - a^2)
  y^2/a^2 - x^2/b^2 = 1

theorem hyperbola_I_equation : 
  ∀ x y : ℝ, hyperbola_I x y ↔ y^2/36 - x^2/28 = 1 :=
sorry

-- Part II
def hyperbola_II (x y : ℝ) : Prop :=
  let c : ℝ := 6  -- distance from center to focus
  let a : ℝ := Real.sqrt (c^2/2)
  x^2/a^2 - y^2/a^2 = 1

theorem hyperbola_II_equation :
  ∀ x y : ℝ, hyperbola_II x y ↔ x^2/18 - y^2/18 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_I_equation_hyperbola_II_equation_l1314_131465


namespace NUMINAMATH_CALUDE_smallest_shift_is_sixty_l1314_131401

/-- A function with period 30 -/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x + 30) = g x

/-- The smallest positive shift for g(2x) -/
def smallest_shift (g : ℝ → ℝ) (b : ℝ) : Prop :=
  (b > 0) ∧
  (∀ x : ℝ, g (2*x + b) = g (2*x)) ∧
  (∀ c : ℝ, c > 0 → (∀ x : ℝ, g (2*x + c) = g (2*x)) → b ≤ c)

theorem smallest_shift_is_sixty (g : ℝ → ℝ) :
  periodic_function g → smallest_shift g 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_shift_is_sixty_l1314_131401


namespace NUMINAMATH_CALUDE_max_value_product_l1314_131464

theorem max_value_product (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27/8 ∧
  ∃ x y z, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧
    (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) = 27/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1314_131464


namespace NUMINAMATH_CALUDE_sin_negative_690_degrees_l1314_131415

theorem sin_negative_690_degrees : Real.sin ((-690 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_690_degrees_l1314_131415


namespace NUMINAMATH_CALUDE_first_number_value_l1314_131427

theorem first_number_value (a b c : ℕ) : 
  a + b + c = 500 → 
  (b = 200 ∨ c = 200 ∨ a = 200) → 
  b = 2 * c → 
  c = 100 → 
  a = 200 := by
sorry

end NUMINAMATH_CALUDE_first_number_value_l1314_131427


namespace NUMINAMATH_CALUDE_circle_area_and_circumference_l1314_131497

/-- Given a circle with diameter endpoints at (1,1) and (8,6), prove its area and circumference -/
theorem circle_area_and_circumference :
  let C : ℝ × ℝ := (1, 1)
  let D : ℝ × ℝ := (8, 6)
  let diameter := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let radius := diameter / 2
  let area := π * radius^2
  let circumference := 2 * π * radius
  (area = 74 * π / 4) ∧ (circumference = Real.sqrt 74 * π) := by
  sorry


end NUMINAMATH_CALUDE_circle_area_and_circumference_l1314_131497


namespace NUMINAMATH_CALUDE_ashton_initial_boxes_l1314_131442

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 14

/-- The number of pencils Ashton gave to his brother -/
def pencils_given : ℕ := 6

/-- The number of pencils Ashton had left after giving some away -/
def pencils_left : ℕ := 22

/-- The number of boxes Ashton had initially -/
def initial_boxes : ℕ := 2

theorem ashton_initial_boxes :
  initial_boxes * pencils_per_box = pencils_left + pencils_given :=
sorry

end NUMINAMATH_CALUDE_ashton_initial_boxes_l1314_131442


namespace NUMINAMATH_CALUDE_rational_sum_problem_l1314_131476

theorem rational_sum_problem (a b c d : ℚ) 
  (h1 : b + c + d = -1)
  (h2 : a + c + d = -3)
  (h3 : a + b + d = 2)
  (h4 : a + b + c = 17) :
  a = 6 ∧ b = 8 ∧ c = 3 ∧ d = -12 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_problem_l1314_131476


namespace NUMINAMATH_CALUDE_sequence_formula_l1314_131474

/-- Given a sequence {a_n} where the sum of the first n terms S_n = 2^n - 1,
    prove that the general formula for the sequence is a_n = 2^(n-1) -/
theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 2^n - 1) : 
    ∀ n : ℕ, a n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l1314_131474


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1314_131444

/-- A regular decagon is a 10-sided polygon -/
def regular_decagon : ℕ := 10

/-- Number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ := choose n 4

theorem decagon_diagonal_intersections :
  interior_intersection_points regular_decagon = 210 :=
sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1314_131444


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1314_131410

-- Problem 1
theorem inequality_solution_1 : 
  {x : ℝ | -x^2 + 3*x + 10 < 0} = {x : ℝ | x > 5 ∨ x < -2} := by sorry

-- Problem 2
theorem inequality_solution_2 (a : ℝ) : 
  {x : ℝ | x^2 - 2*a*x + (a-1)*(a+1) ≤ 0} = {x : ℝ | a-1 ≤ x ∧ x ≤ a+1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1314_131410


namespace NUMINAMATH_CALUDE_linear_function_sum_l1314_131435

/-- A linear function f with specific properties -/
def f (x : ℝ) : ℝ := sorry

/-- The sum of f(2), f(4), ..., f(2n) -/
def sum_f (n : ℕ) : ℝ := sorry

theorem linear_function_sum :
  (f 0 = 1) →
  (∃ r : ℝ, f 1 * r = f 4 ∧ f 4 * r = f 13) →
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) →
  ∀ n : ℕ, sum_f n = n * (2 * n + 3) :=
sorry

end NUMINAMATH_CALUDE_linear_function_sum_l1314_131435


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1314_131425

def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

theorem parallel_vectors_k_value :
  (∃ (c : ℝ), c ≠ 0 ∧ vector_a = c • (vector_b k)) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1314_131425


namespace NUMINAMATH_CALUDE_candy_bar_problem_l1314_131455

theorem candy_bar_problem (fred : ℕ) (bob : ℕ) (jacqueline : ℕ) : 
  fred = 12 →
  bob = fred + 6 →
  jacqueline = 10 * (fred + bob) →
  (40 : ℚ) / 100 * jacqueline = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_problem_l1314_131455


namespace NUMINAMATH_CALUDE_total_marbles_l1314_131436

theorem total_marbles (mary_marbles joan_marbles : ℕ) 
  (h1 : mary_marbles = 9) 
  (h2 : joan_marbles = 3) : 
  mary_marbles + joan_marbles = 12 := by
sorry

end NUMINAMATH_CALUDE_total_marbles_l1314_131436


namespace NUMINAMATH_CALUDE_cos_225_degrees_l1314_131493

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l1314_131493


namespace NUMINAMATH_CALUDE_distance_between_points_l1314_131446

/-- The distance between two points A and B, given the travel time and average speed -/
theorem distance_between_points (time : ℝ) (speed : ℝ) (h1 : time = 4.5) (h2 : speed = 80) :
  time * speed = 360 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1314_131446


namespace NUMINAMATH_CALUDE_A_D_independent_l1314_131419

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | ω.1 = 0}
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- Theorem statement
theorem A_D_independent : P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_A_D_independent_l1314_131419


namespace NUMINAMATH_CALUDE_max_intersection_length_l1314_131420

noncomputable section

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def M : Point := Unit.unit
def N : Point := Unit.unit
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit

-- Define the diameter and its length
def diameter (c : Circle) : ℝ := 2

-- Define the property that MN is a diameter
def is_diameter (c : Circle) (m n : Point) : Prop := True

-- Define A as the midpoint of the semicircular arc
def is_midpoint_arc (c : Circle) (m n a : Point) : Prop := True

-- Define the length of MB
def length_MB : ℝ := 4/7

-- Define C as a point on the other semicircular arc
def on_other_arc (c : Circle) (m n c : Point) : Prop := True

-- Define the intersections of MN with AC and BC
def intersection_AC_MN (c : Circle) (m n a c : Point) : Point := Unit.unit
def intersection_BC_MN (c : Circle) (m n b c : Point) : Point := Unit.unit

-- Define the length of the line segment formed by the intersections
def length_intersections (p q : Point) : ℝ := 0

-- Theorem statement
theorem max_intersection_length (c : Circle) :
  is_diameter c M N →
  is_midpoint_arc c M N A →
  length_MB = 4/7 →
  on_other_arc c M N C →
  ∃ (d : ℝ), d = 10 - 7 * Real.sqrt 3 ∧
    ∀ (V W : Point),
      V = intersection_AC_MN c M N A C →
      W = intersection_BC_MN c M N B C →
      length_intersections V W ≤ d :=
sorry

end

end NUMINAMATH_CALUDE_max_intersection_length_l1314_131420


namespace NUMINAMATH_CALUDE_allison_june_uploads_l1314_131403

/-- Calculates the total number of video hours uploaded by Allison in June -/
def total_video_hours (initial_rate : ℕ) (days_in_june : ℕ) (initial_period : ℕ) : ℕ :=
  let doubled_rate := 2 * initial_rate
  let remaining_period := days_in_june - initial_period
  initial_rate * initial_period + doubled_rate * remaining_period

/-- Theorem stating that Allison's total uploaded video hours in June is 450 -/
theorem allison_june_uploads :
  total_video_hours 10 30 15 = 450 := by
  sorry

end NUMINAMATH_CALUDE_allison_june_uploads_l1314_131403


namespace NUMINAMATH_CALUDE_train_length_calculation_l1314_131421

theorem train_length_calculation (platform_crossing_time platform_length signal_crossing_time : ℝ) 
  (h1 : platform_crossing_time = 27)
  (h2 : platform_length = 150.00000000000006)
  (h3 : signal_crossing_time = 18) :
  ∃ train_length : ℝ, train_length = 300.0000000000001 ∧
    platform_crossing_time * (train_length / signal_crossing_time) = train_length + platform_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1314_131421


namespace NUMINAMATH_CALUDE_most_likely_outcome_l1314_131405

def n : ℕ := 5

def p_boy : ℚ := 1/2
def p_girl : ℚ := 1/2

def prob_all_same_gender : ℚ := p_boy^n + p_girl^n

def prob_three_two : ℚ := (Nat.choose n 3) * (p_boy^3 * p_girl^2 + p_boy^2 * p_girl^3)

theorem most_likely_outcome :
  prob_three_two > prob_all_same_gender ∧
  prob_three_two = 5/16 :=
sorry

end NUMINAMATH_CALUDE_most_likely_outcome_l1314_131405


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l1314_131458

theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l1314_131458


namespace NUMINAMATH_CALUDE_box_volume_increase_l1314_131448

/-- 
Given a rectangular box with dimensions l, w, and h satisfying:
1. Volume is 5400 cubic inches
2. Surface area is 1920 square inches
3. Sum of edge lengths is 240 inches
Prove that increasing each dimension by 2 inches results in a volume of 7568 cubic inches
-/
theorem box_volume_increase (l w h : ℝ) 
  (hvolume : l * w * h = 5400)
  (harea : 2 * (l * w + w * h + h * l) = 1920)
  (hedge : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7568 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l1314_131448


namespace NUMINAMATH_CALUDE_parabola_and_circle_equations_l1314_131481

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Represents a parabola in the form y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Represents a circle in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The line l: y = x - 1 -/
def l : Line := { m := 1, b := -1 }

/-- The focus of parabola C -/
def focus : Point := { x := 1, y := 0 }

/-- The condition that line l passes through the focus of parabola C -/
def line_passes_through_focus (l : Line) (f : Point) : Prop :=
  f.y = l.m * f.x + l.b

/-- The theorem to be proved -/
theorem parabola_and_circle_equations 
  (C : Parabola) 
  (h_p_pos : C.p > 0) 
  (h_focus : line_passes_through_focus l focus) : 
  (∃ (Q : Circle), 
    (∀ (x y : ℝ), y^2 = 4*x ↔ y^2 = 2*C.p*x) ∧ 
    (∀ (x y : ℝ), (x - Q.h)^2 + (y - Q.k)^2 = Q.r^2 ↔ (x - 3)^2 + (y - 2)^2 = 16)) := 
  sorry

end NUMINAMATH_CALUDE_parabola_and_circle_equations_l1314_131481


namespace NUMINAMATH_CALUDE_apple_boxes_bought_l1314_131449

-- Define the variables
variable (cherry_price : ℝ) -- Price of one cherry
variable (apple_price : ℝ) -- Price of one apple
variable (cherry_size : ℝ) -- Size of one cherry
variable (apple_size : ℝ) -- Size of one apple
variable (cherries_per_box : ℕ) -- Number of cherries in a box

-- Define the conditions
axiom price_relation : 2 * cherry_price = 3 * apple_price
axiom size_relation : apple_size = 12 * cherry_size
axiom box_size_equality : cherries_per_box * cherry_size = cherries_per_box * apple_size

-- Define the theorem
theorem apple_boxes_bought (h : cherries_per_box > 0) :
  (cherries_per_box * cherry_price) / apple_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_bought_l1314_131449


namespace NUMINAMATH_CALUDE_table_tennis_cost_calculation_l1314_131441

/-- Represents the cost calculation for table tennis equipment purchase options. -/
def TableTennisCost (x : ℕ) : Prop :=
  (x > 20) →
  let racketPrice : ℕ := 80
  let ballPrice : ℕ := 20
  let racketCount : ℕ := 20
  let option1Cost : ℕ := racketPrice * racketCount + ballPrice * (x - racketCount)
  let option2Cost : ℕ := ((racketPrice * racketCount + ballPrice * x) * 9) / 10
  (option1Cost = 20 * x + 1200) ∧ (option2Cost = 18 * x + 1440)

/-- Theorem stating the cost calculation for both options is correct for any valid x. -/
theorem table_tennis_cost_calculation (x : ℕ) : TableTennisCost x := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_cost_calculation_l1314_131441


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1314_131445

theorem chess_tournament_games (n : ℕ) (h : n = 18) : 
  n * (n - 1) = 306 → 2 * (n * (n - 1)) = 612 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1314_131445


namespace NUMINAMATH_CALUDE_shape_relations_l1314_131453

/-- Given symbols representing geometric shapes with the following relations:
    - triangle + triangle = star
    - circle = square + square
    - triangle = circle + circle + circle + circle
    Prove that star divided by square equals 16 -/
theorem shape_relations (triangle star circle square : ℕ) 
    (h1 : triangle + triangle = star)
    (h2 : circle = square + square)
    (h3 : triangle = circle + circle + circle + circle) :
  star / square = 16 := by sorry

end NUMINAMATH_CALUDE_shape_relations_l1314_131453


namespace NUMINAMATH_CALUDE_optimal_carriages_and_passengers_l1314_131413

/-- The daily round trips as a function of the number of carriages -/
def daily_trips (x : ℕ) : ℝ :=
  -3 * x + 28

/-- The daily operating number of passengers as a function of the number of carriages -/
def daily_passengers (x : ℕ) : ℝ :=
  110 * x * daily_trips x

/-- The set of valid carriage numbers -/
def valid_carriages : Set ℕ :=
  {x | 1 ≤ x ∧ x ≤ 9}

theorem optimal_carriages_and_passengers :
  ∀ x ∈ valid_carriages,
    daily_passengers 5 ≥ daily_passengers x ∧
    daily_passengers 5 = 14300 :=
by sorry

end NUMINAMATH_CALUDE_optimal_carriages_and_passengers_l1314_131413


namespace NUMINAMATH_CALUDE_unique_valid_number_l1314_131447

def is_valid_number (n : ℕ) : Prop :=
  ∃ (p q r s t u : ℕ),
    0 ≤ p ∧ p ≤ 9 ∧
    0 ≤ q ∧ q ≤ 9 ∧
    0 ≤ r ∧ r ≤ 9 ∧
    0 ≤ s ∧ s ≤ 9 ∧
    0 ≤ t ∧ t ≤ 9 ∧
    0 ≤ u ∧ u ≤ 9 ∧
    n = p * 10^7 + q * 10^6 + 7 * 10^5 + 8 * 10^4 + r * 10^3 + s * 10^2 + t * 10 + u ∧
    n % 17 = 0 ∧
    n % 19 = 0 ∧
    p + q + r + s = t + u

theorem unique_valid_number :
  ∃! n, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l1314_131447


namespace NUMINAMATH_CALUDE_platform_length_l1314_131475

/-- Calculates the length of a platform given train specifications -/
theorem platform_length 
  (train_length : ℝ) 
  (time_tree : ℝ) 
  (time_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 210) : 
  ∃ platform_length : ℝ, platform_length = 900 ∧ 
  time_platform = (train_length + platform_length) / (train_length / time_tree) :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1314_131475


namespace NUMINAMATH_CALUDE_k_range_l1314_131452

/-- The logarithm function to base 10 -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The equation lg kx = 2 lg (x+1) has only one real root -/
def has_unique_root (k : ℝ) : Prop :=
  ∃! x : ℝ, log10 (k * x) = 2 * log10 (x + 1)

/-- The range of k values for which the equation has only one real root -/
theorem k_range : ∀ k : ℝ, has_unique_root k ↔ k = 4 ∨ k < 0 := by sorry

end NUMINAMATH_CALUDE_k_range_l1314_131452


namespace NUMINAMATH_CALUDE_expected_value_proof_l1314_131439

/-- The expected value of winning (6-n)^2 dollars when rolling a fair 6-sided die -/
def expected_value : ℚ := 55 / 6

/-- A fair 6-sided die -/
def die : Finset ℕ := Finset.range 6

/-- The probability of rolling any number on a fair 6-sided die -/
def prob (n : ℕ) : ℚ := 1 / 6

/-- The winnings for rolling n on the die -/
def winnings (n : ℕ) : ℚ := (6 - n) ^ 2

theorem expected_value_proof :
  Finset.sum die (λ n => prob n * winnings n) = expected_value :=
sorry

end NUMINAMATH_CALUDE_expected_value_proof_l1314_131439


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1314_131471

/-- Proves that a boat traveling 45 miles upstream in 5 hours and 45 miles downstream in 3 hours has a speed of 12 mph in still water -/
theorem boat_speed_in_still_water : 
  ∀ (upstream_speed downstream_speed : ℝ),
  upstream_speed = 45 / 5 →
  downstream_speed = 45 / 3 →
  ∃ (boat_speed current_speed : ℝ),
  boat_speed - current_speed = upstream_speed ∧
  boat_speed + current_speed = downstream_speed ∧
  boat_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1314_131471


namespace NUMINAMATH_CALUDE_darryl_book_count_l1314_131482

theorem darryl_book_count :
  ∀ (d l m : ℕ),
  l + 3 = m →
  m = 2 * d →
  d + m + l = 97 →
  d = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_darryl_book_count_l1314_131482


namespace NUMINAMATH_CALUDE_total_items_eq_1900_l1314_131426

/-- The number of rows of pencils and crayons. -/
def num_rows : ℕ := 19

/-- The number of pencils in each row. -/
def pencils_per_row : ℕ := 57

/-- The number of crayons in each row. -/
def crayons_per_row : ℕ := 43

/-- The total number of pencils and crayons. -/
def total_items : ℕ := num_rows * (pencils_per_row + crayons_per_row)

theorem total_items_eq_1900 : total_items = 1900 := by
  sorry

end NUMINAMATH_CALUDE_total_items_eq_1900_l1314_131426


namespace NUMINAMATH_CALUDE_ellipse_foci_l1314_131473

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2 / 9 + x^2 / 4 = 1

-- Define the foci coordinates
def foci_coordinates : Set (ℝ × ℝ) := {(0, Real.sqrt 5), (0, -Real.sqrt 5)}

-- Theorem statement
theorem ellipse_foci : 
  ∀ (x y : ℝ), ellipse x y → (x, y) ∈ foci_coordinates ↔ 
  (x = 0 ∧ y = Real.sqrt 5) ∨ (x = 0 ∧ y = -Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l1314_131473


namespace NUMINAMATH_CALUDE_unique_solution_l1314_131463

def problem (a : ℕ) (x : ℕ) : Prop :=
  a > 0 ∧
  x > 0 ∧
  x < a ∧
  71 * x + 69 * (a - x) = 3480

theorem unique_solution :
  ∃! a x, problem a x ∧ a = 50 ∧ x = 15 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1314_131463


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1314_131491

theorem complex_magnitude_problem : ∃ (T : ℂ), 
  T = (1 + Complex.I)^19 + (1 + Complex.I)^19 - (1 - Complex.I)^19 ∧ 
  Complex.abs T = Real.sqrt 5 * 2^(19/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1314_131491


namespace NUMINAMATH_CALUDE_zero_points_count_l1314_131484

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem zero_points_count 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f π) 
  (h_shifted : ∀ x, f (x - π) = f (x + π) ∧ f (x - π) = f x) : 
  ∃ (S : Finset ℝ), S.card = 7 ∧ (∀ x ∈ S, f x = 0 ∧ x ∈ Set.Icc 0 8) ∧
    (∀ x ∈ Set.Icc 0 8, f x = 0 → x ∈ S) :=
sorry

end NUMINAMATH_CALUDE_zero_points_count_l1314_131484


namespace NUMINAMATH_CALUDE_sales_not_notebooks_or_markers_l1314_131433

/-- The percentage of sales that are not notebooks or markers -/
def other_sales_percentage (notebook_percentage marker_percentage : ℝ) : ℝ :=
  100 - (notebook_percentage + marker_percentage)

/-- Theorem stating that the percentage of sales not consisting of notebooks or markers is 33% -/
theorem sales_not_notebooks_or_markers :
  other_sales_percentage 42 25 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sales_not_notebooks_or_markers_l1314_131433


namespace NUMINAMATH_CALUDE_chessboard_pawn_placement_l1314_131414

/-- Represents a chess board configuration -/
structure ChessBoard :=
  (size : Nat)
  (pawns : Nat)

/-- Calculates the number of ways to place distinct pawns on a chess board -/
def placementWays (board : ChessBoard) : Nat :=
  (Nat.factorial board.size) * (Nat.factorial board.size)

/-- Theorem: The number of ways to place 5 distinct pawns on a 5x5 chess board,
    such that no row and no column contains more than one pawn, is 14400 -/
theorem chessboard_pawn_placement :
  let board : ChessBoard := ⟨5, 5⟩
  placementWays board = 14400 := by
  sorry

#eval placementWays ⟨5, 5⟩

end NUMINAMATH_CALUDE_chessboard_pawn_placement_l1314_131414


namespace NUMINAMATH_CALUDE_coordinates_in_new_basis_l1314_131488

open LinearAlgebra

variable {𝕜 : Type*} [Field 𝕜]
variable {E : Type*} [AddCommGroup E] [Module 𝕜 E]

/-- Given a vector space E over a field 𝕜, and two bases e and e' of E, 
    prove that the coordinates of a vector x in the new basis e' are {0, 1, -1} -/
theorem coordinates_in_new_basis 
  (e : Basis (Fin 3) 𝕜 E) 
  (e' : Basis (Fin 3) 𝕜 E) 
  (x : E) :
  (∀ i : Fin 3, e' i = 
    if i = 0 then e 0 + 2 • (e 2)
    else if i = 1 then e 1 + e 2
    else -(e 0) - (e 1) - 2 • (e 2)) →
  (x = e 0 + 2 • (e 1) + 3 • (e 2)) →
  (∃ a b c : 𝕜, x = a • (e' 0) + b • (e' 1) + c • (e' 2) ∧ a = 0 ∧ b = 1 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_in_new_basis_l1314_131488
