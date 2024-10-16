import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l1461_146144

theorem complex_equation_solution (a : ℝ) : (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1461_146144


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_two_l1461_146109

theorem sum_reciprocals_equals_two
  (a b c d : ℝ)
  (ω : ℂ)
  (ha : a ≠ -1)
  (hb : b ≠ -1)
  (hc : c ≠ -1)
  (hd : d ≠ -1)
  (hω1 : ω^4 = 1)
  (hω2 : ω ≠ 1)
  (h_sum : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω^2) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_two_l1461_146109


namespace NUMINAMATH_CALUDE_problem_solution_l1461_146192

def f (x : ℝ) : ℝ := |2*x + 2| + |x - 3|

theorem problem_solution :
  (∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, f x ≥ m) ∧
    (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = m →
      1 / (a + b) + 1 / (b + c) + 1 / (a + c) ≥ 9 / (2 * m))) ∧
  {x : ℝ | f x ≤ 5} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1461_146192


namespace NUMINAMATH_CALUDE_milk_expense_l1461_146107

/-- Proves that the amount spent on milk is 1500, given the total expenses
    excluding milk and savings, the savings amount, and the savings rate. -/
theorem milk_expense (total_expenses_excl_milk : ℕ) (savings : ℕ) (savings_rate : ℚ) :
  total_expenses_excl_milk = 16500 →
  savings = 2000 →
  savings_rate = 1/10 →
  (total_expenses_excl_milk + savings) / (1 - savings_rate) - (total_expenses_excl_milk + savings) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_milk_expense_l1461_146107


namespace NUMINAMATH_CALUDE_girls_in_study_group_l1461_146112

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of students in the study group -/
def total_students : ℕ := 6

/-- The number of students to be selected -/
def selected_students : ℕ := 2

/-- The number of ways to select 2 students with at least 1 girl -/
def ways_with_girl : ℕ := 12

theorem girls_in_study_group :
  ∃ (n : ℕ), n ≤ total_students ∧
  choose total_students selected_students - choose (total_students - n) selected_students = ways_with_girl ∧
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_girls_in_study_group_l1461_146112


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l1461_146187

/-- The orthocenter of a triangle is the point where all three altitudes intersect. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Given three points A, B, and C in 3D space, this theorem states that
    the orthocenter of the triangle formed by these points is (2, 3, 4). -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, -1, 2)
  let C : ℝ × ℝ × ℝ := (1, 6, 5)
  orthocenter A B C = (2, 3, 4) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l1461_146187


namespace NUMINAMATH_CALUDE_council_vote_difference_l1461_146168

theorem council_vote_difference (total_members : ℕ) 
  (initial_for initial_against : ℕ) 
  (revote_for revote_against : ℕ) : 
  total_members = 500 →
  initial_for + initial_against = total_members →
  initial_against > initial_for →
  revote_for + revote_against = total_members →
  revote_for - revote_against = 3 * (initial_against - initial_for) →
  revote_for = (13 * initial_against) / 12 →
  revote_for - initial_for = 40 := by
sorry

end NUMINAMATH_CALUDE_council_vote_difference_l1461_146168


namespace NUMINAMATH_CALUDE_sequence_non_positive_l1461_146134

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k-1) - 2 * a k + a (k+1) ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l1461_146134


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_three_l1461_146121

theorem sum_of_roots_equals_three : ∃ (P Q : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ (x = P ∨ x = Q)) ∧ 
  P + Q = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_three_l1461_146121


namespace NUMINAMATH_CALUDE_shortest_distance_C1_C2_l1461_146162

/-- The curve C1 in Cartesian coordinates -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2

/-- The curve C2 as a line in Cartesian coordinates -/
def C2 (x y : ℝ) : Prop := x + y = 4

/-- The shortest distance between C1 and C2 -/
theorem shortest_distance_C1_C2 :
  ∃ (p q : ℝ × ℝ), C1 p.1 p.2 ∧ C2 q.1 q.2 ∧
    ∀ (p' q' : ℝ × ℝ), C1 p'.1 p'.2 → C2 q'.1 q'.2 →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_C1_C2_l1461_146162


namespace NUMINAMATH_CALUDE_H_range_l1461_146169

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_range : Set.range H = Set.Icc (-5) 5 := by sorry

end NUMINAMATH_CALUDE_H_range_l1461_146169


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l1461_146119

-- Define the function f
def f (x : ℝ) : ℝ := x^6 - 2*x^4 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l1461_146119


namespace NUMINAMATH_CALUDE_smallest_block_volume_l1461_146129

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if the given dimensions satisfy the problem conditions. -/
def satisfiesConditions (d : BlockDimensions) : Prop :=
  (d.length - 1) * (d.width - 1) * (d.height - 1) = 288 ∧
  (d.length + d.width + d.height) % 10 = 0

/-- The volume of the block given its dimensions. -/
def blockVolume (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The theorem stating the smallest possible value of N. -/
theorem smallest_block_volume :
  ∃ (d : BlockDimensions), satisfiesConditions d ∧
    blockVolume d = 455 ∧
    ∀ (d' : BlockDimensions), satisfiesConditions d' → blockVolume d' ≥ 455 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l1461_146129


namespace NUMINAMATH_CALUDE_time_to_return_is_45_minutes_l1461_146155

/-- Represents a hiker's journey on a trail --/
structure HikerJourney where
  rate : Real  -- Minutes per kilometer
  initialDistance : Real  -- Kilometers hiked east initially
  totalDistance : Real  -- Total kilometers hiked east before turning back
  
/-- Calculates the time required for a hiker to return to the start of the trail --/
def timeToReturn (journey : HikerJourney) : Real :=
  sorry

/-- Theorem stating that for the given conditions, the time to return is 45 minutes --/
theorem time_to_return_is_45_minutes (journey : HikerJourney) 
  (h1 : journey.rate = 10)
  (h2 : journey.initialDistance = 2.5)
  (h3 : journey.totalDistance = 3.5) :
  timeToReturn journey = 45 := by
  sorry

end NUMINAMATH_CALUDE_time_to_return_is_45_minutes_l1461_146155


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1461_146175

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1461_146175


namespace NUMINAMATH_CALUDE_lemon_cost_lemon_cost_is_fifty_cents_l1461_146117

/-- The cost of the lemon in Hannah's apple pie recipe -/
theorem lemon_cost (servings : ℕ) (apple_pounds : ℝ) (apple_price : ℝ) 
  (crust_price : ℝ) (butter_price : ℝ) (serving_price : ℝ) : ℝ :=
  let total_cost := servings * serving_price
  let ingredients_cost := apple_pounds * apple_price + crust_price + butter_price
  total_cost - ingredients_cost

/-- The lemon in Hannah's apple pie recipe costs $0.50 -/
theorem lemon_cost_is_fifty_cents : 
  lemon_cost 8 2 2 2 1.5 1 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_lemon_cost_lemon_cost_is_fifty_cents_l1461_146117


namespace NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l1461_146142

theorem square_difference_of_quadratic_solutions : 
  ∀ α β : ℝ, 
  (α^2 = 2*α + 1) → 
  (β^2 = 2*β + 1) → 
  (α ≠ β) → 
  (α - β)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l1461_146142


namespace NUMINAMATH_CALUDE_jack_keeps_half_deer_weight_l1461_146163

/-- Given Jack's hunting habits and the amount of deer he keeps, prove that he keeps half of the total deer weight caught each year. -/
theorem jack_keeps_half_deer_weight 
  (hunts_per_month : ℕ) 
  (hunting_season_months : ℕ) 
  (deers_per_hunt : ℕ) 
  (deer_weight : ℕ) 
  (weight_kept : ℕ) 
  (h1 : hunts_per_month = 6)
  (h2 : hunting_season_months = 3)
  (h3 : deers_per_hunt = 2)
  (h4 : deer_weight = 600)
  (h5 : weight_kept = 10800) : 
  weight_kept / (hunts_per_month * hunting_season_months * deers_per_hunt * deer_weight) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_jack_keeps_half_deer_weight_l1461_146163


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1461_146174

theorem quadratic_equation_real_roots (m n : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁^2 - (m + n)*x₁ + m*n = 0 ∧ x₂^2 - (m + n)*x₂ + m*n = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1461_146174


namespace NUMINAMATH_CALUDE_diagonal_sequence_theorem_l1461_146126

/-- A convex polygon with 1994 sides and 997 diagonals -/
structure ConvexPolygon :=
  (sides : ℕ)
  (diagonals : ℕ)
  (is_convex : Bool)
  (sides_eq : sides = 1994)
  (diagonals_eq : diagonals = 997)
  (convex : is_convex = true)

/-- The length of a diagonal is the number of sides in the smaller part of the perimeter it divides -/
def diagonal_length (p : ConvexPolygon) (d : ℕ) : ℕ := sorry

/-- Each vertex has exactly one diagonal emanating from it -/
def one_diagonal_per_vertex (p : ConvexPolygon) : Prop := sorry

/-- The sequence of diagonal lengths in decreasing order -/
def diagonal_sequence (p : ConvexPolygon) : List ℕ := sorry

theorem diagonal_sequence_theorem (p : ConvexPolygon) 
  (h : one_diagonal_per_vertex p) :
  (∃ (seq : List ℕ), diagonal_sequence p = seq ∧ 
    seq.length = 997 ∧
    seq.count 3 = 991 ∧ 
    seq.count 2 = 6) ∧
  ¬(∃ (seq : List ℕ), diagonal_sequence p = seq ∧ 
    seq.length = 997 ∧
    seq.count 8 = 4 ∧ 
    seq.count 6 = 985 ∧ 
    seq.count 3 = 8) :=
sorry

end NUMINAMATH_CALUDE_diagonal_sequence_theorem_l1461_146126


namespace NUMINAMATH_CALUDE_peaches_sold_to_relatives_l1461_146122

theorem peaches_sold_to_relatives (total_peaches : ℕ) 
                                  (peaches_to_friends : ℕ) 
                                  (price_to_friends : ℚ)
                                  (price_to_relatives : ℚ)
                                  (peaches_kept : ℕ)
                                  (total_sold : ℕ)
                                  (total_earnings : ℚ) :
  total_peaches = 15 →
  peaches_to_friends = 10 →
  price_to_friends = 2 →
  price_to_relatives = 5/4 →
  peaches_kept = 1 →
  total_sold = 14 →
  total_earnings = 25 →
  total_peaches = peaches_to_friends + (total_sold - peaches_to_friends) + peaches_kept →
  total_earnings = peaches_to_friends * price_to_friends + 
                   (total_sold - peaches_to_friends) * price_to_relatives →
  (total_sold - peaches_to_friends) = 4 := by
sorry

end NUMINAMATH_CALUDE_peaches_sold_to_relatives_l1461_146122


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l1461_146131

/-- Represents a collection of lecture notes --/
structure LectureNotes where
  total_sheets : ℕ
  pages_per_sheet : ℕ
  borrowed_sheets : ℕ

/-- Calculates the average of page numbers on remaining sheets --/
def averageOfRemainingSheets (notes : LectureNotes) : ℚ :=
  let total_pages := notes.total_sheets * notes.pages_per_sheet
  let remaining_sheets := notes.total_sheets - notes.borrowed_sheets
  let first_remaining_page := notes.borrowed_sheets * notes.pages_per_sheet + 1
  let last_page := total_pages
  ((first_remaining_page + last_page) * remaining_sheets) / (2 * remaining_sheets)

/-- The main theorem to prove --/
theorem borrowed_sheets_theorem (notes : LectureNotes) :
  notes.total_sheets = 36 ∧
  notes.pages_per_sheet = 2 ∧
  notes.borrowed_sheets = 17 →
  averageOfRemainingSheets notes = 40 := by
  sorry


end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l1461_146131


namespace NUMINAMATH_CALUDE_festival_end_day_l1461_146149

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem stating that 45 days after a 5-day festival starting on Tuesday is Wednesday -/
theorem festival_end_day (startDay : DayOfWeek) 
  (h : startDay = DayOfWeek.Tuesday) : 
  advanceDays startDay (5 + 45) = DayOfWeek.Wednesday := by
  sorry


end NUMINAMATH_CALUDE_festival_end_day_l1461_146149


namespace NUMINAMATH_CALUDE_ratio_equality_sometimes_l1461_146164

/-- An isosceles triangle with side lengths A and base B -/
structure IsoscelesTriangle where
  A : ℝ
  B : ℝ
  h : ℝ  -- height
  K₁ : ℝ  -- area
  β : ℝ  -- base angle
  h_eq : h = Real.sqrt (A^2 - (B/2)^2)
  K₁_eq : K₁ = (1/2) * B * h
  B_ne_A : B ≠ A

/-- An equilateral triangle with side length a -/
structure EquilateralTriangle where
  a : ℝ
  p : ℝ  -- perimeter
  k₁ : ℝ  -- area
  α : ℝ  -- angle
  p_eq : p = 3 * a
  k₁_eq : k₁ = (a^2 * Real.sqrt 3) / 4
  α_eq : α = π / 3

/-- The main theorem stating that the ratio equality holds sometimes but not always -/
theorem ratio_equality_sometimes (iso : IsoscelesTriangle) (equi : EquilateralTriangle)
    (h_eq : iso.A = equi.a) :
    ∃ (iso₁ : IsoscelesTriangle) (equi₁ : EquilateralTriangle),
      iso₁.h / equi₁.p = iso₁.K₁ / equi₁.k₁ ∧
    ∃ (iso₂ : IsoscelesTriangle) (equi₂ : EquilateralTriangle),
      iso₂.h / equi₂.p ≠ iso₂.K₁ / equi₂.k₁ := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_sometimes_l1461_146164


namespace NUMINAMATH_CALUDE_age_problem_l1461_146194

theorem age_problem (a b : ℕ) : 
  (a : ℚ) / b = 5 / 3 →
  ((a + 2) : ℚ) / (b + 2) = 3 / 2 →
  b = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1461_146194


namespace NUMINAMATH_CALUDE_fencing_required_l1461_146166

/-- Calculates the fencing required for a rectangular field with given area and one uncovered side. -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 720 ∧ uncovered_side = 20 →
  uncovered_side + 2 * (area / uncovered_side) = 92 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l1461_146166


namespace NUMINAMATH_CALUDE_octagon_diagonals_l1461_146160

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l1461_146160


namespace NUMINAMATH_CALUDE_complex_number_equivalence_l1461_146154

theorem complex_number_equivalence : (10 * Complex.I) / (1 - 2 * Complex.I) = -4 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equivalence_l1461_146154


namespace NUMINAMATH_CALUDE_infinite_fixed_points_l1461_146132

def is_special_function (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (f n - n < 2021) ∧ (f^[f n] n = n)

theorem infinite_fixed_points (f : ℕ → ℕ) (hf : is_special_function f) :
  Set.Infinite {n : ℕ | f n = n} :=
sorry

end NUMINAMATH_CALUDE_infinite_fixed_points_l1461_146132


namespace NUMINAMATH_CALUDE_problem_l1461_146128

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

theorem problem (a : ℝ) (h1 : a > 1) (h2 : f a 1 = 3) :
  (f a 2 = 7) ∧
  (∀ x₁ x₂, 0 ≤ x₂ ∧ x₂ < x₁ → f a x₁ > f a x₂) ∧
  (∀ m x, 0 ≤ x ∧ x ≤ 1 → 
    f a (2*x) - m * f a x ≥ min (2 - 2*m) (min (-m^2/4 - 2) (7 - 3*m))) := by
  sorry

end NUMINAMATH_CALUDE_problem_l1461_146128


namespace NUMINAMATH_CALUDE_unattainable_y_value_l1461_146113

theorem unattainable_y_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃y : ℝ, y = (2 - x) / (3 * x + 4) ↔ y = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l1461_146113


namespace NUMINAMATH_CALUDE_smallest_n_squared_l1461_146120

theorem smallest_n_squared (n : ℕ+) : 
  (∃ x y z : ℕ+, n.val^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ↔ 
  n.val ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_squared_l1461_146120


namespace NUMINAMATH_CALUDE_solar_panel_installation_time_l1461_146178

/-- Calculates the number of hours needed to install solar panels given the costs of various items --/
def solar_panel_installation_hours (land_acres : ℕ) (land_cost_per_acre : ℕ) 
  (house_cost : ℕ) (cow_count : ℕ) (cow_cost : ℕ) (chicken_count : ℕ) 
  (chicken_cost : ℕ) (solar_panel_hourly_rate : ℕ) (solar_panel_equipment_fee : ℕ) 
  (total_cost : ℕ) : ℕ :=
  let land_cost := land_acres * land_cost_per_acre
  let cows_cost := cow_count * cow_cost
  let chickens_cost := chicken_count * chicken_cost
  let costs_before_solar := land_cost + house_cost + cows_cost + chickens_cost
  let solar_panel_total_cost := total_cost - costs_before_solar
  let installation_cost := solar_panel_total_cost - solar_panel_equipment_fee
  installation_cost / solar_panel_hourly_rate

theorem solar_panel_installation_time : 
  solar_panel_installation_hours 30 20 120000 20 1000 100 5 100 6000 147700 = 6 := by
  sorry

end NUMINAMATH_CALUDE_solar_panel_installation_time_l1461_146178


namespace NUMINAMATH_CALUDE_congruence_problem_l1461_146173

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 151 ∧ (100 * n) % 151 = 93 % 151 → n % 151 = 29 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1461_146173


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1461_146153

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

/-- Given vectors a and b, if they are parallel, then the x-coordinate of b is 2 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (h : parallel a b) :
    a = (1, 2) → b.1 = x → b.2 = 4 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1461_146153


namespace NUMINAMATH_CALUDE_luna_budget_l1461_146106

/-- Luna's monthly budget calculation -/
theorem luna_budget (house_rental food phone : ℝ) : 
  food = 0.6 * house_rental →
  phone = 0.1 * food →
  house_rental + food = 240 →
  house_rental + food + phone = 249 := by
  sorry

end NUMINAMATH_CALUDE_luna_budget_l1461_146106


namespace NUMINAMATH_CALUDE_opposite_sides_of_y_axis_l1461_146140

/-- Given points A and B on opposite sides of the y-axis, with B on the right side,
    prove that the x-coordinate of A is negative. -/
theorem opposite_sides_of_y_axis (a : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (a, 1) ∧ B = (2, a) ∧ 
   (A.1 < 0 ∧ B.1 > 0) ∧ -- A and B are on opposite sides of the y-axis
   B.1 > 0) →              -- B is on the right side of the y-axis
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_of_y_axis_l1461_146140


namespace NUMINAMATH_CALUDE_crocodile_coloring_l1461_146189

theorem crocodile_coloring (m n : ℕ) (h_m : m > 0) (h_n : n > 0) :
  ∃ f : ℤ × ℤ → Bool,
    ∀ x y : ℤ, f (x, y) ≠ f (x + m, y + n) ∧ f (x, y) ≠ f (x + n, y + m) := by
  sorry

end NUMINAMATH_CALUDE_crocodile_coloring_l1461_146189


namespace NUMINAMATH_CALUDE_roses_in_vase_l1461_146171

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 6

/-- The number of roses Mary added to the vase -/
def added_roses : ℕ := 16

/-- The total number of roses in the vase after Mary added more -/
def total_roses : ℕ := initial_roses + added_roses

theorem roses_in_vase : total_roses = 22 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1461_146171


namespace NUMINAMATH_CALUDE_sum_multiple_of_three_l1461_146152

theorem sum_multiple_of_three (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_sum_multiple_of_three_l1461_146152


namespace NUMINAMATH_CALUDE_least_divisible_by_10_to_15_divided_by_26_l1461_146177

theorem least_divisible_by_10_to_15_divided_by_26 :
  let j := Nat.lcm 10 (Nat.lcm 11 (Nat.lcm 12 (Nat.lcm 13 (Nat.lcm 14 15))))
  ∀ k : ℕ, (∀ i ∈ Finset.range 6, k % (i + 10) = 0) → k ≥ j
  → j / 26 = 2310 := by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_10_to_15_divided_by_26_l1461_146177


namespace NUMINAMATH_CALUDE_equation_solution_l1461_146116

theorem equation_solution : ∃! x : ℚ, (3*x^2 + 2*x + 1) / (x - 1) = 3*x + 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1461_146116


namespace NUMINAMATH_CALUDE_cos_36_degrees_l1461_146100

theorem cos_36_degrees : Real.cos (36 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l1461_146100


namespace NUMINAMATH_CALUDE_lab_budget_theorem_l1461_146150

def lab_budget_problem (total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost : ℚ) 
  (min_instruments : ℕ) : Prop :=
  let total_spent := flask_cost + test_tube_cost + safety_gear_cost + chemical_cost + min_instrument_cost
  total_budget = 750 ∧
  flask_cost = 200 ∧
  test_tube_cost = 2/3 * flask_cost ∧
  safety_gear_cost = 1/2 * test_tube_cost ∧
  chemical_cost = 3/4 * flask_cost ∧
  min_instrument_cost ≥ 50 ∧
  min_instruments ≥ 10 ∧
  total_budget - total_spent = 150

theorem lab_budget_theorem :
  ∃ (total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost : ℚ) 
    (min_instruments : ℕ),
  lab_budget_problem total_budget flask_cost test_tube_cost safety_gear_cost chemical_cost min_instrument_cost min_instruments :=
by
  sorry

end NUMINAMATH_CALUDE_lab_budget_theorem_l1461_146150


namespace NUMINAMATH_CALUDE_square_side_increase_l1461_146135

theorem square_side_increase (s : ℝ) (h : s > 0) :
  let new_area := s^2 * (1 + 0.3225)
  let new_side := s * (1 + 0.15)
  new_side^2 = new_area := by sorry

end NUMINAMATH_CALUDE_square_side_increase_l1461_146135


namespace NUMINAMATH_CALUDE_inequality_proof_l1461_146182

theorem inequality_proof (a b c : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n ≥ 2) (habc : a * b * c = 1) :
  (a / (b + c)^(1 / n : ℝ)) + (b / (c + a)^(1 / n : ℝ)) + (c / (a + b)^(1 / n : ℝ)) ≥ 3 / (2^(1 / n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1461_146182


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1461_146141

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1461_146141


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1461_146198

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1461_146198


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1461_146115

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℚ)
  (h_geo : geometric_sequence a)
  (h_1 : a 0 = 32)
  (h_2 : a 1 = -48)
  (h_3 : a 2 = 72)
  (h_4 : a 3 = -108)
  (h_5 : a 4 = 162) :
  ∃ r : ℚ, r = -3/2 ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1461_146115


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1461_146161

/-- Given vectors a and b in ℝ², if a-b is perpendicular to ma+b, then m = 1/4 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (2, 1))
  (h2 : b = (1, -1))
  (h3 : (a.1 - b.1, a.2 - b.2) • (m * a.1 + b.1, m * a.2 + b.2) = 0) :
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1461_146161


namespace NUMINAMATH_CALUDE_percentage_problem_l1461_146185

theorem percentage_problem : 
  ∀ x : ℝ, (120 : ℝ) = 1.5 * x → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1461_146185


namespace NUMINAMATH_CALUDE_nine_integer_lengths_l1461_146110

-- Define the right triangle DEF
def triangle_DEF (DE EF : ℝ) : Prop := DE = 24 ∧ EF = 25

-- Define the function to count integer lengths
def count_integer_lengths (DE EF : ℝ) : ℕ :=
  -- Implementation details are omitted as per instructions
  sorry

-- Theorem statement
theorem nine_integer_lengths :
  ∀ DE EF : ℝ, triangle_DEF DE EF → count_integer_lengths DE EF = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_integer_lengths_l1461_146110


namespace NUMINAMATH_CALUDE_impossible_table_fill_l1461_146137

/-- Represents a table filled with natural numbers -/
def Table (n : ℕ) := Fin n → Fin n → ℕ

/-- Checks if a row in the table satisfies the product condition -/
def RowSatisfiesCondition (row : Fin n → ℕ) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ row i * row j = row k

/-- Checks if all elements in the table are distinct and within the range 1 to n^2 -/
def ValidTable (t : Table n) : Prop :=
  (∀ i j, 1 ≤ t i j ∧ t i j ≤ n^2) ∧
  (∀ i₁ j₁ i₂ j₂, (i₁, j₁) ≠ (i₂, j₂) → t i₁ j₁ ≠ t i₂ j₂)

/-- The main theorem stating the impossibility of filling the table -/
theorem impossible_table_fill (n : ℕ) (h : n ≥ 3) :
  ¬∃ (t : Table n), ValidTable t ∧ (∀ i : Fin n, RowSatisfiesCondition (t i)) :=
sorry

end NUMINAMATH_CALUDE_impossible_table_fill_l1461_146137


namespace NUMINAMATH_CALUDE_triangle_side_length_l1461_146188

-- Define the triangle
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define 60 degree angle
def SixtyDegreeAngle (A B C : ℝ × ℝ) : Prop :=
  ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2))^2 =
  3 * ((B.1 - A.1)^2 + (B.2 - A.2)^2) * ((C.1 - A.1)^2 + (C.2 - A.2)^2) / 4

-- Define inscribed circle radius
def InscribedCircleRadius (A B C : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), 
    (O.1 - A.1)^2 + (O.2 - A.2)^2 = r^2 ∧
    (O.1 - B.1)^2 + (O.2 - B.2)^2 = r^2 ∧
    (O.1 - C.1)^2 + (O.2 - C.2)^2 = r^2

-- Theorem statement
theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : RightAngle B A C)
  (h3 : SixtyDegreeAngle A B C)
  (h4 : InscribedCircleRadius A B C 8) :
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (24 * Real.sqrt 3 + 24)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1461_146188


namespace NUMINAMATH_CALUDE_unique_valid_n_l1461_146197

def is_valid_n (n : ℕ+) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ : ℕ+),
    d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧
    (∀ (d : ℕ+), d ∣ n → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d₄ < d) ∧
    n = d₁^2 + d₂^2 + d₃^2 + d₄^2

theorem unique_valid_n :
  ∃! (n : ℕ+), is_valid_n n ∧ n = 130 := by sorry

end NUMINAMATH_CALUDE_unique_valid_n_l1461_146197


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l1461_146186

-- Define the set M
def M : Set ℝ := {0, 1, 2}

-- Define the set N
def N : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Theorem statement
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l1461_146186


namespace NUMINAMATH_CALUDE_julia_bill_ratio_l1461_146151

/-- Proves the ratio of Julia's Sunday miles to Bill's Sunday miles -/
theorem julia_bill_ratio (bill_sunday : ℕ) (bill_saturday : ℕ) (julia_sunday : ℕ) :
  bill_sunday = 10 →
  bill_sunday = bill_saturday + 4 →
  bill_sunday + bill_saturday + julia_sunday = 36 →
  julia_sunday = 2 * bill_sunday :=
by sorry

end NUMINAMATH_CALUDE_julia_bill_ratio_l1461_146151


namespace NUMINAMATH_CALUDE_james_oranges_l1461_146138

theorem james_oranges :
  ∀ (o : ℕ),
    o ≤ 7 →
    (∃ (a : ℕ), a + o = 7 ∧ (65 * o + 40 * a) % 100 = 0) →
    o = 4 :=
by sorry

end NUMINAMATH_CALUDE_james_oranges_l1461_146138


namespace NUMINAMATH_CALUDE_age_difference_proof_l1461_146118

theorem age_difference_proof (son_age : ℕ) (man_age : ℕ) : son_age = 26 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1461_146118


namespace NUMINAMATH_CALUDE_probability_even_distinct_digits_l1461_146148

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i ≠ j → digits.get i ≠ digits.get j

def count_favorable_outcomes : ℕ := 7 * 8 * 7 * 5

theorem probability_even_distinct_digits :
  (count_favorable_outcomes : ℚ) / (9999 - 2000 + 1 : ℚ) = 49 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_distinct_digits_l1461_146148


namespace NUMINAMATH_CALUDE_expression_always_positive_l1461_146157

theorem expression_always_positive (x y : ℝ) : x^2 - 4*x*y + 6*y^2 - 4*y + 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_positive_l1461_146157


namespace NUMINAMATH_CALUDE_team_leader_selection_l1461_146139

theorem team_leader_selection (n : ℕ) (h : n = 5) : n * (n - 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_team_leader_selection_l1461_146139


namespace NUMINAMATH_CALUDE_miles_french_horns_l1461_146143

/-- Represents the number of musical instruments Miles owns -/
structure MilesInstruments where
  trumpets : ℕ
  guitars : ℕ
  trombones : ℕ
  frenchHorns : ℕ

/-- Represents Miles' body parts relevant to the problem -/
structure MilesAnatomy where
  fingers : ℕ
  hands : ℕ
  head : ℕ

theorem miles_french_horns 
  (anatomy : MilesAnatomy)
  (instruments : MilesInstruments)
  (h1 : anatomy.fingers = 10)
  (h2 : anatomy.hands = 2)
  (h3 : anatomy.head = 1)
  (h4 : instruments.trumpets = anatomy.fingers - 3)
  (h5 : instruments.guitars = anatomy.hands + 2)
  (h6 : instruments.trombones = anatomy.head + 2)
  (h7 : instruments.trumpets + instruments.guitars + instruments.trombones + instruments.frenchHorns = 17)
  (h8 : instruments.frenchHorns = instruments.guitars - 1) :
  instruments.frenchHorns = 3 := by
  sorry

#check miles_french_horns

end NUMINAMATH_CALUDE_miles_french_horns_l1461_146143


namespace NUMINAMATH_CALUDE_amy_biking_distance_l1461_146104

def miles_yesterday : ℕ := 12

def miles_today (y : ℕ) : ℕ := 2 * y - 3

def total_miles (y t : ℕ) : ℕ := y + t

theorem amy_biking_distance :
  total_miles miles_yesterday (miles_today miles_yesterday) = 33 :=
by sorry

end NUMINAMATH_CALUDE_amy_biking_distance_l1461_146104


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l1461_146176

theorem subtraction_of_decimals : (3.156 : ℝ) - (1.029 : ℝ) = 2.127 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l1461_146176


namespace NUMINAMATH_CALUDE_lattice_right_triangles_with_specific_incenter_l1461_146199

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A right triangle with vertices O, A, and B, where O is the origin and the right angle -/
structure LatticeRightTriangle where
  A : LatticePoint
  B : LatticePoint

/-- The incenter of a right triangle -/
def incenter (t : LatticeRightTriangle) : ℚ × ℚ :=
  let a : ℚ := t.A.x
  let b : ℚ := t.B.y
  let c : ℚ := (a^2 + b^2).sqrt
  ((a + b - c) / 2, (a + b - c) / 2)

theorem lattice_right_triangles_with_specific_incenter :
  ∃ (n : ℕ), n > 0 ∧
  ∃ (triangles : Finset LatticeRightTriangle),
    triangles.card = n ∧
    ∀ t ∈ triangles, incenter t = (2015, 14105) := by
  sorry

end NUMINAMATH_CALUDE_lattice_right_triangles_with_specific_incenter_l1461_146199


namespace NUMINAMATH_CALUDE_different_color_picks_count_l1461_146193

/-- Represents a card color -/
inductive CardColor
| Red
| Black
| Colorless

/-- Represents the deck composition -/
structure Deck :=
  (red_cards : Nat)
  (black_cards : Nat)
  (jokers : Nat)

/-- The number of ways to pick two different cards of different colors -/
def different_color_picks (d : Deck) : Nat :=
  -- Red-Black or Black-Red
  2 * d.red_cards * d.black_cards +
  -- Colorless-Red or Colorless-Black
  2 * d.jokers * (d.red_cards + d.black_cards) +
  -- Red-Colorless or Black-Colorless
  2 * (d.red_cards + d.black_cards) * d.jokers

/-- The theorem to be proved -/
theorem different_color_picks_count :
  let d : Deck := { red_cards := 26, black_cards := 26, jokers := 2 }
  different_color_picks d = 1508 := by
  sorry

end NUMINAMATH_CALUDE_different_color_picks_count_l1461_146193


namespace NUMINAMATH_CALUDE_max_trailing_zeros_l1461_146125

/-- Given three natural numbers whose sum is 1003, the maximum number of trailing zeros in their product is 7 -/
theorem max_trailing_zeros (a b c : ℕ) (h_sum : a + b + c = 1003) : 
  (∃ (n : ℕ), a * b * c = n * 10^7 ∧ n % 10 ≠ 0) ∧ 
  ¬(∃ (m : ℕ), a * b * c = m * 10^8) :=
sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_l1461_146125


namespace NUMINAMATH_CALUDE_cube_triangle_areas_sum_l1461_146108

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side_length : ℝ)
  (is_2x2x2 : side_length = 2)

/-- A triangle formed by vertices of the cube -/
structure CubeTriangle :=
  (vertices : Fin 3 → Fin 8)

/-- The area of a triangle formed by vertices of the cube -/
noncomputable def triangle_area (c : Cube) (t : CubeTriangle) : ℝ :=
  sorry

/-- The sum of areas of all triangles formed by vertices of the cube -/
noncomputable def sum_of_triangle_areas (c : Cube) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem cube_triangle_areas_sum (c : Cube) :
  ∃ (m n p : ℕ), sum_of_triangle_areas c = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 121 :=
sorry

end NUMINAMATH_CALUDE_cube_triangle_areas_sum_l1461_146108


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1461_146130

-- Define the set M
def M (a : ℝ) := {x : ℝ | x^2 + (a-4)*x - (a+1)*(2*a-3) < 0}

-- State the theorem
theorem solution_set_characterization (a : ℝ) :
  (0 ∈ M a) →
  ((a < -1 ∨ a > 3/2) ∧
   (a < -1 → M a = Set.Ioo (a+1) (3-2*a)) ∧
   (a > 3/2 → M a = Set.Ioo (3-2*a) (a+1))) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1461_146130


namespace NUMINAMATH_CALUDE_no_infinite_sequence_with_sqrt_property_l1461_146196

theorem no_infinite_sequence_with_sqrt_property :
  ¬ (∃ (a : ℕ → ℕ), ∀ (n : ℕ), a (n + 2) = a (n + 1) + Real.sqrt (a (n + 1) + a n)) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_with_sqrt_property_l1461_146196


namespace NUMINAMATH_CALUDE_right_triangle_circle_ratio_l1461_146184

theorem right_triangle_circle_ratio (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let R := c / 2
  let r := (a + b - c) / 2
  R / r = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_circle_ratio_l1461_146184


namespace NUMINAMATH_CALUDE_square_formation_for_12_and_15_l1461_146156

/-- Given n sticks with lengths 1, 2, ..., n, determine if a square can be formed
    or the minimum number of sticks to be broken in half to form a square. -/
def minSticksToBreak (n : ℕ) : ℕ :=
  let totalLength := n * (n + 1) / 2
  if totalLength % 4 = 0 then 0
  else
    let targetLength := (totalLength / 4 + 1) * 4
    (targetLength - totalLength + 1) / 2

theorem square_formation_for_12_and_15 :
  minSticksToBreak 12 = 2 ∧ minSticksToBreak 15 = 0 := by
  sorry


end NUMINAMATH_CALUDE_square_formation_for_12_and_15_l1461_146156


namespace NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l1461_146158

/-- Given a ratio of milk to flour for pizza dough, calculate the amount of milk needed for a specific amount of flour. -/
theorem pizza_dough_milk_calculation 
  (milk_base : ℚ)  -- Base amount of milk in mL
  (flour_base : ℚ) -- Base amount of flour in mL
  (flour_total : ℚ) -- Total amount of flour to be used in mL
  (h1 : milk_base = 50)  -- Condition 1: Base milk amount
  (h2 : flour_base = 250) -- Condition 1: Base flour amount
  (h3 : flour_total = 750) -- Condition 2: Total flour amount
  : (flour_total / flour_base) * milk_base = 150 := by
  sorry

end NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l1461_146158


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1461_146191

theorem product_of_three_numbers (x y z : ℝ) : 
  x + y + z = 30 → 
  x = 3 * (y + z) → 
  y = 6 * z → 
  x * y * z = 7762.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1461_146191


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l1461_146147

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x₀ : ℝ, |x₀| ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l1461_146147


namespace NUMINAMATH_CALUDE_darnel_sprint_jog_difference_l1461_146190

theorem darnel_sprint_jog_difference : 
  let sprint_distance : ℝ := 0.88
  let jog_distance : ℝ := 0.75
  sprint_distance - jog_distance = 0.13 := by sorry

end NUMINAMATH_CALUDE_darnel_sprint_jog_difference_l1461_146190


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1461_146146

theorem shopkeeper_profit (C : ℝ) (h : C > 0) : 
  ∃ N : ℝ, N > 0 ∧ 12 * C + 0.2 * (N * C) = N * C ∧ N = 15 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1461_146146


namespace NUMINAMATH_CALUDE_height_difference_l1461_146159

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height_m : ℝ := 324

/-- The height of the Eiffel Tower in feet -/
def eiffel_tower_height_ft : ℝ := 1063

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height_m : ℝ := 830

/-- The height of the Burj Khalifa in feet -/
def burj_khalifa_height_ft : ℝ := 2722

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters and feet -/
theorem height_difference :
  (burj_khalifa_height_m - eiffel_tower_height_m = 506) ∧
  (burj_khalifa_height_ft - eiffel_tower_height_ft = 1659) := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1461_146159


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1461_146127

/-- Given two real numbers x and y, where x ≠ 0, y ≠ 0, and x ≠ ±y, 
    prove that the given complex expression simplifies to (x-y)^(1/3) / (x+y) -/
theorem complex_expression_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y ∧ x ≠ -y) :
  let numerator := (x^9 - x^6*y^3)^(1/3) - y^2 * ((8*x^6/y^3 - 8*x^3))^(1/3) + 
                   x*y^3 * (y^3 - y^6/x^3)^(1/2)
  let denominator := x^(8/3)*(x^2 - 2*y^2) + (x^2*y^12)^(1/3)
  numerator / denominator = (x-y)^(1/3) / (x+y) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1461_146127


namespace NUMINAMATH_CALUDE_gcd_143_98_l1461_146180

theorem gcd_143_98 : Nat.gcd 143 98 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_143_98_l1461_146180


namespace NUMINAMATH_CALUDE_parabola_translation_correct_l1461_146136

/-- Represents a parabola in the form y = a(x - h)² + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The original parabola y = 2x² --/
def original_parabola : Parabola := { a := 2, h := 0, k := 0 }

/-- The transformed parabola y = 2(x+4)² + 1 --/
def transformed_parabola : Parabola := { a := 2, h := -4, k := 1 }

/-- Represents a translation in 2D space --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The translation that should transform the original parabola to the transformed parabola --/
def correct_translation : Translation := { dx := -4, dy := 1 }

/-- Applies a translation to a parabola --/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a, h := p.h - t.dx, k := p.k + t.dy }

theorem parabola_translation_correct :
  apply_translation original_parabola correct_translation = transformed_parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_correct_l1461_146136


namespace NUMINAMATH_CALUDE_no_quaint_two_digit_integers_l1461_146105

def is_quaint (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ∃ (a b : ℕ), n = 10 * a + b ∧ a > 0 ∧ b < 10 ∧ n = a + b^3

theorem no_quaint_two_digit_integers : ¬∃ (n : ℕ), is_quaint n := by
  sorry

end NUMINAMATH_CALUDE_no_quaint_two_digit_integers_l1461_146105


namespace NUMINAMATH_CALUDE_gloin_tells_truth_l1461_146181

/-- Represents the type of dwarf: either a knight or a liar -/
inductive DwarfType
  | Knight
  | Liar

/-- Represents a dwarf with their position and type -/
structure Dwarf :=
  (position : Nat)
  (type : DwarfType)

/-- The statement made by a dwarf -/
def statement (d : Dwarf) (line : List Dwarf) : Prop :=
  match d.position with
  | 10 => ∃ (right : Dwarf), right.position > d.position ∧ right.type = DwarfType.Knight
  | _ => ∃ (left : Dwarf), left.position < d.position ∧ left.type = DwarfType.Knight

/-- The main theorem -/
theorem gloin_tells_truth 
  (line : List Dwarf) 
  (h_count : line.length = 10)
  (h_knight : ∃ d ∈ line, d.type = DwarfType.Knight)
  (h_statements : ∀ d ∈ line, d.position ≠ 10 → 
    (d.type = DwarfType.Knight ↔ statement d line))
  (h_gloin : ∃ gloin ∈ line, gloin.position = 10)
  : ∃ gloin ∈ line, gloin.position = 10 ∧ gloin.type = DwarfType.Knight :=
sorry

end NUMINAMATH_CALUDE_gloin_tells_truth_l1461_146181


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1461_146101

theorem perfect_square_trinomial (x y : ℝ) : x^2 + 4*y^2 - 4*x*y = (x - 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1461_146101


namespace NUMINAMATH_CALUDE_ball_box_arrangements_l1461_146172

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of distinct boxes -/
def num_boxes : ℕ := 4

/-- The number of arrangements when exactly one box remains empty -/
def arrangements_one_empty : ℕ := 144

/-- The number of arrangements when exactly two boxes remain empty -/
def arrangements_two_empty : ℕ := 84

/-- Theorem stating the correct number of arrangements for each case -/
theorem ball_box_arrangements :
  (∀ (n : ℕ), n = num_balls → n = num_boxes) →
  (arrangements_one_empty = 144 ∧ arrangements_two_empty = 84) := by
  sorry

end NUMINAMATH_CALUDE_ball_box_arrangements_l1461_146172


namespace NUMINAMATH_CALUDE_apple_pie_calculation_l1461_146133

theorem apple_pie_calculation (total_apples : ℕ) (unripe_apples : ℕ) (apples_per_pie : ℕ) 
  (h1 : total_apples = 34) 
  (h2 : unripe_apples = 6) 
  (h3 : apples_per_pie = 4) :
  (total_apples - unripe_apples) / apples_per_pie = 7 :=
by sorry

end NUMINAMATH_CALUDE_apple_pie_calculation_l1461_146133


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1461_146124

/-- The equation of the tangent line to the curve y = x³ + 2x at the point (1, 3) is 5x - y - 2 = 0 -/
theorem tangent_line_equation (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x = x^3 + 2*x →
  f x₀ = y₀ →
  x₀ = 1 →
  y₀ = 3 →
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 5*x - y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1461_146124


namespace NUMINAMATH_CALUDE_water_bill_calculation_l1461_146114

def weekly_income : ℝ := 500
def tax_rate : ℝ := 0.10
def tithe_rate : ℝ := 0.10
def remaining_amount : ℝ := 345

theorem water_bill_calculation :
  let after_tax := weekly_income * (1 - tax_rate)
  let after_tithe := after_tax - (weekly_income * tithe_rate)
  let water_bill := after_tithe - remaining_amount
  water_bill = 55 := by sorry

end NUMINAMATH_CALUDE_water_bill_calculation_l1461_146114


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1461_146145

theorem solve_linear_equation :
  ∃ x : ℝ, -2 * x - 7 = 7 * x + 2 ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1461_146145


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1461_146165

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1461_146165


namespace NUMINAMATH_CALUDE_beta_max_success_ratio_l1461_146167

/-- Represents a contestant's scores in a two-day math contest -/
structure ContestScores where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- The maximum possible two-day success ratio for Beta -/
def beta_max_ratio : ℚ := 407 / 600

theorem beta_max_success_ratio 
  (alpha : ContestScores)
  (beta : ContestScores)
  (h1 : alpha.day1_score = 180 ∧ alpha.day1_total = 350)
  (h2 : alpha.day2_score = 170 ∧ alpha.day2_total = 250)
  (h3 : beta.day1_score > 0 ∧ beta.day2_score > 0)
  (h4 : beta.day1_total + beta.day2_total = 600)
  (h5 : (beta.day1_score : ℚ) / beta.day1_total < (alpha.day1_score : ℚ) / alpha.day1_total)
  (h6 : (beta.day2_score : ℚ) / beta.day2_total < (alpha.day2_score : ℚ) / alpha.day2_total)
  (h7 : (alpha.day1_score + alpha.day2_score : ℚ) / (alpha.day1_total + alpha.day2_total) = 7 / 12) :
  (∀ b : ContestScores, 
    b.day1_score > 0 ∧ b.day2_score > 0 →
    b.day1_total + b.day2_total = 600 →
    (b.day1_score : ℚ) / b.day1_total < (alpha.day1_score : ℚ) / alpha.day1_total →
    (b.day2_score : ℚ) / b.day2_total < (alpha.day2_score : ℚ) / alpha.day2_total →
    (b.day1_score + b.day2_score : ℚ) / (b.day1_total + b.day2_total) ≤ beta_max_ratio) :=
by
  sorry

end NUMINAMATH_CALUDE_beta_max_success_ratio_l1461_146167


namespace NUMINAMATH_CALUDE_three_a_in_S_implies_a_in_S_l1461_146103

def S : Set ℤ := {n | ∃ x y : ℤ, n = x^2 + 2*y^2}

theorem three_a_in_S_implies_a_in_S (a : ℤ) (h : (3 * a) ∈ S) : a ∈ S := by
  sorry

end NUMINAMATH_CALUDE_three_a_in_S_implies_a_in_S_l1461_146103


namespace NUMINAMATH_CALUDE_total_garbage_accumulation_l1461_146183

/-- Represents the garbage accumulation problem in Daniel's neighborhood --/
def garbage_accumulation (collection_days_per_week : ℕ) (kg_per_collection : ℝ) (weeks : ℕ) (reduction_factor : ℝ) : ℝ :=
  let week1_accumulation := collection_days_per_week * kg_per_collection
  let week2_accumulation := week1_accumulation * reduction_factor
  week1_accumulation + week2_accumulation

/-- Theorem stating the total garbage accumulated over two weeks --/
theorem total_garbage_accumulation :
  garbage_accumulation 3 200 2 0.5 = 900 := by
  sorry

#eval garbage_accumulation 3 200 2 0.5

end NUMINAMATH_CALUDE_total_garbage_accumulation_l1461_146183


namespace NUMINAMATH_CALUDE_x_value_l1461_146123

def A : Set ℝ := {1, 2, 3}
def B (x : ℝ) : Set ℝ := {1, x}

theorem x_value (x : ℝ) : A ∪ B x = A → x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1461_146123


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_4sqrt2_l1461_146195

theorem sqrt_difference_equals_4sqrt2 :
  Real.sqrt (5 + 6 * Real.sqrt 2) - Real.sqrt (5 - 6 * Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_4sqrt2_l1461_146195


namespace NUMINAMATH_CALUDE_hexagonal_prism_vertices_l1461_146111

/-- A prism with hexagonal bases -/
structure HexagonalPrism where
  -- The number of sides in each base
  base_sides : ℕ
  -- The number of rectangular sides
  rect_sides : ℕ
  -- The total number of vertices
  vertices : ℕ

/-- Theorem: A hexagonal prism has 12 vertices -/
theorem hexagonal_prism_vertices (p : HexagonalPrism) 
  (h1 : p.base_sides = 6)
  (h2 : p.rect_sides = 6) : 
  p.vertices = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_vertices_l1461_146111


namespace NUMINAMATH_CALUDE_two_integer_pairs_satisfy_equation_l1461_146102

theorem two_integer_pairs_satisfy_equation :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 2 ∧ 
  (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_two_integer_pairs_satisfy_equation_l1461_146102


namespace NUMINAMATH_CALUDE_number_ordering_eight_ten_equals_four_fifteen_l1461_146170

theorem number_ordering : 8^10 < 3^20 ∧ 3^20 < 4^15 := by
  sorry

-- Additional theorem to establish the given condition
theorem eight_ten_equals_four_fifteen : 8^10 = 4^15 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_eight_ten_equals_four_fifteen_l1461_146170


namespace NUMINAMATH_CALUDE_increasing_square_neg_func_l1461_146179

/-- Given an increasing function f: ℝ → ℝ with f(x) < 0 for all x,
    the function g(x) = x^2 * f(x) is increasing on (-∞, 0) -/
theorem increasing_square_neg_func
  (f : ℝ → ℝ)
  (h_incr : ∀ x y, x < y → f x < f y)
  (h_neg : ∀ x, f x < 0) :
  ∀ x y, x < y → x < 0 → y < 0 → x^2 * f x < y^2 * f y :=
by sorry

end NUMINAMATH_CALUDE_increasing_square_neg_func_l1461_146179
