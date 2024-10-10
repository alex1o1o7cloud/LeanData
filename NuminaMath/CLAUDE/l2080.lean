import Mathlib

namespace decimal_difference_l2080_208020

-- Define the repeating decimal 0.72̄
def repeating_decimal : ℚ := 72 / 99

-- Define the terminating decimal 0.726
def terminating_decimal : ℚ := 726 / 1000

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 14 / 11000 := by
  sorry

end decimal_difference_l2080_208020


namespace arithmetic_sequence_value_l2080_208016

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- State the theorem
theorem arithmetic_sequence_value (a : ℚ) :
  (∃ (seq : ℕ → ℚ), is_arithmetic_sequence seq ∧ 
    seq 0 = a - 1 ∧ seq 1 = 2*a + 1 ∧ seq 2 = a + 4) →
  a = 1/2 :=
by sorry

end arithmetic_sequence_value_l2080_208016


namespace intersection_points_on_line_l2080_208049

-- Define the system of equations
def system (t x y : ℝ) : Prop :=
  (3 * x - 2 * y = 8 * t - 5) ∧
  (2 * x + 3 * y = 6 * t + 9) ∧
  (x + y = 2 * t + 1)

-- Theorem statement
theorem intersection_points_on_line :
  ∀ (t x y : ℝ), system t x y → y = -1/6 * x + 8/5 := by
  sorry

end intersection_points_on_line_l2080_208049


namespace overlap_area_is_three_quarters_l2080_208086

/-- Represents a point on a 3x3 grid --/
structure GridPoint where
  x : Fin 3
  y : Fin 3

/-- Represents a triangle on the grid --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The first triangle connecting top left, middle right, and bottom center --/
def triangle1 : GridTriangle :=
  { p1 := ⟨0, 0⟩, p2 := ⟨2, 1⟩, p3 := ⟨1, 2⟩ }

/-- The second triangle connecting top right, middle left, and bottom center --/
def triangle2 : GridTriangle :=
  { p1 := ⟨2, 0⟩, p2 := ⟨0, 1⟩, p3 := ⟨1, 2⟩ }

/-- Calculates the area of overlap between two triangles on the grid --/
def areaOfOverlap (t1 t2 : GridTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of overlap between the two specific triangles is 0.75 --/
theorem overlap_area_is_three_quarters :
  areaOfOverlap triangle1 triangle2 = 0.75 := by sorry

end overlap_area_is_three_quarters_l2080_208086


namespace line_perp_plane_sufficient_condition_l2080_208076

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- Theorem statement
theorem line_perp_plane_sufficient_condition 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α := by
  sorry

end line_perp_plane_sufficient_condition_l2080_208076


namespace inequality_proof_l2080_208082

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d := by
  sorry

end inequality_proof_l2080_208082


namespace polynomial_factorization_l2080_208034

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end polynomial_factorization_l2080_208034


namespace train_speed_problem_l2080_208036

theorem train_speed_problem (total_distance : ℝ) (speed_increase : ℝ) (distance_difference : ℝ) (time_difference : ℝ) :
  total_distance = 103 ∧ 
  speed_increase = 4 ∧ 
  distance_difference = 23 ∧ 
  time_difference = 1/4 →
  ∃ (initial_speed : ℝ) (initial_time : ℝ),
    initial_speed = 80 ∧
    initial_speed * initial_time + (initial_speed * initial_time + distance_difference) = total_distance ∧
    (initial_speed + speed_increase) * (initial_time + time_difference) = initial_speed * initial_time + distance_difference :=
by sorry

end train_speed_problem_l2080_208036


namespace framing_needed_photo_framing_proof_l2080_208055

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered photo -/
theorem framing_needed (original_width original_height : ℕ) 
  (enlargement_factor : ℕ) (border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  let perimeter_feet := (perimeter_inches + 11) / 12  -- Rounding up to nearest foot
  perimeter_feet

/-- Proves that a 5x7 inch photo, quadrupled and with 3-inch border, requires 10 feet of framing -/
theorem photo_framing_proof :
  framing_needed 5 7 4 3 = 10 := by
  sorry

end framing_needed_photo_framing_proof_l2080_208055


namespace hyperbola_asymptote_implies_m_eq_six_l2080_208024

/-- Given a hyperbola with equation x²/m - y²/6 = 1, where m is a real number,
    if one of its asymptotes is y = x, then m = 6. -/
theorem hyperbola_asymptote_implies_m_eq_six (m : ℝ) :
  (∃ (x y : ℝ), x^2 / m - y^2 / 6 = 1) →
  (∃ (x : ℝ), x = x) →
  m = 6 := by
sorry

end hyperbola_asymptote_implies_m_eq_six_l2080_208024


namespace sequence_a_properties_l2080_208070

def sequence_a (n : ℕ) : ℕ := sorry

theorem sequence_a_properties :
  (∀ n : ℕ, ∃ s t : ℕ, s < t ∧ sequence_a n = 2^s + 2^t) ∧
  (∀ n m : ℕ, n < m → sequence_a n < sequence_a m) ∧
  sequence_a 5 = 10 ∧
  (∃ n : ℕ, sequence_a n = 16640 ∧ n = 100) :=
by sorry

end sequence_a_properties_l2080_208070


namespace square_side_length_l2080_208039

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 9 →
  rectangle_width = 16 →
  square_side * square_side = rectangle_length * rectangle_width →
  square_side = 12 :=
by
  sorry

end square_side_length_l2080_208039


namespace paper_folding_cutting_perimeter_ratio_l2080_208060

theorem paper_folding_cutting_perimeter_ratio :
  let original_length : ℝ := 10
  let original_width : ℝ := 8
  let folded_length : ℝ := original_length / 2
  let folded_width : ℝ := original_width
  let small_rectangle_length : ℝ := folded_length
  let small_rectangle_width : ℝ := folded_width / 2
  let large_rectangle_length : ℝ := folded_length
  let large_rectangle_width : ℝ := folded_width
  let small_rectangle_perimeter : ℝ := 2 * (small_rectangle_length + small_rectangle_width)
  let large_rectangle_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  small_rectangle_perimeter / large_rectangle_perimeter = 9 / 13 := by
  sorry

end paper_folding_cutting_perimeter_ratio_l2080_208060


namespace quadratic_equation_roots_ratio_l2080_208028

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x = 3*y ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → 
  k = 18.75 := by
sorry

end quadratic_equation_roots_ratio_l2080_208028


namespace product_ratio_theorem_l2080_208097

theorem product_ratio_theorem (a b c d e f : ℝ) (X : ℝ) 
  (h1 : a * b * c = X)
  (h2 : b * c * d = X)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 0.25 := by
  sorry

end product_ratio_theorem_l2080_208097


namespace simplify_expression_l2080_208017

theorem simplify_expression (a b : ℝ) :
  (30 * a + 45 * b) + (15 * a + 40 * b) - (20 * a + 55 * b) + (5 * a - 10 * b) = 30 * a + 20 * b := by
  sorry

end simplify_expression_l2080_208017


namespace range_of_a_when_complement_subset_l2080_208021

-- Define the sets A, B, and C
def A : Set ℝ := {x | 0 < 2*x + 4 ∧ 2*x + 4 < 10}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0}

-- State the theorem
theorem range_of_a_when_complement_subset (a : ℝ) :
  (Set.univ \ (A ∪ B) : Set ℝ) ⊆ C a → -2 < a ∧ a < -4/3 :=
by sorry

end range_of_a_when_complement_subset_l2080_208021


namespace face_value_calculation_l2080_208072

/-- Given a banker's discount and true discount, calculate the face value (sum due) -/
def calculate_face_value (bankers_discount true_discount : ℚ) : ℚ :=
  (bankers_discount * true_discount) / (bankers_discount - true_discount)

/-- Theorem stating that given a banker's discount of 144 and a true discount of 120, the face value is 840 -/
theorem face_value_calculation (bankers_discount true_discount : ℚ) 
  (h1 : bankers_discount = 144)
  (h2 : true_discount = 120) :
  calculate_face_value bankers_discount true_discount = 840 := by
sorry

end face_value_calculation_l2080_208072


namespace situps_problem_l2080_208062

/-- Situps problem -/
theorem situps_problem (diana_rate hani_rate total_situps : ℕ) 
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : total_situps = 110) :
  diana_rate * (total_situps / (diana_rate + hani_rate)) = 40 := by
  sorry

#check situps_problem

end situps_problem_l2080_208062


namespace arithmetic_sequence_sum_l2080_208022

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 4 = 4) →
  (a 3 + a 5 = 10) →
  a 5 + a 7 = 22 :=
by
  sorry

end arithmetic_sequence_sum_l2080_208022


namespace inequality_proof_l2080_208002

theorem inequality_proof (x a b : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) :
  x^2 > a*b ∧ a*b > a^2 := by
  sorry

end inequality_proof_l2080_208002


namespace banana_permutations_eq_60_l2080_208023

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end banana_permutations_eq_60_l2080_208023


namespace smallest_positive_solution_l2080_208046

theorem smallest_positive_solution (x : ℝ) : 
  (x^4 - 40*x^2 + 400 = 0 ∧ x > 0 ∧ ∀ y > 0, y^4 - 40*y^2 + 400 = 0 → x ≤ y) → 
  x = 2 * Real.sqrt 5 := by
sorry

end smallest_positive_solution_l2080_208046


namespace sin_cos_difference_l2080_208009

open Real

theorem sin_cos_difference (α : ℝ) 
  (h : 2 * sin α * cos α = (sin α + cos α)^2 - 1)
  (h1 : (sin α + cos α)^2 - 1 = -24/25) : 
  |sin α - cos α| = 7/5 := by
sorry

end sin_cos_difference_l2080_208009


namespace train_crossing_time_l2080_208053

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 → 
  train_speed_kmh = 48 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
sorry

end train_crossing_time_l2080_208053


namespace percentage_boys_school_A_l2080_208018

theorem percentage_boys_school_A (total_boys : ℕ) (boys_A_not_science : ℕ) 
  (h1 : total_boys = 550)
  (h2 : boys_A_not_science = 77)
  (h3 : ∀ P : ℝ, P > 0 → P < 100 → 
    (P / 100) * total_boys * (70 / 100) = boys_A_not_science → P = 20) :
  ∃ P : ℝ, P > 0 ∧ P < 100 ∧ (P / 100) * total_boys * (70 / 100) = boys_A_not_science ∧ P = 20 := by
sorry

end percentage_boys_school_A_l2080_208018


namespace not_divisible_by_n_plus_4_l2080_208005

theorem not_divisible_by_n_plus_4 (n : ℕ) : ¬(∃ k : ℤ, n^2 + 8*n + 15 = (n + 4) * k) := by
  sorry

end not_divisible_by_n_plus_4_l2080_208005


namespace min_interval_number_bound_l2080_208033

/-- Represents a football tournament schedule -/
structure TournamentSchedule (n : ℕ) where
  -- n is the number of teams
  teams : Fin n
  -- schedule is a list of pairs of teams representing matches
  schedule : List (Fin n × Fin n)
  -- Each pair of teams plays exactly one match
  one_match : ∀ i j, i ≠ j → (i, j) ∈ schedule ∨ (j, i) ∈ schedule
  -- One match is scheduled each day
  one_per_day : schedule.length = (n.choose 2)

/-- The interval number between two matches of a team -/
def intervalNumber (s : TournamentSchedule n) (team : Fin n) : ℕ → ℕ → ℕ :=
  sorry

/-- The minimum interval number for a given schedule -/
def minIntervalNumber (s : TournamentSchedule n) : ℕ :=
  sorry

/-- Theorem: The minimum interval number does not exceed ⌊(n-3)/2⌋ -/
theorem min_interval_number_bound {n : ℕ} (hn : n ≥ 5) (s : TournamentSchedule n) :
  minIntervalNumber s ≤ (n - 3) / 2 :=
sorry

end min_interval_number_bound_l2080_208033


namespace min_sum_squares_min_sum_squares_at_m_one_l2080_208052

theorem min_sum_squares (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂ →
  (∃ D : ℝ, D ≥ 0 ∧ D = (m + 3)^2) →
  x₁ + x₂ = -(m + 1) →
  x₁ * x₂ = 2*m - 2 →
  x₁^2 + x₂^2 ≥ 4 :=
by sorry

theorem min_sum_squares_at_m_one (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂ →
  (∃ D : ℝ, D ≥ 0 ∧ D = (m + 3)^2) →
  x₁ + x₂ = -(m + 1) →
  x₁ * x₂ = 2*m - 2 →
  m = 1 →
  x₁^2 + x₂^2 = 4 :=
by sorry

end min_sum_squares_min_sum_squares_at_m_one_l2080_208052


namespace arithmetic_sequence_sum_proof_l2080_208030

theorem arithmetic_sequence_sum_proof : 
  let n : ℕ := 10
  let a : ℕ := 70
  let d : ℕ := 3
  let l : ℕ := 97
  3 * (n / 2 * (a + l)) = 2505 := by sorry

end arithmetic_sequence_sum_proof_l2080_208030


namespace long_distance_call_cost_per_minute_l2080_208042

/-- Calculates the cost per minute of a long distance call given the initial card value,
    call duration, and remaining credit. -/
def cost_per_minute (initial_value : ℚ) (call_duration : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_value - remaining_credit) / call_duration

/-- Proves that the cost per minute for long distance calls is $0.16 given the specified conditions. -/
theorem long_distance_call_cost_per_minute :
  let initial_value : ℚ := 30
  let call_duration : ℚ := 22
  let remaining_credit : ℚ := 26.48
  cost_per_minute initial_value call_duration remaining_credit = 0.16 := by
  sorry

#eval cost_per_minute 30 22 26.48

end long_distance_call_cost_per_minute_l2080_208042


namespace cylinder_volume_from_rectangle_l2080_208041

/-- The volume of a cylinder formed by rotating a rectangle about its lengthwise axis -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_positive : 0 < length) (width_positive : 0 < width) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  (length = 16 ∧ width = 8) → volume = 256 * π := by
  sorry

#check cylinder_volume_from_rectangle

end cylinder_volume_from_rectangle_l2080_208041


namespace constant_term_implies_a_value_l2080_208089

/-- 
Given that the constant term in the expansion of (x + a/x)(2x-1)^5 is 30, 
prove that a = 3.
-/
theorem constant_term_implies_a_value (a : ℝ) : 
  (∃ (f : ℝ → ℝ), 
    (∀ x, f x = (x + a/x) * (2*x - 1)^5) ∧ 
    (∃ c, ∀ x, f x = c + x * (f x - c) ∧ c = 30)) → 
  a = 3 := by
sorry

end constant_term_implies_a_value_l2080_208089


namespace worm_gnawed_pages_in_four_volumes_l2080_208074

/-- Represents a book volume with a specific number of pages -/
structure Volume :=
  (pages : ℕ)

/-- Represents a bookshelf with a list of volumes -/
structure Bookshelf :=
  (volumes : List Volume)

/-- Calculates the number of pages a worm gnaws through in a bookshelf -/
def wormGnawedPages (shelf : Bookshelf) : ℕ :=
  match shelf.volumes with
  | [] => 0
  | [_] => 0
  | v1 :: vs :: tail => 
    (vs.pages + (match tail with
                 | [_] => 0
                 | v3 :: _ => v3.pages
                 | _ => 0))

/-- Theorem stating the number of pages gnawed by the worm -/
theorem worm_gnawed_pages_in_four_volumes : 
  ∀ (shelf : Bookshelf),
    shelf.volumes.length = 4 →
    (∀ v ∈ shelf.volumes, v.pages = 200) →
    wormGnawedPages shelf = 400 := by
  sorry


end worm_gnawed_pages_in_four_volumes_l2080_208074


namespace precision_of_0_598_l2080_208054

/-- Represents the precision of a decimal number -/
inductive Precision
  | Whole
  | Tenth
  | Hundredth
  | Thousandth
  | TenThousandth
  deriving Repr

/-- Determines the precision of an approximate number -/
def precision (x : Float) : Precision :=
  match x.toString.split (· == '.') with
  | [_, decimal] =>
    match decimal.length with
    | 1 => Precision.Tenth
    | 2 => Precision.Hundredth
    | 3 => Precision.Thousandth
    | 4 => Precision.TenThousandth
    | _ => Precision.Whole
  | _ => Precision.Whole

theorem precision_of_0_598 :
  precision 0.598 = Precision.Thousandth := by
  sorry

end precision_of_0_598_l2080_208054


namespace sum_equals_fourteen_thousand_minus_m_l2080_208064

theorem sum_equals_fourteen_thousand_minus_m (M : ℕ) : 
  1989 + 1991 + 1993 + 1995 + 1997 + 1999 + 2001 = 14000 - M → M = 35 := by
  sorry

end sum_equals_fourteen_thousand_minus_m_l2080_208064


namespace circle_equation_l2080_208010

/-- Prove that a circle with given properties has the equation (x+5)^2 + y^2 = 5 -/
theorem circle_equation (a : ℝ) (h1 : a < 0) :
  let O' : ℝ × ℝ := (a, 0)
  let r : ℝ := Real.sqrt 5
  let line : ℝ × ℝ → Prop := λ p => p.1 + 2 * p.2 = 0
  (∀ p, line p → (p.1 - O'.1)^2 + (p.2 - O'.2)^2 = r^2) →
  (∀ x y, (x + 5)^2 + y^2 = 5 ↔ (x - a)^2 + y^2 = 5) :=
by sorry

end circle_equation_l2080_208010


namespace four_isosceles_triangles_l2080_208038

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the 5 triangles
def triangle1 : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 3⟩⟩
def triangle2 : Triangle := ⟨⟨4, 3⟩, ⟨4, 5⟩, ⟨6, 3⟩⟩
def triangle3 : Triangle := ⟨⟨1, 2⟩, ⟨4, 3⟩, ⟨7, 2⟩⟩
def triangle4 : Triangle := ⟨⟨5, 1⟩, ⟨4, 3⟩, ⟨6, 1⟩⟩
def triangle5 : Triangle := ⟨⟨3, 1⟩, ⟨4, 3⟩, ⟨5, 1⟩⟩

-- Theorem to prove
theorem four_isosceles_triangles :
  (isIsosceles triangle1) ∧
  (isIsosceles triangle2) ∧
  (isIsosceles triangle3) ∧
  (¬ isIsosceles triangle4) ∧
  (isIsosceles triangle5) :=
sorry

end four_isosceles_triangles_l2080_208038


namespace cards_distribution_l2080_208006

theorem cards_distribution (total_cards : Nat) (num_people : Nat) (h1 : total_cards = 100) (h2 : num_people = 15) :
  let cards_per_person := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_fewer := num_people - people_with_extra
  cards_per_person = 6 ∧ remainder = 10 ∧ people_with_fewer = 5 := by
  sorry

end cards_distribution_l2080_208006


namespace jose_profit_share_l2080_208013

/-- Calculates the share of profit for an investor based on their investment amount, 
    investment duration, and the total profit. -/
def calculate_profit_share (investment : ℕ) (duration : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_investment_months

theorem jose_profit_share :
  let tom_investment := 30000
  let tom_duration := 12
  let jose_investment := 45000
  let jose_duration := 10
  let total_profit := 72000
  let total_investment_months := tom_investment * tom_duration + jose_investment * jose_duration
  calculate_profit_share jose_investment jose_duration total_investment_months total_profit = 40000 := by
sorry

#eval calculate_profit_share 45000 10 810000 72000

end jose_profit_share_l2080_208013


namespace trig_expression_equals_negative_four_l2080_208015

theorem trig_expression_equals_negative_four :
  1 / Real.sin (70 * π / 180) - Real.sqrt 3 / Real.cos (70 * π / 180) = -4 := by
  sorry

end trig_expression_equals_negative_four_l2080_208015


namespace rectangle_circle_overlap_area_l2080_208075

/-- The area of overlap between a rectangle and a circle with shared center -/
theorem rectangle_circle_overlap_area 
  (rect_length : ℝ) 
  (rect_width : ℝ) 
  (circle_radius : ℝ) 
  (h_length : rect_length = 10) 
  (h_width : rect_width = 4) 
  (h_radius : circle_radius = 3) : 
  ∃ (overlap_area : ℝ), 
    overlap_area = 9 * Real.pi - 8 * Real.sqrt 5 + 12 :=
sorry

end rectangle_circle_overlap_area_l2080_208075


namespace complex_magnitude_problem_l2080_208026

theorem complex_magnitude_problem (z : ℂ) (h : (1 - z) / (1 + z) = Complex.I) :
  Complex.abs (1 + z) = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l2080_208026


namespace coordinates_of_C_l2080_208057

-- Define the points
def A : ℝ × ℝ := (11, 9)
def B : ℝ × ℝ := (2, -3)
def D : ℝ × ℝ := (-1, 3)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on BC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
  -- AD is perpendicular to BC
  (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0

-- Theorem statement
theorem coordinates_of_C :
  ∃ C : ℝ × ℝ, triangle_ABC C ∧ C = (-4, 9) := by sorry

end coordinates_of_C_l2080_208057


namespace opposite_pairs_l2080_208027

theorem opposite_pairs : 
  ((-3)^2 = -(-3^2)) ∧ 
  ((-3)^2 ≠ -(3^2)) ∧ 
  ((-2)^3 ≠ -(-2^3)) ∧ 
  (|-2|^3 ≠ -(|-2^3|)) := by
  sorry

end opposite_pairs_l2080_208027


namespace sum_simplification_l2080_208051

theorem sum_simplification :
  (296 + 297 + 298 + 299 + 1 + 2 + 3 + 4 = 1200) ∧
  (457 + 458 + 459 + 460 + 461 + 462 + 463 = 3220) := by
  sorry

end sum_simplification_l2080_208051


namespace no_power_of_three_l2080_208045

theorem no_power_of_three (a b : ℕ+) : ¬∃ k : ℕ, (15 * a + b) * (a + 15 * b) = 3^k := by
  sorry

end no_power_of_three_l2080_208045


namespace shaded_area_equals_unshaded_triangle_area_l2080_208080

/-- The area of the shaded region in a rectangular grid with an unshaded right triangle -/
theorem shaded_area_equals_unshaded_triangle_area (width height : ℝ) :
  width = 14 ∧ height = 5 →
  let grid_area := width * height
  let triangle_area := (1 / 2) * width * height
  let shaded_area := grid_area - triangle_area
  shaded_area = triangle_area := by sorry

end shaded_area_equals_unshaded_triangle_area_l2080_208080


namespace valid_combinations_l2080_208008

/-- The number of elective courses available -/
def total_courses : ℕ := 6

/-- The number of courses to be chosen -/
def courses_to_choose : ℕ := 2

/-- The number of pairs of courses that cannot be taken together -/
def conflicting_pairs : ℕ := 2

/-- The function to calculate the number of combinations -/
def calculate_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Theorem stating that the number of valid course combinations is 13 -/
theorem valid_combinations : 
  calculate_combinations total_courses courses_to_choose - conflicting_pairs = 13 := by
  sorry

end valid_combinations_l2080_208008


namespace simplify_expression_l2080_208011

theorem simplify_expression (x y : ℝ) :
  3 * x - 5 * (2 - x + y) + 4 * (1 - x - 2 * y) - 6 * (2 + 3 * x - y) = -14 * x - 7 * y - 18 := by
  sorry

end simplify_expression_l2080_208011


namespace linear_equation_solution_l2080_208040

theorem linear_equation_solution (k : ℝ) : 
  (-1 : ℝ) - k * 2 = 7 → k = -4 := by
  sorry

end linear_equation_solution_l2080_208040


namespace arithmetic_sequence_property_l2080_208000

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_9 < 0 and a_1 + a_18 > 0, then a_10 > 0 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a9_neg : a 9 < 0)
  (h_sum_pos : a 1 + a 18 > 0) : 
  a 10 > 0 := by
  sorry


end arithmetic_sequence_property_l2080_208000


namespace zeros_imply_sum_l2080_208096

/-- A quadratic function with zeros at -2 and 3 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- Theorem stating that if f has zeros at -2 and 3, then a + b = -7 -/
theorem zeros_imply_sum (a b : ℝ) :
  f a b (-2) = 0 ∧ f a b 3 = 0 → a + b = -7 :=
by sorry

end zeros_imply_sum_l2080_208096


namespace product_of_sums_equal_difference_of_powers_l2080_208003

theorem product_of_sums_equal_difference_of_powers : 
  (5 + 3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * 
  (5^16 + 3^16) * (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := by
  sorry

end product_of_sums_equal_difference_of_powers_l2080_208003


namespace phil_initial_books_l2080_208099

def initial_book_count (pages_per_book : ℕ) (books_lost : ℕ) (pages_left : ℕ) : ℕ :=
  (pages_left / pages_per_book) + books_lost

theorem phil_initial_books :
  initial_book_count 100 2 800 = 10 :=
by
  sorry

end phil_initial_books_l2080_208099


namespace parabola_coefficient_b_l2080_208092

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, 2p), where p ≠ 0,
    the coefficient b is equal to -2. -/
theorem parabola_coefficient_b (a b c p : ℝ) : p ≠ 0 →
  (∀ x, a * x^2 + b * x + c = (x - p)^2 / p + p) →
  a * 0^2 + b * 0 + c = 2 * p →
  b = -2 := by sorry

end parabola_coefficient_b_l2080_208092


namespace max_tshirts_purchased_l2080_208079

def tshirt_cost : ℚ := 915 / 100
def total_spent : ℚ := 201

theorem max_tshirts_purchased : 
  ⌊total_spent / tshirt_cost⌋ = 21 := by sorry

end max_tshirts_purchased_l2080_208079


namespace red_marble_fraction_l2080_208063

theorem red_marble_fraction (total : ℝ) (h_total_pos : total > 0) : 
  let blue := (2/3) * total
  let red := (1/3) * total
  let new_blue := 3 * blue
  let new_total := new_blue + red
  red / new_total = 1/7 := by
sorry

end red_marble_fraction_l2080_208063


namespace wrong_observation_value_l2080_208056

theorem wrong_observation_value (n : ℕ) (original_mean new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 30)
  (h3 : new_mean = 30.5) :
  ∃ (wrong_value correct_value : ℝ),
    (n : ℝ) * original_mean = (n - 1 : ℝ) * original_mean + wrong_value ∧
    (n : ℝ) * new_mean = (n - 1 : ℝ) * original_mean + correct_value ∧
    wrong_value = 73 :=
by
  sorry

end wrong_observation_value_l2080_208056


namespace certain_number_problem_l2080_208019

theorem certain_number_problem (x : ℝ) (h : x + 33 + 333 + 33.3 = 399.6) : x = 0.3 := by
  sorry

end certain_number_problem_l2080_208019


namespace peters_pizza_fraction_l2080_208048

theorem peters_pizza_fraction (total_slices : ℕ) (peters_solo_slices : ℕ) (shared_slices : ℕ) :
  total_slices = 16 →
  peters_solo_slices = 3 →
  shared_slices = 2 →
  (peters_solo_slices : ℚ) / total_slices + (shared_slices : ℚ) / (2 * total_slices) = 5 / 16 := by
  sorry

end peters_pizza_fraction_l2080_208048


namespace tan_two_beta_l2080_208066

theorem tan_two_beta (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (α - β) = 3) : 
  Real.tan (2 * β) = -1/7 := by
  sorry

end tan_two_beta_l2080_208066


namespace gcd_45_75_l2080_208032

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l2080_208032


namespace greatest_integer_fraction_inequality_l2080_208068

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end greatest_integer_fraction_inequality_l2080_208068


namespace equation_solution_l2080_208058

theorem equation_solution : ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ x = 2 := by
  sorry

end equation_solution_l2080_208058


namespace polynomial_expansion_equality_l2080_208031

/-- Proves that the expansion of (3y-2)*(5y^12+3y^11+5y^10+3y^9) equals 15y^13 - y^12 + 9y^11 - y^10 + 6y^9 for all real y. -/
theorem polynomial_expansion_equality (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^11 + 5*y^10 + 3*y^9) = 
  15*y^13 - y^12 + 9*y^11 - y^10 + 6*y^9 := by
  sorry

end polynomial_expansion_equality_l2080_208031


namespace doras_stickers_solve_l2080_208083

/-- The number of packs of stickers Dora gets -/
def doras_stickers (allowance : ℕ) (card_cost : ℕ) (sticker_box_cost : ℕ) : ℕ :=
  let total_money := 2 * allowance
  let remaining_money := total_money - card_cost
  let boxes_bought := remaining_money / sticker_box_cost
  boxes_bought / 2

theorem doras_stickers_solve :
  doras_stickers 9 10 2 = 2 := by
  sorry

#eval doras_stickers 9 10 2

end doras_stickers_solve_l2080_208083


namespace x_intercept_ratio_l2080_208091

/-- Two lines intersecting the y-axis at the same non-zero point -/
structure IntersectingLines where
  b : ℝ
  s : ℝ
  t : ℝ
  hb : b ≠ 0
  h1 : 0 = (5/2) * s + b
  h2 : 0 = (7/3) * t + b

/-- The ratio of x-intercepts is 14/15 -/
theorem x_intercept_ratio (l : IntersectingLines) : l.s / l.t = 14 / 15 := by
  sorry

#check x_intercept_ratio

end x_intercept_ratio_l2080_208091


namespace minimum_value_and_inequality_l2080_208050

def f (x m : ℝ) : ℝ := |x - m| + |x + 1|

theorem minimum_value_and_inequality {m a b c : ℝ} (h_min : ∀ x, f x m ≥ 4) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + 2*b + 3*c = m) :
  (m = -5 ∨ m = 3) ∧ (1/a + 1/(2*b) + 1/(3*c) ≥ 3) := by
  sorry

end minimum_value_and_inequality_l2080_208050


namespace log_23_between_consecutive_integers_sum_l2080_208043

theorem log_23_between_consecutive_integers_sum : ∃ (c d : ℤ), 
  c + 1 = d ∧ 
  (c : ℝ) < Real.log 23 / Real.log 10 ∧ 
  Real.log 23 / Real.log 10 < (d : ℝ) ∧ 
  c + d = 3 := by sorry

end log_23_between_consecutive_integers_sum_l2080_208043


namespace inequality_proof_l2080_208007

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) : 
  (1 / Real.sqrt (1/2 + a + a*b + a*b*c)) + 
  (1 / Real.sqrt (1/2 + b + b*c + b*c*d)) + 
  (1 / Real.sqrt (1/2 + c + c*d + c*d*a)) + 
  (1 / Real.sqrt (1/2 + d + d*a + d*a*b)) ≥ Real.sqrt 2 := by
  sorry

end inequality_proof_l2080_208007


namespace inequality_holds_function_increasing_l2080_208098

theorem inequality_holds (x : ℝ) (h : x ≥ 1) : x / 2 ≥ (x - 1) / (x + 1) := by
  sorry

theorem function_increasing (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hxy : x < y) :
  (x / 2 - (x - 1) / (x + 1)) < (y / 2 - (y - 1) / (y + 1)) := by
  sorry

end inequality_holds_function_increasing_l2080_208098


namespace max_questions_is_13_l2080_208004

/-- Represents a quiz with questions and student solutions -/
structure Quiz where
  questions : Nat
  students : Nat
  solvedBy : Nat → Finset Nat  -- For each question, the set of students who solved it
  solvedQuestions : Nat → Finset Nat  -- For each student, the set of questions they solved

/-- Properties that must hold for a valid quiz configuration -/
def ValidQuiz (q : Quiz) : Prop :=
  (∀ i : Nat, i < q.questions → (q.solvedBy i).card = 4) ∧
  (∀ i j : Nat, i < q.questions → j < q.questions → i ≠ j →
    (q.solvedBy i ∩ q.solvedBy j).card = 1) ∧
  (∀ s : Nat, s < q.students → (q.solvedQuestions s).card < q.questions)

/-- The maximum number of questions possible in a valid quiz configuration -/
def MaxQuestions : Nat := 13

/-- Theorem stating that 13 is the maximum number of questions in a valid quiz -/
theorem max_questions_is_13 :
  ∀ q : Quiz, ValidQuiz q → q.questions ≤ MaxQuestions :=
sorry

end max_questions_is_13_l2080_208004


namespace lock_settings_count_l2080_208047

/-- The number of digits on each dial of the lock -/
def numDigits : ℕ := 10

/-- The number of dials on the lock -/
def numDials : ℕ := 4

/-- The set of all possible digits -/
def digitSet : Finset ℕ := Finset.range numDigits

/-- The set of valid first digits (excluding zero) -/
def validFirstDigits : Finset ℕ := digitSet.filter (λ x => x ≠ 0)

/-- The number of different settings possible for the lock -/
def numSettings : ℕ := validFirstDigits.card * (numDigits - 1) * (numDigits - 2) * (numDigits - 3)

theorem lock_settings_count :
  numSettings = 4536 :=
sorry

end lock_settings_count_l2080_208047


namespace community_pantry_fraction_l2080_208077

theorem community_pantry_fraction (total_donation : ℚ) 
  (crisis_fund_fraction : ℚ) (livelihood_fraction : ℚ) (contingency_amount : ℚ) :
  total_donation = 240 →
  crisis_fund_fraction = 1/2 →
  livelihood_fraction = 1/4 →
  contingency_amount = 30 →
  (total_donation - crisis_fund_fraction * total_donation - 
   livelihood_fraction * (total_donation - crisis_fund_fraction * total_donation) - 
   contingency_amount) / total_donation = 1/4 := by
  sorry

end community_pantry_fraction_l2080_208077


namespace correct_evolution_process_l2080_208078

-- Define the types of population growth models
inductive PopulationGrowthModel
| Primitive
| Traditional
| Modern

-- Define the characteristics of each model
structure ModelCharacteristics where
  productiveForces : ℕ
  disasterResistance : ℕ
  birthRate : ℕ
  deathRate : ℕ
  economicLevel : ℕ
  socialSecurity : ℕ

-- Define the evolution process
def evolutionProcess : List PopulationGrowthModel :=
  [PopulationGrowthModel.Primitive, PopulationGrowthModel.Traditional, PopulationGrowthModel.Modern]

-- Define the characteristics for each model
def primitiveCharacteristics : ModelCharacteristics :=
  { productiveForces := 1, disasterResistance := 1, birthRate := 3, deathRate := 3,
    economicLevel := 1, socialSecurity := 1 }

def traditionalCharacteristics : ModelCharacteristics :=
  { productiveForces := 2, disasterResistance := 2, birthRate := 3, deathRate := 1,
    economicLevel := 2, socialSecurity := 2 }

def modernCharacteristics : ModelCharacteristics :=
  { productiveForces := 3, disasterResistance := 3, birthRate := 1, deathRate := 1,
    economicLevel := 3, socialSecurity := 3 }

-- Theorem stating that the evolution process is correct
theorem correct_evolution_process :
  evolutionProcess = [PopulationGrowthModel.Primitive, PopulationGrowthModel.Traditional, PopulationGrowthModel.Modern] :=
by sorry

end correct_evolution_process_l2080_208078


namespace vector_magnitude_problem_l2080_208012

/-- Given two vectors a and b in a real inner product space, 
    if |a| = 2, |b| = 3, and |a + b| = √19, then |a - b| = √7. -/
theorem vector_magnitude_problem 
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 := by
  sorry

end vector_magnitude_problem_l2080_208012


namespace probability_not_snow_l2080_208035

theorem probability_not_snow (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end probability_not_snow_l2080_208035


namespace container_volume_increase_l2080_208065

/-- Given a cylindrical container with volume V = πr²h that holds 3 gallons,
    prove that a new container with triple the radius and double the height holds 54 gallons. -/
theorem container_volume_increase (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 3 → π * (3*r)^2 * (2*h) = 54 := by
  sorry

end container_volume_increase_l2080_208065


namespace milk_price_proof_l2080_208081

/-- The original price of milk before discount -/
def milk_original_price : ℝ := 10

/-- Lily's initial budget -/
def initial_budget : ℝ := 60

/-- Cost of celery -/
def celery_cost : ℝ := 5

/-- Original price of cereal -/
def cereal_original_price : ℝ := 12

/-- Discount rate for cereal -/
def cereal_discount_rate : ℝ := 0.5

/-- Cost of bread -/
def bread_cost : ℝ := 8

/-- Discount rate for milk -/
def milk_discount_rate : ℝ := 0.1

/-- Cost of one potato -/
def potato_unit_cost : ℝ := 1

/-- Number of potatoes bought -/
def potato_quantity : ℕ := 6

/-- Amount left after buying all items -/
def amount_left : ℝ := 26

theorem milk_price_proof :
  let cereal_cost := cereal_original_price * (1 - cereal_discount_rate)
  let potato_cost := potato_unit_cost * potato_quantity
  let other_items_cost := celery_cost + cereal_cost + bread_cost + potato_cost
  let total_spent := initial_budget - amount_left
  let milk_discounted_price := total_spent - other_items_cost
  milk_original_price = milk_discounted_price / (1 - milk_discount_rate) :=
by sorry

end milk_price_proof_l2080_208081


namespace line_separate_from_circle_l2080_208073

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a point being inside a circle -/
def inside_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

/-- Definition of a line being separate from a circle -/
def separate_from_circle (l : Line) (c : Circle) : Prop :=
  let d := |l.a * c.center.1 + l.b * c.center.2 + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d > c.radius

/-- Main theorem -/
theorem line_separate_from_circle 
  (a : ℝ) 
  (h_a : a > 0) 
  (M : ℝ × ℝ) 
  (h_M : inside_circle M ⟨⟨0, 0⟩, a, h_a⟩) 
  (h_M_not_center : M ≠ (0, 0)) :
  separate_from_circle ⟨M.1, M.2, -a^2⟩ ⟨⟨0, 0⟩, a, h_a⟩ :=
sorry

end line_separate_from_circle_l2080_208073


namespace friendly_angle_values_l2080_208025

-- Define a friendly triangle
def is_friendly_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ (a = 2*b ∨ b = 2*c ∨ c = 2*a)

-- Theorem statement
theorem friendly_angle_values :
  ∀ a b c : ℝ,
  is_friendly_triangle a b c →
  (a = 42 ∨ b = 42 ∨ c = 42) →
  (a = 42 ∨ a = 84 ∨ a = 92 ∨
   b = 42 ∨ b = 84 ∨ b = 92 ∨
   c = 42 ∨ c = 84 ∨ c = 92) :=
by sorry

end friendly_angle_values_l2080_208025


namespace symmetric_complex_division_l2080_208059

/-- Two complex numbers are symmetric with respect to y = x if their real and imaginary parts are swapped -/
def symmetric_wrt_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

/-- The main theorem -/
theorem symmetric_complex_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_wrt_y_eq_x z₁ z₂) (h_z₁ : z₁ = 1 + 2*I) : 
  z₁ / z₂ = 4/5 + 3/5*I :=
sorry

end symmetric_complex_division_l2080_208059


namespace specific_qiandu_surface_area_l2080_208088

/-- Represents a right-angled triangular prism ("堑堵") with an isosceles right-angled triangle base -/
structure QianDu where
  hypotenuse : ℝ
  height : ℝ

/-- Calculates the surface area of a QianDu -/
def surface_area (qd : QianDu) : ℝ := sorry

/-- Theorem stating the surface area of a specific QianDu -/
theorem specific_qiandu_surface_area :
  ∃ (qd : QianDu), qd.hypotenuse = 2 ∧ qd.height = 2 ∧ surface_area qd = 4 * Real.sqrt 2 + 6 := by
  sorry

end specific_qiandu_surface_area_l2080_208088


namespace coffee_stock_proof_l2080_208084

/-- Represents the initial stock of coffee in pounds -/
def initial_stock : ℝ := 400

/-- Represents the percentage of decaffeinated coffee in the initial stock -/
def initial_decaf_percent : ℝ := 0.25

/-- Represents the additional coffee purchase in pounds -/
def additional_purchase : ℝ := 100

/-- Represents the percentage of decaffeinated coffee in the additional purchase -/
def additional_decaf_percent : ℝ := 0.60

/-- Represents the final percentage of decaffeinated coffee in the total stock -/
def final_decaf_percent : ℝ := 0.32

theorem coffee_stock_proof :
  initial_stock * initial_decaf_percent + additional_purchase * additional_decaf_percent =
  final_decaf_percent * (initial_stock + additional_purchase) :=
by sorry

end coffee_stock_proof_l2080_208084


namespace count_special_numbers_l2080_208067

theorem count_special_numbers : ∃ (S : Finset Nat),
  (∀ n ∈ S, n < 500 ∧ n % 5 = 0 ∧ n % 10 ≠ 0 ∧ n % 15 ≠ 0) ∧
  (∀ n < 500, n % 5 = 0 ∧ n % 10 ≠ 0 ∧ n % 15 ≠ 0 → n ∈ S) ∧
  S.card = 33 :=
by
  sorry

#check count_special_numbers

end count_special_numbers_l2080_208067


namespace cow_to_horse_ratio_l2080_208044

def total_animals : ℕ := 168
def num_cows : ℕ := 140

theorem cow_to_horse_ratio :
  let num_horses := total_animals - num_cows
  num_cows / num_horses = 5 := by
sorry

end cow_to_horse_ratio_l2080_208044


namespace all_propositions_false_l2080_208001

-- Define the basic geometric objects
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Line : Type)
variable (Plane : Type)

-- Define the geometric relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (intersection_plane : Plane → Plane → Line)

-- Theorem statement
theorem all_propositions_false :
  (∀ (a b : Line) (p : Plane), parallel_line a b → line_in_plane b p → parallel_line_plane a p) = False ∧
  (∀ (a b : Line) (α : Plane), parallel_line_plane a α → parallel_line_plane b α → parallel_line a b) = False ∧
  (∀ (a b : Line) (α β : Plane), parallel_line_plane a α → parallel_line_plane b β → perpendicular_plane α β → perpendicular_line a b) = False ∧
  (∀ (a b : Line) (α β : Plane), intersection_plane α β = a → parallel_line_plane b α → parallel_line b a) = False :=
sorry

end all_propositions_false_l2080_208001


namespace percentage_decrease_of_b_l2080_208061

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 → 
  b > 0 → 
  a / b = 4 / 5 → 
  x = a * 1.25 → 
  m = b * (1 - p / 100) → 
  m / x = 0.6 → 
  p = 40 := by
sorry

end percentage_decrease_of_b_l2080_208061


namespace original_class_strength_l2080_208014

/-- Given an adult class, prove that the original strength was 18 students -/
theorem original_class_strength
  (original_avg : ℝ)
  (new_students : ℕ)
  (new_avg : ℝ)
  (avg_decrease : ℝ)
  (h1 : original_avg = 40)
  (h2 : new_students = 18)
  (h3 : new_avg = 32)
  (h4 : avg_decrease = 4)
  : ∃ (x : ℕ), x * original_avg + new_students * new_avg = (x + new_students) * (original_avg - avg_decrease) ∧ x = 18 := by
  sorry

end original_class_strength_l2080_208014


namespace max_value_of_expression_l2080_208029

theorem max_value_of_expression (n : ℤ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  3 * (500 - n) ≤ 1200 ∧ ∃ (m : ℤ), 100 ≤ m ∧ m ≤ 999 ∧ 3 * (500 - m) = 1200 :=
by sorry

end max_value_of_expression_l2080_208029


namespace function_properties_l2080_208090

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x * Real.exp 1

-- State the theorem
theorem function_properties :
  -- Part 1: The constant a is 2
  (∃ a : ℝ, (deriv (f a)) 0 = -1 ∧ a = 2) ∧
  -- Part 2: For x > 0, x^2 < e^x
  (∀ x : ℝ, x > 0 → x^2 < Real.exp x) ∧
  -- Part 3: For any positive c, there exists x₀ such that for x ∈ (x₀, +∞), x^2 < ce^x
  (∀ c : ℝ, c > 0 → ∃ x₀ : ℝ, ∀ x : ℝ, x > x₀ → x^2 < c * Real.exp x) :=
by sorry

end

end function_properties_l2080_208090


namespace matching_shoes_probability_l2080_208037

/-- The probability of selecting a matching pair of shoes from a box with 7 pairs -/
theorem matching_shoes_probability (n : ℕ) (total : ℕ) (pairs : ℕ) : 
  n = 7 → total = 2 * n → pairs = n → 
  (pairs : ℚ) / (total.choose 2 : ℚ) = 1 / 13 := by
  sorry

#check matching_shoes_probability

end matching_shoes_probability_l2080_208037


namespace min_champion_wins_l2080_208094

theorem min_champion_wins (n : ℕ) (h : n = 10 ∨ n = 11) :
  let min_wins := (n / 2 : ℚ).ceil.toNat + 1
  ∀ k : ℕ, (∀ i : ℕ, i < n → i ≠ k → (n - 1).choose 2 ≤ k + i * (k - 1)) →
    min_wins ≤ k := by
  sorry

end min_champion_wins_l2080_208094


namespace quadratic_inequality_range_l2080_208085

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 :=
by sorry

end quadratic_inequality_range_l2080_208085


namespace sphere_volume_ratio_l2080_208093

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4/9 →
  ((4/3) * Real.pi * r₁^3) / ((4/3) * Real.pi * r₂^3) = 8/27 := by
sorry

end sphere_volume_ratio_l2080_208093


namespace speedster_convertibles_l2080_208095

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  4 * speedsters = 3 * total →
  5 * convertibles = 3 * speedsters →
  total - speedsters = 30 →
  convertibles = 54 := by
  sorry

end speedster_convertibles_l2080_208095


namespace division_remainder_problem_l2080_208069

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 1565)
  (h2 : divisor = 24)
  (h3 : quotient = 65) :
  dividend = divisor * quotient + 5 := by
sorry

end division_remainder_problem_l2080_208069


namespace intersection_rectangular_prisms_cubes_l2080_208087

-- Define the set of all rectangular prisms
def rectangular_prisms : Set (ℝ × ℝ × ℝ) := {p | ∃ l w h, p = (l, w, h) ∧ l > 0 ∧ w > 0 ∧ h > 0}

-- Define the set of all cubes
def cubes : Set (ℝ × ℝ × ℝ) := {c | ∃ s, c = (s, s, s) ∧ s > 0}

-- Theorem statement
theorem intersection_rectangular_prisms_cubes :
  rectangular_prisms ∩ cubes = cubes :=
by sorry

end intersection_rectangular_prisms_cubes_l2080_208087


namespace at_least_one_fraction_less_than_two_l2080_208071

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := by
  sorry

end at_least_one_fraction_less_than_two_l2080_208071
