import Mathlib

namespace NUMINAMATH_CALUDE_average_speed_calculation_l721_72100

/-- Calculate the average speed of a round trip with different speeds and wind conditions -/
theorem average_speed_calculation (outward_speed return_speed : ℝ) 
  (tailwind headwind : ℝ) : 
  outward_speed = 110 →
  tailwind = 15 →
  return_speed = 72 →
  headwind = 10 →
  (2 * (outward_speed + tailwind) * (return_speed - headwind)) / 
  ((outward_speed + tailwind) + (return_speed - headwind)) = 250 * 62 / 187 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l721_72100


namespace NUMINAMATH_CALUDE_circle_center_l721_72180

/-- The center of a circle given by the equation 3x^2 - 6x + 3y^2 + 12y - 75 = 0 is (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (3 * x^2 - 6 * x + 3 * y^2 + 12 * y - 75 = 0) → 
  (∃ r : ℝ, (x - 1)^2 + (y - (-2))^2 = r^2) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l721_72180


namespace NUMINAMATH_CALUDE_problem_solution_l721_72190

theorem problem_solution :
  (∀ x : ℝ, x^2 - x ≥ x - 1) ∧
  (∃ x : ℝ, x > 1 ∧ x + 4 / (x - 1) = 6) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → b / a < (b + 1) / (a + 1)) ∧
  (∀ x : ℝ, (x^2 + 10) / Real.sqrt (x^2 + 9) > 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l721_72190


namespace NUMINAMATH_CALUDE_product_67_63_l721_72138

theorem product_67_63 : 67 * 63 = 4221 := by
  sorry

end NUMINAMATH_CALUDE_product_67_63_l721_72138


namespace NUMINAMATH_CALUDE_triangle_heights_order_l721_72143

/-- Given a triangle with sides a, b, c and corresponding heights ha, hb, hc,
    if a > b > c, then ha < hb < hc -/
theorem triangle_heights_order (a b c ha hb hc : ℝ) :
  a > 0 → b > 0 → c > 0 →  -- positive sides
  ha > 0 → hb > 0 → hc > 0 →  -- positive heights
  a > b → b > c →  -- order of sides
  a * ha = b * hb →  -- area equality
  b * hb = c * hc →  -- area equality
  ha < hb ∧ hb < hc := by
  sorry


end NUMINAMATH_CALUDE_triangle_heights_order_l721_72143


namespace NUMINAMATH_CALUDE_sqrt_5_simplest_l721_72161

/-- Represents a square root expression -/
inductive SqrtExpr
| Frac : Rat → SqrtExpr
| Dec : Rat → SqrtExpr
| Int : Nat → SqrtExpr
| Sqrt : Nat → SqrtExpr

/-- Checks if a SqrtExpr is in its simplest form -/
def isSimplest : SqrtExpr → Prop
| SqrtExpr.Frac _ => false
| SqrtExpr.Dec _ => false
| SqrtExpr.Int n => true
| SqrtExpr.Sqrt n => n > 1 ∧ ¬ (∃ m : Nat, m * m = n)

/-- The given options -/
def options : List SqrtExpr :=
  [SqrtExpr.Frac (1/2), SqrtExpr.Dec (4/5), SqrtExpr.Int 9, SqrtExpr.Sqrt 5]

/-- The theorem stating that √5 is the simplest among the options -/
theorem sqrt_5_simplest :
  ∃ e ∈ options, isSimplest e ∧ ∀ e' ∈ options, isSimplest e' → e = e' :=
  sorry

end NUMINAMATH_CALUDE_sqrt_5_simplest_l721_72161


namespace NUMINAMATH_CALUDE_min_value_of_max_sum_l721_72197

theorem min_value_of_max_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → 
  a + b + c + d = 4 → 
  let M := max (max (a + b + c) (a + b + d)) (max (a + c + d) (b + c + d))
  3 ≤ M ∧ ∀ (M' : ℝ), (∀ (a' b' c' d' : ℝ), 
    a' > 0 → b' > 0 → c' > 0 → d' > 0 → 
    a' + b' + c' + d' = 4 → 
    let M'' := max (max (a' + b' + c') (a' + b' + d')) (max (a' + c' + d') (b' + c' + d'))
    M'' ≤ M') → 
  3 ≤ M' := by
sorry

end NUMINAMATH_CALUDE_min_value_of_max_sum_l721_72197


namespace NUMINAMATH_CALUDE_inequality_proof_l721_72194

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + a^2) 
  ≥ Real.sqrt 2 * (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l721_72194


namespace NUMINAMATH_CALUDE_baseball_average_hits_l721_72118

theorem baseball_average_hits (first_games : Nat) (first_avg : Nat) (remaining_games : Nat) (remaining_avg : Nat) : 
  first_games = 20 →
  first_avg = 2 →
  remaining_games = 10 →
  remaining_avg = 5 →
  let total_games := first_games + remaining_games
  let total_hits := first_games * first_avg + remaining_games * remaining_avg
  (total_hits : Rat) / total_games = 3 := by sorry

end NUMINAMATH_CALUDE_baseball_average_hits_l721_72118


namespace NUMINAMATH_CALUDE_bell_weight_ratio_l721_72135

/-- Given three bells with specific weight relationships, prove the ratio of the third to second bell's weight --/
theorem bell_weight_ratio :
  ∀ (bell1 bell2 bell3 : ℝ),
  bell1 = 50 →
  bell2 = 2 * bell1 →
  bell1 + bell2 + bell3 = 550 →
  bell3 / bell2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bell_weight_ratio_l721_72135


namespace NUMINAMATH_CALUDE_smallest_a_equals_36_l721_72187

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_double (x : ℝ) (h : x > 0) : f (2 * x) = 2 * f x

axiom f_interval (x : ℝ) (h : 1 < x ∧ x ≤ 2) : f x = 2 - x

-- Define the theorem
theorem smallest_a_equals_36 :
  ∃ a : ℝ, a > 0 ∧ f a = f 2020 ∧ ∀ b : ℝ, b > 0 → f b = f 2020 → a ≤ b :=
sorry

end NUMINAMATH_CALUDE_smallest_a_equals_36_l721_72187


namespace NUMINAMATH_CALUDE_johns_dad_age_l721_72109

theorem johns_dad_age (j d : ℕ) : j + 28 = d → j + d = 76 → d = 52 := by sorry

end NUMINAMATH_CALUDE_johns_dad_age_l721_72109


namespace NUMINAMATH_CALUDE_min_distance_complex_l721_72173

theorem min_distance_complex (Z : ℂ) (h : Complex.abs (Z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (W : ℂ), Complex.abs (W + 2 - 2*I) = 1 → Complex.abs (W - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l721_72173


namespace NUMINAMATH_CALUDE_annual_increase_fraction_l721_72133

theorem annual_increase_fraction (initial_amount final_amount : ℝ) 
  (h1 : initial_amount > 0)
  (h2 : final_amount > initial_amount)
  (h3 : initial_amount * (1 + f)^2 = final_amount)
  (h4 : initial_amount = 57600)
  (h5 : final_amount = 72900) : 
  f = 0.125 := by
sorry

end NUMINAMATH_CALUDE_annual_increase_fraction_l721_72133


namespace NUMINAMATH_CALUDE_codger_shoe_purchase_l721_72127

/-- Represents the number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- Represents the number of shoes in a complete set for a sloth -/
def complete_set : ℕ := 3

/-- Represents the number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- Represents the number of complete sets Codger wants to have -/
def desired_sets : ℕ := 7

/-- Represents the number of shoes Codger already owns -/
def owned_shoes : ℕ := 3

/-- Represents the constraint that shoes must be bought in even-numbered sets of pairs -/
def even_numbered_pairs (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- The main theorem -/
theorem codger_shoe_purchase :
  ∃ (pairs_to_buy : ℕ),
    pairs_to_buy * shoes_per_pair + owned_shoes ≥ desired_sets * complete_set ∧
    even_numbered_pairs pairs_to_buy ∧
    ∀ (n : ℕ), n < pairs_to_buy →
      n * shoes_per_pair + owned_shoes < desired_sets * complete_set ∨
      ¬(even_numbered_pairs n) :=
sorry

end NUMINAMATH_CALUDE_codger_shoe_purchase_l721_72127


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l721_72113

theorem pizza_slices_per_person 
  (small_pizza_slices : ℕ) 
  (large_pizza_slices : ℕ) 
  (slices_eaten_per_person : ℕ) 
  (num_people : ℕ) :
  small_pizza_slices = 8 →
  large_pizza_slices = 14 →
  slices_eaten_per_person = 9 →
  num_people = 2 →
  (small_pizza_slices + large_pizza_slices - slices_eaten_per_person * num_people) / num_people = 2 := by
sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l721_72113


namespace NUMINAMATH_CALUDE_rectangle_area_change_l721_72165

/-- Given a rectangle with width s and height h, where increasing the width by 3 and
    decreasing the height by 3 doesn't change the area, prove that decreasing the width
    by 4 and increasing the height by 4 results in an area decrease of 28 square units. -/
theorem rectangle_area_change (s h : ℝ) (h_area : (s + 3) * (h - 3) = s * h) :
  s * h - (s - 4) * (h + 4) = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l721_72165


namespace NUMINAMATH_CALUDE_lcm_36_45_l721_72145

theorem lcm_36_45 : Nat.lcm 36 45 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_45_l721_72145


namespace NUMINAMATH_CALUDE_fifth_teapot_volume_l721_72193

theorem fifth_teapot_volume
  (a : ℕ → ℚ)  -- arithmetic sequence of rational numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_length : ∀ n, n ≥ 9 → a n = a 9)  -- sequence has 9 terms
  (h_sum_first_three : a 1 + a 2 + a 3 = 1/2)  -- sum of first three terms
  (h_sum_last_three : a 7 + a 8 + a 9 = 5/2)  -- sum of last three terms
  : a 5 = 1/2 := by sorry

end NUMINAMATH_CALUDE_fifth_teapot_volume_l721_72193


namespace NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_l721_72174

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f'_derivative (a : ℝ) :
  (∃ x, f' a x = 0 ∧ x = 2) →  -- f has an extremum at x = 2
  (∃ m n : ℝ, m ∈ Set.Icc (-1) 1 ∧ n ∈ Set.Icc (-1) 1 ∧
    ∀ p q : ℝ, p ∈ Set.Icc (-1) 1 → q ∈ Set.Icc (-1) 1 →
      f a m + f' a n ≤ f a p + f' a q) →
  (∃ m n : ℝ, m ∈ Set.Icc (-1) 1 ∧ n ∈ Set.Icc (-1) 1 ∧
    f a m + f' a n = -13) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_l721_72174


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l721_72186

theorem line_ellipse_intersection (m : ℝ) : 
  (∃! x y : ℝ, y = m * x + 2 ∧ x^2 + 6 * y^2 = 4) → m^2 = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l721_72186


namespace NUMINAMATH_CALUDE_election_votes_theorem_l721_72176

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : (13 : ℚ) / 20 * total_votes = 39 + (total_votes - 39)) : 
  total_votes = 60 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l721_72176


namespace NUMINAMATH_CALUDE_count_even_factors_l721_72101

def M : ℕ := 2^5 * 3^4 * 5^3 * 7^3 * 11^1

/-- The number of even factors of M -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors M = 800 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_l721_72101


namespace NUMINAMATH_CALUDE_sum_of_v_at_specific_points_l721_72163

-- Define the function v
def v (x : ℝ) : ℝ := x^3 - 3*x + 1

-- State the theorem
theorem sum_of_v_at_specific_points : 
  v 2 + v (-2) + v 1 + v (-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_v_at_specific_points_l721_72163


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l721_72164

theorem complex_fraction_simplification :
  let z : ℂ := (1 - Complex.I * Real.sqrt 3) / (Complex.I + Real.sqrt 3) ^ 2
  z = -1/4 - (Complex.I * Real.sqrt 3) / 4 := by
    sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l721_72164


namespace NUMINAMATH_CALUDE_bowling_score_ratio_l721_72141

theorem bowling_score_ratio (total_score : ℕ) (third_score : ℕ) : 
  total_score = 810 →
  third_score = 162 →
  ∃ (first_score second_score : ℕ),
    first_score + second_score + third_score = total_score ∧
    first_score = second_score / 3 →
    second_score / third_score = 3 := by
sorry

end NUMINAMATH_CALUDE_bowling_score_ratio_l721_72141


namespace NUMINAMATH_CALUDE_square_area_difference_l721_72154

theorem square_area_difference (area_B : ℝ) (side_diff : ℝ) : 
  area_B = 81 → 
  side_diff = 4 → 
  let side_B := Real.sqrt area_B
  let side_A := side_B + side_diff
  side_A * side_A = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_difference_l721_72154


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_l721_72166

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : y^2 = 4*x

/-- Theorem: For a parabola y^2 = 4x, if a line passing through its focus intersects 
    the parabola at points A and B, and the distance |AB| = 12, then x₁ + x₂ = 10 -/
theorem parabola_intersection_sum (A B : ParabolaPoint) 
  (focus_line : A.x ≠ B.x → (A.y - B.y) / (A.x - B.x) = (A.y + B.y) / (A.x + B.x - 2))
  (distance : (A.x - B.x)^2 + (A.y - B.y)^2 = 12^2) :
  A.x + B.x = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_l721_72166


namespace NUMINAMATH_CALUDE_fish_after_ten_years_l721_72191

def initial_fish : ℕ := 6

def fish_added (year : ℕ) : ℕ :=
  if year ≤ 10 then year + 1 else 0

def fish_died (year : ℕ) : ℕ :=
  if year ≤ 10 then
    if year ≤ 4 then 5 - year
    else year - 3
  else 0

def fish_count (year : ℕ) : ℕ :=
  if year = 0 then initial_fish
  else (fish_count (year - 1) + fish_added year - fish_died year)

theorem fish_after_ten_years :
  fish_count 10 = 34 := by sorry

end NUMINAMATH_CALUDE_fish_after_ten_years_l721_72191


namespace NUMINAMATH_CALUDE_oil_fraction_after_replacements_l721_72157

def tank_capacity : ℚ := 20
def replacement_amount : ℚ := 5
def num_replacements : ℕ := 5

def fraction_remaining (n : ℕ) : ℚ := (3/4) ^ n

theorem oil_fraction_after_replacements :
  fraction_remaining num_replacements = 243/1024 := by
  sorry

end NUMINAMATH_CALUDE_oil_fraction_after_replacements_l721_72157


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l721_72178

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → 
  ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l721_72178


namespace NUMINAMATH_CALUDE_complement_union_theorem_l721_72111

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l721_72111


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l721_72125

def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 9 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l721_72125


namespace NUMINAMATH_CALUDE_quadratic_congruence_solution_l721_72148

theorem quadratic_congruence_solution (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℤ, (6 * n^2 + 5 * n + 1) % p = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_congruence_solution_l721_72148


namespace NUMINAMATH_CALUDE_carnival_tickets_total_l721_72122

/-- Represents the number of tickets used for a carnival ride -/
structure RideTickets where
  ferrisWheel : ℕ
  bumperCars : ℕ
  rollerCoaster : ℕ

/-- Calculates the total number of tickets used for a set of rides -/
def totalTickets (rides : RideTickets) (ferrisWheelCost bumperCarsCost rollerCoasterCost : ℕ) : ℕ :=
  rides.ferrisWheel * ferrisWheelCost + rides.bumperCars * bumperCarsCost + rides.rollerCoaster * rollerCoasterCost

/-- Theorem stating the total number of tickets used by Oliver, Emma, and Sophia -/
theorem carnival_tickets_total : 
  let ferrisWheelCost := 7
  let bumperCarsCost := 5
  let rollerCoasterCost := 9
  let oliver := RideTickets.mk 5 4 0
  let emma := RideTickets.mk 0 6 3
  let sophia := RideTickets.mk 3 2 2
  totalTickets oliver ferrisWheelCost bumperCarsCost rollerCoasterCost +
  totalTickets emma ferrisWheelCost bumperCarsCost rollerCoasterCost +
  totalTickets sophia ferrisWheelCost bumperCarsCost rollerCoasterCost = 161 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_total_l721_72122


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l721_72103

theorem sum_with_radical_conjugate :
  let x : ℝ := 12 - Real.sqrt 2023
  let y : ℝ := 12 + Real.sqrt 2023
  x + y = 24 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l721_72103


namespace NUMINAMATH_CALUDE_log_equality_l721_72168

theorem log_equality (x : ℝ) (h : x > 0) :
  (Real.log (2 * x) / Real.log (5 * x) = Real.log (8 * x) / Real.log (625 * x)) →
  (Real.log x / Real.log 2 = Real.log 5 / (3 * (Real.log 2 - Real.log 5))) := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l721_72168


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_remaining_books_l721_72140

/-- Given a series of books, calculate the number of books remaining to be read -/
def booksRemaining (totalBooks readBooks : ℕ) : ℕ :=
  totalBooks - readBooks

/-- Theorem: In a series of 32 books, if 17 have been read, 15 remain to be read -/
theorem crazy_silly_school_series_remaining_books :
  booksRemaining 32 17 = 15 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_remaining_books_l721_72140


namespace NUMINAMATH_CALUDE_freds_basketball_games_l721_72196

theorem freds_basketball_games 
  (missed_this_year : ℕ) 
  (attended_last_year : ℕ) 
  (total_attended : ℕ) 
  (h1 : missed_this_year = 35)
  (h2 : attended_last_year = 11)
  (h3 : total_attended = 47) :
  total_attended - attended_last_year = 36 :=
by sorry

end NUMINAMATH_CALUDE_freds_basketball_games_l721_72196


namespace NUMINAMATH_CALUDE_division_remainder_problem_l721_72106

theorem division_remainder_problem (x y : ℕ) (hx : x > 0) :
  (∃ q : ℕ, x = 10 * y + 3) →
  (∃ r : ℕ, 2 * x = 7 * (3 * y) + r ∧ r < 7) →
  11 * y - x = 2 →
  ∃ r : ℕ, 2 * x = 7 * (3 * y) + r ∧ r = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l721_72106


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l721_72195

theorem polynomial_equation_solution (p : Polynomial ℝ) :
  (∀ x : ℝ, x ≠ 0 → p.eval x ^ 2 + p.eval (1 / x) ^ 2 = p.eval (x ^ 2) * p.eval (1 / x ^ 2)) →
  p = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l721_72195


namespace NUMINAMATH_CALUDE_dot_product_specific_vectors_l721_72130

theorem dot_product_specific_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-3, 1)
  (a.1 * b.1 + a.2 * b.2) = -1 := by sorry

end NUMINAMATH_CALUDE_dot_product_specific_vectors_l721_72130


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l721_72126

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 6*x - 7 = 0 ↔ (x - 3)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l721_72126


namespace NUMINAMATH_CALUDE_intersection_theorem_l721_72142

/-- The curve C₁ in Cartesian coordinates -/
def C₁ (k : ℝ) : ℝ → ℝ := λ x ↦ k * |x| + 2

/-- The curve C₂ in Cartesian coordinates -/
def C₂ : ℝ × ℝ → Prop := λ p ↦ (p.1 + 1)^2 + p.2^2 = 4

/-- The number of intersection points between C₁ and C₂ -/
def numIntersections (k : ℝ) : ℕ := sorry

theorem intersection_theorem (k : ℝ) :
  numIntersections k = 3 → k = -4/3 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l721_72142


namespace NUMINAMATH_CALUDE_ball_bounces_on_table_l721_72156

/-- Represents a rectangular table -/
structure Table where
  length : ℕ
  width : ℕ

/-- Calculates the number of bounces required for a ball to travel
    from one corner to the opposite corner of a rectangular table,
    moving at a 45° angle and bouncing off sides at 45° -/
def numberOfBounces (t : Table) : ℕ :=
  t.length + t.width - 2

theorem ball_bounces_on_table (t : Table) (h1 : t.length = 5) (h2 : t.width = 2) :
  numberOfBounces t = 5 := by
  sorry

#eval numberOfBounces { length := 5, width := 2 }

end NUMINAMATH_CALUDE_ball_bounces_on_table_l721_72156


namespace NUMINAMATH_CALUDE_train_crossing_time_l721_72131

/-- The time it takes for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 50 ∧ 
  train_speed = 24.997600191984645 ∧ 
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l721_72131


namespace NUMINAMATH_CALUDE_kirills_height_l721_72107

/-- Proves that Kirill's height is 49 cm given the conditions -/
theorem kirills_height (brother_height : ℕ) 
  (h1 : brother_height - 14 + brother_height = 112) : 
  brother_height - 14 = 49 := by
  sorry

#check kirills_height

end NUMINAMATH_CALUDE_kirills_height_l721_72107


namespace NUMINAMATH_CALUDE_geometric_sequence_y_value_l721_72159

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_y_value (x y z : ℝ) :
  is_geometric_sequence 1 x y z 9 → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_y_value_l721_72159


namespace NUMINAMATH_CALUDE_wilsons_theorem_l721_72152

theorem wilsons_theorem (p : ℕ) (h : p ≥ 2) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l721_72152


namespace NUMINAMATH_CALUDE_marys_remaining_money_l721_72134

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary's remaining money is 50 - 12p -/
theorem marys_remaining_money (p : ℝ) : money_left p = 50 - 12 * p := by
  sorry

end NUMINAMATH_CALUDE_marys_remaining_money_l721_72134


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l721_72189

/-- The total number of pencils after adding more -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Proof that the total number of pencils is 215 -/
theorem pencils_in_drawer : total_pencils 115 100 = 215 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l721_72189


namespace NUMINAMATH_CALUDE_type_q_machine_time_l721_72162

theorem type_q_machine_time (q : ℝ) (h1 : q > 0) 
  (h2 : 2 / q + 3 / 7 = 1 / 1.2) : q = 84 / 17 := by
  sorry

end NUMINAMATH_CALUDE_type_q_machine_time_l721_72162


namespace NUMINAMATH_CALUDE_at_least_two_acute_angles_l721_72112

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define an acute angle
def is_acute (angle : ℝ) : Prop := angle < 90

-- Define the theorem
theorem at_least_two_acute_angles (t : Triangle) : 
  ∃ i j, i ≠ j ∧ is_acute (t.angles i) ∧ is_acute (t.angles j) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_acute_angles_l721_72112


namespace NUMINAMATH_CALUDE_polynomial_equality_constant_l721_72175

theorem polynomial_equality_constant (s : ℚ) : 
  (∀ x : ℚ, (3 * x^2 - 8 * x + 9) * (5 * x^2 + s * x + 15) = 
    15 * x^4 - 71 * x^3 + 174 * x^2 - 215 * x + 135) → 
  s = -95/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_constant_l721_72175


namespace NUMINAMATH_CALUDE_equation_solution_l721_72147

theorem equation_solution : ∃ x : ℚ, (5*x + 9*x = 450 - 10*(x - 5)) ∧ x = 125/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l721_72147


namespace NUMINAMATH_CALUDE_quadratic_properties_l721_72185

def f (x : ℝ) := -x^2 + 4*x + 3

theorem quadratic_properties :
  let a := 2
  let b := 7
  (∀ x, f x = -(x - a)^2 + b) ∧
  (∀ x ∈ Set.Icc 1 4, f x ≤ b) ∧
  (∃ x ∈ Set.Icc 1 4, f x = b) ∧
  (∃ x ∈ Set.Icc 1 4, f x = 3) ∧
  (∀ x ∈ Set.Icc 1 4, f x ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l721_72185


namespace NUMINAMATH_CALUDE_candy_cost_approx_ten_cents_l721_72183

/-- The cost of one candy given the number of people, candies per person, leftover candies, and total spent -/
def candy_cost (people : ℕ) (candies_per_person : ℕ) (leftover : ℕ) (total_spent : ℚ) : ℚ :=
  total_spent / (people * candies_per_person + leftover)

/-- Theorem stating that the cost of one candy is approximately $0.10 -/
theorem candy_cost_approx_ten_cents 
  (people : ℕ) (candies_per_person : ℕ) (leftover : ℕ) (total_spent : ℚ)
  (h1 : people = 35)
  (h2 : candies_per_person = 2)
  (h3 : leftover = 12)
  (h4 : total_spent = 8) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |candy_cost people candies_per_person leftover total_spent - 1/10| < ε :=
sorry

end NUMINAMATH_CALUDE_candy_cost_approx_ten_cents_l721_72183


namespace NUMINAMATH_CALUDE_total_heads_is_48_l721_72151

/-- Represents the number of feet an animal has -/
def feet_count (animal : String) : ℕ :=
  if animal = "hen" then 2 else 4

/-- The total number of animals -/
def total_animals (hens cows : ℕ) : ℕ := hens + cows

/-- The total number of feet -/
def total_feet (hens cows : ℕ) : ℕ := feet_count "hen" * hens + feet_count "cow" * cows

/-- Theorem stating that the total number of heads is 48 -/
theorem total_heads_is_48 (hens cows : ℕ) 
  (h1 : total_feet hens cows = 140) 
  (h2 : hens = 26) : 
  total_animals hens cows = 48 := by sorry

end NUMINAMATH_CALUDE_total_heads_is_48_l721_72151


namespace NUMINAMATH_CALUDE_basketball_handshakes_l721_72177

/-- Calculates the total number of handshakes in a basketball game scenario --/
def total_handshakes (players_per_team : ℕ) (num_referees : ℕ) (num_coaches : ℕ) : ℕ :=
  let player_handshakes := players_per_team * players_per_team
  let player_referee_handshakes := 2 * players_per_team * num_referees
  let coach_handshakes := num_coaches * (2 * players_per_team + num_referees)
  player_handshakes + player_referee_handshakes + coach_handshakes

/-- Theorem stating that the total number of handshakes in the given scenario is 102 --/
theorem basketball_handshakes :
  total_handshakes 6 3 2 = 102 := by
  sorry

#eval total_handshakes 6 3 2

end NUMINAMATH_CALUDE_basketball_handshakes_l721_72177


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l721_72171

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + Complex.I) / Complex.I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l721_72171


namespace NUMINAMATH_CALUDE_grape_to_fruit_ratio_l721_72124

def red_apples : ℕ := 9
def green_apples : ℕ := 4
def grape_bunches : ℕ := 3
def grapes_per_bunch : ℕ := 15
def yellow_bananas : ℕ := 6
def orange_oranges : ℕ := 2
def kiwis : ℕ := 5
def blueberries : ℕ := 30

def total_grapes : ℕ := grape_bunches * grapes_per_bunch

def total_fruits : ℕ := red_apples + green_apples + total_grapes + yellow_bananas + orange_oranges + kiwis + blueberries

theorem grape_to_fruit_ratio :
  (total_grapes : ℚ) / (total_fruits : ℚ) = 45 / 101 := by
  sorry

end NUMINAMATH_CALUDE_grape_to_fruit_ratio_l721_72124


namespace NUMINAMATH_CALUDE_log_xy_value_l721_72120

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x^3 * y^2) = 2) (h2 : Real.log (x^2 * y^3) = 2) : 
  Real.log (x * y) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_log_xy_value_l721_72120


namespace NUMINAMATH_CALUDE_definite_integral_2x_plus_1_over_x_l721_72102

theorem definite_integral_2x_plus_1_over_x :
  ∫ x in (1 : ℝ)..2, (2 * x + 1 / x) = 3 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_2x_plus_1_over_x_l721_72102


namespace NUMINAMATH_CALUDE_probability_exactly_once_l721_72149

theorem probability_exactly_once (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - (1 - p)^3 = 26/27) →
  3 * p * (1 - p)^2 = 2/9 :=
by sorry

end NUMINAMATH_CALUDE_probability_exactly_once_l721_72149


namespace NUMINAMATH_CALUDE_tanC_over_tanA_max_tanB_l721_72188

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + 2*t.b^2 = t.c^2

-- Theorem 1: If the condition is satisfied, then tan C / tan A = -3
theorem tanC_over_tanA (t : Triangle) (h : satisfiesCondition t) :
  Real.tan t.C / Real.tan t.A = -3 :=
sorry

-- Theorem 2: If the condition is satisfied, then the maximum value of tan B is √3/3
theorem max_tanB (t : Triangle) (h : satisfiesCondition t) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 3 ∧ Real.tan t.B ≤ max :=
sorry

end NUMINAMATH_CALUDE_tanC_over_tanA_max_tanB_l721_72188


namespace NUMINAMATH_CALUDE_jerry_cereal_calories_l721_72108

/-- Represents the calorie content of Jerry's breakfast items -/
structure BreakfastCalories where
  pancakeCount : ℕ
  pancakeCalories : ℕ
  baconCount : ℕ
  baconCalories : ℕ
  totalCalories : ℕ

/-- Calculates the calories in the cereal bowl given the breakfast composition -/
def cerealCalories (b : BreakfastCalories) : ℕ :=
  b.totalCalories - (b.pancakeCount * b.pancakeCalories + b.baconCount * b.baconCalories)

/-- Theorem stating that Jerry's cereal bowl contains 200 calories -/
theorem jerry_cereal_calories :
  let jerryBreakfast : BreakfastCalories := {
    pancakeCount := 6,
    pancakeCalories := 120,
    baconCount := 2,
    baconCalories := 100,
    totalCalories := 1120
  }
  cerealCalories jerryBreakfast = 200 := by
  sorry

end NUMINAMATH_CALUDE_jerry_cereal_calories_l721_72108


namespace NUMINAMATH_CALUDE_line_equation_through_points_line_equation_specific_points_l721_72182

/-- The equation of a line passing through two points -/
theorem line_equation_through_points (x₁ y₁ x₂ y₂ : ℝ) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (x₂ ≠ x₁) →
  (∀ x y : ℝ, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)) :=
by sorry

/-- The equation of the line passing through (0, -5) and (1, 0) is y = 5x - 5 -/
theorem line_equation_specific_points :
  ∀ x y : ℝ, y = 5 * x - 5 ↔ (x = 0 ∧ y = -5) ∨ (x = 1 ∧ y = 0) ∨ (y - (-5)) * (1 - 0) = (x - 0) * (0 - (-5)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_line_equation_specific_points_l721_72182


namespace NUMINAMATH_CALUDE_xy_value_l721_72169

theorem xy_value (x y : ℝ) (h : x * (x - y) = x^2 - 6) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l721_72169


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l721_72137

/-- The distance between the vertices of a hyperbola with equation x²/144 - y²/49 = 1 is 24 -/
theorem hyperbola_vertex_distance : 
  ∀ (x y : ℝ), x^2/144 - y^2/49 = 1 → ∃ (d : ℝ), d = 24 ∧ d = 2 * (Real.sqrt 144) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l721_72137


namespace NUMINAMATH_CALUDE_correct_statements_count_l721_72172

/-- Represents a statistical statement about a population or sample -/
inductive StatStatement
| population_statement
| individual_statement
| sample_statement
| sample_size_statement

/-- Represents the correctness of a statement -/
def is_correct : StatStatement → Bool
| StatStatement.population_statement => true
| StatStatement.individual_statement => false
| StatStatement.sample_statement => true
| StatStatement.sample_size_statement => true

/-- The total number of students in the population -/
def population_size : Nat := 14000

/-- The number of students in the sample -/
def sample_size : Nat := 1000

/-- The list of all statements -/
def all_statements : List StatStatement := [
  StatStatement.population_statement,
  StatStatement.individual_statement,
  StatStatement.sample_statement,
  StatStatement.sample_size_statement
]

/-- Counts the number of correct statements -/
def count_correct_statements (statements : List StatStatement) : Nat :=
  statements.filter is_correct |>.length

/-- Theorem stating that the number of correct statements is 3 -/
theorem correct_statements_count :
  count_correct_statements all_statements = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l721_72172


namespace NUMINAMATH_CALUDE_inequality_proof_l721_72150

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (b / a + c / b + a / c) ≥ (1 / 3) * (a + b + c) * (1 / a + 1 / b + 1 / c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l721_72150


namespace NUMINAMATH_CALUDE_circle_radii_sum_l721_72179

theorem circle_radii_sum : ∀ r : ℝ, 
  r > 0 →
  (r - 4)^2 + r^2 = (r + 2)^2 →
  ∃ r' : ℝ, r' > 0 ∧ (r' - 4)^2 + r'^2 = (r' + 2)^2 ∧ r + r' = 12 :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_sum_l721_72179


namespace NUMINAMATH_CALUDE_alien_attack_probability_l721_72139

/-- The number of aliens attacking --/
def num_aliens : ℕ := 3

/-- The number of galaxies being attacked --/
def num_galaxies : ℕ := 4

/-- The number of days of the attack --/
def num_days : ℕ := 3

/-- The probability that a specific galaxy is not chosen by any alien on a given day --/
def prob_not_chosen_day : ℚ := (3/4)^num_aliens

/-- The probability that a specific galaxy is not destroyed over all days --/
def prob_not_destroyed : ℚ := prob_not_chosen_day^num_days

/-- The probability that at least one galaxy is not destroyed --/
def prob_at_least_one_not_destroyed : ℚ := num_galaxies * prob_not_destroyed

/-- The probability that all galaxies are destroyed --/
def prob_all_destroyed : ℚ := 1 - prob_at_least_one_not_destroyed

theorem alien_attack_probability : prob_all_destroyed = 45853/65536 := by
  sorry

end NUMINAMATH_CALUDE_alien_attack_probability_l721_72139


namespace NUMINAMATH_CALUDE_smallest_possible_b_l721_72153

theorem smallest_possible_b (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1/b + 1/a ≤ 2) →
  b ≥ 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l721_72153


namespace NUMINAMATH_CALUDE_retail_price_is_1_04a_l721_72114

/-- The retail price of a washing machine after markup and discount -/
def retail_price (a : ℝ) : ℝ :=
  a * (1 + 0.3) * (1 - 0.2)

/-- Theorem stating that the retail price is 1.04 times the initial cost -/
theorem retail_price_is_1_04a (a : ℝ) : retail_price a = 1.04 * a := by
  sorry

end NUMINAMATH_CALUDE_retail_price_is_1_04a_l721_72114


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l721_72181

theorem sum_of_two_numbers (x y : ℝ) : 
  y = 2 * x + 3 ∧ y = 19 → x + y = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l721_72181


namespace NUMINAMATH_CALUDE_divisors_of_m_squared_l721_72117

def m : ℕ := 2^42 * 3^26 * 5^12

theorem divisors_of_m_squared (d : ℕ) : 
  (d ∣ m^2) ∧ (d < m) ∧ ¬(d ∣ m) → 
  (Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1))).card = 38818 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_m_squared_l721_72117


namespace NUMINAMATH_CALUDE_sprint_tournament_races_l721_72129

theorem sprint_tournament_races (total_sprinters : ℕ) (lanes_per_race : ℕ) : 
  total_sprinters = 320 →
  lanes_per_race = 8 →
  (∃ (num_races : ℕ), 
    num_races = 46 ∧
    num_races = (total_sprinters - 1) / (lanes_per_race - 1) + 
      (if (total_sprinters - 1) % (lanes_per_race - 1) = 0 then 0 else 1)) :=
by
  sorry

#check sprint_tournament_races

end NUMINAMATH_CALUDE_sprint_tournament_races_l721_72129


namespace NUMINAMATH_CALUDE_interest_rates_equality_l721_72146

theorem interest_rates_equality (initial_savings : ℝ) 
  (simple_interest : ℝ) (compound_interest : ℝ) : 
  initial_savings = 1000 ∧ 
  simple_interest = 100 ∧ 
  compound_interest = 105 →
  ∃ (r : ℝ), 
    simple_interest = (initial_savings / 2) * r * 2 ∧
    compound_interest = (initial_savings / 2) * ((1 + r)^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_interest_rates_equality_l721_72146


namespace NUMINAMATH_CALUDE_janes_change_is_correct_l721_72132

/-- The change Jane receives when buying an apple -/
def janes_change (apple_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - apple_price

/-- Theorem: Jane receives $4.25 in change -/
theorem janes_change_is_correct : 
  janes_change 0.75 5.00 = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_janes_change_is_correct_l721_72132


namespace NUMINAMATH_CALUDE_exam_students_count_l721_72136

theorem exam_students_count (N : ℕ) (T : ℕ) : 
  N * 85 = T ∧
  (N - 5) * 90 = T - 300 ∧
  (N - 8) * 95 = T - 465 ∧
  (N - 15) * 100 = T - 955 →
  N = 30 := by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l721_72136


namespace NUMINAMATH_CALUDE_inscribed_rhombus_radius_l721_72199

/-- A rhombus inscribed in the intersection of two equal circles -/
structure InscribedRhombus where
  /-- The length of one diagonal of the rhombus -/
  diagonal1 : ℝ
  /-- The length of the other diagonal of the rhombus -/
  diagonal2 : ℝ
  /-- The radius of the circles -/
  radius : ℝ
  /-- The diagonals are positive -/
  diagonal1_pos : diagonal1 > 0
  diagonal2_pos : diagonal2 > 0
  /-- The radius is positive -/
  radius_pos : radius > 0
  /-- The relationship between the diagonals and the radius -/
  radius_eq : radius^2 = (radius - diagonal1/2)^2 + (diagonal2/2)^2

/-- The theorem stating that a rhombus with diagonals 12 and 6 inscribed in two equal circles implies the radius is 7.5 -/
theorem inscribed_rhombus_radius (r : InscribedRhombus) (h1 : r.diagonal1 = 6) (h2 : r.diagonal2 = 12) : 
  r.radius = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_radius_l721_72199


namespace NUMINAMATH_CALUDE_gretas_hourly_wage_l721_72160

/-- Proves that Greta's hourly wage is $12 given the conditions of the problem -/
theorem gretas_hourly_wage (greta_hours : ℕ) (lisa_wage : ℕ) (lisa_hours : ℕ) 
  (h1 : greta_hours = 40)
  (h2 : lisa_wage = 15)
  (h3 : lisa_hours = 32)
  (h4 : greta_hours * gretas_wage = lisa_hours * lisa_wage) :
  gretas_wage = 12 := by
  sorry

#check gretas_hourly_wage

end NUMINAMATH_CALUDE_gretas_hourly_wage_l721_72160


namespace NUMINAMATH_CALUDE_equation_solutions_l721_72198

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧
    x₁^2 - 4*x₁ + 1 = 0 ∧ x₂^2 - 4*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 2/3 ∧
    3*y₁*(2*y₁ + 1) = 4*y₁ + 2 ∧ 3*y₂*(2*y₂ + 1) = 4*y₂ + 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l721_72198


namespace NUMINAMATH_CALUDE_sequence_problem_l721_72170

def D (A : ℕ → ℝ) : ℕ → ℝ := λ n => A (n + 1) - A n

theorem sequence_problem (A : ℕ → ℝ) 
  (h1 : ∀ n, D (D A) n = 1) 
  (h2 : A 19 = 0) 
  (h3 : A 92 = 0) : 
  A 1 = 819 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l721_72170


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_zero_l721_72155

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given complex number z as a function of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a) (a - 2)

/-- Theorem: If z(a) is a pure imaginary number, then a = 0 -/
theorem pure_imaginary_implies_a_zero : 
  ∀ a : ℝ, is_pure_imaginary (z a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_zero_l721_72155


namespace NUMINAMATH_CALUDE_square_fold_distance_l721_72110

/-- Given a square ABCD with side length 4, folded along diagonal BD to form a dihedral angle of 60°,
    the distance between the midpoint of BC and point A is 2√2. -/
theorem square_fold_distance (A B C D : ℝ × ℝ) : 
  let side_length : ℝ := 4
  let dihedral_angle : ℝ := 60
  let is_square := (A.1 = 0 ∧ A.2 = 0) ∧ 
                   (B.1 = side_length ∧ B.2 = 0) ∧ 
                   (C.1 = side_length ∧ C.2 = side_length) ∧ 
                   (D.1 = 0 ∧ D.2 = side_length)
  let midpoint_BC := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let distance := Real.sqrt ((A.1 - midpoint_BC.1)^2 + (A.2 - midpoint_BC.2)^2)
  is_square → distance = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_fold_distance_l721_72110


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l721_72115

/-- 
Given a quadratic equation x^2 + 6x + m = 0 with two equal real roots,
prove that m = 9.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + 6*y + m = 0 → y = x) → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l721_72115


namespace NUMINAMATH_CALUDE_area_ratio_value_l721_72192

/-- Represents a sequence of circles touching a right angle -/
structure CircleSequence where
  -- The ratio of radii between consecutive circles
  radius_ratio : ℝ
  -- Assumption that the ratio is equal to (√2 - 1)²
  h_ratio : radius_ratio = (Real.sqrt 2 - 1)^2

/-- The ratio of the area of the first circle to the sum of areas of all subsequent circles -/
def area_ratio (seq : CircleSequence) : ℝ := sorry

/-- Theorem stating the area ratio for the given circle sequence -/
theorem area_ratio_value (seq : CircleSequence) :
  area_ratio seq = 16 + 12 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_area_ratio_value_l721_72192


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l721_72128

/-- Given a quadratic function f(x) = 3x^2 + 2x + 1, when shifted 5 units to the right,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 41 -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, (3 * (x - 5)^2 + 2 * (x - 5) + 1) = (a * x^2 + b * x + c)) →
  a + b + c = 41 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l721_72128


namespace NUMINAMATH_CALUDE_wire_division_proof_l721_72167

/-- Calculates the number of equal parts a wire can be divided into -/
def wire_parts (total_length : ℕ) (part_length : ℕ) : ℕ :=
  total_length / part_length

/-- Proves that a wire of 64 inches divided into 16-inch parts results in 4 parts -/
theorem wire_division_proof :
  wire_parts 64 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_wire_division_proof_l721_72167


namespace NUMINAMATH_CALUDE_casey_pumping_rate_l721_72123

def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def water_per_corn_plant : ℚ := 1/2
def num_pigs : ℕ := 10
def water_per_pig : ℚ := 4
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4
def pumping_time : ℕ := 25

theorem casey_pumping_rate :
  let total_corn_plants := corn_rows * corn_plants_per_row
  let water_for_corn := (total_corn_plants : ℚ) * water_per_corn_plant
  let water_for_pigs := (num_pigs : ℚ) * water_per_pig
  let water_for_ducks := (num_ducks : ℚ) * water_per_duck
  let total_water := water_for_corn + water_for_pigs + water_for_ducks
  total_water / (pumping_time : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_casey_pumping_rate_l721_72123


namespace NUMINAMATH_CALUDE_twelve_person_tournament_matches_l721_72144

/-- Calculate the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 12-person round-robin tournament has 66 matches -/
theorem twelve_person_tournament_matches : 
  roundRobinMatches 12 = 66 := by
sorry

end NUMINAMATH_CALUDE_twelve_person_tournament_matches_l721_72144


namespace NUMINAMATH_CALUDE_min_participants_quiz_l721_72119

-- Define the number of correct answers for each question
def correct_q1 : ℕ := 90
def correct_q2 : ℕ := 50
def correct_q3 : ℕ := 40
def correct_q4 : ℕ := 20

-- Define the maximum number of questions a participant can answer correctly
def max_correct_per_participant : ℕ := 2

-- Define the total number of correct answers
def total_correct_answers : ℕ := correct_q1 + correct_q2 + correct_q3 + correct_q4

-- Theorem stating the minimum number of participants
theorem min_participants_quiz : 
  ∀ n : ℕ, 
  (n * max_correct_per_participant ≥ total_correct_answers) → 
  (∀ m : ℕ, m < n → m * max_correct_per_participant < total_correct_answers) → 
  n = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_participants_quiz_l721_72119


namespace NUMINAMATH_CALUDE_g_fixed_points_l721_72158

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_fixed_points_l721_72158


namespace NUMINAMATH_CALUDE_profit_percentage_l721_72105

theorem profit_percentage (selling_price cost_price : ℝ) 
  (h : cost_price = 0.92 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 92 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l721_72105


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l721_72184

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) (h1 : n ≥ 2) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a = k * b) (h5 : k > 0) :
  (-n * a^(n-1) * b + (n * (n-1) / 2) * a^(n-2) * b^2 = 0) → n = 2 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l721_72184


namespace NUMINAMATH_CALUDE_mice_eaten_in_decade_l721_72121

/-- Represents the number of weeks in a year -/
def weeksInYear : ℕ := 52

/-- Represents the eating frequency (in weeks) for the snake in its first year -/
def firstYearFrequency : ℕ := 4

/-- Represents the eating frequency (in weeks) for the snake in its second year -/
def secondYearFrequency : ℕ := 3

/-- Represents the eating frequency (in weeks) for the snake after its second year -/
def laterYearsFrequency : ℕ := 2

/-- Calculates the number of mice eaten in the first year -/
def miceEatenFirstYear : ℕ := weeksInYear / firstYearFrequency

/-- Calculates the number of mice eaten in the second year -/
def miceEatenSecondYear : ℕ := weeksInYear / secondYearFrequency

/-- Calculates the number of mice eaten in one year after the second year -/
def miceEatenPerLaterYear : ℕ := weeksInYear / laterYearsFrequency

/-- Represents the number of years in a decade -/
def yearsInDecade : ℕ := 10

/-- Theorem stating the total number of mice eaten over a decade -/
theorem mice_eaten_in_decade : 
  miceEatenFirstYear + miceEatenSecondYear + (yearsInDecade - 2) * miceEatenPerLaterYear = 238 := by
  sorry

end NUMINAMATH_CALUDE_mice_eaten_in_decade_l721_72121


namespace NUMINAMATH_CALUDE_tangent_line_equation_l721_72104

-- Define the curve
def f (x : ℝ) : ℝ := -x^2 + 4

-- Define the point of interest
def x₀ : ℝ := -1

-- Define the slope of the tangent line
def k : ℝ := -2 * x₀

-- Define the y-coordinate of the point on the curve
def y₀ : ℝ := f x₀

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y = k * (x - x₀) + y₀ ↔ y = 2*x + 5 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l721_72104


namespace NUMINAMATH_CALUDE_product_xy_value_l721_72116

/-- A parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  is_parallelogram : EF = GH 1 ∧ FG 1 = HE

/-- The product of x and y in the given parallelogram -/
def product_xy (p : Parallelogram) (x y : ℝ) : ℝ := x * y

/-- Theorem: The product of x and y in the given parallelogram is 18 * ∛4 -/
theorem product_xy_value (p : Parallelogram) 
  (h1 : p.EF = 110)
  (h2 : p.FG = fun y => 16 * y^3)
  (h3 : p.GH = fun x => 6 * x + 2)
  (h4 : p.HE = 64)
  : ∃ x y, product_xy p x y = 18 * (4 ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_product_xy_value_l721_72116
