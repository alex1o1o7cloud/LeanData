import Mathlib

namespace cups_filled_l1465_146543

-- Define the volume of water in milliliters
def water_volume : ℕ := 1000

-- Define the cup size in milliliters
def cup_size : ℕ := 200

-- Theorem to prove
theorem cups_filled (water_volume : ℕ) (cup_size : ℕ) :
  water_volume = 1000 → cup_size = 200 → water_volume / cup_size = 5 := by
  sorry

end cups_filled_l1465_146543


namespace pen_price_relationship_l1465_146584

/-- Given a box of pens with a selling price of $16 and containing 10 pens,
    prove that the relationship between the selling price of one pen (y)
    and the number of pens (x) is y = 1.6x. -/
theorem pen_price_relationship (box_price : ℝ) (pens_per_box : ℕ) (x y : ℝ) :
  box_price = 16 →
  pens_per_box = 10 →
  y = (box_price / pens_per_box) * x →
  y = 1.6 * x :=
by
  sorry


end pen_price_relationship_l1465_146584


namespace birds_in_marsh_l1465_146530

theorem birds_in_marsh (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end birds_in_marsh_l1465_146530


namespace winter_olympics_volunteer_allocation_l1465_146536

theorem winter_olympics_volunteer_allocation :
  let n_volunteers : ℕ := 5
  let n_events : ℕ := 4
  let allocation_schemes : ℕ := (n_volunteers.choose 2) * n_events.factorial
  allocation_schemes = 240 :=
by sorry

end winter_olympics_volunteer_allocation_l1465_146536


namespace sock_pairs_count_l1465_146500

def total_socks : ℕ := 12
def white_socks : ℕ := 5
def brown_socks : ℕ := 5
def blue_socks : ℕ := 2

def same_color_pairs : ℕ := Nat.choose white_socks 2 + Nat.choose brown_socks 2 + Nat.choose blue_socks 2

theorem sock_pairs_count : same_color_pairs = 21 := by
  sorry

end sock_pairs_count_l1465_146500


namespace polynomial_sum_theorem_l1465_146553

theorem polynomial_sum_theorem (p q r s : ℤ) :
  (∀ x : ℝ, (x^2 + p*x + q) * (x^2 + r*x + s) = x^4 + 3*x^3 - 4*x^2 + 9*x + 7) →
  p + q + r + s = 11 := by
sorry

end polynomial_sum_theorem_l1465_146553


namespace carol_wins_probability_l1465_146515

/-- Represents the probability of tossing a six -/
def prob_six : ℚ := 1 / 6

/-- Represents the probability of not tossing a six -/
def prob_not_six : ℚ := 1 - prob_six

/-- Represents the number of players -/
def num_players : ℕ := 4

/-- The probability of Carol winning in one cycle -/
def prob_carol_win_cycle : ℚ := prob_not_six^2 * prob_six * prob_not_six

/-- The probability of no one winning in one cycle -/
def prob_no_win_cycle : ℚ := prob_not_six^num_players

/-- Theorem: The probability of Carol being the first to toss a six 
    in a repeated die-tossing game with four players is 125/671 -/
theorem carol_wins_probability : 
  prob_carol_win_cycle / (1 - prob_no_win_cycle) = 125 / 671 := by
  sorry

end carol_wins_probability_l1465_146515


namespace range_of_m_l1465_146596

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/3)^(-x) - 2 else 2 * Real.log (-x) / Real.log 3

theorem range_of_m (m : ℝ) (h : f m > 1) :
  m ∈ Set.Ioi 1 ∪ Set.Iic (-Real.sqrt 3) :=
sorry

end range_of_m_l1465_146596


namespace sample_size_from_model_a_l1465_146510

/-- Represents the ratio of quantities for models A, B, and C -/
structure ProductRatio :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Represents a stratified sample -/
structure StratifiedSample :=
  (size : ℕ) (model_a_count : ℕ)

/-- Theorem: Given the product ratio and model A count in a stratified sample, 
    prove the total sample size -/
theorem sample_size_from_model_a
  (ratio : ProductRatio)
  (sample : StratifiedSample)
  (h_ratio : ratio = ⟨3, 4, 7⟩)
  (h_model_a : sample.model_a_count = 15) :
  sample.size = 70 :=
sorry

end sample_size_from_model_a_l1465_146510


namespace complement_I_M_l1465_146519

def M : Set ℕ := {0, 1}
def I : Set ℕ := {0, 1, 2, 3, 4, 5}

theorem complement_I_M : (I \ M) = {2, 3, 4, 5} := by sorry

end complement_I_M_l1465_146519


namespace average_weight_of_a_and_b_l1465_146526

theorem average_weight_of_a_and_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 47 →
  b = 39 →
  (a + b) / 2 = 40 :=
by sorry

end average_weight_of_a_and_b_l1465_146526


namespace x_value_when_y_is_3_l1465_146502

/-- The inverse square relationship between x and y -/
def inverse_square_relation (x y : ℝ) (k : ℝ) : Prop :=
  x = k / (y ^ 2)

/-- Theorem: Given the inverse square relationship between x and y,
    and the condition that x ≈ 0.1111111111111111 when y = 9,
    prove that x = 1 when y = 3 -/
theorem x_value_when_y_is_3
  (h1 : ∃ k, ∀ x y, inverse_square_relation x y k)
  (h2 : ∃ x, inverse_square_relation x 9 (9 * 0.1111111111111111)) :
  ∃ x, inverse_square_relation x 3 1 ∧ x = 1 := by
  sorry

end x_value_when_y_is_3_l1465_146502


namespace joseph_running_distance_l1465_146534

/-- Calculates the total distance run over a number of days with a given initial distance and daily increase. -/
def totalDistance (initialDistance : ℕ) (dailyIncrease : ℕ) (days : ℕ) : ℕ :=
  (days * (2 * initialDistance + (days - 1) * dailyIncrease)) / 2

/-- Proves that given an initial distance of 900 meters, a daily increase of 200 meters,
    and running for 3 days, the total distance run is 3300 meters. -/
theorem joseph_running_distance :
  totalDistance 900 200 3 = 3300 := by
  sorry

end joseph_running_distance_l1465_146534


namespace square_triangle_equal_area_l1465_146597

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * triangle_base →
  triangle_base = 16 := by
  sorry

end square_triangle_equal_area_l1465_146597


namespace monotonic_exp_minus_mx_l1465_146572

/-- If f(x) = e^x - mx is monotonically increasing on [0, +∞), then m ≤ 1 -/
theorem monotonic_exp_minus_mx (m : ℝ) :
  (∀ x : ℝ, x ≥ 0 → Monotone (fun x : ℝ ↦ Real.exp x - m * x)) →
  m ≤ 1 := by
  sorry

end monotonic_exp_minus_mx_l1465_146572


namespace kayak_rental_cost_l1465_146548

/-- Represents the daily rental business for canoes and kayaks -/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_count : ℕ
  kayak_count : ℕ
  total_revenue : ℕ

/-- The rental business satisfies the given conditions -/
def valid_rental_business (rb : RentalBusiness) : Prop :=
  rb.canoe_cost = 9 ∧
  rb.canoe_count = rb.kayak_count + 6 ∧
  4 * rb.kayak_count = 3 * rb.canoe_count ∧
  rb.total_revenue = rb.canoe_cost * rb.canoe_count + rb.kayak_cost * rb.kayak_count ∧
  rb.total_revenue = 432

/-- The theorem stating that under the given conditions, the kayak rental cost is $12 per day -/
theorem kayak_rental_cost (rb : RentalBusiness) 
  (h : valid_rental_business rb) : rb.kayak_cost = 12 := by
  sorry

end kayak_rental_cost_l1465_146548


namespace wang_hua_practice_days_l1465_146577

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in the Gregorian calendar -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInMonth (year : Nat) (month : Nat) : Nat :=
  match month with
  | 2 => if isLeapYear year then 29 else 28
  | 4 | 6 | 9 | 11 => 30
  | _ => 31

def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def countPracticeDays (year : Nat) (month : Nat) : Nat :=
  sorry

theorem wang_hua_practice_days :
  let newYearsDay2016 := Date.mk 2016 1 1
  let augustFirst2016 := Date.mk 2016 8 1
  dayOfWeek newYearsDay2016 = DayOfWeek.Friday →
  isLeapYear 2016 = true →
  countPracticeDays 2016 8 = 9 :=
by sorry

end wang_hua_practice_days_l1465_146577


namespace smallest_sum_proof_l1465_146535

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/6, 1/3 + 1/9]
  (∀ x ∈ sums, 1/3 + 1/9 ≤ x) ∧ (1/3 + 1/9 = 4/9) := by
  sorry

end smallest_sum_proof_l1465_146535


namespace exists_integer_root_polynomial_l1465_146538

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Function to evaluate a quadratic polynomial at a given x -/
def evaluate (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate to check if a quadratic polynomial has integer roots -/
def has_integer_roots (p : QuadraticPolynomial) : Prop :=
  ∃ (r₁ r₂ : ℤ), p.a * r₁^2 + p.b * r₁ + p.c = 0 ∧ p.a * r₂^2 + p.b * r₂ + p.c = 0

/-- The main theorem -/
theorem exists_integer_root_polynomial :
  ∃ (p : QuadraticPolynomial),
    p.a = 1 ∧
    (evaluate p (-1) ≤ evaluate ⟨1, 10, 20⟩ (-1) ∧ evaluate p (-1) ≥ evaluate ⟨1, 20, 10⟩ (-1)) ∧
    has_integer_roots p :=
by
  sorry

end exists_integer_root_polynomial_l1465_146538


namespace puppy_group_arrangements_eq_2520_l1465_146528

/-- The number of ways to divide 12 puppies into groups of 4, 6, and 2,
    with Coco in the 4-puppy group and Rocky in the 6-puppy group. -/
def puppy_group_arrangements : ℕ :=
  Nat.choose 10 3 * Nat.choose 7 5

/-- Theorem stating that the number of puppy group arrangements is 2520. -/
theorem puppy_group_arrangements_eq_2520 :
  puppy_group_arrangements = 2520 := by
  sorry

#eval puppy_group_arrangements

end puppy_group_arrangements_eq_2520_l1465_146528


namespace triangle_sides_from_heights_and_median_l1465_146540

/-- Given a triangle with heights m₁ and m₂ corresponding to sides a and b respectively,
    and median k₃ corresponding to side c, prove that the sides a and b can be expressed as:
    a = m₂ / sin(γ) and b = m₁ / sin(γ), where γ is the angle opposite to side c. -/
theorem triangle_sides_from_heights_and_median 
  (m₁ m₂ k₃ : ℝ) (γ : ℝ) (hm₁ : m₁ > 0) (hm₂ : m₂ > 0) (hk₃ : k₃ > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ (a b : ℝ), a = m₂ / Real.sin γ ∧ b = m₁ / Real.sin γ := by
  sorry

end triangle_sides_from_heights_and_median_l1465_146540


namespace delivery_driver_stops_l1465_146509

theorem delivery_driver_stops (initial_stops additional_stops : ℕ) 
  (h1 : initial_stops = 3) 
  (h2 : additional_stops = 4) : 
  initial_stops + additional_stops = 7 := by
  sorry

end delivery_driver_stops_l1465_146509


namespace black_ribbon_count_l1465_146591

theorem black_ribbon_count (total : ℕ) (silver : ℕ) : 
  silver = 40 →
  (1 : ℚ) / 4 + 1 / 3 + 1 / 6 + 1 / 12 + (silver : ℚ) / total = 1 →
  (total : ℚ) / 12 = 20 :=
by sorry

end black_ribbon_count_l1465_146591


namespace third_quiz_score_l1465_146563

theorem third_quiz_score (score1 score2 score3 : ℕ) : 
  score1 = 91 → 
  score2 = 90 → 
  (score1 + score2 + score3) / 3 = 91 → 
  score3 = 92 := by
sorry

end third_quiz_score_l1465_146563


namespace complex_square_value_l1465_146582

theorem complex_square_value : ((1 - Complex.I * Real.sqrt 3) / Complex.I) ^ 2 = 2 + Complex.I * (2 * Real.sqrt 3) := by
  sorry

end complex_square_value_l1465_146582


namespace y1_greater_than_y2_l1465_146505

/-- Given a line y = -3x + 5 and two points (-6, y₁) and (3, y₂) on this line,
    prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -3 * (-6) + 5) →  -- Point (-6, y₁) lies on the line
  (y₂ = -3 * 3 + 5) →     -- Point (3, y₂) lies on the line
  y₁ > y₂ := by
sorry

end y1_greater_than_y2_l1465_146505


namespace flower_bed_fraction_l1465_146550

/-- Represents a rectangular yard with flower beds -/
structure FlowerYard where
  /-- Length of the yard -/
  length : ℝ
  /-- Width of the yard -/
  width : ℝ
  /-- Radius of the circular flower bed -/
  circle_radius : ℝ
  /-- Length of the shorter parallel side of the trapezoidal remainder -/
  trapezoid_short_side : ℝ
  /-- Length of the longer parallel side of the trapezoidal remainder -/
  trapezoid_long_side : ℝ

/-- Theorem stating the fraction of the yard occupied by flower beds -/
theorem flower_bed_fraction (yard : FlowerYard) 
  (h1 : yard.trapezoid_short_side = 20)
  (h2 : yard.trapezoid_long_side = 40)
  (h3 : yard.circle_radius = 2)
  (h4 : yard.length = yard.trapezoid_long_side)
  (h5 : yard.width = (yard.trapezoid_long_side - yard.trapezoid_short_side) / 2) :
  (100 + 4 * Real.pi) / 400 = 
    ((yard.trapezoid_long_side - yard.trapezoid_short_side)^2 / 4 + yard.circle_radius^2 * Real.pi) / 
    (yard.length * yard.width) :=
by sorry

end flower_bed_fraction_l1465_146550


namespace min_S_proof_l1465_146522

/-- The number of dice rolled -/
def n : ℕ := 333

/-- The target sum -/
def target_sum : ℕ := 1994

/-- The minimum value of S -/
def min_S : ℕ := 334

/-- The probability of obtaining a sum of k when rolling n standard dice -/
noncomputable def prob_sum (k : ℕ) : ℝ := sorry

theorem min_S_proof :
  (prob_sum target_sum > 0) ∧
  (prob_sum target_sum = prob_sum min_S) ∧
  (∀ S : ℕ, S < min_S → prob_sum target_sum ≠ prob_sum S) :=
sorry

end min_S_proof_l1465_146522


namespace polygon_existence_l1465_146598

/-- A polygon on a unit grid --/
structure GridPolygon where
  sides : ℕ
  area : ℕ
  vertices : List (ℕ × ℕ)

/-- Predicate to check if a GridPolygon is valid --/
def isValidGridPolygon (p : GridPolygon) : Prop :=
  p.sides = p.vertices.length ∧
  p.area ≤ (List.maximum (p.vertices.map Prod.fst)).getD 0 * 
           (List.maximum (p.vertices.map Prod.snd)).getD 0

theorem polygon_existence : 
  (∃ p : GridPolygon, p.sides = 20 ∧ p.area = 9 ∧ isValidGridPolygon p) ∧
  (∃ p : GridPolygon, p.sides = 100 ∧ p.area = 49 ∧ isValidGridPolygon p) := by
  sorry

end polygon_existence_l1465_146598


namespace absolute_value_not_positive_l1465_146586

theorem absolute_value_not_positive (x : ℚ) : |4*x - 8| ≤ 0 ↔ x = 2 := by
  sorry

end absolute_value_not_positive_l1465_146586


namespace derivative_y_l1465_146537

noncomputable def y (x : ℝ) : ℝ := Real.cos x / x

theorem derivative_y (x : ℝ) (hx : x ≠ 0) :
  deriv y x = -((x * Real.sin x + Real.cos x) / x^2) := by
  sorry

end derivative_y_l1465_146537


namespace optimal_cylinder_ratio_l1465_146552

/-- The optimal ratio of height to radius for a cylinder with minimal surface area --/
theorem optimal_cylinder_ratio (V : ℝ) (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  V = π * r^2 * h → (∀ h' r', h' > 0 → r' > 0 → V = π * r'^2 * h' → 
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') → 
  h / r = 2 :=
sorry

end optimal_cylinder_ratio_l1465_146552


namespace sphere_radius_non_uniform_l1465_146579

/-- The radius of a sphere given its curved surface area in a non-uniform coordinate system -/
theorem sphere_radius_non_uniform (surface_area : ℝ) (k1 k2 k3 : ℝ) (h : surface_area = 64 * Real.pi) :
  ∃ (r : ℝ), r = 4 ∧ surface_area = 4 * Real.pi * r^2 := by
  sorry

end sphere_radius_non_uniform_l1465_146579


namespace ten_point_circle_triangles_l1465_146575

/-- The number of triangles that can be formed from n distinct points on a circle's circumference -/
def total_triangles (n : ℕ) : ℕ := Nat.choose n 3

/-- The number of triangles where one side subtends an arc greater than 180 degrees -/
def long_arc_triangles (n : ℕ) : ℕ := 2 * n

/-- The number of valid triangles that can be formed from n distinct points on a circle's circumference,
    where no side subtends an arc greater than 180 degrees -/
def valid_triangles (n : ℕ) : ℕ := total_triangles n - long_arc_triangles n

theorem ten_point_circle_triangles :
  valid_triangles 10 = 100 := by sorry

end ten_point_circle_triangles_l1465_146575


namespace first_four_terms_l1465_146524

def a (n : ℕ) : ℚ := (1 + (-1)^(n+1)) / 2

theorem first_four_terms :
  (a 1 = 1) ∧ (a 2 = 0) ∧ (a 3 = 1) ∧ (a 4 = 0) := by
  sorry

end first_four_terms_l1465_146524


namespace women_count_l1465_146564

/-- Represents the work done by one woman in one day -/
def W : ℝ := sorry

/-- Represents the work done by one child in one day -/
def C : ℝ := sorry

/-- Represents the number of women working initially -/
def x : ℝ := sorry

/-- The total work to be completed -/
def total_work : ℝ := sorry

theorem women_count : x = 10 := by
  have h1 : 5 * x * W = total_work := sorry
  have h2 : 100 * C = total_work := sorry
  have h3 : 5 * (5 * W + 10 * C) = total_work := sorry
  sorry

end women_count_l1465_146564


namespace product_of_fractions_l1465_146546

theorem product_of_fractions (p : ℝ) (hp : p ≠ 0) :
  (p^3 + 4*p^2 + 10*p + 12) / (p^3 - p^2 + 2*p + 16) *
  (p^3 - 3*p^2 + 8*p) / (p^2 + 2*p + 6) =
  ((p^3 + 4*p^2 + 10*p + 12) * (p^3 - 3*p^2 + 8*p)) /
  ((p^3 - p^2 + 2*p + 16) * (p^2 + 2*p + 6)) :=
by sorry

end product_of_fractions_l1465_146546


namespace inequality_range_l1465_146508

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 - Real.log x / Real.log m < 0) ↔ 
  m ∈ Set.Icc (1/16) 1 ∧ m ≠ 1 :=
sorry

end inequality_range_l1465_146508


namespace arithmetic_calculation_l1465_146513

theorem arithmetic_calculation : 12 - (-18) + (-7) = 23 := by
  sorry

end arithmetic_calculation_l1465_146513


namespace polygon_sides_l1465_146589

theorem polygon_sides (n : ℕ) : 
  (((n - 2) * 180) / 360 : ℚ) = 3 / 1 → n = 8 := by
  sorry

end polygon_sides_l1465_146589


namespace hyperbolas_same_asymptotes_l1465_146569

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - (x²/M) = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) →
  (∀ x y : ℝ, y = 4/3 * x ↔ y = 5/Real.sqrt M * x) →
  M = 225/16 := by
sorry

end hyperbolas_same_asymptotes_l1465_146569


namespace quadratic_equation_roots_l1465_146503

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + a*x₁ + 5 = 0 ∧ 
    x₂^2 + a*x₂ + 5 = 0 ∧ 
    x₁^2 + 250/(19*x₂^3) = x₂^2 + 250/(19*x₁^3)) → 
  a = 10 := by sorry

end quadratic_equation_roots_l1465_146503


namespace bus_meeting_problem_l1465_146587

theorem bus_meeting_problem (n k : ℕ) : n > 3 → 
  (n * (n - 1) * (2 * k - 1) = 600) → 
  ((n = 4 ∧ k = 13) ∨ (n = 5 ∧ k = 8)) := by
  sorry

end bus_meeting_problem_l1465_146587


namespace probability_two_red_balls_l1465_146541

def total_balls : ℕ := 6 + 5 + 2

def red_balls : ℕ := 6

theorem probability_two_red_balls :
  let prob_first_red : ℚ := red_balls / total_balls
  let prob_second_red : ℚ := (red_balls - 1) / (total_balls - 1)
  prob_first_red * prob_second_red = 5 / 26 := by sorry

end probability_two_red_balls_l1465_146541


namespace min_packs_needed_l1465_146567

/-- Represents the number of cans in each pack type -/
def pack_sizes : Fin 3 → ℕ
  | 0 => 8
  | 1 => 15
  | 2 => 18

/-- The total number of cans needed -/
def total_cans : ℕ := 95

/-- The maximum number of packs allowed for each type -/
def max_packs : ℕ := 4

/-- A function to calculate the total number of cans from a given combination of packs -/
def total_from_packs (x y z : ℕ) : ℕ :=
  x * pack_sizes 0 + y * pack_sizes 1 + z * pack_sizes 2

/-- The main theorem to prove -/
theorem min_packs_needed :
  ∃ (x y z : ℕ),
    x ≤ max_packs ∧ y ≤ max_packs ∧ z ≤ max_packs ∧
    total_from_packs x y z = total_cans ∧
    x + y + z = 6 ∧
    (∀ (a b c : ℕ),
      a ≤ max_packs → b ≤ max_packs → c ≤ max_packs →
      total_from_packs a b c = total_cans →
      a + b + c ≥ 6) :=
sorry

end min_packs_needed_l1465_146567


namespace bart_mixtape_length_l1465_146580

/-- Calculates the total length of a mixtape in minutes -/
def mixtape_length (songs_side1 : ℕ) (songs_side2 : ℕ) (song_duration : ℕ) : ℕ :=
  (songs_side1 + songs_side2) * song_duration

/-- Proves that the total length of Bart's mixtape is 40 minutes -/
theorem bart_mixtape_length :
  mixtape_length 6 4 4 = 40 := by
  sorry

end bart_mixtape_length_l1465_146580


namespace inequality_solution_interval_l1465_146583

theorem inequality_solution_interval (a : ℝ) : 
  (∀ x, Real.sqrt (x + a) ≥ x) ∧ 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    Real.sqrt (x₁ + a) = x₁ ∧ 
    Real.sqrt (x₂ + a) = x₂ ∧ 
    |x₁ - x₂| = 4 * |a|) →
  a = 4/9 ∨ a = (1 - Real.sqrt 5) / 8 :=
by sorry

end inequality_solution_interval_l1465_146583


namespace number_of_nickels_l1465_146518

def quarter_value : Rat := 25 / 100
def dime_value : Rat := 10 / 100
def nickel_value : Rat := 5 / 100
def penny_value : Rat := 1 / 100

def num_quarters : Nat := 10
def num_dimes : Nat := 3
def num_pennies : Nat := 200
def total_amount : Rat := 5

theorem number_of_nickels : 
  ∃ (num_nickels : Nat), 
    (num_quarters : Nat) * quarter_value + 
    (num_dimes : Nat) * dime_value + 
    (num_nickels : Nat) * nickel_value + 
    (num_pennies : Nat) * penny_value = total_amount ∧ 
    num_nickels = 4 := by
  sorry

end number_of_nickels_l1465_146518


namespace parabola_transformation_transformation_is_right_shift_2_l1465_146527

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ := (x + 5) * (x - 3)

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ := (x + 3) * (x - 5)

-- Define the transformation
def transformation (x : ℝ) : ℝ := x + 2

-- Theorem stating the transformation between the two parabolas
theorem parabola_transformation :
  ∀ x : ℝ, parabola1 x = parabola2 (transformation x) :=
by
  sorry

-- Theorem stating that the transformation is a shift of 2 units to the right
theorem transformation_is_right_shift_2 :
  ∀ x : ℝ, transformation x = x + 2 :=
by
  sorry

end parabola_transformation_transformation_is_right_shift_2_l1465_146527


namespace elderly_in_sample_is_18_l1465_146517

/-- Represents the distribution of employees in a company and their sampling --/
structure EmployeeSampling where
  total : ℕ
  young : ℕ
  elderly : ℕ
  sampledYoung : ℕ
  middleAged : ℕ := 2 * elderly
  youngRatio : ℚ := young / total
  elderlyRatio : ℚ := elderly / total

/-- The number of elderly employees in the sample given the conditions --/
def elderlyInSample (e : EmployeeSampling) : ℚ :=
  e.elderlyRatio * (e.sampledYoung / e.youngRatio)

/-- Theorem stating the number of elderly employees in the sample --/
theorem elderly_in_sample_is_18 (e : EmployeeSampling) 
    (h1 : e.total = 430)
    (h2 : e.young = 160)
    (h3 : e.sampledYoung = 32)
    (h4 : e.total = e.young + e.middleAged + e.elderly) :
  elderlyInSample e = 18 := by
  sorry

#eval elderlyInSample { total := 430, young := 160, elderly := 90, sampledYoung := 32 }

end elderly_in_sample_is_18_l1465_146517


namespace cos_alpha_value_l1465_146570

theorem cos_alpha_value (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) :
  Real.cos α = 1 / 5 := by
  sorry

end cos_alpha_value_l1465_146570


namespace cosine_sum_simplification_l1465_146578

theorem cosine_sum_simplification (x : ℝ) (k : ℤ) :
  Real.cos ((6 * k + 1) * π / 3 + x) + Real.cos ((6 * k - 1) * π / 3 + x) = Real.cos x :=
by sorry

end cosine_sum_simplification_l1465_146578


namespace office_chairs_probability_l1465_146565

theorem office_chairs_probability (black_chairs brown_chairs : ℕ) 
  (h1 : black_chairs = 15)
  (h2 : brown_chairs = 18) :
  let total_chairs := black_chairs + brown_chairs
  let prob_same_color := (black_chairs * (black_chairs - 1) + brown_chairs * (brown_chairs - 1)) / (total_chairs * (total_chairs - 1))
  prob_same_color = 43 / 88 := by
sorry

end office_chairs_probability_l1465_146565


namespace share_multiple_l1465_146594

theorem share_multiple (a b c k : ℚ) : 
  a + b + c = 585 →
  4 * a = 6 * b →
  4 * a = k * c →
  c = 260 →
  k = 3 := by sorry

end share_multiple_l1465_146594


namespace seating_arrangements_count_l1465_146571

/-- The number of ways three people can sit in a row of six chairs -/
def seating_arrangements : ℕ := 6 * 5 * 4

/-- Theorem stating that the number of seating arrangements is 120 -/
theorem seating_arrangements_count : seating_arrangements = 120 := by
  sorry

end seating_arrangements_count_l1465_146571


namespace dot_product_v_w_l1465_146551

def v : Fin 3 → ℝ := ![(-5 : ℝ), 2, -3]
def w : Fin 3 → ℝ := ![7, -4, 6]

theorem dot_product_v_w :
  (Finset.univ.sum fun i => v i * w i) = -61 := by sorry

end dot_product_v_w_l1465_146551


namespace cos_sin_sum_equals_sqrt3_over_2_l1465_146533

theorem cos_sin_sum_equals_sqrt3_over_2 :
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end cos_sin_sum_equals_sqrt3_over_2_l1465_146533


namespace mms_per_pack_is_40_l1465_146574

/-- The number of sundaes made on Monday -/
def monday_sundaes : ℕ := 40

/-- The number of m&ms per sundae on Monday -/
def monday_mms_per_sundae : ℕ := 6

/-- The number of sundaes made on Tuesday -/
def tuesday_sundaes : ℕ := 20

/-- The number of m&ms per sundae on Tuesday -/
def tuesday_mms_per_sundae : ℕ := 10

/-- The total number of m&m packs used -/
def total_packs : ℕ := 11

/-- The number of m&ms in each pack -/
def mms_per_pack : ℕ := (monday_sundaes * monday_mms_per_sundae + tuesday_sundaes * tuesday_mms_per_sundae) / total_packs

theorem mms_per_pack_is_40 : mms_per_pack = 40 := by
  sorry

end mms_per_pack_is_40_l1465_146574


namespace eighth_fibonacci_term_l1465_146554

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_fibonacci_term :
  fibonacci 7 = 21 :=
by sorry

end eighth_fibonacci_term_l1465_146554


namespace vine_paint_time_l1465_146568

/-- Time to paint different flowers and total painting time -/
def paint_problem (lily_time rose_time orchid_time vine_time : ℕ) 
  (total_time lily_count rose_count orchid_count vine_count : ℕ) : Prop :=
  lily_time * lily_count + rose_time * rose_count + 
  orchid_time * orchid_count + vine_time * vine_count = total_time

/-- Theorem stating the time to paint a vine -/
theorem vine_paint_time : 
  ∃ (vine_time : ℕ), 
    paint_problem 5 7 3 vine_time 213 17 10 6 20 ∧ 
    vine_time = 2 := by
  sorry

end vine_paint_time_l1465_146568


namespace fraction_sum_power_six_l1465_146558

theorem fraction_sum_power_six : (5 / 3 : ℚ)^6 + (2 / 3 : ℚ)^6 = 15689 / 729 := by
  sorry

end fraction_sum_power_six_l1465_146558


namespace missing_number_equation_l1465_146561

theorem missing_number_equation (x : ℤ) : 10010 - x * 3 * 2 = 9938 ↔ x = 12 := by
  sorry

end missing_number_equation_l1465_146561


namespace cost_calculation_l1465_146504

/-- The cost of buying pens and notebooks -/
def total_cost (pen_price notebook_price : ℝ) : ℝ :=
  5 * pen_price + 8 * notebook_price

/-- Theorem: The total cost of 5 pens at 'a' yuan each and 8 notebooks at 'b' yuan each is 5a + 8b yuan -/
theorem cost_calculation (a b : ℝ) : total_cost a b = 5 * a + 8 * b := by
  sorry

end cost_calculation_l1465_146504


namespace point_displacement_on_line_l1465_146595

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the line x = (y / 2) - (2 / 5) -/
def onLine (p : Point) : Prop :=
  p.x = p.y / 2 - 2 / 5

theorem point_displacement_on_line (m n p : ℝ) :
  onLine ⟨m, n⟩ ∧ onLine ⟨m + p, n + 4⟩ → p = 2 := by
  sorry

end point_displacement_on_line_l1465_146595


namespace complex_equality_implies_ratio_l1465_146573

theorem complex_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 →
  b / a = 1 := by
sorry

end complex_equality_implies_ratio_l1465_146573


namespace equal_interest_principal_second_amount_calculation_l1465_146593

/-- Given two investments with equal interest, calculate the principal of the second investment -/
theorem equal_interest_principal (p₁ r₁ t₁ r₂ t₂ : ℚ) (hp₁ : p₁ > 0) (hr₁ : r₁ > 0) (ht₁ : t₁ > 0) (hr₂ : r₂ > 0) (ht₂ : t₂ > 0) :
  p₁ * r₁ * t₁ = (p₁ * r₁ * t₁ / (r₂ * t₂)) * r₂ * t₂ :=
by sorry

/-- The second amount that produces the same interest as Rs 200 at 10% for 12 years, when invested at 12% for 5 years, is Rs 400 -/
theorem second_amount_calculation :
  let p₁ : ℚ := 200
  let r₁ : ℚ := 10 / 100
  let t₁ : ℚ := 12
  let r₂ : ℚ := 12 / 100
  let t₂ : ℚ := 5
  (p₁ * r₁ * t₁ / (r₂ * t₂)) = 400 :=
by sorry

end equal_interest_principal_second_amount_calculation_l1465_146593


namespace distance_between_locations_l1465_146556

theorem distance_between_locations (speed_A speed_B : ℝ) (time : ℝ) (remaining_distance : ℝ) :
  speed_B = (4/5) * speed_A →
  time = 3 →
  remaining_distance = 3 →
  ∃ (distance_AB : ℝ),
    distance_AB = speed_A * time + speed_B * time + remaining_distance :=
by sorry

end distance_between_locations_l1465_146556


namespace equation_solution_l1465_146516

theorem equation_solution : ∃ x : ℝ, 13 + Real.sqrt (x + 5 * 3 - 3 * 3) = 14 ∧ x = -5 := by
  sorry

end equation_solution_l1465_146516


namespace triangle_segment_relation_l1465_146547

/-- Given a triangle ABC with point D on AB and point E on AD, 
    prove the relation for FC where F is on AC. -/
theorem triangle_segment_relation 
  (A B C D E F : ℝ × ℝ) 
  (h1 : dist D C = 6)
  (h2 : dist C B = 9)
  (h3 : dist A B = 1/5 * dist A D)
  (h4 : dist E D = 2/3 * dist A D) :
  dist F C = (dist E D * dist C A) / dist D A :=
sorry

end triangle_segment_relation_l1465_146547


namespace pure_imaginary_solutions_l1465_146539

theorem pure_imaginary_solutions (x : ℂ) :
  (x^4 - 5*x^3 + 10*x^2 - 50*x - 75 = 0) ∧ (∃ k : ℝ, x = k * I) ↔
  (x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
sorry

end pure_imaginary_solutions_l1465_146539


namespace perpendicular_vectors_y_value_l1465_146529

theorem perpendicular_vectors_y_value :
  let a : Fin 3 → ℝ := ![1, 2, 6]
  let b : Fin 3 → ℝ := ![2, y, -1]
  (∀ i : Fin 3, (a • b) = 0) → y = 2 := by
  sorry

end perpendicular_vectors_y_value_l1465_146529


namespace equation_satisfied_l1465_146506

theorem equation_satisfied (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end equation_satisfied_l1465_146506


namespace geometric_sequence_tan_property_l1465_146523

/-- Given a geometric sequence {aₙ} satisfying certain conditions, 
    prove that tan(a₁a₁₃) = √3 -/
theorem geometric_sequence_tan_property 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_condition : a 3 * a 11 + 2 * (a 7)^2 = 4 * Real.pi) : 
  Real.tan (a 1 * a 13) = Real.sqrt 3 := by
  sorry

end geometric_sequence_tan_property_l1465_146523


namespace arithmetic_geometric_sequence_problem_l1465_146566

/-- Given three real numbers forming an arithmetic sequence with sum 12,
    and their translations forming a geometric sequence,
    prove that the only solutions are (1, 4, 7) and (10, 4, -2) -/
theorem arithmetic_geometric_sequence_problem (a b c : ℝ) : 
  (∃ d : ℝ, b - a = d ∧ c - b = d) →  -- arithmetic sequence condition
  (a + b + c = 12) →                  -- sum condition
  (∃ r : ℝ, (b + 2) = (a + 2) * r ∧ (c + 5) = (b + 2) * r) →  -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by sorry

end arithmetic_geometric_sequence_problem_l1465_146566


namespace jackie_exercise_hours_l1465_146588

/-- Represents Jackie's daily schedule --/
structure DailySchedule where
  total_hours : ℕ
  work_hours : ℕ
  sleep_hours : ℕ
  free_hours : ℕ

/-- Calculates the number of hours Jackie spends exercising --/
def exercise_hours (schedule : DailySchedule) : ℕ :=
  schedule.total_hours - (schedule.work_hours + schedule.sleep_hours + schedule.free_hours)

/-- Theorem stating that Jackie spends 3 hours exercising --/
theorem jackie_exercise_hours :
  let schedule : DailySchedule := {
    total_hours := 24,
    work_hours := 8,
    sleep_hours := 8,
    free_hours := 5
  }
  exercise_hours schedule = 3 := by sorry

end jackie_exercise_hours_l1465_146588


namespace table_covering_l1465_146559

/-- Represents a cell in the table -/
inductive Cell
| Zero
| One

/-- Represents the 1000x1000 table -/
def Table := Fin 1000 → Fin 1000 → Cell

/-- Checks if a set of rows covers all columns with at least one 1 -/
def coversColumnsWithOnes (t : Table) (rows : Finset (Fin 1000)) : Prop :=
  ∀ j : Fin 1000, ∃ i ∈ rows, t i j = Cell.One

/-- Checks if a set of columns covers all rows with at least one 0 -/
def coversRowsWithZeros (t : Table) (cols : Finset (Fin 1000)) : Prop :=
  ∀ i : Fin 1000, ∃ j ∈ cols, t i j = Cell.Zero

/-- The main theorem -/
theorem table_covering (t : Table) :
  (∃ rows : Finset (Fin 1000), rows.card = 10 ∧ coversColumnsWithOnes t rows) ∨
  (∃ cols : Finset (Fin 1000), cols.card = 10 ∧ coversRowsWithZeros t cols) :=
sorry

end table_covering_l1465_146559


namespace largest_whole_number_satisfying_inequality_l1465_146576

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/5 < 3/2 ↔ x ≤ 6 :=
sorry

end largest_whole_number_satisfying_inequality_l1465_146576


namespace ellipse_equation_l1465_146542

noncomputable section

-- Define the ellipse C
def C (x y : ℝ) (a b : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

-- Define the foci
def F₁ (c : ℝ) : ℝ × ℝ := (-c, 0)
def F₂ (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define point A
def A : ℝ × ℝ := (2, 0)

-- Define the slope product condition
def slope_product (c : ℝ) : Prop :=
  let k_AF₁ := (0 - (-c)) / (2 - 0)
  let k_AF₂ := (0 - c) / (2 - 0)
  k_AF₁ * k_AF₂ = -1/4

-- Define the distance sum condition for point B
def distance_sum (a : ℝ) : Prop := 2*a = 2*Real.sqrt 2

-- Main theorem
theorem ellipse_equation (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (∃ c : ℝ, slope_product c ∧ distance_sum a) →
  (∀ x y : ℝ, C x y a b ↔ y^2/2 + x^2 = 1) :=
sorry

end ellipse_equation_l1465_146542


namespace relationship_l1465_146525

-- Define the real numbers a, b, and c
variable (a b c : ℝ)

-- Define the conditions
axiom eq_a : 2 * a^3 + a = 2
axiom eq_b : b * Real.log b / Real.log 2 = 1
axiom eq_c : c * Real.log c / Real.log 5 = 1

-- State the theorem to be proved
theorem relationship : c > b ∧ b > a := by
  sorry

end relationship_l1465_146525


namespace paint_cans_theorem_l1465_146511

/-- Represents the number of rooms that can be painted with the initial amount of paint -/
def initial_rooms : ℕ := 50

/-- Represents the number of rooms that can be painted after losing two cans -/
def remaining_rooms : ℕ := 42

/-- Represents the number of cans lost -/
def lost_cans : ℕ := 2

/-- Calculates the number of cans used to paint the remaining rooms -/
def cans_used : ℕ := 
  let rooms_per_can := (initial_rooms - remaining_rooms) / lost_cans
  (remaining_rooms + rooms_per_can - 1) / rooms_per_can

theorem paint_cans_theorem : cans_used = 11 := by
  sorry

end paint_cans_theorem_l1465_146511


namespace evaluate_expression_l1465_146521

theorem evaluate_expression : 8^8 * 27^8 * 8^27 * 27^27 = 216^35 := by
  sorry

end evaluate_expression_l1465_146521


namespace cloth_sale_problem_l1465_146520

/-- Proves that the number of meters of cloth sold is 45 given the specified conditions -/
theorem cloth_sale_problem (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 4500 →
  profit_per_meter = 14 →
  cost_price_per_meter = 86 →
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 45 := by
  sorry

#check cloth_sale_problem

end cloth_sale_problem_l1465_146520


namespace circle_center_l1465_146592

/-- The center of the circle x^2 + y^2 - 2x + 4y + 3 = 0 is at the point (1, -2). -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 3 = 0) → 
  ∃ (h : ℝ), (x - 1)^2 + (y + 2)^2 = h^2 :=
by sorry

end circle_center_l1465_146592


namespace batsman_average_after_20th_innings_l1465_146531

/-- Represents a batsman's innings record -/
structure BatsmanRecord where
  innings : ℕ
  totalScore : ℕ
  avgIncrease : ℚ
  lastScore : ℕ

/-- Calculates the average score of a batsman -/
def calculateAverage (record : BatsmanRecord) : ℚ :=
  record.totalScore / record.innings

theorem batsman_average_after_20th_innings 
  (record : BatsmanRecord)
  (h1 : record.innings = 20)
  (h2 : record.lastScore = 90)
  (h3 : record.avgIncrease = 2)
  : calculateAverage record = 52 := by
  sorry

end batsman_average_after_20th_innings_l1465_146531


namespace rotation_result_l1465_146599

/-- Rotation of a vector about the origin -/
def rotate90 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Check if a vector passes through the y-axis -/
def passesYAxis (v : ℝ × ℝ × ℝ) : Prop := sorry

theorem rotation_result :
  let v₀ : ℝ × ℝ × ℝ := (2, 1, 1)
  let v₁ := rotate90 v₀
  passesYAxis v₁ →
  v₁ = (Real.sqrt (6/11), -3 * Real.sqrt (6/11), Real.sqrt (6/11)) :=
by sorry

end rotation_result_l1465_146599


namespace f_properties_l1465_146557

noncomputable def f (b c x : ℝ) : ℝ := |x| * x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x y : ℝ, x < y → b > 0 → f b c x < f b c y) ∧
  (∀ x : ℝ, f b c x = f b c (-x)) ∧
  (∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0) :=
by sorry

end f_properties_l1465_146557


namespace squared_differences_inequality_l1465_146585

theorem squared_differences_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  min ((a - b)^2) (min ((b - c)^2) ((c - a)^2)) ≤ (a^2 + b^2 + c^2) / 2 := by
  sorry

end squared_differences_inequality_l1465_146585


namespace xiao_ming_correct_count_l1465_146549

/-- Represents a math problem with a given answer --/
structure MathProblem where
  given_answer : Int
  correct_answer : Int

/-- Checks if a math problem is answered correctly --/
def is_correct (problem : MathProblem) : Bool :=
  problem.given_answer = problem.correct_answer

/-- Counts the number of correctly answered problems --/
def count_correct (problems : List MathProblem) : Nat :=
  (problems.filter is_correct).length

/-- The list of math problems Xiao Ming solved --/
def xiao_ming_problems : List MathProblem := [
  { given_answer := 0, correct_answer := -4 },
  { given_answer := -4, correct_answer := 0 },
  { given_answer := -4, correct_answer := -4 }
]

theorem xiao_ming_correct_count :
  count_correct xiao_ming_problems = 1 := by
  sorry

end xiao_ming_correct_count_l1465_146549


namespace base15_divisible_by_9_l1465_146514

/-- Converts a base-15 integer to decimal --/
def base15ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (15 ^ i)) 0

/-- The base-15 representation of 2643₁₅ --/
def base15Number : List Nat := [3, 4, 6, 2]

/-- Theorem stating that 2643₁₅ divided by 9 has a remainder of 0 --/
theorem base15_divisible_by_9 :
  (base15ToDecimal base15Number) % 9 = 0 := by
  sorry

end base15_divisible_by_9_l1465_146514


namespace sum_a_plus_d_equals_six_l1465_146581

theorem sum_a_plus_d_equals_six (a b c d e : ℝ)
  (eq1 : a + b = 12)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3)
  (eq4 : d + e = 7)
  (eq5 : e + a = 10) :
  a + d = 6 := by sorry

end sum_a_plus_d_equals_six_l1465_146581


namespace squarable_numbers_l1465_146507

def isSquarable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ (i : Fin n), ∃ (k : ℕ), (p i).val + i.val + 1 = k^2

theorem squarable_numbers : 
  (¬ isSquarable 7) ∧ 
  (isSquarable 9) ∧ 
  (¬ isSquarable 11) ∧ 
  (isSquarable 15) := by sorry

end squarable_numbers_l1465_146507


namespace symmetry_axes_count_other_rotation_axes_count_l1465_146532

/-- Enumeration of regular polyhedra -/
inductive RegularPolyhedron
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

/-- Function to calculate the number of symmetry axes for a regular polyhedron -/
def symmetryAxes (p : RegularPolyhedron) : Nat :=
  match p with
  | RegularPolyhedron.Tetrahedron => 3
  | RegularPolyhedron.Cube => 9
  | RegularPolyhedron.Octahedron => 9
  | RegularPolyhedron.Dodecahedron => 16
  | RegularPolyhedron.Icosahedron => 16

/-- Function to calculate the number of other rotation axes for a regular polyhedron -/
def otherRotationAxes (p : RegularPolyhedron) : Nat :=
  match p with
  | RegularPolyhedron.Tetrahedron => 4
  | RegularPolyhedron.Cube => 10
  | RegularPolyhedron.Octahedron => 10
  | RegularPolyhedron.Dodecahedron => 16
  | RegularPolyhedron.Icosahedron => 16

/-- Theorem stating the number of symmetry axes for each regular polyhedron -/
theorem symmetry_axes_count :
  (∀ p : RegularPolyhedron, symmetryAxes p = 
    match p with
    | RegularPolyhedron.Tetrahedron => 3
    | RegularPolyhedron.Cube => 9
    | RegularPolyhedron.Octahedron => 9
    | RegularPolyhedron.Dodecahedron => 16
    | RegularPolyhedron.Icosahedron => 16) :=
by sorry

/-- Theorem stating the number of other rotation axes for each regular polyhedron -/
theorem other_rotation_axes_count :
  (∀ p : RegularPolyhedron, otherRotationAxes p = 
    match p with
    | RegularPolyhedron.Tetrahedron => 4
    | RegularPolyhedron.Cube => 10
    | RegularPolyhedron.Octahedron => 10
    | RegularPolyhedron.Dodecahedron => 16
    | RegularPolyhedron.Icosahedron => 16) :=
by sorry

end symmetry_axes_count_other_rotation_axes_count_l1465_146532


namespace eventual_bounded_groups_l1465_146560

/-- Represents a group distribution in the society -/
def GroupDistribution := List Nat

/-- The redistribution process for a week -/
def redistribute : GroupDistribution → GroupDistribution := sorry

/-- Checks if all groups in a distribution have size at most 1 + √(2n) -/
def all_groups_bounded (n : Nat) (dist : GroupDistribution) : Prop := sorry

/-- Theorem: Eventually, all groups will be bounded by 1 + √(2n) -/
theorem eventual_bounded_groups (n : Nat) :
  ∃ (k : Nat), all_groups_bounded n ((redistribute^[k]) [n]) := by
  sorry

end eventual_bounded_groups_l1465_146560


namespace max_value_of_expression_achievable_max_value_l1465_146555

theorem max_value_of_expression (n : ℕ) : 
  10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ 870 := by
  sorry

theorem achievable_max_value : 
  ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - n) = 870 := by
  sorry

end max_value_of_expression_achievable_max_value_l1465_146555


namespace age_difference_l1465_146512

/-- Proves that the difference between Rahul's and Sachin's ages is 9 years -/
theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 31.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 9 :=
by
  sorry


end age_difference_l1465_146512


namespace opaque_arrangements_count_l1465_146562

/-- Represents a glass piece with one painted triangular section -/
structure GlassPiece where
  rotation : Fin 4  -- 0, 1, 2, 3 representing 0°, 90°, 180°, 270°

/-- Represents a stack of glass pieces -/
def GlassStack := List GlassPiece

/-- Checks if a given stack of glass pieces is completely opaque -/
def is_opaque (stack : GlassStack) : Bool :=
  sorry

/-- Counts the number of opaque arrangements for 5 glass pieces -/
def count_opaque_arrangements : Nat :=
  sorry

/-- The main theorem stating the correct number of opaque arrangements -/
theorem opaque_arrangements_count :
  count_opaque_arrangements = 7200 :=
sorry

end opaque_arrangements_count_l1465_146562


namespace estimate_larger_than_original_l1465_146501

theorem estimate_larger_than_original 
  (x y ε δ : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x > y) 
  (hε : ε > 0) 
  (hδ : δ > 0) 
  (hεδ : ε ≠ δ) : 
  (x + ε) - (y - δ) > x - y := by
sorry

end estimate_larger_than_original_l1465_146501


namespace product_112_54_l1465_146590

theorem product_112_54 : 112 * 54 = 6048 := by
  sorry

end product_112_54_l1465_146590


namespace sum_of_integers_l1465_146544

theorem sum_of_integers (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = y.val * z.val + x.val)
  (h2 : y.val * z.val + x.val = x.val * z.val + y.val)
  (h3 : x.val * z.val + y.val = 55)
  (h4 : Even x.val ∨ Even y.val ∨ Even z.val) :
  x.val + y.val + z.val = 56 := by
  sorry

end sum_of_integers_l1465_146544


namespace expression_eval_zero_l1465_146545

theorem expression_eval_zero (a : ℚ) (h : a = 4/3) :
  (6 * a^2 - 15 * a + 5) * (3 * a - 4) = 0 := by
  sorry

end expression_eval_zero_l1465_146545
