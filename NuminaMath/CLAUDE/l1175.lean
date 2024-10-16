import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_perpendicular_intersection_l1175_117558

-- Define the line l
def line_l (x : ℝ) : ℝ := -x + 3

-- Define the ellipse C
def ellipse_C (m n x y : ℝ) : Prop := m * x^2 + n * y^2 = 1

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define the line l'
def line_l' (b x : ℝ) : ℝ := -x + b

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_and_line_intersection :
  ∀ m n : ℝ, n > m → m > 0 →
  (∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 1 ∧ line_l p.1 = p.2 ∧ ellipse_C m n p.1 p.2) →
  (∀ x y : ℝ, ellipse_C m n x y ↔ standard_ellipse x y) :=
sorry

theorem perpendicular_intersection :
  ∀ b : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    standard_ellipse A.1 A.2 ∧ standard_ellipse B.1 B.2 ∧
    line_l' b A.1 = A.2 ∧ line_l' b B.1 = B.2 ∧
    perpendicular A.1 A.2 B.1 B.2) →
  b = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_perpendicular_intersection_l1175_117558


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l1175_117537

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def yellow_balls : ℕ := 5
def balls_per_box : ℕ := 5
def red_in_box_A : ℕ := 3
def yellow_in_box_A : ℕ := 2
def exchanged_balls : ℕ := 3

def probability_3_red_2_yellow : ℚ := 25 / 63

def mathematical_expectation : ℚ := 12 / 5

theorem ball_distribution_theorem :
  (probability_3_red_2_yellow = 25 / 63) ∧
  (mathematical_expectation = 12 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l1175_117537


namespace NUMINAMATH_CALUDE_shop_monthly_rent_l1175_117586

/-- Calculates the monthly rent of a shop given its dimensions and annual rent per square foot. -/
def monthly_rent (length width annual_rent_per_sqft : ℕ) : ℕ :=
  let area := length * width
  let annual_rent := area * annual_rent_per_sqft
  annual_rent / 12

/-- Theorem stating that for a shop with given dimensions and annual rent per square foot,
    the monthly rent is 3600. -/
theorem shop_monthly_rent :
  monthly_rent 20 15 144 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_shop_monthly_rent_l1175_117586


namespace NUMINAMATH_CALUDE_square_feet_per_acre_l1175_117566

/-- Represents the area of a rectangle in square feet -/
def rectangle_area (length width : ℝ) : ℝ := length * width

/-- Represents the total number of acres rented -/
def total_acres : ℝ := 10

/-- Represents the monthly rent for the entire plot -/
def total_rent : ℝ := 300

/-- Represents the length of the rectangular plot in feet -/
def plot_length : ℝ := 360

/-- Represents the width of the rectangular plot in feet -/
def plot_width : ℝ := 1210

theorem square_feet_per_acre :
  (rectangle_area plot_length plot_width) / total_acres = 43560 := by
  sorry

#check square_feet_per_acre

end NUMINAMATH_CALUDE_square_feet_per_acre_l1175_117566


namespace NUMINAMATH_CALUDE_sticker_distribution_solution_l1175_117547

/-- Represents the sticker distribution problem --/
structure StickerDistribution where
  space : ℕ := 120
  cat : ℕ := 80
  dinosaur : ℕ := 150
  superhero : ℕ := 45
  space_given : ℕ := 25
  cat_given : ℕ := 13
  dinosaur_given : ℕ := 33
  superhero_given : ℕ := 29

/-- Calculates the total number of stickers left after initial distribution --/
def remaining_stickers (sd : StickerDistribution) : ℕ :=
  (sd.space - sd.space_given) + (sd.cat - sd.cat_given) + 
  (sd.dinosaur - sd.dinosaur_given) + (sd.superhero - sd.superhero_given)

/-- Theorem stating the solution to the sticker distribution problem --/
theorem sticker_distribution_solution (sd : StickerDistribution) :
  ∃ (X : ℕ), X = 3 ∧ (remaining_stickers sd - X) / 4 = 73 := by
  sorry


end NUMINAMATH_CALUDE_sticker_distribution_solution_l1175_117547


namespace NUMINAMATH_CALUDE_women_in_second_group_l1175_117563

/-- Represents the work rate of a man -/
def man_rate : ℝ := sorry

/-- Represents the work rate of a woman -/
def woman_rate : ℝ := sorry

/-- The number of women in the second group -/
def x : ℝ := sorry

/-- First condition: 3 men and 8 women complete a task in the same time as 6 men and x women -/
axiom condition1 : 3 * man_rate + 8 * woman_rate = 6 * man_rate + x * woman_rate

/-- Second condition: 2 men and 3 women complete half the work in the same time as the first group -/
axiom condition2 : 2 * man_rate + 3 * woman_rate = 0.5 * (3 * man_rate + 8 * woman_rate)

/-- The theorem to be proved -/
theorem women_in_second_group : x = 2 := by sorry

end NUMINAMATH_CALUDE_women_in_second_group_l1175_117563


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1175_117575

theorem quadratic_roots_property (a b : ℝ) : 
  (2 * a^2 + 6 * a - 14 = 0) → 
  (2 * b^2 + 6 * b - 14 = 0) → 
  (2 * a - 3) * (4 * b - 6) = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1175_117575


namespace NUMINAMATH_CALUDE_permutation_count_equals_fibonacci_l1175_117573

/-- The number of permutations satisfying the given condition -/
def P (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else P (n - 1) + P (n - 2)

/-- The nth Fibonacci number -/
def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

/-- Theorem stating the equivalence between P(n) and the (n+1)th Fibonacci number -/
theorem permutation_count_equals_fibonacci (n : ℕ) :
  P n = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_permutation_count_equals_fibonacci_l1175_117573


namespace NUMINAMATH_CALUDE_cinema_seating_l1175_117580

/-- The number of people sitting between the far right and far left audience members -/
def people_between : ℕ := 30

/-- The total number of people sitting in the chairs -/
def total_people : ℕ := people_between + 2

theorem cinema_seating : total_people = 32 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seating_l1175_117580


namespace NUMINAMATH_CALUDE_all_hyperprimes_l1175_117593

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def isSegmentPrime (n : ℕ) : Prop :=
  ∀ start len : ℕ, len > 0 → start + len ≤ (Nat.digits 10 n).length →
    isPrime (Nat.digits 10 n |> List.take len |> List.drop start |> List.foldl (· * 10 + ·) 0)

def isHyperprime (n : ℕ) : Prop := n > 0 ∧ isSegmentPrime n

theorem all_hyperprimes :
  {n : ℕ | isHyperprime n} = {2, 3, 5, 7, 23, 37, 53, 73, 373} := by sorry

end NUMINAMATH_CALUDE_all_hyperprimes_l1175_117593


namespace NUMINAMATH_CALUDE_initial_men_count_initial_men_count_is_seven_l1175_117595

/-- Proves that the initial number of men in a group is 7 given specific conditions about age changes. -/
theorem initial_men_count : ℕ :=
  let initial_average : ℝ := sorry
  let final_average : ℝ := initial_average + 4
  let replaced_men_ages : Fin 2 → ℕ := ![26, 30]
  let women_average_age : ℝ := 42
  let men_count : ℕ := sorry
  have h1 : final_average * men_count = initial_average * men_count + 4 * men_count := sorry
  have h2 : (men_count - 2) * initial_average + 2 * women_average_age = men_count * final_average := sorry
  have h3 : 2 * women_average_age - (replaced_men_ages 0 + replaced_men_ages 1) = 4 * men_count := sorry
  7

theorem initial_men_count_is_seven : initial_men_count = 7 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_initial_men_count_is_seven_l1175_117595


namespace NUMINAMATH_CALUDE_even_mono_decreasing_order_l1175_117529

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def isMonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_mono_decreasing_order (f : ℝ → ℝ) 
  (h_even : isEven f) 
  (h_mono : isMonoDecreasing f 0 3) : 
  f (-1) > f 2 ∧ f 2 > f 3 := by
  sorry

end NUMINAMATH_CALUDE_even_mono_decreasing_order_l1175_117529


namespace NUMINAMATH_CALUDE_bus_distance_problem_l1175_117591

/-- Proves that given a total distance of 250 km, covered partly at 40 kmph and partly at 60 kmph,
    with a total travel time of 6 hours, the distance covered at 40 kmph is 220 km. -/
theorem bus_distance_problem (x : ℝ) 
    (h1 : x ≥ 0) 
    (h2 : x ≤ 250) 
    (h3 : x / 40 + (250 - x) / 60 = 6) : x = 220 := by
  sorry

#check bus_distance_problem

end NUMINAMATH_CALUDE_bus_distance_problem_l1175_117591


namespace NUMINAMATH_CALUDE_perpendicular_lines_n_value_l1175_117596

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_lines_n_value (m n : ℝ) (p : ℝ) :
  let l₁ : Line := ⟨m, 4, -2⟩
  let l₂ : Line := ⟨2, -5, n⟩
  let foot : Point := ⟨1, p⟩
  perpendicular (m / -4) (2 / 5) →
  point_on_line foot l₁ →
  point_on_line foot l₂ →
  n = -12 := by
  sorry

#check perpendicular_lines_n_value

end NUMINAMATH_CALUDE_perpendicular_lines_n_value_l1175_117596


namespace NUMINAMATH_CALUDE_max_pairs_with_distinct_sums_l1175_117582

/-- Given a set of integers from 1 to 2010, we can choose at most 803 pairs
    such that the elements of each pair are distinct, no two pairs share an element,
    and the sum of each pair is unique and not greater than 2010. -/
theorem max_pairs_with_distinct_sums :
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 803 ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 2010 ∧ p.2 ∈ Finset.range 2010) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 2010) ∧
    pairs.card = k ∧
    (∀ (m : ℕ) (other_pairs : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 ∈ Finset.range 2010 ∧ p.2 ∈ Finset.range 2010) →
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ other_pairs → q ∈ other_pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ other_pairs → q ∈ other_pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 + p.2 ≤ 2010) →
      other_pairs.card = m →
      m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_with_distinct_sums_l1175_117582


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1175_117587

theorem unique_integer_solution : 
  ∃! (n : ℤ), (n^2 + 3*n + 5) / (n + 2 : ℚ) = 1 + Real.sqrt (6 - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1175_117587


namespace NUMINAMATH_CALUDE_cincinnati_to_nyc_distance_l1175_117553

/-- The total distance between Cincinnati and New York City -/
def total_distance (day1 day2 day3 remaining : ℕ) : ℕ :=
  day1 + day2 + day3 + remaining

/-- The distance walked on the second day -/
def day2_distance (day1 : ℕ) : ℕ :=
  day1 / 2 - 6

theorem cincinnati_to_nyc_distance :
  total_distance 20 (day2_distance 20) 10 36 = 70 := by
  sorry

end NUMINAMATH_CALUDE_cincinnati_to_nyc_distance_l1175_117553


namespace NUMINAMATH_CALUDE_ray_return_characterization_l1175_117503

/-- Represents a point in the triangular grid --/
structure GridPoint where
  a : ℕ
  b : ℕ

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Checks if a GridPoint is on the triangular grid --/
def isOnGrid (p : GridPoint) : Prop :=
  p.a ≡ p.b [MOD 3]

/-- Checks if a line from origin to GridPoint doesn't pass through other grid points --/
def isDirectPath (p : GridPoint) : Prop :=
  Nat.gcd p.a p.b = 1

/-- Calculates the number of bounces for a ray to reach a GridPoint --/
def numberOfBounces (p : GridPoint) : ℕ :=
  2 * (p.a + p.b) - 3

/-- Theorem: Characterization of valid number of bounces for ray to return to A --/
theorem ray_return_characterization (n : ℕ) :
  (∃ (t : EquilateralTriangle) (p : GridPoint), 
    isOnGrid p ∧ isDirectPath p ∧ numberOfBounces p = n) ↔ 
  (n ≡ 1 [MOD 6] ∨ n ≡ 5 [MOD 6]) ∧ n ≠ 5 ∧ n ≠ 17 :=
sorry

end NUMINAMATH_CALUDE_ray_return_characterization_l1175_117503


namespace NUMINAMATH_CALUDE_equal_distribution_of_items_l1175_117570

theorem equal_distribution_of_items (pencils erasers friends : ℕ) 
  (h1 : pencils = 35) 
  (h2 : erasers = 5) 
  (h3 : friends = 5) : 
  (pencils + erasers) / friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_items_l1175_117570


namespace NUMINAMATH_CALUDE_total_bowling_balls_l1175_117572

theorem total_bowling_balls (red : ℕ) (green : ℕ) (blue : ℕ) : 
  red = 30 →
  green = red + 6 →
  blue = 2 * green →
  red + green + blue = 138 := by
sorry

end NUMINAMATH_CALUDE_total_bowling_balls_l1175_117572


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1175_117565

-- Define the set of possible initial compositions
inductive InitialComposition
| NoWhite
| OneWhite
| TwoWhite

-- Define the probability of drawing a white ball given an initial composition
def probWhiteGivenComposition (ic : InitialComposition) : ℚ :=
  match ic with
  | InitialComposition.NoWhite => 1/3
  | InitialComposition.OneWhite => 2/3
  | InitialComposition.TwoWhite => 1

-- Define the theorem
theorem probability_of_white_ball :
  let initialCompositions := [InitialComposition.NoWhite, InitialComposition.OneWhite, InitialComposition.TwoWhite]
  let numCompositions := initialCompositions.length
  let probEachComposition := 1 / numCompositions
  let totalProb := (initialCompositions.map probWhiteGivenComposition).sum * probEachComposition
  totalProb = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1175_117565


namespace NUMINAMATH_CALUDE_union_of_sets_l1175_117560

theorem union_of_sets : 
  let A : Set ℕ := {1, 3, 7, 8}
  let B : Set ℕ := {1, 5, 8}
  A ∪ B = {1, 3, 5, 7, 8} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1175_117560


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l1175_117561

theorem pen_pencil_ratio : 
  ∀ (num_pencils num_pens : ℕ),
  num_pencils = 24 →
  num_pencils = num_pens + 4 →
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l1175_117561


namespace NUMINAMATH_CALUDE_jogging_track_circumference_l1175_117504

/-- The circumference of a circular jogging track given two people walking in opposite directions --/
theorem jogging_track_circumference 
  (deepak_speed : ℝ) 
  (wife_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : deepak_speed = 4.5) 
  (h2 : wife_speed = 3.75) 
  (h3 : meeting_time = 4.32) : 
  deepak_speed * meeting_time + wife_speed * meeting_time = 35.64 := by
  sorry

#check jogging_track_circumference

end NUMINAMATH_CALUDE_jogging_track_circumference_l1175_117504


namespace NUMINAMATH_CALUDE_sum_of_number_and_predecessor_l1175_117567

theorem sum_of_number_and_predecessor : ∃ n : ℤ, (6 * n - 2 = 100) ∧ (n + (n - 1) = 33) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_predecessor_l1175_117567


namespace NUMINAMATH_CALUDE_definite_integral_2x_minus_3x_squared_l1175_117520

theorem definite_integral_2x_minus_3x_squared : 
  ∫ x in (0:ℝ)..1, (2*x - 3*x^2) = 0 := by sorry

end NUMINAMATH_CALUDE_definite_integral_2x_minus_3x_squared_l1175_117520


namespace NUMINAMATH_CALUDE_largest_square_from_rectangle_l1175_117557

/-- Given a rectangular paper of length 54 cm and width 20 cm, 
    the largest side length of three equal squares that can be cut from this paper is 18 cm. -/
theorem largest_square_from_rectangle : ∀ (side_length : ℝ), 
  side_length > 0 ∧ 
  3 * side_length ≤ 54 ∧ 
  side_length ≤ 20 →
  side_length ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_from_rectangle_l1175_117557


namespace NUMINAMATH_CALUDE_simplification_value_at_3_value_at_negative_3_even_function_l1175_117550

-- Define the original expression
def original_expression (x : ℝ) : ℝ :=
  6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2)

-- Define the simplified expression
def simplified_expression (x : ℝ) : ℝ :=
  2 * x^2 + 2

-- Theorem stating that the original expression simplifies to the simplified expression
theorem simplification : 
  ∀ x : ℝ, original_expression x = simplified_expression x :=
sorry

-- Theorem stating that the simplified expression equals 20 when x = 3
theorem value_at_3 : simplified_expression 3 = 20 :=
sorry

-- Theorem stating that the simplified expression equals 20 when x = -3
theorem value_at_negative_3 : simplified_expression (-3) = 20 :=
sorry

-- Theorem stating that the simplified expression is an even function
theorem even_function :
  ∀ x : ℝ, simplified_expression x = simplified_expression (-x) :=
sorry

end NUMINAMATH_CALUDE_simplification_value_at_3_value_at_negative_3_even_function_l1175_117550


namespace NUMINAMATH_CALUDE_complex_modulus_power_four_l1175_117540

theorem complex_modulus_power_four : Complex.abs ((2 + Complex.I) ^ 4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_power_four_l1175_117540


namespace NUMINAMATH_CALUDE_may_savings_l1175_117517

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l1175_117517


namespace NUMINAMATH_CALUDE_vidya_mother_age_l1175_117590

theorem vidya_mother_age (vidya_age : ℕ) (mother_age : ℕ) : 
  vidya_age = 13 → 
  mother_age = 3 * vidya_age + 5 → 
  mother_age = 44 := by
sorry

end NUMINAMATH_CALUDE_vidya_mother_age_l1175_117590


namespace NUMINAMATH_CALUDE_actual_journey_equation_hypothetical_journey_equation_distance_AB_l1175_117513

/-- The distance between dock A and dock B in kilometers -/
def distance : ℝ := 270

/-- The initial speed of the steamboat in km/hr -/
noncomputable def initial_speed : ℝ := distance / 22.5

/-- Time equation for the actual journey -/
theorem actual_journey_equation :
  distance / initial_speed + 3.5 = 3 + (distance - 2 * initial_speed) / (0.8 * initial_speed) :=
sorry

/-- Time equation for the hypothetical journey with later stop -/
theorem hypothetical_journey_equation :
  distance / initial_speed + 1.5 = 3 + 180 / initial_speed + (distance - 2 * initial_speed - 180) / (0.8 * initial_speed) :=
sorry

/-- The distance AB is 270 km -/
theorem distance_AB : distance = 270 :=
sorry

end NUMINAMATH_CALUDE_actual_journey_equation_hypothetical_journey_equation_distance_AB_l1175_117513


namespace NUMINAMATH_CALUDE_feathers_per_crown_l1175_117508

theorem feathers_per_crown (total_feathers : ℕ) (total_crowns : ℕ) 
  (h1 : total_feathers = 6538) 
  (h2 : total_crowns = 934) : 
  total_feathers / total_crowns = 7 := by
  sorry

end NUMINAMATH_CALUDE_feathers_per_crown_l1175_117508


namespace NUMINAMATH_CALUDE_cat_resisting_time_l1175_117542

/-- Proves that given a total time of 28 minutes, a walking distance of 64 feet,
    and a walking rate of 8 feet/minute, the time spent resisting is 20 minutes. -/
theorem cat_resisting_time
  (total_time : ℕ)
  (walking_distance : ℕ)
  (walking_rate : ℕ)
  (h1 : total_time = 28)
  (h2 : walking_distance = 64)
  (h3 : walking_rate = 8)
  : total_time - walking_distance / walking_rate = 20 := by
  sorry

#check cat_resisting_time

end NUMINAMATH_CALUDE_cat_resisting_time_l1175_117542


namespace NUMINAMATH_CALUDE_equivalence_conditions_l1175_117531

theorem equivalence_conditions (n : ℕ) :
  (∀ (a : ℕ+), n ∣ a^n - a) ↔
  (∀ (p : ℕ), Prime p → p ∣ n → (¬(p^2 ∣ n) ∧ (p - 1 ∣ n - 1))) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_conditions_l1175_117531


namespace NUMINAMATH_CALUDE_line_intersects_circle_intersection_point_polar_coordinates_l1175_117549

-- Define the line l
def line_l (x y : ℝ) : Prop := y - 1 = 2 * (x + 1)

-- Define the circle C₁
def circle_C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 4

-- Define the curve C₂
def curve_C₂ (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Theorem 1: Line l intersects circle C₁
theorem line_intersects_circle : ∃ (x y : ℝ), line_l x y ∧ circle_C₁ x y := by sorry

-- Theorem 2: The intersection point of C₁ and C₂ is (2, 2) in Cartesian coordinates
theorem intersection_point : ∃! (x y : ℝ), circle_C₁ x y ∧ curve_C₂ x y ∧ x = 2 ∧ y = 2 := by sorry

-- Theorem 3: The polar coordinates of the intersection point are (2√2, π/4)
theorem polar_coordinates : 
  let (x, y) := (2, 2)
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  ρ = 2 * Real.sqrt 2 ∧ θ = π / 4 := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_intersection_point_polar_coordinates_l1175_117549


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1175_117544

theorem complex_equation_solution (z : ℂ) 
  (h : 10 * Complex.normSq z = 2 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 16) + 40) : 
  z + 9 / z = -3 / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1175_117544


namespace NUMINAMATH_CALUDE_inequality_range_theorem_l1175_117523

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The theorem statement -/
theorem inequality_range_theorem (m : ℝ) :
  (∀ x ∈ Set.Ici (2/3), f (x/m) - 4*m^2*f x ≤ f (x-1) + 4*f m) →
  m ∈ Set.Iic (-Real.sqrt 3 / 2) ∪ Set.Ici (Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_theorem_l1175_117523


namespace NUMINAMATH_CALUDE_special_function_property_l1175_117584

/-- A continuously differentiable function satisfying f'(t) > f(f(t)) for all t ∈ ℝ -/
structure SpecialFunction where
  f : ℝ → ℝ
  cont_diff : ContDiff ℝ 1 f
  property : ∀ t : ℝ, deriv f t > f (f t)

/-- The main theorem -/
theorem special_function_property (sf : SpecialFunction) :
  ∀ t : ℝ, t ≥ 0 → sf.f (sf.f (sf.f t)) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l1175_117584


namespace NUMINAMATH_CALUDE_pension_calculation_l1175_117548

/-- Represents the pension calculation problem -/
theorem pension_calculation
  (c d r s y : ℝ)
  (h_cd : c ≠ d)
  (h_c : ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x + c - y) = t * Real.sqrt (x - y) + r)
  (h_d : ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x + d - y) = t * Real.sqrt (x - y) + s) :
  ∃ (t : ℝ), ∀ (x : ℝ), t * Real.sqrt (x - y) = (c * s^2 - d * r^2) / (2 * (d * r - c * s)) :=
sorry

end NUMINAMATH_CALUDE_pension_calculation_l1175_117548


namespace NUMINAMATH_CALUDE_emilys_marbles_l1175_117568

theorem emilys_marbles (jake_marbles : ℕ) (emily_scale : ℕ) : 
  jake_marbles = 216 → 
  emily_scale = 3 → 
  (emily_scale ^ 3) * jake_marbles = 5832 :=
by sorry

end NUMINAMATH_CALUDE_emilys_marbles_l1175_117568


namespace NUMINAMATH_CALUDE_total_marbles_l1175_117592

def marbles_bought : ℝ := 5423.6
def marbles_before : ℝ := 12834.9

theorem total_marbles :
  marbles_bought + marbles_before = 18258.5 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l1175_117592


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1175_117539

def U : Finset Nat := {1,2,3,4,5,6,7}
def A : Finset Nat := {2,4,5,7}
def B : Finset Nat := {3,4,5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {1,2,3,6,7} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1175_117539


namespace NUMINAMATH_CALUDE_inverse_function_point_l1175_117597

/-- Given a function f(x) = 2^x + m, prove that if its inverse passes through (3,1), then m = 1 -/
theorem inverse_function_point (m : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 2^x + m) ∧ (∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ g 3 = 1)) → 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_point_l1175_117597


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1175_117546

theorem quadratic_integer_roots (m : ℝ) :
  (∃ x : ℤ, (m + 1) * x^2 + 2 * x - 5 * m - 13 = 0) ↔
  (m = -1 ∨ m = -11/10 ∨ m = -1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1175_117546


namespace NUMINAMATH_CALUDE_surface_area_of_rmon_l1175_117533

/-- Right prism with equilateral triangle base -/
structure RightPrism :=
  (height : ℝ)
  (baseSideLength : ℝ)

/-- Point on an edge of the prism -/
structure EdgePoint :=
  (position : ℝ)

/-- The solid RMON created by slicing the prism -/
structure SlicedSolid :=
  (prism : RightPrism)
  (m : EdgePoint)
  (n : EdgePoint)
  (o : EdgePoint)

/-- Calculate the surface area of the sliced solid -/
noncomputable def surfaceArea (solid : SlicedSolid) : ℝ :=
  sorry

/-- Main theorem: The surface area of RMON is 30.62 square units -/
theorem surface_area_of_rmon (solid : SlicedSolid) 
  (h1 : solid.prism.height = 10)
  (h2 : solid.prism.baseSideLength = 10)
  (h3 : solid.m.position = 1/4)
  (h4 : solid.n.position = 1/4)
  (h5 : solid.o.position = 1/4) :
  surfaceArea solid = 30.62 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_rmon_l1175_117533


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1175_117579

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) passing through the center of the circle x² + y² + 4x - 4y - 1 = 0,
    the minimum value of 2/a + 3/b is 5 + 2√6 -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : ∃ (x y : ℝ), a * x - b * y + 2 = 0)
    (h_circle : ∃ (x y : ℝ), x^2 + y^2 + 4*x - 4*y - 1 = 0)
    (h_center : ∃ (x y : ℝ), (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (a * x - b * y + 2 = 0)) :
    (∀ (a' b' : ℝ), (a' > 0 ∧ b' > 0) → (2/a' + 3/b' ≥ 5 + 2 * Real.sqrt 6)) ∧
    (∃ (a' b' : ℝ), (a' > 0 ∧ b' > 0) ∧ (2/a' + 3/b' = 5 + 2 * Real.sqrt 6)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1175_117579


namespace NUMINAMATH_CALUDE_shop_dimension_example_l1175_117532

/-- Calculates the dimension of a shop given its monthly rent and annual rent per square foot. -/
def shopDimension (monthlyRent : ℕ) (annualRentPerSqFt : ℕ) : ℕ :=
  (monthlyRent * 12) / annualRentPerSqFt

/-- Theorem stating that for a shop with a monthly rent of 1300 and an annual rent per square foot of 156, the dimension is 100 square feet. -/
theorem shop_dimension_example : shopDimension 1300 156 = 100 := by
  sorry

end NUMINAMATH_CALUDE_shop_dimension_example_l1175_117532


namespace NUMINAMATH_CALUDE_jellybean_count_l1175_117598

theorem jellybean_count (total blue purple red orange : ℕ) : 
  total = 200 →
  blue = 14 →
  purple = 26 →
  red = 120 →
  total = blue + purple + red + orange →
  orange = 40 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1175_117598


namespace NUMINAMATH_CALUDE_triangle_parallelogram_relation_l1175_117589

theorem triangle_parallelogram_relation (triangle_area : ℝ) (parallelogram_height : ℝ) : 
  triangle_area = 15 → parallelogram_height = 5 → 
  ∃ (parallelogram_area parallelogram_base : ℝ),
    parallelogram_area = 2 * triangle_area ∧
    parallelogram_area = parallelogram_height * parallelogram_base ∧
    parallelogram_area = 30 ∧
    parallelogram_base = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_relation_l1175_117589


namespace NUMINAMATH_CALUDE_geometric_sequence_relation_l1175_117507

/-- A geometric sequence with five terms -/
structure GeometricSequence :=
  (a b c : ℝ)
  (isGeometric : ∃ r : ℝ, r ≠ 0 ∧ a = -2 * r ∧ b = a * r ∧ c = b * r ∧ -8 = c * r)

/-- The theorem stating the relationship between b and ac in the geometric sequence -/
theorem geometric_sequence_relation (seq : GeometricSequence) : seq.b = -4 ∧ seq.a * seq.c = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_relation_l1175_117507


namespace NUMINAMATH_CALUDE_inverse_proportion_values_l1175_117571

/-- α is inversely proportional to β with α = 5 when β = -4 -/
def inverse_proportion (α β : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ α * β = k ∧ 5 * (-4) = k

theorem inverse_proportion_values (α β : ℝ) (h : inverse_proportion α β) :
  (β = -10 → α = 2) ∧ (β = 2 → α = -10) := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_values_l1175_117571


namespace NUMINAMATH_CALUDE_friday_snowfall_l1175_117552

-- Define the snowfall amounts
def total_snowfall : Float := 0.89
def wednesday_snowfall : Float := 0.33
def thursday_snowfall : Float := 0.33

-- Define the theorem
theorem friday_snowfall :
  total_snowfall - (wednesday_snowfall + thursday_snowfall) = 0.23 := by
  sorry

end NUMINAMATH_CALUDE_friday_snowfall_l1175_117552


namespace NUMINAMATH_CALUDE_pecan_mixture_amount_l1175_117577

/-- Prove that the amount of pecans in a mixture is correct given the specified conditions. -/
theorem pecan_mixture_amount 
  (cashew_amount : ℝ) 
  (cashew_price : ℝ) 
  (mixture_price : ℝ) 
  (pecan_amount : ℝ) :
  cashew_amount = 2 ∧ 
  cashew_price = 3.5 ∧ 
  mixture_price = 4.34 ∧
  pecan_amount = 1.33333333333 →
  pecan_amount = 1.33333333333 :=
by sorry

end NUMINAMATH_CALUDE_pecan_mixture_amount_l1175_117577


namespace NUMINAMATH_CALUDE_not_all_rectangles_similar_l1175_117511

/-- A rectangle is a parallelogram with all interior angles equal to 90 degrees. -/
structure Rectangle where
  sides : Fin 4 → ℝ
  angle_measure : ℝ
  is_parallelogram : True
  right_angles : angle_measure = 90

/-- Similarity in shapes means corresponding angles are equal and ratios of corresponding sides are constant. -/
def are_similar (r1 r2 : Rectangle) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 4, r1.sides i = k * r2.sides i

/-- Theorem: Not all rectangles are similar to each other. -/
theorem not_all_rectangles_similar : ¬ ∀ r1 r2 : Rectangle, are_similar r1 r2 := by
  sorry

end NUMINAMATH_CALUDE_not_all_rectangles_similar_l1175_117511


namespace NUMINAMATH_CALUDE_S_infinite_l1175_117501

/-- The set of positive integers n for which the number of positive divisors of 2^n - 1 is greater than n -/
def S : Set Nat :=
  {n : Nat | n > 0 ∧ (Nat.divisors (2^n - 1)).card > n}

/-- Theorem stating that the set S is infinite -/
theorem S_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_infinite_l1175_117501


namespace NUMINAMATH_CALUDE_basketball_team_points_l1175_117525

theorem basketball_team_points (x : ℚ) (y : ℕ) : 
  (1 / 3 : ℚ) * x + (1 / 5 : ℚ) * x + 18 + y = x → 
  y ≤ 21 → 
  y = 15 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_points_l1175_117525


namespace NUMINAMATH_CALUDE_final_position_of_942nd_square_l1175_117505

/-- Represents the state of a square after folding -/
structure SquareState where
  position : ℕ
  below : ℕ

/-- Calculates the new state of a square after a fold -/
def fold (state : SquareState) (stripLength : ℕ) : SquareState :=
  if state.position ≤ stripLength then
    state
  else
    { position := 2 * stripLength + 1 - state.position,
      below := stripLength - (2 * stripLength + 1 - state.position) }

/-- Performs multiple folds on a square -/
def foldMultiple (initialState : SquareState) (numFolds : ℕ) : SquareState :=
  match numFolds with
  | 0 => initialState
  | n + 1 => fold (foldMultiple initialState n) (1024 / 2^(n + 1))

theorem final_position_of_942nd_square :
  (foldMultiple { position := 942, below := 0 } 10).below = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_position_of_942nd_square_l1175_117505


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1175_117554

theorem fraction_multiplication :
  (2 : ℚ) / 3 * 5 / 7 * 9 / 13 * 4 / 11 = 120 / 1001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1175_117554


namespace NUMINAMATH_CALUDE_gcd_15_2015_l1175_117502

theorem gcd_15_2015 : Nat.gcd 15 2015 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15_2015_l1175_117502


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1175_117509

theorem ferris_wheel_capacity (total_capacity : ℕ) (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : total_capacity = 4)
  (h2 : num_seats = 2)
  (h3 : people_per_seat * num_seats = total_capacity) :
  people_per_seat = 2 := by
sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1175_117509


namespace NUMINAMATH_CALUDE_sequence_general_term_l1175_117527

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 2 → a n / a (n - 1) = 2^(n - 1)) →
  a 1 = 1 →
  ∀ n : ℕ, n > 0 → a n = 2^(n * (n - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1175_117527


namespace NUMINAMATH_CALUDE_smallest_marble_count_l1175_117599

theorem smallest_marble_count : ∃ N : ℕ, 
  N > 1 ∧ 
  N % 9 = 1 ∧ 
  N % 10 = 1 ∧ 
  N % 11 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 9 = 1 ∧ m % 10 = 1 ∧ m % 11 = 1 → m ≥ N) ∧
  N = 991 := by
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l1175_117599


namespace NUMINAMATH_CALUDE_baker_pastries_l1175_117564

theorem baker_pastries (cakes_made : ℕ) (pastries_sold : ℕ) (total_cakes_sold : ℕ) (difference : ℕ) :
  cakes_made = 14 →
  pastries_sold = 8 →
  total_cakes_sold = 97 →
  total_cakes_sold - pastries_sold = difference →
  difference = 89 →
  pastries_sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_baker_pastries_l1175_117564


namespace NUMINAMATH_CALUDE_ramsey_r33_l1175_117528

/-- Represents the relationship between two people -/
inductive Relationship
| Acquaintance
| Stranger

/-- A group of people -/
def People := Fin 6

/-- The relationship between each pair of people -/
def RelationshipMap := People → People → Relationship

/-- Checks if three people are mutual acquaintances -/
def areMutualAcquaintances (rel : RelationshipMap) (a b c : People) : Prop :=
  rel a b = Relationship.Acquaintance ∧
  rel a c = Relationship.Acquaintance ∧
  rel b c = Relationship.Acquaintance

/-- Checks if three people are mutual strangers -/
def areMutualStrangers (rel : RelationshipMap) (a b c : People) : Prop :=
  rel a b = Relationship.Stranger ∧
  rel a c = Relationship.Stranger ∧
  rel b c = Relationship.Stranger

/-- Main theorem: In a group of 6 people, there are either 3 mutual acquaintances or 3 mutual strangers -/
theorem ramsey_r33 (rel : RelationshipMap) :
  (∃ a b c : People, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ areMutualAcquaintances rel a b c) ∨
  (∃ a b c : People, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ areMutualStrangers rel a b c) :=
sorry

end NUMINAMATH_CALUDE_ramsey_r33_l1175_117528


namespace NUMINAMATH_CALUDE_decimal_85_equals_base7_151_l1175_117555

/-- Converts a number from decimal to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to decimal --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem decimal_85_equals_base7_151 : fromBase7 [1, 5, 1] = 85 := by
  sorry

#eval toBase7 85  -- Should output [1, 5, 1]
#eval fromBase7 [1, 5, 1]  -- Should output 85

end NUMINAMATH_CALUDE_decimal_85_equals_base7_151_l1175_117555


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1175_117594

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 16 + a 30 = 60 →
  a 10 + a 22 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1175_117594


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1175_117569

theorem arithmetic_mean_problem (x : ℚ) : 
  (x + 10 + 20 + 3*x + 18 + 3*x + 6) / 5 = 32 → x = 106/7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1175_117569


namespace NUMINAMATH_CALUDE_pascal_triangle_odd_rows_l1175_117524

/-- Represents a row in Pascal's triangle -/
def PascalRow := List Nat

/-- Generates the nth row of Pascal's triangle -/
def generatePascalRow (n : Nat) : PascalRow := sorry

/-- Checks if a row has all odd numbers except for the ends -/
def isAllOddExceptEnds (row : PascalRow) : Bool := sorry

/-- Counts the number of rows up to n that have all odd numbers except for the ends -/
def countAllOddExceptEndsRows (n : Nat) : Nat := sorry

/-- The main theorem to be proved -/
theorem pascal_triangle_odd_rows :
  countAllOddExceptEndsRows 30 = 3 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_odd_rows_l1175_117524


namespace NUMINAMATH_CALUDE_next_simultaneous_ringing_l1175_117585

def town_hall_period : ℕ := 18
def university_tower_period : ℕ := 24
def fire_station_period : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_ringing :
  ∃ (n : ℕ), n > 0 ∧ 
    n % town_hall_period = 0 ∧
    n % university_tower_period = 0 ∧
    n % fire_station_period = 0 ∧
    n / minutes_in_hour = 6 :=
sorry

end NUMINAMATH_CALUDE_next_simultaneous_ringing_l1175_117585


namespace NUMINAMATH_CALUDE_tuesday_books_brought_back_l1175_117559

/-- Calculates the number of books brought back on Tuesday given the initial number of books,
    the number of books taken out on Monday, and the final number of books on Tuesday. -/
def books_brought_back (initial : ℕ) (taken_out : ℕ) (final : ℕ) : ℕ :=
  final - (initial - taken_out)

/-- Theorem stating that 22 books were brought back on Tuesday given the specified conditions. -/
theorem tuesday_books_brought_back :
  books_brought_back 336 124 234 = 22 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_books_brought_back_l1175_117559


namespace NUMINAMATH_CALUDE_initial_bags_calculation_l1175_117512

/-- Given the total number of cookies, total number of candies, and current number of bags,
    calculate the initial number of bags. -/
def initialBags (totalCookies : ℕ) (totalCandies : ℕ) (currentBags : ℕ) : ℕ :=
  sorry

theorem initial_bags_calculation (totalCookies totalCandies currentBags : ℕ) 
    (h1 : totalCookies = 28)
    (h2 : totalCandies = 86)
    (h3 : currentBags = 2)
    (h4 : totalCookies % currentBags = 0)  -- Ensures equal distribution of cookies
    (h5 : totalCandies % (initialBags totalCookies totalCandies currentBags) = 0)  -- Ensures equal distribution of candies
    (h6 : totalCookies / currentBags = totalCandies / (initialBags totalCookies totalCandies currentBags))  -- Cookies per bag equals candies per bag
    : initialBags totalCookies totalCandies currentBags = 6 :=
  sorry

end NUMINAMATH_CALUDE_initial_bags_calculation_l1175_117512


namespace NUMINAMATH_CALUDE_house_transaction_result_l1175_117551

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : ℝ
  hasHouse : Bool

/-- Represents the state of the house -/
structure HouseState where
  value : ℝ
  owner : String

def initial_mr_a : FinancialState := { cash := 15000, hasHouse := true }
def initial_mr_b : FinancialState := { cash := 20000, hasHouse := false }
def initial_house : HouseState := { value := 15000, owner := "A" }

def house_sale_price : ℝ := 20000
def depreciation_rate : ℝ := 0.15

theorem house_transaction_result :
  let first_transaction_mr_a : FinancialState :=
    { cash := initial_mr_a.cash + house_sale_price, hasHouse := false }
  let first_transaction_mr_b : FinancialState :=
    { cash := initial_mr_b.cash - house_sale_price, hasHouse := true }
  let depreciated_house_value : ℝ := initial_house.value * (1 - depreciation_rate)
  let final_mr_a : FinancialState :=
    { cash := first_transaction_mr_a.cash - depreciated_house_value, hasHouse := true }
  let final_mr_b : FinancialState :=
    { cash := first_transaction_mr_b.cash + depreciated_house_value, hasHouse := false }
  let mr_a_net_gain : ℝ := final_mr_a.cash + depreciated_house_value - (initial_mr_a.cash + initial_house.value)
  let mr_b_net_gain : ℝ := final_mr_b.cash - initial_mr_b.cash
  mr_a_net_gain = 5000 ∧ mr_b_net_gain = -7250 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_result_l1175_117551


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l1175_117530

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l1175_117530


namespace NUMINAMATH_CALUDE_book_purchase_change_l1175_117534

/-- The change received when buying two books with given prices and paying with a fixed amount. -/
theorem book_purchase_change (book1_price book2_price payment : ℝ) 
  (h1 : book1_price = 5.5)
  (h2 : book2_price = 6.5)
  (h3 : payment = 20) : 
  payment - (book1_price + book2_price) = 8 := by
sorry

end NUMINAMATH_CALUDE_book_purchase_change_l1175_117534


namespace NUMINAMATH_CALUDE_min_value_theorem_l1175_117526

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (min : ℝ), min = 5 + 2 * Real.sqrt 6 ∧ ∀ (x : ℝ), (3 / a + 2 / b) ≥ x := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1175_117526


namespace NUMINAMATH_CALUDE_parabolas_intersection_l1175_117535

/-- Parabola 1 equation -/
def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 7

/-- Parabola 2 equation -/
def parabola2 (x : ℝ) : ℝ := 2 * x^2 + 5

/-- The intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) := {(-4, 37), (3/2, 9.5)}

theorem parabolas_intersection :
  ∀ p : ℝ × ℝ, (parabola1 p.1 = parabola2 p.1) ↔ p ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l1175_117535


namespace NUMINAMATH_CALUDE_fourth_number_12th_row_l1175_117588

/-- Given a number pattern where each row has 8 numbers, and the last number of each row is 8 times the row number, this function calculates the nth number in the mth row. -/
def patternNumber (m n : ℕ) : ℕ :=
  8 * (m - 1) + n

/-- Theorem stating that the fourth number in the 12th row of the described pattern is 92. -/
theorem fourth_number_12th_row : patternNumber 12 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_12th_row_l1175_117588


namespace NUMINAMATH_CALUDE_inverse_proposition_l1175_117578

theorem inverse_proposition (a b : ℝ) :
  (∀ x y : ℝ, (|x| > |y| → x > y)) →
  (a > b → |a| > |b|) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_l1175_117578


namespace NUMINAMATH_CALUDE_fraction_value_l1175_117545

theorem fraction_value (a b : ℚ) (h1 : a = 7) (h2 : b = 2) : 3 / (a + b) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1175_117545


namespace NUMINAMATH_CALUDE_two_integer_k_values_for_nontrivial_solution_l1175_117518

/-- The system of equations has a non-trivial solution for exactly two integer values of k. -/
theorem two_integer_k_values_for_nontrivial_solution :
  ∃! (s : Finset ℤ), (∀ k ∈ s, ∃ a b c : ℝ, (a, b, c) ≠ (0, 0, 0) ∧
    a^2 + b^2 = k * c * (a + b) ∧
    b^2 + c^2 = k * a * (b + c) ∧
    c^2 + a^2 = k * b * (c + a)) ∧
  s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_integer_k_values_for_nontrivial_solution_l1175_117518


namespace NUMINAMATH_CALUDE_sector_central_angle_l1175_117536

/-- Given a sector with radius 8 and area 32, prove that its central angle in radians is 1 -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 8) (h2 : area = 32) :
  let α := 2 * area / (r * r)
  α = 1 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1175_117536


namespace NUMINAMATH_CALUDE_chad_age_l1175_117562

theorem chad_age (diana fabian eduardo chad : ℕ) 
  (h1 : diana = fabian - 5)
  (h2 : fabian = eduardo + 2)
  (h3 : chad = eduardo + 3)
  (h4 : diana = 15) : 
  chad = 21 := by
  sorry

end NUMINAMATH_CALUDE_chad_age_l1175_117562


namespace NUMINAMATH_CALUDE_stating_course_selection_schemes_l1175_117521

/-- Represents the number of elective courses -/
def num_courses : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of courses with no students -/
def courses_with_no_students : ℕ := 2

/-- Represents the number of courses with students -/
def courses_with_students : ℕ := num_courses - courses_with_no_students

/-- 
  Theorem stating that the number of ways to distribute students among courses
  under the given conditions is 18
-/
theorem course_selection_schemes : 
  (num_courses.choose courses_with_students) * 
  ((num_students.choose courses_with_students) / 2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_stating_course_selection_schemes_l1175_117521


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1175_117583

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 30 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1175_117583


namespace NUMINAMATH_CALUDE_diagonal_length_l1175_117506

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_special_quadrilateral (q : Quadrilateral) : Prop :=
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  dist q.A q.B = 12 ∧
  dist q.B q.C = 12 ∧
  dist q.C q.D = 15 ∧
  dist q.D q.A = 15 ∧
  let angle := λ p₁ p₂ p₃ : ℝ × ℝ => Real.arccos (
    ((p₁.1 - p₂.1) * (p₃.1 - p₂.1) + (p₁.2 - p₂.2) * (p₃.2 - p₂.2)) /
    (dist p₁ p₂ * dist p₂ p₃)
  )
  angle q.A q.D q.C = 2 * Real.pi / 3

-- Theorem statement
theorem diagonal_length (q : Quadrilateral) (h : is_special_quadrilateral q) :
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  dist q.A q.C = 15 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_l1175_117506


namespace NUMINAMATH_CALUDE_same_color_probability_l1175_117519

theorem same_color_probability (total_pieces : ℕ) (black_pieces : ℕ) (white_pieces : ℕ)
  (prob_two_black : ℚ) (prob_two_white : ℚ) :
  total_pieces = 15 →
  black_pieces = 6 →
  white_pieces = 9 →
  prob_two_black = 1/7 →
  prob_two_white = 12/35 →
  prob_two_black + prob_two_white = 17/35 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1175_117519


namespace NUMINAMATH_CALUDE_max_value_3m_4n_l1175_117515

theorem max_value_3m_4n (m n : ℕ+) : 
  (m.val * (m.val + 1) + n.val^2 = 1987) → 
  (∀ k l : ℕ+, k.val * (k.val + 1) + l.val^2 = 1987 → 3 * k.val + 4 * l.val ≤ 3 * m.val + 4 * n.val) →
  3 * m.val + 4 * n.val = 221 :=
by sorry

end NUMINAMATH_CALUDE_max_value_3m_4n_l1175_117515


namespace NUMINAMATH_CALUDE_loss_equals_cost_of_five_balls_l1175_117541

def number_of_balls : ℕ := 13
def selling_price : ℕ := 720
def cost_per_ball : ℕ := 90

theorem loss_equals_cost_of_five_balls :
  (number_of_balls * cost_per_ball - selling_price) / cost_per_ball = 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_equals_cost_of_five_balls_l1175_117541


namespace NUMINAMATH_CALUDE_fraction_zero_iff_x_plus_minus_five_l1175_117543

theorem fraction_zero_iff_x_plus_minus_five (x : ℝ) :
  (x^2 - 25) / (4 * x^2 - 2 * x) = 0 ↔ x = 5 ∨ x = -5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_iff_x_plus_minus_five_l1175_117543


namespace NUMINAMATH_CALUDE_smallest_three_digit_sum_of_powers_l1175_117538

/-- A function that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A function that checks if a number is a one-digit positive integer -/
def isOneDigitPositive (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- The main theorem statement -/
theorem smallest_three_digit_sum_of_powers :
  ∃ (K a b : ℕ), 
    isThreeDigit K ∧
    isOneDigitPositive a ∧
    isOneDigitPositive b ∧
    K = a^b + b^a ∧
    (∀ (K' a' b' : ℕ), 
      isThreeDigit K' ∧ 
      isOneDigitPositive a' ∧ 
      isOneDigitPositive b' ∧ 
      K' = a'^b' + b'^a' → 
      K ≤ K') ∧
    K = 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_sum_of_powers_l1175_117538


namespace NUMINAMATH_CALUDE_intersection_complement_eq_l1175_117576

open Set

def U : Set ℝ := univ
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement_eq : A ∩ (U \ B) = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_eq_l1175_117576


namespace NUMINAMATH_CALUDE_smallest_hypotenuse_right_triangle_isosceles_right_triangle_minimizes_hypotenuse_l1175_117556

theorem smallest_hypotenuse_right_triangle (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = 8 →
  c ≥ 4 * Real.sqrt 2 :=
by sorry

theorem isosceles_right_triangle_minimizes_hypotenuse :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    a + b + c = 8 ∧
    c = 4 * Real.sqrt 2 ∧
    a = b :=
by sorry

end NUMINAMATH_CALUDE_smallest_hypotenuse_right_triangle_isosceles_right_triangle_minimizes_hypotenuse_l1175_117556


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1175_117510

theorem contrapositive_equivalence (a b : ℝ) :
  (((a + b = 1) → (a^2 + b^2 ≥ 1/2)) ↔ ((a^2 + b^2 < 1/2) → (a + b ≠ 1))) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1175_117510


namespace NUMINAMATH_CALUDE_polar_bear_club_time_l1175_117500

/-- Represents the time spent in the pool by each person -/
structure PoolTime where
  jerry : ℕ
  elaine : ℕ
  george : ℕ
  kramer : ℕ

/-- Calculates the total time spent in the pool -/
def total_time (pt : PoolTime) : ℕ :=
  pt.jerry + pt.elaine + pt.george + pt.kramer

/-- Theorem stating the total time spent in the pool is 11 minutes -/
theorem polar_bear_club_time : ∃ (pt : PoolTime),
  pt.jerry = 3 ∧
  pt.elaine = 2 * pt.jerry ∧
  pt.george = pt.elaine / 3 ∧
  pt.kramer = 0 ∧
  total_time pt = 11 := by
  sorry

end NUMINAMATH_CALUDE_polar_bear_club_time_l1175_117500


namespace NUMINAMATH_CALUDE_french_toast_loaves_l1175_117516

/-- Calculates the number of loaves of bread needed for french toast over a given number of weeks -/
def loaves_needed (slices_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (slices_per_loaf : ℕ) : ℕ :=
  (slices_per_day * days_per_week * weeks + slices_per_loaf - 1) / slices_per_loaf

theorem french_toast_loaves :
  let slices_per_day : ℕ := 3  -- Suzanne (1) + husband (1) + daughters (0.5 + 0.5)
  let days_per_week : ℕ := 2   -- Saturday and Sunday
  let weeks : ℕ := 52
  let slices_per_loaf : ℕ := 12
  loaves_needed slices_per_day days_per_week weeks slices_per_loaf = 26 := by
  sorry

#eval loaves_needed 3 2 52 12

end NUMINAMATH_CALUDE_french_toast_loaves_l1175_117516


namespace NUMINAMATH_CALUDE_investment_growth_l1175_117581

/-- Calculates the total amount after compound interest is applied --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_growth :
  let initial_investment : ℝ := 300
  let monthly_rate : ℝ := 0.1
  let months : ℕ := 2
  compound_interest initial_investment monthly_rate months = 363 := by
sorry

end NUMINAMATH_CALUDE_investment_growth_l1175_117581


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1175_117514

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1175_117514


namespace NUMINAMATH_CALUDE_exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l1175_117574

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := 180 * (n - 2)
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The exterior angle of a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon_is_45 : 
  exterior_angle_regular_octagon = 45 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l1175_117574


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1175_117522

theorem fraction_equation_solution (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 20 * b) / (b + 20 * a) = 3) : 
  a / b = 0.33 := by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1175_117522
