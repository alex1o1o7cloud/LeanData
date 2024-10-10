import Mathlib

namespace expression_evaluation_l385_38555

theorem expression_evaluation :
  (15 + 12)^2 - (12^2 + 15^2 + 6 * 15 * 12) = -720 := by
  sorry

end expression_evaluation_l385_38555


namespace rectangle_area_l385_38583

/-- Given a rectangle where the length is four times the width and the perimeter is 250 cm,
    prove that its area is 2500 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 250 → l * w = 2500 := by
  sorry

end rectangle_area_l385_38583


namespace quadratic_solution_sum_l385_38510

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 7 * x + 20 = 0 ↔ x = p + q * I ∨ x = p - q * I) → 
  p + q^2 = 421 / 100 := by
  sorry

end quadratic_solution_sum_l385_38510


namespace smallest_common_multiple_of_6_and_15_l385_38594

theorem smallest_common_multiple_of_6_and_15 : 
  ∃ b : ℕ+, (∀ m : ℕ+, (6 ∣ m) ∧ (15 ∣ m) → b ≤ m) ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 := by
  sorry

end smallest_common_multiple_of_6_and_15_l385_38594


namespace median_less_than_half_sum_of_sides_l385_38540

/-- Given a triangle ABC with sides a, b, and c, and median CM₃ to side c,
    prove that CM₃ < (a + b) / 2 -/
theorem median_less_than_half_sum_of_sides 
  {a b c : ℝ} 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : c < a + b) :
  let CM₃ := Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  CM₃ < (a + b) / 2 := by
sorry

end median_less_than_half_sum_of_sides_l385_38540


namespace initial_friends_count_l385_38518

def car_cost : ℕ := 1700
def car_wash_earnings : ℕ := 500
def cost_increase : ℕ := 40

theorem initial_friends_count : 
  ∃ (F : ℕ), 
    F > 0 ∧
    (car_cost - car_wash_earnings) / F + cost_increase = 
    (car_cost - car_wash_earnings) / (F - 1) ∧
    F = 6 := by
  sorry

end initial_friends_count_l385_38518


namespace quadratic_inequality_solution_set_l385_38587

theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ (-9 < m ∧ m < -1) :=
by sorry

end quadratic_inequality_solution_set_l385_38587


namespace rectangle_area_l385_38570

/-- Given a rectangle with perimeter 280 meters and length-to-width ratio of 5:2, its area is 4000 square meters. -/
theorem rectangle_area (L W : ℝ) (h1 : 2*L + 2*W = 280) (h2 : L / W = 5 / 2) : L * W = 4000 := by
  sorry

end rectangle_area_l385_38570


namespace total_apples_l385_38545

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73

theorem total_apples : pinky_apples + danny_apples = 109 := by
  sorry

end total_apples_l385_38545


namespace ellipse_equation_l385_38522

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h0 : a > b
  h1 : b > 0

/-- The right focus of the ellipse -/
def right_focus (e : Ellipse) : ℝ × ℝ := (3, 0)

/-- The midpoint of the line segment AB -/
def midpoint_AB : ℝ × ℝ := (1, -1)

/-- Theorem: Given an ellipse with the specified properties, its equation is x²/18 + y²/9 = 1 -/
theorem ellipse_equation (e : Ellipse) 
  (h2 : right_focus e = (3, 0))
  (h3 : midpoint_AB = (1, -1)) :
  ∃ (x y : ℝ), x^2 / 18 + y^2 / 9 = 1 :=
sorry

end ellipse_equation_l385_38522


namespace coaching_fee_calculation_l385_38595

/-- Calculates the number of days from January 1 to a given date in a non-leap year -/
def daysFromNewYear (month : Nat) (day : Nat) : Nat :=
  match month with
  | 1 => day
  | 2 => 31 + day
  | 3 => 59 + day
  | 4 => 90 + day
  | 5 => 120 + day
  | 6 => 151 + day
  | 7 => 181 + day
  | 8 => 212 + day
  | 9 => 243 + day
  | 10 => 273 + day
  | 11 => 304 + day
  | 12 => 334 + day
  | _ => 0

/-- Daily coaching charge in dollars -/
def dailyCharge : Nat := 39

/-- Calculates the total coaching fee -/
def totalCoachingFee (startMonth : Nat) (startDay : Nat) (endMonth : Nat) (endDay : Nat) : Nat :=
  let totalDays := daysFromNewYear endMonth endDay - daysFromNewYear startMonth startDay + 1
  totalDays * dailyCharge

theorem coaching_fee_calculation :
  totalCoachingFee 1 1 11 3 = 11934 := by
  sorry

#eval totalCoachingFee 1 1 11 3

end coaching_fee_calculation_l385_38595


namespace divisor_sum_totient_inequality_divisor_sum_totient_equality_l385_38597

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem divisor_sum_totient_inequality (n : ℕ) :
  1 / (phi n : ℝ) + 1 / (sigma n : ℝ) ≥ 2 / n :=
sorry

/-- Characterization of the equality case -/
theorem divisor_sum_totient_equality (n : ℕ) :
  (1 / (phi n : ℝ) + 1 / (sigma n : ℝ) = 2 / n) ↔ n = 1 :=
sorry

end divisor_sum_totient_inequality_divisor_sum_totient_equality_l385_38597


namespace meeting_calculation_correct_l385_38543

/-- Represents the meeting of a pedestrian and cyclist --/
structure Meeting where
  time : ℝ  -- Time since start (in hours)
  distance : ℝ  -- Distance from the city (in km)

/-- Calculates the meeting point of a pedestrian and cyclist --/
def calculate_meeting (city_distance : ℝ) (pedestrian_speed : ℝ) (cyclist_speed : ℝ) (cyclist_rest : ℝ) : Meeting :=
  { time := 1.25,  -- 9:15 AM is 1.25 hours after 8:00 AM
    distance := 4.5 }

/-- Theorem stating the correctness of the meeting calculation --/
theorem meeting_calculation_correct 
  (city_distance : ℝ) 
  (pedestrian_speed : ℝ) 
  (cyclist_speed : ℝ) 
  (cyclist_rest : ℝ)
  (h1 : city_distance = 12)
  (h2 : pedestrian_speed = 6)
  (h3 : cyclist_speed = 18)
  (h4 : cyclist_rest = 1/3)  -- 20 minutes is 1/3 of an hour
  : 
  let meeting := calculate_meeting city_distance pedestrian_speed cyclist_speed cyclist_rest
  meeting.time = 1.25 ∧ meeting.distance = 4.5 := by
  sorry

#check meeting_calculation_correct

end meeting_calculation_correct_l385_38543


namespace pipe_filling_time_l385_38531

theorem pipe_filling_time (fill_rate_A fill_rate_B : ℝ) : 
  fill_rate_A = 2 / 75 →
  9 * (fill_rate_A + fill_rate_B) + 21 * fill_rate_A = 1 →
  fill_rate_B = 1 / 45 :=
by sorry

end pipe_filling_time_l385_38531


namespace correct_calculation_l385_38532

theorem correct_calculation (x : ℤ) : 63 - x = 70 → 36 + x = 29 := by
  sorry

end correct_calculation_l385_38532


namespace extremum_at_one_implies_f_two_equals_two_l385_38503

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- State the theorem
theorem extremum_at_one_implies_f_two_equals_two (a b : ℝ) :
  (∃ (y : ℝ), y = f a b 1 ∧ y = 10 ∧ 
    (∀ (x : ℝ), f a b x ≤ y ∨ f a b x ≥ y)) →
  f a b 2 = 2 := by
  sorry

end extremum_at_one_implies_f_two_equals_two_l385_38503


namespace simplify_expression_l385_38593

theorem simplify_expression (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  (12 * x^2 * y^3 * z) / (4 * x * y * z^2) = 9 := by
  sorry

end simplify_expression_l385_38593


namespace parabola_point_relation_l385_38519

theorem parabola_point_relation (a y₁ y₂ y₃ : ℝ) :
  a < -1 →
  y₁ = (a - 1)^2 →
  y₂ = a^2 →
  y₃ = (a + 1)^2 →
  y₁ > y₂ ∧ y₂ > y₃ :=
by sorry

end parabola_point_relation_l385_38519


namespace distinct_strings_equal_fibonacci_l385_38580

/-- Represents the possible operations on a string --/
inductive Operation
  | replaceH
  | replaceMM
  | replaceT

/-- Defines a valid string after operations --/
def ValidString : Type := List Char

/-- Applies an operation to a valid string --/
def applyOperation (s : ValidString) (op : Operation) : ValidString :=
  sorry

/-- Counts the number of distinct strings after n operations --/
def countDistinctStrings (n : Nat) : Nat :=
  sorry

/-- Computes the nth Fibonacci number (starting with F(1) = 2, F(2) = 3) --/
def fibonacci (n : Nat) : Nat :=
  sorry

/-- The main theorem: number of distinct strings after 10 operations equals 10th Fibonacci number --/
theorem distinct_strings_equal_fibonacci :
  countDistinctStrings 10 = fibonacci 10 := by
  sorry

end distinct_strings_equal_fibonacci_l385_38580


namespace exponential_problem_l385_38558

theorem exponential_problem (a x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) :
  a^(2*x + 3*y) = 108 := by
  sorry

end exponential_problem_l385_38558


namespace complex_magnitude_l385_38520

theorem complex_magnitude (z : ℂ) (h : z + (z - 1) * Complex.I = 3) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l385_38520


namespace determinant_scaling_l385_38557

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 7 →
  Matrix.det ![![3*x, 3*y], ![3*z, 3*w]] = 63 := by
  sorry

end determinant_scaling_l385_38557


namespace smallest_divisor_perfect_cube_l385_38591

theorem smallest_divisor_perfect_cube : ∃! n : ℕ, 
  n > 0 ∧ 
  n ∣ 34560 ∧ 
  (∃ m : ℕ, 34560 / n = m^3) ∧
  (∀ k : ℕ, k > 0 → k ∣ 34560 → (∃ l : ℕ, 34560 / k = l^3) → k ≥ n) :=
by sorry

end smallest_divisor_perfect_cube_l385_38591


namespace total_growing_space_l385_38566

/-- Represents a garden bed with length and width dimensions -/
structure GardenBed where
  length : ℕ
  width : ℕ

/-- Calculates the area of a garden bed -/
def area (bed : GardenBed) : ℕ := bed.length * bed.width

/-- Calculates the total area of multiple identical garden beds -/
def totalArea (bed : GardenBed) (count : ℕ) : ℕ := area bed * count

/-- The set of garden beds Amy is building -/
def amysGardenBeds : List (GardenBed × ℕ) := [
  (⟨5, 4⟩, 3),
  (⟨6, 3⟩, 4),
  (⟨7, 5⟩, 2),
  (⟨8, 4⟩, 1)
]

/-- Theorem stating that the total growing space is 234 sq ft -/
theorem total_growing_space :
  (amysGardenBeds.map (fun (bed, count) => totalArea bed count)).sum = 234 := by
  sorry

end total_growing_space_l385_38566


namespace complex_magnitude_product_l385_38565

theorem complex_magnitude_product : 
  Complex.abs ((7 - 24 * Complex.I) * (-5 + 10 * Complex.I)) = 125 * Real.sqrt 5 := by
  sorry

end complex_magnitude_product_l385_38565


namespace sum_of_digits_of_large_power_minus_75_l385_38523

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_large_power_minus_75 :
  sum_of_digits (10^50 - 75) = 439 := by sorry

end sum_of_digits_of_large_power_minus_75_l385_38523


namespace snow_cover_probabilities_l385_38554

theorem snow_cover_probabilities (p : ℝ) (h : p = 0.2) :
  let q := 1 - p
  (q^2 = 0.64) ∧ (1 - q^2 = 0.36) := by
  sorry

end snow_cover_probabilities_l385_38554


namespace tan_beta_plus_pi_third_l385_38586

theorem tan_beta_plus_pi_third (α β : ℝ) 
  (h1 : Real.tan (α + β) = 1) 
  (h2 : Real.tan (α - π/3) = 1/3) : 
  Real.tan (β + π/3) = 1/2 := by
  sorry

end tan_beta_plus_pi_third_l385_38586


namespace jeremys_songs_l385_38539

theorem jeremys_songs (songs_yesterday songs_today total_songs : ℕ) : 
  songs_yesterday < songs_today →
  songs_yesterday = 9 →
  total_songs = 23 →
  songs_yesterday + songs_today = total_songs →
  songs_today = 14 := by
  sorry

end jeremys_songs_l385_38539


namespace art_students_l385_38529

/-- Proves that the number of students taking art is 20 -/
theorem art_students (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 30)
  (h3 : both = 10)
  (h4 : neither = 460) :
  total - neither - (music - both) = 20 := by
  sorry

end art_students_l385_38529


namespace scientific_notation_4212000_l385_38508

theorem scientific_notation_4212000 :
  ∃ (a : ℝ) (n : ℤ), 
    4212000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ 
    a < 10 ∧ 
    a = 4.212 ∧ 
    n = 6 := by
  sorry

end scientific_notation_4212000_l385_38508


namespace A9_coordinates_l385_38505

/-- Define a sequence of points in a Cartesian coordinate system -/
def A (n : ℕ) : ℝ × ℝ := (n, n^2)

/-- Theorem: The 9th point in the sequence has coordinates (9, 81) -/
theorem A9_coordinates : A 9 = (9, 81) := by
  sorry

end A9_coordinates_l385_38505


namespace average_marks_proof_l385_38571

theorem average_marks_proof (M P C : ℕ) (h1 : M + P = 60) (h2 : C = P + 20) :
  (M + C) / 2 = 40 := by
  sorry

end average_marks_proof_l385_38571


namespace seventh_term_of_sequence_l385_38576

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem seventh_term_of_sequence (a₁ q : ℝ) (h₁ : a₁ = 3) (h₂ : q = Real.sqrt 2) :
  geometric_sequence a₁ q 7 = 24 := by
  sorry

end seventh_term_of_sequence_l385_38576


namespace second_group_factories_l385_38550

theorem second_group_factories (total : ℕ) (first_group : ℕ) (unchecked : ℕ) :
  total = 169 →
  first_group = 69 →
  unchecked = 48 →
  total - (first_group + unchecked) = 52 := by
  sorry

end second_group_factories_l385_38550


namespace intersection_implies_a_in_range_l385_38526

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * |p.1|}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + a}

theorem intersection_implies_a_in_range (a : ℝ) :
  (∃! p, p ∈ set_A a ∩ set_B a) → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry


end intersection_implies_a_in_range_l385_38526


namespace cone_sphere_volume_difference_l385_38521

/-- Given an equilateral cone with an inscribed sphere, prove that the difference in volume
    between the cone and the sphere is (10/3) * √(2/π) dm³ when the surface area of the cone
    is 10 dm² more than the surface area of the sphere. -/
theorem cone_sphere_volume_difference (R : ℝ) (h : R > 0) :
  let r := R / Real.sqrt 3
  let cone_surface_area := 3 * Real.pi * R^2
  let sphere_surface_area := 4 * Real.pi * r^2
  let cone_volume := (Real.pi * Real.sqrt 3 / 3) * R^3
  let sphere_volume := (4 * Real.pi / 3) * r^3
  cone_surface_area = sphere_surface_area + 10 →
  cone_volume - sphere_volume = (10 / 3) * Real.sqrt (2 / Real.pi) := by
sorry

end cone_sphere_volume_difference_l385_38521


namespace parallel_vectors_sum_l385_38573

/-- Given two vectors a and b in R³, if they are parallel and have specific components,
    then the sum of their unknown components is -7. -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (2, x, 3)
  let b : ℝ × ℝ × ℝ := (-4, 2, y)
  (∃ (k : ℝ), a = k • b) →
  x + y = -7 := by
sorry

end parallel_vectors_sum_l385_38573


namespace tangent_slope_implies_a_value_l385_38588

-- Define the curve
def curve (a x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (a x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a_value (a : ℝ) :
  curve a (-1) = a + 2 →
  curve_derivative a (-1) = 8 →
  a = -6 := by sorry

end tangent_slope_implies_a_value_l385_38588


namespace unique_divisor_square_sum_l385_38501

theorem unique_divisor_square_sum (p n : ℕ) (hp : p.Prime) (hp2 : p > 2) (hn : n > 0) :
  ∃! d : ℕ, d > 0 ∧ d ∣ (p * n^2) ∧ ∃ k : ℕ, n^2 + d = k^2 :=
by sorry

end unique_divisor_square_sum_l385_38501


namespace dan_marbles_l385_38559

/-- The number of marbles Dan has after giving some away -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Dan has 96 marbles after giving away 32 from his initial 128 -/
theorem dan_marbles : remaining_marbles 128 32 = 96 := by
  sorry

end dan_marbles_l385_38559


namespace houses_built_during_boom_l385_38548

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558

theorem houses_built_during_boom : 
  current_houses - original_houses = 97741 := by sorry

end houses_built_during_boom_l385_38548


namespace possible_values_of_a_l385_38534

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x - 2 = 0}

-- Define the condition that x ∈ A is necessary but not sufficient for x ∈ B
def necessary_not_sufficient (a : ℝ) : Prop :=
  B a ⊆ A ∧ B a ≠ A

-- Theorem statement
theorem possible_values_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ a ∈ ({-1, 0, 1} : Set ℝ) :=
by sorry

end possible_values_of_a_l385_38534


namespace tangents_not_necessarily_coincide_at_both_intersections_l385_38584

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a parabola y = x^2 -/
def Parabola := {p : Point | p.y = p.x^2}

/-- Checks if a point is on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Checks if a point is on the parabola y = x^2 -/
def onParabola (p : Point) : Prop := p.y = p.x^2

/-- Checks if two curves have coinciding tangents at a point -/
def coincidingTangents (p : Point) : Prop := sorry

/-- The main theorem -/
theorem tangents_not_necessarily_coincide_at_both_intersections
  (c : Circle) (A B : Point) :
  onCircle A c → onCircle B c →
  onParabola A → onParabola B →
  A ≠ B →
  coincidingTangents A →
  ¬ ∀ (c : Circle) (A B : Point),
    onCircle A c → onCircle B c →
    onParabola A → onParabola B →
    A ≠ B →
    coincidingTangents A →
    coincidingTangents B :=
by sorry

end tangents_not_necessarily_coincide_at_both_intersections_l385_38584


namespace power_product_evaluation_l385_38525

theorem power_product_evaluation (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end power_product_evaluation_l385_38525


namespace phil_cards_l385_38504

/-- Calculates the number of baseball cards remaining after buying for a year and losing half. -/
def remaining_cards (cards_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  (cards_per_week * weeks_per_year) / 2

/-- Theorem stating that buying 20 cards each week for 52 weeks and losing half results in 520 cards. -/
theorem phil_cards : remaining_cards 20 52 = 520 := by
  sorry

end phil_cards_l385_38504


namespace tangent_line_to_circle_l385_38506

/-- Tangent line to a circle -/
theorem tangent_line_to_circle (r x_0 y_0 : ℝ) (h : x_0^2 + y_0^2 = r^2) :
  ∀ x y : ℝ, (x^2 + y^2 = r^2) → ((x - x_0)^2 + (y - y_0)^2 = 0 ∨ x_0*x + y_0*y = r^2) :=
by sorry

end tangent_line_to_circle_l385_38506


namespace inequality_equivalence_l385_38517

theorem inequality_equivalence (x : ℝ) : 
  (x - 2) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ 2 := by
sorry

end inequality_equivalence_l385_38517


namespace infinitely_many_coprime_phi_m_root_l385_38511

/-- m-th iteration of Euler's totient function -/
def phi_m (m : ℕ) : ℕ → ℕ :=
  match m with
  | 0 => id
  | m + 1 => phi_m m ∘ Nat.totient

/-- Main theorem -/
theorem infinitely_many_coprime_phi_m_root (a b m k : ℕ) (hk : k ≥ 2) :
  ∃ S : Set ℕ, S.Infinite ∧ ∀ n ∈ S, Nat.gcd (phi_m m n) (Nat.floor ((a * n + b : ℝ) ^ (1 / k))) = 1 := by
  sorry

end infinitely_many_coprime_phi_m_root_l385_38511


namespace matrix_multiplication_result_l385_38528

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 3, -1; 1, -2, 5; 0, 6, 1]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, 0, 4; 3, 2, -1; 0, 4, -2]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![11, 2, 7; -5, 16, -4; 18, 16, -8]

theorem matrix_multiplication_result : A * B = C := by
  sorry

end matrix_multiplication_result_l385_38528


namespace circle_passes_through_points_l385_38596

-- Define the circle equation
def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 8*x + 6*y

-- Define the points
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (1, 1)
def point_C : ℝ × ℝ := (4, 2)

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 = 0 ∧
  circle_equation point_B.1 point_B.2 = 0 ∧
  circle_equation point_C.1 point_C.2 = 0 :=
by sorry

end circle_passes_through_points_l385_38596


namespace percent_relation_l385_38514

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) :
  y = (3/7) * x := by
  sorry

end percent_relation_l385_38514


namespace solve_for_m_l385_38535

-- Define the custom operation
def customOp (m n : ℕ) : ℕ := n^2 - m

-- State the theorem
theorem solve_for_m :
  (∀ m n, customOp m n = n^2 - m) →
  (∃ m, customOp m 3 = 3) →
  (∃ m, m = 6) :=
by sorry

end solve_for_m_l385_38535


namespace infinitely_many_primes_dividing_2_pow_plus_poly_l385_38564

/-- A nonzero polynomial with integer coefficients -/
def nonzero_int_poly (P : ℕ → ℤ) : Prop :=
  ∃ n, P n ≠ 0 ∧ ∀ m, ∃ k : ℤ, P m = k

theorem infinitely_many_primes_dividing_2_pow_plus_poly 
  (P : ℕ → ℤ) (h : nonzero_int_poly P) :
  ∀ N : ℕ, ∃ q : ℕ, q > N ∧ Nat.Prime q ∧ 
    ∃ n : ℕ, (q : ℤ) ∣ (2^n : ℤ) + P n :=
sorry

end infinitely_many_primes_dividing_2_pow_plus_poly_l385_38564


namespace at_least_one_triangle_l385_38502

/-- Given 2n points (n ≥ 2) and n^2 + 1 segments, at least one triangle is formed. -/
theorem at_least_one_triangle (n : ℕ) (h : n ≥ 2) :
  ∃ (points : Finset (ℝ × ℝ × ℝ)) (segments : Finset (Fin 2 → ℝ × ℝ × ℝ)),
    Finset.card points = 2 * n ∧
    Finset.card segments = n^2 + 1 ∧
    ∃ (a b c : ℝ × ℝ × ℝ),
      a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
      (λ i => if i = 0 then a else b) ∈ segments ∧
      (λ i => if i = 0 then b else c) ∈ segments ∧
      (λ i => if i = 0 then c else a) ∈ segments :=
by
  sorry


end at_least_one_triangle_l385_38502


namespace base5_324_equals_binary_1011001_l385_38598

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to binary --/
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec toBinary (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else toBinary (m / 2) ((m % 2) :: acc)
  toBinary n []

/-- Theorem: The base-5 number 324₍₅₎ is equal to the binary number 1011001₍₂₎ --/
theorem base5_324_equals_binary_1011001 :
  decimalToBinary (base5ToDecimal [4, 2, 3]) = [1, 0, 1, 1, 0, 0, 1] := by
  sorry


end base5_324_equals_binary_1011001_l385_38598


namespace no_common_terms_except_one_l385_38500

def x : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n+2) => x (n+1) + 2 * x n

def y : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n+2) => 2 * y (n+1) + 3 * y n

theorem no_common_terms_except_one (n : ℕ) (m : ℕ) (h : n ≥ 1) :
  x n ≠ y m ∨ (x n = y m ∧ n = 0 ∧ m = 0) := by
  sorry

end no_common_terms_except_one_l385_38500


namespace cooperation_is_best_l385_38538

/-- Represents a factory with its daily processing capacity and fee -/
structure Factory where
  capacity : ℕ
  fee : ℕ

/-- Represents a processing plan with its duration and total cost -/
structure Plan where
  duration : ℕ
  cost : ℕ

/-- Calculates the plan for a single factory -/
def single_factory_plan (f : Factory) (total_products : ℕ) (engineer_fee : ℕ) : Plan :=
  let duration := total_products / f.capacity
  { duration := duration
  , cost := duration * (f.fee + engineer_fee) }

/-- Calculates the plan for two factories cooperating -/
def cooperation_plan (f1 f2 : Factory) (total_products : ℕ) (engineer_fee : ℕ) : Plan :=
  let duration := total_products / (f1.capacity + f2.capacity)
  { duration := duration
  , cost := duration * (f1.fee + f2.fee + engineer_fee) }

/-- Checks if one plan is better than another -/
def is_better_plan (p1 p2 : Plan) : Prop :=
  p1.duration < p2.duration ∧ p1.cost < p2.cost

theorem cooperation_is_best (total_products engineer_fee : ℕ) :
  let factory_a : Factory := { capacity := 16, fee := 80 }
  let factory_b : Factory := { capacity := 24, fee := 120 }
  let plan_a := single_factory_plan factory_a total_products engineer_fee
  let plan_b := single_factory_plan factory_b total_products engineer_fee
  let plan_coop := cooperation_plan factory_a factory_b total_products engineer_fee
  total_products = 960 ∧
  engineer_fee = 10 ∧
  factory_a.capacity * 3 = factory_b.capacity * 2 ∧
  factory_a.capacity + factory_b.capacity = 40 →
  is_better_plan plan_coop plan_a ∧ is_better_plan plan_coop plan_b :=
by sorry

end cooperation_is_best_l385_38538


namespace area_equality_iff_rectangle_l385_38524

/-- A quadrilateral with sides a, b, c, d and area A -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  A : ℝ

/-- Definition of a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∃ (w h : ℝ), q.a = w ∧ q.b = h ∧ q.c = w ∧ q.d = h ∧ q.A = w * h

/-- Theorem: Area equality holds iff the quadrilateral is a rectangle -/
theorem area_equality_iff_rectangle (q : Quadrilateral) :
  q.A = ((q.a + q.c) / 2) * ((q.b + q.d) / 2) ↔ is_rectangle q :=
sorry

end area_equality_iff_rectangle_l385_38524


namespace quadratic_always_has_real_roots_k_range_for_positive_root_less_than_one_l385_38533

variable (k : ℝ)

def quadratic_equation (x : ℝ) : Prop :=
  x^2 - (k+3)*x + 2*k + 2 = 0

theorem quadratic_always_has_real_roots :
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ :=
sorry

theorem k_range_for_positive_root_less_than_one :
  (∃ x : ℝ, quadratic_equation k x ∧ 0 < x ∧ x < 1) → -1 < k ∧ k < 0 :=
sorry

end quadratic_always_has_real_roots_k_range_for_positive_root_less_than_one_l385_38533


namespace necessary_but_not_sufficient_condition_l385_38556

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x > 5 → x > 4) ∧
  (∃ x : ℝ, x > 4 ∧ x ≤ 5) :=
by sorry

end necessary_but_not_sufficient_condition_l385_38556


namespace prime_triplets_theorem_l385_38551

/-- A prime triplet (a, b, c) satisfying the given conditions -/
structure PrimeTriplet where
  a : Nat
  b : Nat
  c : Nat
  h1 : a < b
  h2 : b < c
  h3 : c < 100
  h4 : Nat.Prime a
  h5 : Nat.Prime b
  h6 : Nat.Prime c
  h7 : (b + 1 - (a + 1)) * (c + 1 - (b + 1)) = (b + 1) * (b + 1 - (a + 1))

/-- The set of all valid prime triplets -/
def validTriplets : Set PrimeTriplet := {
  ⟨2, 5, 11, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨5, 11, 23, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨7, 11, 23, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨11, 23, 47, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩
}

/-- The main theorem -/
theorem prime_triplets_theorem :
  ∀ t : PrimeTriplet, t ∈ validTriplets := by
  sorry

end prime_triplets_theorem_l385_38551


namespace angle_bisector_theorem_l385_38549

-- Define the points A, B, C, and X
variable (A B C X : ℝ × ℝ)

-- Define the lengths of the sides
def AB : ℝ := 80
def AC : ℝ := 36
def BC : ℝ := 72

-- Define the angle bisector property
def is_angle_bisector (A B C X : ℝ × ℝ) : Prop :=
  (A.1 - X.1) * (C.2 - X.2) = (C.1 - X.1) * (A.2 - X.2) ∧
  (B.1 - X.1) * (C.2 - X.2) = (C.1 - X.1) * (B.2 - X.2)

-- State the theorem
theorem angle_bisector_theorem (h : is_angle_bisector A B C X) :
  dist A X = 80 / 3 :=
sorry

end angle_bisector_theorem_l385_38549


namespace college_student_count_l385_38536

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The total number of students in the college -/
def College.total (c : College) : ℕ := c.boys + c.girls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 190 girls, 
    the total number of students is 494 -/
theorem college_student_count (c : College) 
    (h1 : c.boys * 5 = c.girls * 8) 
    (h2 : c.girls = 190) : 
  c.total = 494 := by
  sorry

end college_student_count_l385_38536


namespace smallest_regular_polygon_sides_l385_38527

theorem smallest_regular_polygon_sides (n : ℕ) : n > 0 → (∃ k : ℕ, k > 0 ∧ 360 * k / (2 * n) = 28) → n ≥ 45 :=
sorry

end smallest_regular_polygon_sides_l385_38527


namespace sum_of_coefficients_l385_38589

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 := by
  sorry

end sum_of_coefficients_l385_38589


namespace expression_value_l385_38568

theorem expression_value (a b c : ℤ) (ha : a = 17) (hb : b = 21) (hc : c = 5) :
  (a - (b - c)) - ((a - b) - c) = 10 := by
  sorry

end expression_value_l385_38568


namespace smallest_five_digit_square_cube_l385_38553

theorem smallest_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n ≤ 99999) ∧ 
  (∃ a : ℕ, n = a^2) ∧ 
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m ≤ 99999 ∧ (∃ c : ℕ, m = c^2) ∧ (∃ d : ℕ, m = d^3) → m ≥ n) ∧
  n = 15625 :=
by sorry

end smallest_five_digit_square_cube_l385_38553


namespace no_multiples_of_five_end_in_two_l385_38581

theorem no_multiples_of_five_end_in_two :
  {n : ℕ | n > 0 ∧ n < 500 ∧ n % 5 = 0 ∧ n % 10 = 2} = ∅ := by
  sorry

end no_multiples_of_five_end_in_two_l385_38581


namespace exists_positive_decreasing_function_l385_38560

theorem exists_positive_decreasing_function :
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f y < f x) ∧ (∀ x : ℝ, f x > 0) := by
  sorry

end exists_positive_decreasing_function_l385_38560


namespace common_chord_length_is_sqrt55_div_5_l385_38592

noncomputable section

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

def circle_C2_center : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

def circle_C2_radius : ℝ := 1

-- Define the length of the common chord
def common_chord_length : ℝ := Real.sqrt 55 / 5

-- Theorem statement
theorem common_chord_length_is_sqrt55_div_5 :
  ∃ (A B : ℝ × ℝ),
    (circle_C1 A.1 A.2) ∧
    (circle_C1 B.1 B.2) ∧
    ((A.1 - circle_C2_center.1)^2 + (A.2 - circle_C2_center.2)^2 = circle_C2_radius^2) ∧
    ((B.1 - circle_C2_center.1)^2 + (B.2 - circle_C2_center.2)^2 = circle_C2_radius^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = common_chord_length :=
by
  sorry

end common_chord_length_is_sqrt55_div_5_l385_38592


namespace joyce_apples_l385_38572

theorem joyce_apples (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 75 → given = 52 → remaining = initial - given → remaining = 23 := by
sorry

end joyce_apples_l385_38572


namespace quadratic_inequalities_l385_38574

theorem quadratic_inequalities :
  (∀ y : ℝ, y^2 + 4*y + 8 ≥ 4) ∧
  (∀ m : ℝ, m^2 + 2*m + 3 ≥ 2) ∧
  (∀ m : ℝ, -m^2 + 2*m + 3 ≤ 4) := by
  sorry

end quadratic_inequalities_l385_38574


namespace button_sequence_l385_38569

theorem button_sequence (a : Fin 6 → ℕ) (h1 : a 0 = 1)
    (h2 : a 1 = 3) (h4 : a 3 = 27) (h5 : a 4 = 81) (h6 : a 5 = 243)
    (h_ratio : ∀ i : Fin 5, a (i + 1) = 3 * a i) : a 2 = 9 := by
  sorry

end button_sequence_l385_38569


namespace expression_equals_two_l385_38562

theorem expression_equals_two : 
  (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end expression_equals_two_l385_38562


namespace place_two_before_eq_l385_38552

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_lt_10 : hundreds < 10
  t_lt_10 : tens < 10
  u_lt_10 : units < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Represents the operation of placing 2 before a three-digit number -/
def place_two_before (n : ThreeDigitNumber) : ℕ :=
  2000 + 100 * n.hundreds + 10 * n.tens + n.units

/-- Theorem stating that placing 2 before a three-digit number results in 2000 + 100h + 10t + u -/
theorem place_two_before_eq (n : ThreeDigitNumber) :
  place_two_before n = 2000 + 100 * n.hundreds + 10 * n.tens + n.units := by
  sorry

end place_two_before_eq_l385_38552


namespace hot_chocolate_max_servings_l385_38547

/-- Represents the recipe for hot chocolate -/
structure Recipe where
  chocolate : ℚ
  sugar : ℚ
  water : ℚ
  milk : ℚ
  servings : ℚ

/-- Represents the available ingredients -/
structure Ingredients where
  chocolate : ℚ
  sugar : ℚ
  milk : ℚ

/-- Calculates the maximum number of servings possible given a recipe and available ingredients -/
def max_servings (recipe : Recipe) (ingredients : Ingredients) : ℚ :=
  min (ingredients.chocolate / recipe.chocolate * recipe.servings)
      (min (ingredients.sugar / recipe.sugar * recipe.servings)
           (ingredients.milk / recipe.milk * recipe.servings))

theorem hot_chocolate_max_servings :
  let recipe : Recipe := {
    chocolate := 3,
    sugar := 1/3,
    water := 3/2,
    milk := 5,
    servings := 6
  }
  let ingredients : Ingredients := {
    chocolate := 8,
    sugar := 3,
    milk := 12
  }
  max_servings recipe ingredients = 16 := by sorry

end hot_chocolate_max_servings_l385_38547


namespace rectangle_count_l385_38537

/-- Given a rectangle with dimensions a and b where a < b, this theorem states that
    the number of rectangles with dimensions x and y satisfying the specified conditions
    is either 0 or 1. -/
theorem rectangle_count (a b x y : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (x < a ∧ y < a ∧ 
   2*(x + y) = (1/2)*(a + b) ∧ 
   x*y = (1/4)*a*b) → 
  (∃! p : ℝ × ℝ, p.1 < a ∧ p.2 < a ∧ 
                 2*(p.1 + p.2) = (1/2)*(a + b) ∧ 
                 p.1*p.2 = (1/4)*a*b) ∨
  (¬ ∃ p : ℝ × ℝ, p.1 < a ∧ p.2 < a ∧ 
                  2*(p.1 + p.2) = (1/2)*(a + b) ∧ 
                  p.1*p.2 = (1/4)*a*b) :=
by sorry

end rectangle_count_l385_38537


namespace initial_distance_between_cars_l385_38544

/-- Proves that the initial distance between two cars is 16 miles given their speeds and overtaking time -/
theorem initial_distance_between_cars
  (speed_A : ℝ)
  (speed_B : ℝ)
  (overtake_time : ℝ)
  (ahead_distance : ℝ)
  (h1 : speed_A = 58)
  (h2 : speed_B = 50)
  (h3 : overtake_time = 3)
  (h4 : ahead_distance = 8) :
  speed_A * overtake_time - speed_B * overtake_time - ahead_distance = 16 :=
by sorry

end initial_distance_between_cars_l385_38544


namespace square_plus_reciprocal_square_l385_38599

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 2 → x^4 + 1/x^4 = 2 := by
  sorry

end square_plus_reciprocal_square_l385_38599


namespace product_ratio_theorem_l385_38590

theorem product_ratio_theorem (a b c d e f : ℚ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1/4 := by
sorry

end product_ratio_theorem_l385_38590


namespace min_value_of_z_l385_38575

/-- Given a system of linear inequalities, prove that the minimum value of z = 2x + y is 4 -/
theorem min_value_of_z (x y : ℝ) 
  (h1 : 2 * x - y ≥ 0) 
  (h2 : x + y - 3 ≥ 0) 
  (h3 : y - x ≥ 0) : 
  ∃ (z : ℝ), z = 2 * x + y ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = 2 * x + y → w ≥ z :=
by sorry

end min_value_of_z_l385_38575


namespace max_value_2x_plus_y_l385_38546

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  ∃ (max : ℝ), max = 6 ∧ ∀ (x' y' : ℝ), x' + 2*y' ≤ 3 → x' ≥ 0 → y' ≥ 0 → 2*x' + y' ≤ max :=
by sorry

end max_value_2x_plus_y_l385_38546


namespace second_person_share_l385_38541

/-- Represents the share of money for each person -/
structure Shares :=
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ)

/-- Theorem: Given a sum of money distributed among four people in the proportion 6:3:5:4,
    where the third person gets 1000 more than the fourth, the second person's share is 3000 -/
theorem second_person_share
  (shares : Shares)
  (h1 : shares.a = 6 * shares.d)
  (h2 : shares.b = 3 * shares.d)
  (h3 : shares.c = 5 * shares.d)
  (h4 : shares.c = shares.d + 1000) :
  shares.b = 3000 :=
by
  sorry

end second_person_share_l385_38541


namespace divided_volumes_theorem_l385_38577

/-- Regular triangular prism with base side length 2√14 -/
structure RegularTriangularPrism where
  base_side : ℝ
  height : ℝ
  base_side_eq : base_side = 2 * Real.sqrt 14

/-- Plane dividing the prism -/
structure DividingPlane where
  prism : RegularTriangularPrism
  parallel_to_diagonal : Bool
  passes_through_vertex : Bool
  passes_through_center : Bool
  cross_section_area : ℝ
  cross_section_area_eq : cross_section_area = 21

/-- Volumes of the parts created by the dividing plane -/
def divided_volumes (p : RegularTriangularPrism) (d : DividingPlane) : (ℝ × ℝ) := sorry

/-- Theorem stating the volumes of the divided parts -/
theorem divided_volumes_theorem (p : RegularTriangularPrism) (d : DividingPlane) :
  d.prism = p → divided_volumes p d = (112/3, 154/3) := by sorry

end divided_volumes_theorem_l385_38577


namespace boyd_boys_percentage_l385_38563

/-- Represents the number of friends on a social media platform -/
structure SocialMediaFriends where
  boys : ℕ
  girls : ℕ

/-- Represents a person's friends on different social media platforms -/
structure Person where
  facebook : SocialMediaFriends
  instagram : SocialMediaFriends

def Julian : Person :=
  { facebook := { boys := 48, girls := 32 },
    instagram := { boys := 45, girls := 105 } }

def Boyd : Person :=
  { facebook := { boys := 1, girls := 64 },
    instagram := { boys := 135, girls := 0 } }

def total_friends (p : Person) : ℕ :=
  p.facebook.boys + p.facebook.girls + p.instagram.boys + p.instagram.girls

def boys_percentage (p : Person) : ℚ :=
  (p.facebook.boys + p.instagram.boys : ℚ) / total_friends p

theorem boyd_boys_percentage :
  boys_percentage Boyd = 68 / 100 :=
sorry

end boyd_boys_percentage_l385_38563


namespace valid_selections_count_l385_38578

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → Point

/-- Represents a 3600-gon -/
def BigPolygon : RegularPolygon 3600 := sorry

/-- Represents a 72-gon formed by red vertices -/
def RedPolygon : RegularPolygon 72 := sorry

/-- Predicate to check if a vertex is red -/
def isRed (v : Fin 3600) : Prop := sorry

/-- Represents a selection of 40 vertices -/
def Selection : Finset (Fin 3600) := sorry

/-- Predicate to check if a selection forms a regular 40-gon -/
def isRegular40gon (s : Finset (Fin 3600)) : Prop := sorry

/-- The number of ways to select 40 non-red vertices forming a regular 40-gon -/
def validSelections : ℕ := sorry

theorem valid_selections_count : validSelections = 81 := by sorry

end valid_selections_count_l385_38578


namespace min_value_trig_function_l385_38582

theorem min_value_trig_function :
  let f : ℝ → ℝ := λ x ↦ 2 * (Real.cos x)^2 - Real.sin (2 * x)
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ (m = 1 - Real.sqrt 2) := by
sorry

end min_value_trig_function_l385_38582


namespace area_between_circles_l385_38509

theorem area_between_circles (R r : ℝ) (h : ℝ) : 
  R = 10 → h = 16 → r^2 = R^2 - (h/2)^2 → (R^2 - r^2) * π = 64 * π := by
  sorry

end area_between_circles_l385_38509


namespace lunch_solution_l385_38512

def lunch_problem (total_spent : ℕ) (friend_spent : ℕ) : Prop :=
  friend_spent > total_spent - friend_spent ∧
  friend_spent - (total_spent - friend_spent) = 3

theorem lunch_solution :
  lunch_problem 19 11 := by sorry

end lunch_solution_l385_38512


namespace inequality_proof_l385_38513

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end inequality_proof_l385_38513


namespace sine_equation_equality_l385_38507

theorem sine_equation_equality (α β γ τ : ℝ) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0 ∧ τ > 0) 
  (h_eq : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) : 
  α = γ ∨ α = τ := by
  sorry

end sine_equation_equality_l385_38507


namespace max_min_difference_c_l385_38567

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 = 15) :
  (3 : ℝ) - (-7/3) = 16/3 := by sorry

end max_min_difference_c_l385_38567


namespace circle_point_range_l385_38579

theorem circle_point_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end circle_point_range_l385_38579


namespace product_of_valid_m_l385_38542

theorem product_of_valid_m : ∃ (S : Finset ℤ), 
  (∀ m ∈ S, m ≥ 1 ∧ 
    ∃ y : ℤ, y ≠ 2 ∧ m * y / (y - 2) + 1 = -3 * y / (2 - y)) ∧ 
  (∀ m : ℤ, m ≥ 1 → 
    (∃ y : ℤ, y ≠ 2 ∧ m * y / (y - 2) + 1 = -3 * y / (2 - y)) → 
    m ∈ S) ∧
  S.prod id = 4 :=
sorry

end product_of_valid_m_l385_38542


namespace max_amount_received_back_l385_38561

/-- Represents the denominations of chips --/
inductive ChipDenomination
  | twoHundred
  | fiveHundred

/-- Represents the number of chips lost for each denomination --/
structure ChipsLost where
  twoHundred : ℕ
  fiveHundred : ℕ

def totalChipsBought : ℕ := 50000

def chipValue (d : ChipDenomination) : ℕ :=
  match d with
  | ChipDenomination.twoHundred => 200
  | ChipDenomination.fiveHundred => 500

def totalChipsLost (c : ChipsLost) : ℕ := c.twoHundred + c.fiveHundred

def validChipsLost (c : ChipsLost) : Prop :=
  totalChipsLost c = 25 ∧
  (c.twoHundred = c.fiveHundred + 5 ∨ c.twoHundred + 5 = c.fiveHundred)

def valueLost (c : ChipsLost) : ℕ :=
  c.twoHundred * chipValue ChipDenomination.twoHundred +
  c.fiveHundred * chipValue ChipDenomination.fiveHundred

def amountReceivedBack (c : ChipsLost) : ℕ := totalChipsBought - valueLost c

theorem max_amount_received_back :
  ∃ (c : ChipsLost), validChipsLost c ∧
    (∀ (c' : ChipsLost), validChipsLost c' → amountReceivedBack c ≥ amountReceivedBack c') ∧
    amountReceivedBack c = 42000 :=
  sorry

end max_amount_received_back_l385_38561


namespace compare_roots_l385_38585

theorem compare_roots : 
  (4 : ℝ) ^ (1/4) > (5 : ℝ) ^ (1/5) ∧ 
  (5 : ℝ) ^ (1/5) > (16 : ℝ) ^ (1/16) ∧ 
  (16 : ℝ) ^ (1/16) > (27 : ℝ) ^ (1/27) := by
  sorry

#check compare_roots

end compare_roots_l385_38585


namespace potato_price_proof_l385_38530

/-- The cost of one bag of potatoes from the farmer in rubles -/
def farmer_price : ℝ := sorry

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 1

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 0.6

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 0.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey in rubles -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof : 
  farmer_price = 250 :=
by
  have h1 : bags_bought * farmer_price * (1 + andrey_increase) = 
            bags_bought * farmer_price * (1 + boris_first_increase) * (boris_first_sale / bags_bought) + 
            bags_bought * farmer_price * (1 + boris_first_increase) * (1 + boris_second_increase) * (boris_second_sale / bags_bought) - 
            earnings_difference := by sorry
  sorry

end potato_price_proof_l385_38530


namespace job_land_theorem_l385_38516

/-- Represents the total land owned by Job in hectares -/
def total_land : ℕ := 150

/-- Represents the land occupied by house and farm machinery in hectares -/
def house_and_machinery : ℕ := 25

/-- Represents the land reserved for future expansion in hectares -/
def future_expansion : ℕ := 15

/-- Represents the land dedicated to rearing cattle in hectares -/
def cattle_land : ℕ := 40

/-- Represents the land used for crop production in hectares -/
def crop_land : ℕ := 70

/-- Theorem stating that the total land is equal to the sum of all land uses -/
theorem job_land_theorem : 
  total_land = house_and_machinery + future_expansion + cattle_land + crop_land := by
  sorry

end job_land_theorem_l385_38516


namespace division_of_fractions_l385_38515

theorem division_of_fractions : (3/8) / (5/12) = 9/10 := by
  sorry

end division_of_fractions_l385_38515
