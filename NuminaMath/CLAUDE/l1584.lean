import Mathlib

namespace real_part_of_i_squared_times_one_minus_two_i_l1584_158471

theorem real_part_of_i_squared_times_one_minus_two_i : 
  Complex.re (Complex.I^2 * (1 - 2*Complex.I)) = -1 := by
sorry

end real_part_of_i_squared_times_one_minus_two_i_l1584_158471


namespace smallest_number_of_cubes_for_given_box_l1584_158470

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of cubes of different sizes needed to fill a box -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  sorry

/-- The theorem stating the smallest number of cubes needed for the given box dimensions -/
theorem smallest_number_of_cubes_for_given_box :
  let box : BoxDimensions := { length := 98, width := 77, depth := 35 }
  smallestNumberOfCubes box = 770 := by
  sorry

end smallest_number_of_cubes_for_given_box_l1584_158470


namespace negation_of_exponential_proposition_l1584_158483

theorem negation_of_exponential_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.exp x ≥ 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.exp x < 1) := by
  sorry

end negation_of_exponential_proposition_l1584_158483


namespace acid_mixture_concentration_l1584_158420

/-- Calculates the final acid concentration when replacing part of a solution with another -/
def finalAcidConcentration (initialConcentration replacementConcentration : ℚ) 
  (replacementFraction : ℚ) : ℚ :=
  (1 - replacementFraction) * initialConcentration + 
  replacementFraction * replacementConcentration

/-- Proves that replacing half of a 50% acid solution with a 30% acid solution results in a 40% solution -/
theorem acid_mixture_concentration : 
  finalAcidConcentration (1/2) (3/10) (1/2) = 2/5 := by
  sorry

end acid_mixture_concentration_l1584_158420


namespace equation_solution_l1584_158443

theorem equation_solution (x : ℝ) : 
  (x + 1)^5 + (x + 1)^4 * (x - 1) + (x + 1)^3 * (x - 1)^2 + 
  (x + 1)^2 * (x - 1)^3 + (x + 1) * (x - 1)^4 + (x - 1)^5 = 0 ↔ x = 0 :=
by sorry

end equation_solution_l1584_158443


namespace lower_limit_proof_l1584_158406

theorem lower_limit_proof (x : ℤ) (y : ℝ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : y < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4)
  (h6 : x = 1) :
  y < 1 := by
sorry

end lower_limit_proof_l1584_158406


namespace charlie_feather_count_l1584_158454

/-- The number of feathers Charlie already has -/
def feathers_already_has : ℕ := 387

/-- The number of feathers Charlie needs to collect -/
def feathers_to_collect : ℕ := 513

/-- The total number of feathers Charlie needs for his wings -/
def total_feathers_needed : ℕ := feathers_already_has + feathers_to_collect

theorem charlie_feather_count : total_feathers_needed = 900 := by
  sorry

end charlie_feather_count_l1584_158454


namespace cube_split_with_31_l1584_158404

/-- For a natural number m > 1, if 31 is one of the odd numbers in the sum that equals m^3, then m = 6. -/
theorem cube_split_with_31 (m : ℕ) (h1 : m > 1) : 
  (∃ (k : ℕ) (l : List ℕ), 
    (∀ n ∈ l, Odd n) ∧ 
    (List.sum l = m^3) ∧
    (31 ∈ l) ∧
    (List.length l = m)) → 
  m = 6 := by
sorry

end cube_split_with_31_l1584_158404


namespace candy_distribution_l1584_158431

/-- Given 27.5 candy bars divided among 8.3 people, each person receives approximately 3.313 candy bars -/
theorem candy_distribution (total_candy : ℝ) (num_people : ℝ) (candy_per_person : ℝ) 
  (h1 : total_candy = 27.5)
  (h2 : num_people = 8.3)
  (h3 : candy_per_person = total_candy / num_people) :
  ∃ ε > 0, |candy_per_person - 3.313| < ε :=
by sorry

end candy_distribution_l1584_158431


namespace harmonic_sum_terms_added_l1584_158408

theorem harmonic_sum_terms_added (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end harmonic_sum_terms_added_l1584_158408


namespace x_intercept_of_line_l1584_158477

/-- The x-intercept of the line 4x + 7y = 28 is the point (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  (4 * x + 7 * y = 28) → (x = 7 ∧ y = 0 → 4 * x + 7 * y = 28) := by
  sorry

end x_intercept_of_line_l1584_158477


namespace correct_parking_methods_l1584_158458

/-- Represents the number of consecutive parking spaces -/
def total_spaces : ℕ := 7

/-- Represents the number of cars to be parked -/
def cars_to_park : ℕ := 3

/-- Represents the number of consecutive empty spaces required -/
def required_empty_spaces : ℕ := 4

/-- Calculates the number of different parking methods -/
def parking_methods : ℕ := 24

/-- Theorem stating that the number of parking methods is correct -/
theorem correct_parking_methods :
  ∀ (total : ℕ) (cars : ℕ) (empty : ℕ),
    total = total_spaces →
    cars = cars_to_park →
    empty = required_empty_spaces →
    total - cars = empty →
    parking_methods = 24 :=
by
  sorry

end correct_parking_methods_l1584_158458


namespace family_trip_arrangements_l1584_158491

theorem family_trip_arrangements (n : Nat) (k : Nat) : 
  n = 4 ∧ k = 3 → k^n = 81 := by
  sorry

end family_trip_arrangements_l1584_158491


namespace painting_time_equation_l1584_158479

theorem painting_time_equation (doug_time dave_time t : ℝ) :
  doug_time = 6 →
  dave_time = 8 →
  (1 / doug_time + 1 / dave_time) * (t - 1) = 1 :=
by sorry

end painting_time_equation_l1584_158479


namespace popsicle_sticks_theorem_l1584_158499

def steve_sticks : ℕ := 12

def sid_sticks : ℕ := 2 * steve_sticks

def sam_sticks : ℕ := 3 * sid_sticks

def total_sticks : ℕ := steve_sticks + sid_sticks + sam_sticks

theorem popsicle_sticks_theorem : total_sticks = 108 := by
  sorry

end popsicle_sticks_theorem_l1584_158499


namespace geometric_sequence_bounded_l1584_158418

theorem geometric_sequence_bounded (n k : ℕ) (a : ℕ → ℝ) : 
  n > 0 → k > 0 → 
  (∀ i ∈ Finset.range (k+1), n^k ≤ a i ∧ a i ≤ (n+1)^k) →
  (∀ i ∈ Finset.range k, ∃ q : ℝ, a (i+1) = a i * q) →
  (∀ i ∈ Finset.range (k+1), a i = n^k * ((n+1)/n)^i ∨ a i = (n+1)^k * (n/(n+1))^i) :=
by sorry

end geometric_sequence_bounded_l1584_158418


namespace consecutive_numbers_theorem_l1584_158476

theorem consecutive_numbers_theorem 
  (a b c d e f g : ℤ) 
  (consecutive : b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 ∧ g = a + 6)
  (average_9 : (a + b + c + d + e + f + g) / 7 = 9)
  (a_half_of_g : 2 * a = g) : 
  ∃ (n : ℕ), n = 7 ∧ g - a + 1 = n :=
by sorry

end consecutive_numbers_theorem_l1584_158476


namespace unique_solution_condition_l1584_158496

/-- The equation has exactly one real solution if and only if b < -4 -/
theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0) ↔ b < -4 :=
sorry

end unique_solution_condition_l1584_158496


namespace square_traffic_sign_perimeter_l1584_158482

/-- A square traffic sign with sides of 4 feet has a perimeter of 16 feet. -/
theorem square_traffic_sign_perimeter : 
  ∀ (side_length : ℝ), side_length = 4 → 4 * side_length = 16 := by
  sorry

end square_traffic_sign_perimeter_l1584_158482


namespace original_number_proof_l1584_158409

theorem original_number_proof (x : ℕ) : (10 * x + 9) + 2 * x = 633 → x = 52 := by
  sorry

end original_number_proof_l1584_158409


namespace f_is_odd_iff_a_eq_one_l1584_158486

/-- A function f is odd if f(-x) = -f(x) for all x in its domain. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x(x-1)(x+a) -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - 1) * (x + a)

/-- Theorem: f(x) = x(x-1)(x+a) is an odd function if and only if a = 1 -/
theorem f_is_odd_iff_a_eq_one (a : ℝ) : IsOdd (f a) ↔ a = 1 := by
  sorry


end f_is_odd_iff_a_eq_one_l1584_158486


namespace expedition_duration_proof_l1584_158456

theorem expedition_duration_proof (first_expedition : ℕ) 
  (h1 : first_expedition = 3)
  (second_expedition : ℕ) 
  (h2 : second_expedition = first_expedition + 2)
  (third_expedition : ℕ) 
  (h3 : third_expedition = 2 * second_expedition) : 
  (first_expedition + second_expedition + third_expedition) * 7 = 126 := by
  sorry

end expedition_duration_proof_l1584_158456


namespace extremum_of_f_on_M_l1584_158490

def M : Set ℝ := {x | x^2 + 4*x ≤ 0}

def f (x : ℝ) : ℝ := -x^2 - 6*x + 1

theorem extremum_of_f_on_M :
  ∃ (min max : ℝ), 
    (∀ x ∈ M, f x ≥ min) ∧ 
    (∃ x ∈ M, f x = min) ∧
    (∀ x ∈ M, f x ≤ max) ∧ 
    (∃ x ∈ M, f x = max) ∧
    min = 1 ∧ max = 10 :=
sorry

end extremum_of_f_on_M_l1584_158490


namespace book_purchase_problem_l1584_158457

/-- Represents the number of books purchased -/
def num_books : ℕ := 8

/-- Represents the number of albums purchased -/
def num_albums : ℕ := num_books - 6

/-- Represents the price of a book in kopecks -/
def price_book : ℕ := 1056 / num_books

/-- Represents the price of an album in kopecks -/
def price_album : ℕ := 56 / num_albums

/-- Theorem stating that the given conditions are satisfied by the defined values -/
theorem book_purchase_problem :
  (num_books : ℤ) = (num_albums : ℤ) + 6 ∧
  num_books * price_book = 1056 ∧
  num_albums * price_album = 56 ∧
  price_book > price_album + 100 :=
by sorry

end book_purchase_problem_l1584_158457


namespace equation_solution_l1584_158427

theorem equation_solution : ∃ x : ℝ, 61 + x * 12 / (180 / 3) = 62 ∧ x = 5 := by
  sorry

end equation_solution_l1584_158427


namespace unique_single_digit_square_l1584_158415

theorem unique_single_digit_square (A : ℕ) : A < 10 ∧ (10 * A + A) * (10 * A + A) = 5929 ↔ A = 7 := by sorry

end unique_single_digit_square_l1584_158415


namespace point_on_line_point_twelve_seven_on_line_l1584_158488

/-- Given three points in the plane, this theorem states that if the first two points
    determine a line, then the third point lies on that line. -/
theorem point_on_line (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁) →
  ∃ (m b : ℝ), y₁ = m * x₁ + b ∧ y₂ = m * x₂ + b ∧ y₃ = m * x₃ + b :=
by sorry

/-- The point (12,7) lies on the line passing through (0,1) and (-6,-2). -/
theorem point_twelve_seven_on_line : 
  ∃ (m b : ℝ), 1 = m * 0 + b ∧ -2 = m * (-6) + b ∧ 7 = m * 12 + b :=
by
  apply point_on_line 0 1 (-6) (-2) 12 7
  -- Proof that the points are collinear
  sorry

end point_on_line_point_twelve_seven_on_line_l1584_158488


namespace factorization_problems_l1584_158473

variable (x y : ℝ)

theorem factorization_problems :
  (x^2 + 3*x = x*(x + 3)) ∧ (x^2 - 2*x*y + y^2 = (x - y)^2) := by
  sorry

end factorization_problems_l1584_158473


namespace simple_interest_rate_problem_l1584_158412

theorem simple_interest_rate_problem (P A T : ℕ) (h1 : P = 25000) (h2 : A = 35500) (h3 : T = 12) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 35 / 10 := by sorry

end simple_interest_rate_problem_l1584_158412


namespace pencils_per_row_l1584_158400

theorem pencils_per_row (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) 
  (h1 : packs = 28) 
  (h2 : pencils_per_pack = 24) 
  (h3 : rows = 42) :
  (packs * pencils_per_pack) / rows = 16 := by
  sorry

#check pencils_per_row

end pencils_per_row_l1584_158400


namespace ten_digit_number_divisibility_l1584_158428

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem ten_digit_number_divisibility (a b : ℕ) :
  a < 10 → b < 10 →
  is_divisible_by_99 (2016 * 10000 + a * 1000 + b * 100 + 2017) →
  a + b = 8 := by sorry

end ten_digit_number_divisibility_l1584_158428


namespace monotonic_decreasing_interval_l1584_158413

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 2) ↔ (f' x < 0) :=
sorry

end monotonic_decreasing_interval_l1584_158413


namespace xy_range_l1584_158419

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.exp x = x * y * (2 * Real.log x + Real.log y)) : 
  x * y ≥ Real.exp 1 := by
sorry

end xy_range_l1584_158419


namespace sine_cosine_sum_l1584_158451

theorem sine_cosine_sum (α : Real) : 
  (∃ (x y : Real), x = 3/5 ∧ y = 4/5 ∧ x^2 + y^2 = 1 ∧ 
    Real.cos α = x ∧ Real.sin α = y) → 
  Real.sin α + 2 * Real.cos α = 2 := by
sorry

end sine_cosine_sum_l1584_158451


namespace seven_rows_five_seats_l1584_158417

-- Define a movie ticket as a pair of natural numbers
def MovieTicket : Type := ℕ × ℕ

-- Define a function to create a movie ticket representation
def createTicket (rows : ℕ) (seats : ℕ) : MovieTicket := (rows, seats)

-- Theorem statement
theorem seven_rows_five_seats :
  createTicket 7 5 = (7, 5) := by sorry

end seven_rows_five_seats_l1584_158417


namespace volume_removed_percentage_l1584_158424

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ :=
  side ^ 3

/-- Theorem: The percentage of volume removed from a box with dimensions 20x15x10,
    by removing a 4cm cube from each of its 8 corners, is equal to (512/3000) * 100% -/
theorem volume_removed_percentage :
  let originalBox : BoxDimensions := ⟨20, 15, 10⟩
  let removedCubeSide : ℝ := 4
  let numCorners : ℕ := 8
  let originalVolume := boxVolume originalBox
  let removedVolume := numCorners * (cubeVolume removedCubeSide)
  (removedVolume / originalVolume) * 100 = (512 / 3000) * 100 := by
  sorry

end volume_removed_percentage_l1584_158424


namespace independent_events_probability_l1584_158464

theorem independent_events_probability (a b : Set ℝ) (p : Set ℝ → ℝ) 
  (h1 : p a = 4/5)
  (h2 : p b = 2/5)
  (h3 : p (a ∩ b) = 0.32)
  (h4 : p (a ∩ b) = p a * p b) : 
  p b = 2/5 := by
sorry

end independent_events_probability_l1584_158464


namespace unique_prime_triple_l1584_158468

theorem unique_prime_triple : 
  ∀ p q r : ℕ+, 
    Prime p.val → Prime q.val → 
    (r.val^2 - 5*q.val^2) / (p.val^2 - 1) = 2 → 
    (p, q, r) = (3, 2, 6) := by
  sorry

end unique_prime_triple_l1584_158468


namespace undetermined_disjunction_l1584_158450

theorem undetermined_disjunction (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), (¬p ∧ ¬(p ∧ q)) → (p ∨ q) := by
sorry

end undetermined_disjunction_l1584_158450


namespace zoo_meat_amount_l1584_158448

/-- The amount of meat (in kg) that lasts for a given number of days for a lion and a tiger -/
def meatAmount (lionConsumption tigerConsumption daysLasting : ℕ) : ℕ :=
  (lionConsumption + tigerConsumption) * daysLasting

theorem zoo_meat_amount :
  meatAmount 25 20 2 = 90 := by
  sorry

end zoo_meat_amount_l1584_158448


namespace family_composition_l1584_158495

/-- A family where one member has an equal number of brothers and sisters,
    and another member has twice as many brothers as sisters. -/
structure Family where
  boys : ℕ
  girls : ℕ
  tony_equal_siblings : boys - 1 = girls
  alice_double_brothers : boys = 2 * (girls - 1)

/-- The family has 4 boys and 3 girls. -/
theorem family_composition (f : Family) : f.boys = 4 ∧ f.girls = 3 := by
  sorry

end family_composition_l1584_158495


namespace triangle_problem_l1584_158422

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (2 * cos C * (a * cos B + b * cos A) = c) →
  (c = Real.sqrt 7) →
  (1/2 * a * b * sin C = 3 * Real.sqrt 3 / 2) →
  (C = π/3 ∧ a + b + c = 5 + Real.sqrt 7) :=
by sorry

end triangle_problem_l1584_158422


namespace is_arithmetic_sequence_pn_plus_q_l1584_158407

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) (p q : ℝ) : ℝ := p * n + q

/-- Theorem: A sequence with general term a_n = pn + q is an arithmetic sequence. -/
theorem is_arithmetic_sequence_pn_plus_q (p q : ℝ) :
  IsArithmeticSequence (a · p q) := by
  sorry

end is_arithmetic_sequence_pn_plus_q_l1584_158407


namespace max_area_parallelogram_in_circle_l1584_158494

/-- A right-angled parallelogram inscribed in a circle of radius r has maximum area when its sides are r√2 -/
theorem max_area_parallelogram_in_circle (r : ℝ) (h : r > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a * b ≤ x * y) ∧
  (x^2 + y^2 = (2*r)^2) ∧
  x = r * Real.sqrt 2 ∧ y = r * Real.sqrt 2 :=
sorry

end max_area_parallelogram_in_circle_l1584_158494


namespace madan_age_is_five_l1584_158485

-- Define the ages as natural numbers
def arun_age : ℕ := 60

-- Define Gokul's age as a function of Arun's age
def gokul_age (a : ℕ) : ℕ := (a - 6) / 18

-- Define Madan's age as a function of Gokul's age
def madan_age (g : ℕ) : ℕ := g + 2

-- Theorem to prove
theorem madan_age_is_five :
  madan_age (gokul_age arun_age) = 5 := by
  sorry

end madan_age_is_five_l1584_158485


namespace probability_green_or_white_specific_l1584_158447

/-- The probability of drawing either a green or white marble from a bag -/
def probability_green_or_white (green white black : ℕ) : ℚ :=
  (green + white) / (green + white + black)

/-- Theorem stating the probability of drawing a green or white marble -/
theorem probability_green_or_white_specific :
  probability_green_or_white 4 3 8 = 7 / 15 := by
  sorry

end probability_green_or_white_specific_l1584_158447


namespace ball_hitting_ground_time_l1584_158410

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hitting_ground_time : 
  ∃ t : ℝ, t = 1 + (Real.sqrt 19) / 2 ∧ 
  (∀ y : ℝ, y = -16 * t^2 + 32 * t + 60 → y = 0) := by
  sorry

end ball_hitting_ground_time_l1584_158410


namespace remainder_790123_div_15_l1584_158461

theorem remainder_790123_div_15 : 790123 % 15 = 13 := by
  sorry

end remainder_790123_div_15_l1584_158461


namespace sum_of_coefficients_l1584_158497

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by
  sorry

end sum_of_coefficients_l1584_158497


namespace magic_square_x_value_l1584_158426

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℤ)
  (row_sum : a + b + c = d + e + f ∧ d + e + f = g + h + i)
  (col_sum : a + d + g = b + e + h ∧ b + e + h = c + f + i)
  (diag_sum : a + e + i = c + e + g)

/-- The theorem stating the value of x in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a = x)
  (h2 : ms.b = 19)
  (h3 : ms.c = 96)
  (h4 : ms.d = 1) :
  x = 200 := by
  sorry

end magic_square_x_value_l1584_158426


namespace simplify_trig_expression_l1584_158465

theorem simplify_trig_expression :
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4 := by sorry

end simplify_trig_expression_l1584_158465


namespace tangent_line_parallelism_l1584_158472

theorem tangent_line_parallelism (a : ℝ) :
  a > -2 * Real.sqrt 2 →
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ 2 * x₁ + 1 / x₁ = Real.exp x₂ - a :=
sorry

end tangent_line_parallelism_l1584_158472


namespace sample_is_weights_l1584_158441

/-- Represents a student in the survey -/
structure Student where
  weight : ℝ

/-- Represents the survey conducted by the city -/
structure Survey where
  students : Finset Student
  grade : Nat

/-- Definition of a sample in this context -/
def Sample (survey : Survey) : Set ℝ :=
  {w | ∃ s ∈ survey.students, w = s.weight}

/-- The theorem stating that the sample is the weight of 100 students -/
theorem sample_is_weights (survey : Survey) 
    (h1 : survey.grade = 9) 
    (h2 : survey.students.card = 100) : 
  Sample survey = {w | ∃ s ∈ survey.students, w = s.weight} := by
  sorry

end sample_is_weights_l1584_158441


namespace no_solution_floor_equation_l1584_158493

theorem no_solution_floor_equation :
  ¬ ∃ (x : ℤ), (⌊x⌋ : ℤ) + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ = 12345 := by
  sorry

end no_solution_floor_equation_l1584_158493


namespace quadratic_not_equal_linear_l1584_158480

theorem quadratic_not_equal_linear : ¬∃ (a b c A B : ℝ), a ≠ 0 ∧ ∀ x, a * x^2 + b * x + c = A * x + B := by
  sorry

end quadratic_not_equal_linear_l1584_158480


namespace greatest_whole_number_satisfying_inequality_l1584_158436

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5 * x - 4 < 3 - 2 * x := by sorry

end greatest_whole_number_satisfying_inequality_l1584_158436


namespace max_hawthorns_l1584_158440

theorem max_hawthorns (x : ℕ) : 
  x > 100 ∧
  x % 3 = 1 ∧
  x % 4 = 2 ∧
  x % 5 = 3 ∧
  x % 6 = 4 →
  x ≤ 178 ∧ 
  ∃ y : ℕ, y > 100 ∧ 
    y % 3 = 1 ∧ 
    y % 4 = 2 ∧ 
    y % 5 = 3 ∧ 
    y % 6 = 4 ∧ 
    y = 178 :=
by sorry

end max_hawthorns_l1584_158440


namespace prime_square_sum_l1584_158416

theorem prime_square_sum (p q r : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r →
  (∃ (n : ℕ), p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3)) :=
by sorry

end prime_square_sum_l1584_158416


namespace product_of_three_numbers_l1584_158462

theorem product_of_three_numbers 
  (x y z : ℝ) 
  (sum_eq : x + y = 18) 
  (sum_squares_eq : x^2 + y^2 = 220) 
  (diff_eq : z = x - y) : 
  x * y * z = 104 * Real.sqrt 29 := by
  sorry

end product_of_three_numbers_l1584_158462


namespace subset_sum_partition_l1584_158446

theorem subset_sum_partition (n : ℕ) (S : Finset ℝ) (h_pos : ∀ x ∈ S, 0 < x) (h_card : S.card = n) :
  ∃ (P : Finset (Finset ℝ)), 
    P.card = n ∧ 
    (∀ X ∈ P, ∃ (min max : ℝ), 
      (∀ y ∈ X, min ≤ y ∧ y ≤ max) ∧ 
      max < 2 * min) ∧
    (∀ A : Finset ℝ, A.Nonempty → A ⊆ S → ∃ X ∈ P, (A.sum id) ∈ X) :=
sorry

end subset_sum_partition_l1584_158446


namespace max_sum_of_factors_l1584_158475

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 42 → (∀ x y : ℕ, x * y = 42 → x + y ≤ heart + club) → heart + club = 43 :=
sorry

end max_sum_of_factors_l1584_158475


namespace line_parallel_perpendicular_implies_planes_perpendicular_l1584_158481

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l1584_158481


namespace fraction_simplification_l1584_158489

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
  (x - 3) / (2 * x * (x - 3)) = 1 / (2 * x) := by
  sorry

end fraction_simplification_l1584_158489


namespace adjacent_sum_divisible_by_four_l1584_158467

/-- A type representing a cell in the grid -/
structure Cell where
  row : Fin 22
  col : Fin 22

/-- A function representing the number in each cell -/
def gridValue : Cell → Fin (22^2) := sorry

/-- Two cells are adjacent if they share an edge or vertex -/
def adjacent (c1 c2 : Cell) : Prop := sorry

theorem adjacent_sum_divisible_by_four :
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (gridValue c1 + gridValue c2) % 4 = 0 := by sorry

end adjacent_sum_divisible_by_four_l1584_158467


namespace complex_equation_solution_l1584_158498

theorem complex_equation_solution (z : ℂ) : 
  z * (1 - 2 * Complex.I) = 2 + 4 * Complex.I → 
  z = -2/5 + 8/5 * Complex.I :=
by
  sorry

end complex_equation_solution_l1584_158498


namespace sequence_sum_l1584_158402

theorem sequence_sum (seq : Fin 10 → ℝ) 
  (h1 : seq 2 = 5)
  (h2 : ∀ i : Fin 8, seq i + seq (i + 1) + seq (i + 2) = 25) :
  seq 0 + seq 9 = 30 := by
  sorry

end sequence_sum_l1584_158402


namespace functions_equality_l1584_158405

theorem functions_equality (x : ℝ) : 2 * |x| = Real.sqrt (4 * x^2) := by
  sorry

end functions_equality_l1584_158405


namespace min_value_ab_l1584_158453

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3/a + 2/b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3/x + 2/y = 2 → a * b ≤ x * y :=
by sorry

end min_value_ab_l1584_158453


namespace junior_score_l1584_158429

theorem junior_score (n : ℕ) (h : n > 0) : 
  let junior_count : ℝ := 0.2 * n
  let senior_count : ℝ := 0.8 * n
  let total_score : ℝ := 85 * n
  let senior_score : ℝ := 82 * senior_count
  let junior_total_score : ℝ := total_score - senior_score
  junior_total_score / junior_count = 97 := by sorry

end junior_score_l1584_158429


namespace problem_solution_l1584_158442

theorem problem_solution (y : ℝ) (h1 : y > 0) (h2 : y / 100 * y + 6 = 10) : y = 20 := by
  sorry

end problem_solution_l1584_158442


namespace extremum_point_implies_a_and_minimum_value_l1584_158459

/-- The function f(x) = (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  ((2*x + a) + (x^2 + a*x - 1)) * Real.exp (x - 1)

theorem extremum_point_implies_a_and_minimum_value 
  (a : ℝ) 
  (h1 : f_derivative a (-2) = 0) :
  a = -1 ∧ ∀ x, f (-1) x ≥ -1 := by
  sorry

end extremum_point_implies_a_and_minimum_value_l1584_158459


namespace unique_number_with_three_prime_factors_l1584_158401

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 := by
  sorry

end unique_number_with_three_prime_factors_l1584_158401


namespace bottle_cap_count_l1584_158433

theorem bottle_cap_count : 
  ∀ (cost_per_cap total_cost num_caps : ℕ),
  cost_per_cap = 2 →
  total_cost = 12 →
  total_cost = cost_per_cap * num_caps →
  num_caps = 6 :=
by
  sorry

end bottle_cap_count_l1584_158433


namespace quadratic_real_roots_when_ac_negative_l1584_158455

theorem quadratic_real_roots_when_ac_negative 
  (a b c : ℝ) (h : a * c < 0) : 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end quadratic_real_roots_when_ac_negative_l1584_158455


namespace walking_speed_equation_l1584_158478

theorem walking_speed_equation (x : ℝ) 
  (h1 : x > 0) -- Xiao Wang's speed is positive
  (h2 : x + 1 > 0) -- Xiao Zhang's speed is positive
  : 15 / x - 15 / (x + 1) = 1 / 2 ↔ 
    (15 / x = 15 / (x + 1) + 1 / 2 ∧ 
     15 / (x + 1) < 15 / x) :=
by sorry

end walking_speed_equation_l1584_158478


namespace flash_overtakes_ace_l1584_158403

/-- The distance Flash needs to jog to overtake Ace -/
def overtake_distance (v y t : ℝ) : ℝ :=
  2 * (y + 60 * v * t)

/-- Theorem stating the distance Flash needs to jog to overtake Ace -/
theorem flash_overtakes_ace (v y t : ℝ) (hv : v > 0) (hy : y ≥ 0) (ht : t ≥ 0) :
  ∃ d : ℝ, d = overtake_distance v y t ∧ d > 0 :=
by
  sorry


end flash_overtakes_ace_l1584_158403


namespace no_three_distinct_rational_roots_l1584_158492

theorem no_three_distinct_rational_roots (a b : ℝ) : 
  ¬ (∃ (u v w : ℚ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
    (u^3 + (2*a+1)*u^2 + (2*a^2+2*a-3)*u + b = 0) ∧
    (v^3 + (2*a+1)*v^2 + (2*a^2+2*a-3)*v + b = 0) ∧
    (w^3 + (2*a+1)*w^2 + (2*a^2+2*a-3)*w + b = 0)) :=
by
  sorry

end no_three_distinct_rational_roots_l1584_158492


namespace floor_of_4_7_l1584_158445

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end floor_of_4_7_l1584_158445


namespace special_cone_vertex_angle_l1584_158449

/-- A right circular cone with three pairwise perpendicular generatrices -/
structure SpecialCone where
  /-- The angle at the vertex of the axial section -/
  vertex_angle : ℝ
  /-- The condition that three generatrices are pairwise perpendicular -/
  perpendicular_generatrices : Prop

/-- Theorem: The angle at the vertex of the axial section of a special cone is 2 * arcsin(√6 / 3) -/
theorem special_cone_vertex_angle (cone : SpecialCone) :
  cone.perpendicular_generatrices →
  cone.vertex_angle = 2 * Real.arcsin (Real.sqrt 6 / 3) := by
  sorry

end special_cone_vertex_angle_l1584_158449


namespace collinear_points_k_value_l1584_158432

/-- Given three points on a line, prove the value of k --/
theorem collinear_points_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 8 = m * 2 + b ∧ k = m * 10 + b ∧ 2 = m * 16 + b) → k = 32/7 := by
  sorry

end collinear_points_k_value_l1584_158432


namespace range_of_f_l1584_158411

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 2 ≤ y ∧ y ≤ 6} := by sorry

end range_of_f_l1584_158411


namespace arithmetic_sequence_sum_l1584_158452

/-- An arithmetic sequence with a_6 = 1 -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ a1 d : ℚ, ∀ n : ℕ, a n = a1 + (n - 1) * d) ∧ a 6 = 1

/-- For any arithmetic sequence with a_6 = 1, a_2 + a_10 = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  a 2 + a 10 = 2 := by
  sorry

end arithmetic_sequence_sum_l1584_158452


namespace circular_track_length_l1584_158414

-- Define the track length
def track_length : ℝ := 350

-- Define the constants given in the problem
def first_meeting_distance : ℝ := 80
def second_meeting_distance : ℝ := 140

-- Theorem statement
theorem circular_track_length :
  ∀ (brenda_speed sally_speed : ℝ),
  brenda_speed > 0 ∧ sally_speed > 0 →
  ∃ (t₁ t₂ : ℝ),
  t₁ > 0 ∧ t₂ > 0 ∧
  brenda_speed * t₁ = first_meeting_distance ∧
  sally_speed * t₁ = track_length / 2 - first_meeting_distance ∧
  brenda_speed * (t₁ + t₂) = track_length / 2 + first_meeting_distance ∧
  sally_speed * (t₁ + t₂) = track_length / 2 + second_meeting_distance →
  track_length = 350 :=
by
  sorry -- Proof omitted

end circular_track_length_l1584_158414


namespace largest_p_value_l1584_158437

theorem largest_p_value (m n p : ℕ) : 
  m ≥ 3 → n ≥ 3 → p ≥ 3 →
  (1 : ℚ) / m + (1 : ℚ) / n + (1 : ℚ) / p = (1 : ℚ) / 2 →
  p ≤ 42 :=
sorry

end largest_p_value_l1584_158437


namespace solve_linear_equation_l1584_158430

theorem solve_linear_equation :
  ∀ x : ℚ, -3 * x - 8 = 4 * x + 3 → x = -11/7 := by
  sorry

end solve_linear_equation_l1584_158430


namespace factor_x8_minus_81_l1584_158444

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x^2 - 3) := by
  sorry

end factor_x8_minus_81_l1584_158444


namespace convex_figure_integer_points_l1584_158460

/-- A convex figure in the plane -/
structure ConvexFigure where
  -- We don't need to define the structure fully, just declare it
  dummy : Unit

/-- The area of a convex figure -/
noncomputable def area (φ : ConvexFigure) : ℝ := sorry

/-- The semiperimeter of a convex figure -/
noncomputable def semiperimeter (φ : ConvexFigure) : ℝ := sorry

/-- The number of integer points contained in a convex figure -/
noncomputable def integerPoints (φ : ConvexFigure) : ℕ := sorry

/-- 
If the area of a convex figure is greater than n times its semiperimeter,
then it contains at least n integer points.
-/
theorem convex_figure_integer_points (φ : ConvexFigure) (n : ℕ) :
  area φ > n • (semiperimeter φ) → integerPoints φ ≥ n := by sorry

end convex_figure_integer_points_l1584_158460


namespace value_of_a_minus_b_l1584_158421

theorem value_of_a_minus_b (a b c : ℚ) 
  (eq1 : 2011 * a + 2015 * b + c = 2021)
  (eq2 : 2013 * a + 2017 * b + c = 2023)
  (eq3 : 2012 * a + 2016 * b + 2 * c = 2026) :
  a - b = -2 := by
  sorry

end value_of_a_minus_b_l1584_158421


namespace ratio_chain_l1584_158474

theorem ratio_chain (a b c d e f g h : ℝ) 
  (hab : a / b = 7 / 3)
  (hbc : b / c = 5 / 2)
  (hcd : c / d = 2)
  (hde : d / e = 3 / 2)
  (hef : e / f = 4 / 3)
  (hfg : f / g = 1 / 4)
  (hgh : g / h = 3 / 5) :
  a * b * c * d * e * f * g / (d * e * f * g * h * i * j) = 15.75 :=
by sorry

end ratio_chain_l1584_158474


namespace rectangle_area_l1584_158463

/-- Given a rectangle ABCD divided into four identical squares with side length s,
    prove that its area is 2500 square centimeters when three of its sides total 100 cm. -/
theorem rectangle_area (s : ℝ) : 
  s > 0 →                            -- s is positive (implied by the context)
  4 * s = 100 →                      -- three sides total 100 cm
  (2 * s) * (2 * s) = 2500 :=        -- area of ABCD is 2500 sq cm
by
  sorry

#check rectangle_area

end rectangle_area_l1584_158463


namespace total_spent_is_83_50_l1584_158434

-- Define the ticket prices
def adult_ticket_price : ℚ := 5.5
def child_ticket_price : ℚ := 3.5

-- Define the total number of tickets and number of adult tickets
def total_tickets : ℕ := 21
def adult_tickets : ℕ := 5

-- Define the function to calculate total spent
def total_spent : ℚ :=
  (adult_tickets : ℚ) * adult_ticket_price + 
  ((total_tickets - adult_tickets) : ℚ) * child_ticket_price

-- Theorem statement
theorem total_spent_is_83_50 : total_spent = 83.5 := by
  sorry

end total_spent_is_83_50_l1584_158434


namespace quadratic_ratio_l1584_158425

/-- The quadratic function f(x) = x^2 + 1600x + 1607 -/
def f (x : ℝ) : ℝ := x^2 + 1600*x + 1607

/-- The constant b in the completed square form (x+b)^2 + c -/
def b : ℝ := 800

/-- The constant c in the completed square form (x+b)^2 + c -/
def c : ℝ := -638393

/-- Theorem stating that c/b equals -797.99125 for the given quadratic -/
theorem quadratic_ratio : c / b = -797.99125 := by sorry

end quadratic_ratio_l1584_158425


namespace evaluate_expression_l1584_158469

theorem evaluate_expression (x : ℝ) (h : x = 2) : (3 * x^2 - 8 * x + 5) * (4 * x - 7) = 1 := by
  sorry

end evaluate_expression_l1584_158469


namespace max_sum_product_sqrt_l1584_158438

theorem max_sum_product_sqrt (x₁ x₂ x₃ x₄ : ℝ) 
  (non_neg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0) 
  (sum_one : x₁ + x₂ + x₃ + x₄ = 1) :
  (x₁ + x₂) * Real.sqrt (x₁ * x₂) +
  (x₁ + x₃) * Real.sqrt (x₁ * x₃) +
  (x₁ + x₄) * Real.sqrt (x₁ * x₄) +
  (x₂ + x₃) * Real.sqrt (x₂ * x₃) +
  (x₂ + x₄) * Real.sqrt (x₂ * x₄) +
  (x₃ + x₄) * Real.sqrt (x₃ * x₄) ≤ 3/4 ∧
  (x₁ = 1/4 ∧ x₂ = 1/4 ∧ x₃ = 1/4 ∧ x₄ = 1/4 →
    (x₁ + x₂) * Real.sqrt (x₁ * x₂) +
    (x₁ + x₃) * Real.sqrt (x₁ * x₃) +
    (x₁ + x₄) * Real.sqrt (x₁ * x₄) +
    (x₂ + x₃) * Real.sqrt (x₂ * x₃) +
    (x₂ + x₄) * Real.sqrt (x₂ * x₄) +
    (x₃ + x₄) * Real.sqrt (x₃ * x₄) = 3/4) :=
by sorry

end max_sum_product_sqrt_l1584_158438


namespace sports_competition_team_sizes_l1584_158487

theorem sports_competition_team_sizes :
  ∀ (boys girls : ℕ),
  (boys + 48 : ℚ) / 6 + (girls + 50 : ℚ) / 7 = 48 - (boys : ℚ) / 6 + 50 - (girls : ℚ) / 7 →
  boys - 48 = (girls - 50) / 2 →
  boys = 72 ∧ girls = 98 :=
by
  sorry

end sports_competition_team_sizes_l1584_158487


namespace ice_cream_theorem_l1584_158466

/-- The number of ways to distribute n indistinguishable objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ice cream flavor combinations -/
def ice_cream_combinations : ℕ := distribute 4 4

theorem ice_cream_theorem : ice_cream_combinations = 35 := by sorry

end ice_cream_theorem_l1584_158466


namespace expand_expression_l1584_158435

theorem expand_expression (x y : ℝ) : (x + 12) * (3 * y + 8) = 3 * x * y + 8 * x + 36 * y + 96 := by
  sorry

end expand_expression_l1584_158435


namespace people_per_bus_l1584_158423

/-- Given a field trip with vans and buses, calculate the number of people per bus -/
theorem people_per_bus 
  (total_people : ℕ) 
  (num_vans : ℕ) 
  (people_per_van : ℕ) 
  (num_buses : ℕ) 
  (h1 : total_people = 342)
  (h2 : num_vans = 9)
  (h3 : people_per_van = 8)
  (h4 : num_buses = 10)
  : (total_people - num_vans * people_per_van) / num_buses = 27 := by
  sorry

end people_per_bus_l1584_158423


namespace square_sum_inequality_l1584_158484

theorem square_sum_inequality (a b : ℝ) : a^2 + b^2 - 1 - a^2*b^2 ≥ 0 := by
  sorry

end square_sum_inequality_l1584_158484


namespace icosikaipentagon_diagonals_l1584_158439

/-- The number of diagonals that can be drawn from a single vertex of an n-sided polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

theorem icosikaipentagon_diagonals :
  diagonals_from_vertex 25 = 22 :=
by sorry

end icosikaipentagon_diagonals_l1584_158439
