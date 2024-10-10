import Mathlib

namespace congruence_problem_l1463_146360

theorem congruence_problem (x : ℤ) 
  (h1 : (2 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (4 + x) % (3^3) = 2^2 % (3^3))
  (h3 : (6 + x) % (5^3) = 7^2 % (5^3)) :
  x % 120 = 103 := by
sorry

end congruence_problem_l1463_146360


namespace chosen_number_proof_l1463_146362

theorem chosen_number_proof (x : ℝ) : (x / 12) - 240 = 8 ↔ x = 2976 := by
  sorry

end chosen_number_proof_l1463_146362


namespace general_admission_price_is_21_85_l1463_146304

/-- Represents the ticket sales data for a snooker tournament --/
structure TicketSales where
  totalTickets : ℕ
  totalRevenue : ℚ
  vipPrice : ℚ
  vipGenDifference : ℕ

/-- Calculates the price of a general admission ticket --/
def generalAdmissionPrice (sales : TicketSales) : ℚ :=
  let genTickets := (sales.totalTickets + sales.vipGenDifference) / 2
  let vipTickets := sales.totalTickets - genTickets
  (sales.totalRevenue - sales.vipPrice * vipTickets) / genTickets

/-- Theorem stating that the general admission price is $21.85 --/
theorem general_admission_price_is_21_85 (sales : TicketSales) 
  (h1 : sales.totalTickets = 320)
  (h2 : sales.totalRevenue = 7500)
  (h3 : sales.vipPrice = 45)
  (h4 : sales.vipGenDifference = 276) :
  generalAdmissionPrice sales = 21.85 := by
  sorry

end general_admission_price_is_21_85_l1463_146304


namespace area_of_triangle_AEB_main_theorem_l1463_146319

/-- Rectangle ABCD with given dimensions and points -/
structure Rectangle :=
  (A B C D F G E : ℝ × ℝ)
  (ab_length : ℝ)
  (bc_length : ℝ)
  (df_length : ℝ)
  (gc_length : ℝ)

/-- Conditions for the rectangle -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  rect.ab_length = 7 ∧
  rect.bc_length = 4 ∧
  rect.df_length = 2 ∧
  rect.gc_length = 1 ∧
  rect.F.2 = rect.C.2 ∧
  rect.G.2 = rect.C.2 ∧
  rect.A.1 = rect.D.1 ∧
  rect.B.1 = rect.C.1 ∧
  rect.A.2 = rect.B.2 ∧
  rect.C.2 = rect.D.2 ∧
  (rect.E.1 - rect.A.1) / (rect.B.1 - rect.A.1) = (rect.F.1 - rect.D.1) / (rect.C.1 - rect.D.1) ∧
  (rect.E.1 - rect.B.1) / (rect.A.1 - rect.B.1) = (rect.G.1 - rect.C.1) / (rect.D.1 - rect.C.1)

/-- Theorem: The area of triangle AEB is 22.4 -/
theorem area_of_triangle_AEB (rect : Rectangle) 
  (h : rectangle_conditions rect) : ℝ :=
  22.4

/-- Main theorem: If the rectangle satisfies the given conditions, 
    then the area of triangle AEB is 22.4 -/
theorem main_theorem (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  area_of_triangle_AEB rect h = 22.4 := by
  sorry

end area_of_triangle_AEB_main_theorem_l1463_146319


namespace unique_two_digit_number_with_remainder_one_l1463_146350

theorem unique_two_digit_number_with_remainder_one : 
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 4 = 1 ∧ n % 17 = 1 :=
by sorry

end unique_two_digit_number_with_remainder_one_l1463_146350


namespace chessboard_sum_zero_l1463_146399

/-- Represents a chessboard with signed numbers -/
def SignedChessboard := Fin 8 → Fin 8 → Int

/-- Checks if a row has exactly four positive and four negative numbers -/
def valid_row (board : SignedChessboard) (row : Fin 8) : Prop :=
  (Finset.filter (λ col => board row col > 0) Finset.univ).card = 4 ∧
  (Finset.filter (λ col => board row col < 0) Finset.univ).card = 4

/-- Checks if a column has exactly four positive and four negative numbers -/
def valid_column (board : SignedChessboard) (col : Fin 8) : Prop :=
  (Finset.filter (λ row => board row col > 0) Finset.univ).card = 4 ∧
  (Finset.filter (λ row => board row col < 0) Finset.univ).card = 4

/-- Checks if the board contains numbers from 1 to 64 with signs -/
def valid_numbers (board : SignedChessboard) : Prop :=
  ∀ n : Fin 64, ∃ (i j : Fin 8), |board i j| = n.val + 1

/-- The main theorem: sum of all numbers on a valid chessboard is zero -/
theorem chessboard_sum_zero (board : SignedChessboard)
  (h_rows : ∀ row, valid_row board row)
  (h_cols : ∀ col, valid_column board col)
  (h_nums : valid_numbers board) :
  (Finset.univ.sum (λ (i : Fin 8) => Finset.univ.sum (λ (j : Fin 8) => board i j))) = 0 :=
sorry

end chessboard_sum_zero_l1463_146399


namespace negation_of_forall_positive_l1463_146339

theorem negation_of_forall_positive (R : Type) [OrderedRing R] :
  (¬ (∀ x : R, x^2 + x + 1 > 0)) ↔ (∃ x : R, x^2 + x + 1 ≤ 0) := by
  sorry

end negation_of_forall_positive_l1463_146339


namespace prob_two_adjacent_is_one_fifth_l1463_146383

def num_knights : ℕ := 30
def num_selected : ℕ := 3

def prob_at_least_two_adjacent : ℚ :=
  1 - (num_knights * (num_knights - 3) * (num_knights - 4) - num_knights * 2 * (num_knights - 3)) / (num_knights.choose num_selected)

theorem prob_two_adjacent_is_one_fifth :
  prob_at_least_two_adjacent = 1 / 5 := by
  sorry

end prob_two_adjacent_is_one_fifth_l1463_146383


namespace flensburgian_iff_even_l1463_146354

/-- A system of equations is Flensburgian if there exists a variable that is always greater than the others for all pairwise different solutions. -/
def isFlensburgian (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ i : Fin 3, ∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x → f x y z →
    match i with
    | 0 => x > y ∧ x > z
    | 1 => y > x ∧ y > z
    | 2 => z > x ∧ z > y

/-- The system of equations for the Flensburgian problem. -/
def flensburgSystem (n : ℕ) (a b c : ℝ) : Prop :=
  a^n + b = a ∧ c^(n+1) + b^2 = a*b

/-- The main theorem stating that the system is Flensburgian if and only if n is even. -/
theorem flensburgian_iff_even (n : ℕ) (h : n ≥ 2) :
  isFlensburgian (flensburgSystem n) ↔ Even n :=
sorry

end flensburgian_iff_even_l1463_146354


namespace gcd_f_x_l1463_146332

def f (x : ℤ) : ℤ := (3*x+4)*(7*x+1)*(13*x+6)*(2*x+9)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 15336 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 216 := by
  sorry

end gcd_f_x_l1463_146332


namespace business_profit_share_l1463_146348

/-- Calculates the profit share of a partner given the total capital, partner's capital, and total profit -/
def profitShare (totalCapital : ℚ) (partnerCapital : ℚ) (totalProfit : ℚ) : ℚ :=
  (partnerCapital / totalCapital) * totalProfit

theorem business_profit_share 
  (capitalA capitalB capitalC : ℚ)
  (profitDifferenceAC : ℚ)
  (h1 : capitalA = 8000)
  (h2 : capitalB = 10000)
  (h3 : capitalC = 12000)
  (h4 : profitDifferenceAC = 760) :
  ∃ (totalProfit : ℚ), 
    profitShare (capitalA + capitalB + capitalC) capitalB totalProfit = 1900 :=
by
  sorry

#check business_profit_share

end business_profit_share_l1463_146348


namespace billion_two_hundred_million_scientific_notation_l1463_146335

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_two_hundred_million_scientific_notation :
  toScientificNotation 1200000000 = ScientificNotation.mk 1.2 9 (by norm_num) :=
sorry

end billion_two_hundred_million_scientific_notation_l1463_146335


namespace expression_simplification_l1463_146393

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  sorry

end expression_simplification_l1463_146393


namespace problem_solution_l1463_146313

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 7 - b) 
  (h2 : 3 + b = 8 + a) : 
  4 - a = 11/2 := by
sorry

end problem_solution_l1463_146313


namespace parabola_transformation_l1463_146318

-- Define the original function
def original_function (x : ℝ) : ℝ := (x - 1)^2 + 2

-- Define the transformation
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + 1) - 1

-- State the theorem
theorem parabola_transformation :
  ∀ x : ℝ, transform original_function x = x^2 + 1 :=
by sorry

end parabola_transformation_l1463_146318


namespace linear_regression_at_25_l1463_146377

/-- Linear regression function -/
def linear_regression (x : ℝ) : ℝ := 0.50 * x - 0.81

/-- Theorem: The linear regression equation y = 0.50x - 0.81 yields y = 11.69 when x = 25 -/
theorem linear_regression_at_25 : linear_regression 25 = 11.69 := by
  sorry

end linear_regression_at_25_l1463_146377


namespace parabola_focus_l1463_146386

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 2 * x^2

/-- The focus of a parabola -/
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (a : ℝ), a ≠ 0 ∧ 
  ∀ (x y : ℝ), parabola x y ↔ (x - p.1)^2 = 4 * a * (y - p.2)

/-- Theorem: The focus of the parabola y = 2x² is at the point (0, 1/8) -/
theorem parabola_focus :
  focus (0, 1/8) parabola_equation :=
sorry

end parabola_focus_l1463_146386


namespace dave_tickets_l1463_146320

theorem dave_tickets (won lost used : ℕ) (h1 : won = 14) (h2 : lost = 2) (h3 : used = 10) :
  won - lost - used = 2 := by
  sorry

end dave_tickets_l1463_146320


namespace mba_committee_size_l1463_146390

theorem mba_committee_size 
  (total_mbas : ℕ) 
  (num_committees : ℕ) 
  (prob_same_committee : ℚ) :
  total_mbas = 6 ∧ 
  num_committees = 2 ∧ 
  prob_same_committee = 2/5 →
  ∃ (committee_size : ℕ), 
    committee_size * num_committees = total_mbas ∧
    committee_size = 3 :=
by sorry

end mba_committee_size_l1463_146390


namespace mans_walking_rate_l1463_146394

/-- The problem of finding a man's initial walking rate given certain conditions. -/
theorem mans_walking_rate (distance : ℝ) (early_speed : ℝ) (early_time : ℝ) (late_time : ℝ) :
  distance = 6.000000000000001 →
  early_speed = 6 →
  early_time = 5 / 60 →
  late_time = 7 / 60 →
  ∃ (initial_speed : ℝ),
    initial_speed = distance / (distance / early_speed + early_time + late_time) ∧
    initial_speed = 5 := by
  sorry

#eval 6.000000000000001 / (6.000000000000001 / 6 + 5 / 60 + 7 / 60)

end mans_walking_rate_l1463_146394


namespace problem_1_problem_2_problem_3_problem_4_l1463_146338

-- 1. Prove that 522 - 112 ÷ 4 = 494
theorem problem_1 : 522 - 112 / 4 = 494 := by
  sorry

-- 2. Prove that (603 - 587) × 80 = 1280
theorem problem_2 : (603 - 587) * 80 = 1280 := by
  sorry

-- 3. Prove that 26 × 18 + 463 = 931
theorem problem_3 : 26 * 18 + 463 = 931 := by
  sorry

-- 4. Prove that 400 × (45 ÷ 9) = 2000
theorem problem_4 : 400 * (45 / 9) = 2000 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l1463_146338


namespace geometric_sequence_property_l1463_146316

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : is_positive_geometric_sequence a) 
  (h_prod : a 4 * a 8 = 9) : 
  a 6 = 3 := by
sorry

end geometric_sequence_property_l1463_146316


namespace negation_of_universal_statement_l1463_146367

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x : ℝ, x < 2) := by sorry

end negation_of_universal_statement_l1463_146367


namespace angle_between_quito_and_kampala_l1463_146334

/-- The angle at the center of a spherical Earth between two points on the equator -/
def angle_at_center (west_longitude east_longitude : ℝ) : ℝ :=
  west_longitude + east_longitude

/-- Theorem: The angle at the center of a spherical Earth between two points,
    one at 78° W and the other at 32° E, both on the equator, is 110°. -/
theorem angle_between_quito_and_kampala :
  angle_at_center 78 32 = 110 := by
  sorry

end angle_between_quito_and_kampala_l1463_146334


namespace polar_to_cartesian_line_l1463_146358

/-- The polar equation r = 2 / (2sin θ - cos θ) represents a line in Cartesian coordinates. -/
theorem polar_to_cartesian_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (x y : ℝ), (∃ (r θ : ℝ), r > 0 ∧
    r = 2 / (2 * Real.sin θ - Real.cos θ) ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ) →
  a * x + b * y = c :=
sorry

end polar_to_cartesian_line_l1463_146358


namespace disneyland_attractions_permutations_l1463_146361

theorem disneyland_attractions_permutations : Nat.factorial 6 = 720 := by
  sorry

end disneyland_attractions_permutations_l1463_146361


namespace mildred_orange_collection_l1463_146310

/-- Mildred's orange collection problem -/
theorem mildred_orange_collection (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 77 → additional = 2 → total = initial + additional → total = 79 := by
  sorry

end mildred_orange_collection_l1463_146310


namespace revenue_increase_percentage_l1463_146353

/-- Calculates the percentage increase in revenue given initial and new package volumes and prices. -/
theorem revenue_increase_percentage
  (initial_volume : ℝ)
  (initial_price : ℝ)
  (new_volume : ℝ)
  (new_price : ℝ)
  (h1 : initial_volume = 1)
  (h2 : initial_price = 60)
  (h3 : new_volume = 0.9)
  (h4 : new_price = 81) :
  (new_price / new_volume - initial_price / initial_volume) / (initial_price / initial_volume) * 100 = 50 := by
sorry


end revenue_increase_percentage_l1463_146353


namespace andrew_total_donation_l1463_146306

/-- Calculates the total donation amount for a geometric series of donations -/
def totalDonation (initialAmount : ℕ) (commonRatio : ℕ) (startAge : ℕ) (currentAge : ℕ) : ℕ :=
  let numberOfTerms := currentAge - startAge + 1
  initialAmount * (commonRatio ^ numberOfTerms - 1) / (commonRatio - 1)

/-- Theorem stating that Andrew's total donation equals 3,669,609k -/
theorem andrew_total_donation :
  totalDonation 7000 2 11 29 = 3669609000 := by
  sorry


end andrew_total_donation_l1463_146306


namespace print_350_pages_time_l1463_146382

/-- Calculates the time needed to print a given number of pages with a printer that has a specified
printing rate and pause interval. -/
def print_time (total_pages : ℕ) (pages_per_minute : ℕ) (pause_interval : ℕ) (pause_duration : ℕ) : ℕ :=
  let num_pauses := (total_pages / pause_interval) - 1
  let pause_time := num_pauses * pause_duration
  let print_time := (total_pages + pages_per_minute - 1) / pages_per_minute
  print_time + pause_time

/-- Theorem stating that printing 350 pages with the given printer specifications
takes approximately 27 minutes. -/
theorem print_350_pages_time :
  print_time 350 23 50 2 = 27 := by
  sorry

end print_350_pages_time_l1463_146382


namespace license_plate_increase_l1463_146365

theorem license_plate_increase : 
  let old_plates := 26 * (10 ^ 3)
  let new_plates := (26 ^ 4) * (10 ^ 4)
  (new_plates / old_plates : ℚ) = 175760 := by
sorry

end license_plate_increase_l1463_146365


namespace congruent_mod_divisor_congruent_mod_polynomial_l1463_146364

/-- Definition of congruence modulo m -/
def congruent_mod (a b m : ℤ) : Prop :=
  ∃ k : ℤ, a - b = m * k

/-- Statement 1 -/
theorem congruent_mod_divisor (a b m d : ℤ) (hm : 0 < m) (hd : 0 < d) (hdiv : d ∣ m) 
    (h : congruent_mod a b m) : congruent_mod a b d := by
  sorry

/-- Definition of the polynomial f(x) = x³ - 2x + 5 -/
def f (x : ℤ) : ℤ := x^3 - 2*x + 5

/-- Statement 4 -/
theorem congruent_mod_polynomial (a b m : ℤ) (hm : 0 < m) 
    (h : congruent_mod a b m) : congruent_mod (f a) (f b) m := by
  sorry

end congruent_mod_divisor_congruent_mod_polynomial_l1463_146364


namespace n_divisible_by_40_l1463_146369

theorem n_divisible_by_40 (n : ℤ) 
  (h1 : ∃ k : ℤ, 2 * n + 1 = k ^ 2) 
  (h2 : ∃ m : ℤ, 3 * n + 1 = m ^ 2) : 
  40 ∣ n := by
sorry

end n_divisible_by_40_l1463_146369


namespace triangle_area_after_median_division_l1463_146355

-- Define a triangle type
structure Triangle where
  area : ℝ

-- Define a function that represents dividing a triangle by a median
def divideByMedian (t : Triangle) : (Triangle × Triangle) :=
  sorry

-- Theorem statement
theorem triangle_area_after_median_division (t : Triangle) :
  let (t1, t2) := divideByMedian t
  t1.area = 7 → t.area = 14 := by
  sorry

end triangle_area_after_median_division_l1463_146355


namespace number_of_factors_of_48_l1463_146343

theorem number_of_factors_of_48 : Nat.card (Nat.divisors 48) = 10 := by
  sorry

end number_of_factors_of_48_l1463_146343


namespace factorization_equality_l1463_146302

theorem factorization_equality (x : ℝ) : x * (x + 2) - x - 2 = (x + 2) * (x - 1) := by
  sorry

end factorization_equality_l1463_146302


namespace cube_volume_from_side_area_l1463_146388

theorem cube_volume_from_side_area (side_area : ℝ) (volume : ℝ) :
  side_area = 64 →
  volume = (side_area ^ (1/2 : ℝ)) ^ 3 →
  volume = 512 :=
by sorry

end cube_volume_from_side_area_l1463_146388


namespace balls_in_boxes_with_empty_l1463_146385

/-- The number of ways to put n distinguishable balls in k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to put n distinguishable balls in k distinguishable boxes
    with at least one box empty -/
def ways_with_empty_box (n : ℕ) (k : ℕ) : ℕ :=
  ways_to_put_balls_in_boxes n k -
  (Nat.choose k 1) * ways_to_put_balls_in_boxes n (k-1) +
  (Nat.choose k 2) * ways_to_put_balls_in_boxes n (k-2) -
  (Nat.choose k 3) * ways_to_put_balls_in_boxes n (k-3)

/-- Theorem: There are 240 ways to put 5 distinguishable balls in 4 distinguishable boxes
    with at least one box remaining empty -/
theorem balls_in_boxes_with_empty : ways_with_empty_box 5 4 = 240 := by
  sorry

end balls_in_boxes_with_empty_l1463_146385


namespace solve_equation_l1463_146366

theorem solve_equation (x : ℝ) : 3 * x = (36 - x) + 16 → x = 13 := by
  sorry

end solve_equation_l1463_146366


namespace dance_team_new_members_l1463_146324

/-- Calculates the number of new people who joined a dance team given the initial size, number of people who quit, and final size. -/
def new_members (initial_size quit_count final_size : ℕ) : ℕ :=
  final_size - (initial_size - quit_count)

/-- Proves that 13 new people joined the dance team given the specific conditions. -/
theorem dance_team_new_members :
  let initial_size := 25
  let quit_count := 8
  let final_size := 30
  new_members initial_size quit_count final_size = 13 := by
  sorry

end dance_team_new_members_l1463_146324


namespace square_sum_inequality_l1463_146392

theorem square_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧
  ((a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end square_sum_inequality_l1463_146392


namespace line_through_point_l1463_146346

/-- Given a line with equation 5x + by + 2 = d passing through the point (40, 5),
    prove that d = 202 + 5b -/
theorem line_through_point (b : ℝ) : 
  ∃ (d : ℝ), 5 * 40 + b * 5 + 2 = d ∧ d = 202 + 5 * b := by
  sorry

end line_through_point_l1463_146346


namespace complex_absolute_value_l1463_146349

theorem complex_absolute_value (z : ℂ) :
  (3 + 4*I) / z = 5*I → Complex.abs z = 1 := by sorry

end complex_absolute_value_l1463_146349


namespace last_two_digits_sum_factorials_l1463_146344

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials :
  last_two_digits (sum_factorials 50) = last_two_digits (sum_factorials 9) := by
sorry

end last_two_digits_sum_factorials_l1463_146344


namespace smallest_integer_l1463_146384

theorem smallest_integer (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 3) →
  (Nat.lcm a b = x * (x + 3)) →
  (b = 36) →
  (∀ y : ℕ+, y < x → ¬(∃ c : ℕ+, 
    Nat.gcd c 36 = y + 3 ∧ 
    Nat.lcm c 36 = y * (y + 3))) →
  (a = 108) :=
by sorry

end smallest_integer_l1463_146384


namespace solution_set_inequality_l1463_146305

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_inequality_l1463_146305


namespace min_value_expression_l1463_146370

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 8 ∧
  ((x^3 / (y - 1)) + (y^3 / (x - 1)) = 8 ↔ x = 2 ∧ y = 2) :=
by sorry

end min_value_expression_l1463_146370


namespace first_year_rate_is_12_percent_l1463_146317

/-- Profit rate in the first year -/
def first_year_rate : ℝ := 0.12

/-- Initial investment in millions of yuan -/
def initial_investment : ℝ := 5

/-- Profit rate increase in the second year -/
def rate_increase : ℝ := 0.08

/-- Net profit in the second year in millions of yuan -/
def second_year_profit : ℝ := 1.12

theorem first_year_rate_is_12_percent :
  (initial_investment + initial_investment * first_year_rate) * 
  (first_year_rate + rate_increase) = second_year_profit := by sorry

end first_year_rate_is_12_percent_l1463_146317


namespace simplify_expression_l1463_146345

theorem simplify_expression (x : ℝ) : (3*x - 6)*(2*x + 8) - (x + 6)*(3*x + 1) = 3*x^2 - 7*x - 54 := by
  sorry

end simplify_expression_l1463_146345


namespace jason_picked_46_pears_l1463_146373

/-- Calculates the number of pears Jason picked given the number of pears Keith picked,
    the number of pears Mike ate, and the number of pears left. -/
def jasons_pears (keith_pears mike_ate pears_left : ℕ) : ℕ :=
  (mike_ate + pears_left) - keith_pears

/-- Proves that Jason picked 46 pears given the problem conditions. -/
theorem jason_picked_46_pears :
  jasons_pears 47 12 81 = 46 := by
  sorry

end jason_picked_46_pears_l1463_146373


namespace no_extreme_points_iff_m_range_l1463_146336

/-- The function f(x) = x ln x + mx² - m has no extreme points in its domain
    if and only if m ∈ (-∞, -1/2] --/
theorem no_extreme_points_iff_m_range (m : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε),
    (y * Real.log y + m * y^2 - m ≠ x * Real.log x + m * x^2 - m)) ↔
  m ≤ -1/2 :=
sorry

end no_extreme_points_iff_m_range_l1463_146336


namespace correct_transformation_l1463_146329

theorem correct_transformation (x : ℝ) : 2*x = 3*x + 4 → 2*x - 3*x = 4 := by
  sorry

end correct_transformation_l1463_146329


namespace g_properties_l1463_146396

noncomputable def g (x : ℝ) : ℝ :=
  (4 * Real.sin x ^ 4 + 5 * Real.cos x ^ 2) / (4 * Real.cos x ^ 4 + 3 * Real.sin x ^ 2)

theorem g_properties :
  (∀ k : ℤ, g (π/4 + k*π) = 7/5 ∧ g (π/3 + 2*k*π) = 7/5 ∧ g (-π/3 + 2*k*π) = 7/5) ∧
  (∀ x : ℝ, g x ≤ 71/55) ∧
  (∀ x : ℝ, g x ≥ 5/4) ∧
  (∃ x : ℝ, g x = 71/55) ∧
  (∃ x : ℝ, g x = 5/4) :=
by sorry

end g_properties_l1463_146396


namespace perfect_square_addition_l1463_146301

theorem perfect_square_addition (n : Nat) : ∃ (m : Nat), (n + 49)^2 = 4440 + 49 := by
  sorry

end perfect_square_addition_l1463_146301


namespace candidate_vote_percentage_l1463_146359

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000) 
  (h2 : invalid_percentage = 15 / 100) 
  (h3 : candidate_valid_votes = 357000) : 
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 := by
sorry

end candidate_vote_percentage_l1463_146359


namespace sum_of_digits_of_large_number_l1463_146303

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_large_number : sumOfDigits (2^2010 * 5^2008 * 7) = 10 := by sorry

end sum_of_digits_of_large_number_l1463_146303


namespace combined_boys_avg_is_67_l1463_146300

/-- Represents a high school with average scores for boys, girls, and combined --/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools --/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_girls_avg : ℝ

/-- Calculates the combined average score for boys given two schools --/
def combined_boys_avg (schools : CombinedSchools) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the combined average score for boys is 67 --/
theorem combined_boys_avg_is_67 (schools : CombinedSchools) 
  (h1 : schools.school1 = ⟨65, 75, 68⟩)
  (h2 : schools.school2 = ⟨70, 85, 75⟩)
  (h3 : schools.combined_girls_avg = 80) :
  combined_boys_avg schools = 67 := by
  sorry

end combined_boys_avg_is_67_l1463_146300


namespace x_plus_y_values_l1463_146376

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x < y) :
  x + y = -5 ∨ x + y = -1 := by
sorry

end x_plus_y_values_l1463_146376


namespace quadratic_decreasing_condition_l1463_146309

/-- A quadratic function f(x) = x² - mx + c -/
def f (m c x : ℝ) : ℝ := x^2 - m*x + c

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 2*x - m

theorem quadratic_decreasing_condition (m c : ℝ) :
  (∀ x < 1, (f' m x) < 0) → m ≥ 2 := by
  sorry

end quadratic_decreasing_condition_l1463_146309


namespace even_z_dominoes_l1463_146378

/-- Represents a lattice polygon that can be covered by quad-dominoes -/
structure LatticePolygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents an S-quad-domino -/
inductive SQuadDomino

/-- Represents a Z-quad-domino -/
inductive ZQuadDomino

/-- Represents a covering of a lattice polygon with quad-dominoes -/
structure Covering (P : LatticePolygon) where
  s_dominoes : List SQuadDomino
  z_dominoes : List ZQuadDomino
  is_valid : Bool -- Indicates if the covering is valid (no overlap and complete)

/-- Checks if a lattice polygon can be completely covered by S-quad-dominoes -/
def can_cover_with_s (P : LatticePolygon) : Prop :=
  ∃ (c : Covering P), c.z_dominoes.length = 0 ∧ c.is_valid

/-- Main theorem: If a lattice polygon can be covered by S-quad-dominoes,
    then any valid covering with S and Z quad-dominoes uses an even number of Z-quad-dominoes -/
theorem even_z_dominoes (P : LatticePolygon) 
  (h : can_cover_with_s P) : 
  ∀ (c : Covering P), c.is_valid → Even c.z_dominoes.length :=
sorry

end even_z_dominoes_l1463_146378


namespace lowest_score_within_two_std_dev_l1463_146363

/-- Given a mean score and standard deviation, calculate the lowest score within a certain number of standard deviations from the mean. -/
def lowest_score (mean : ℝ) (std_dev : ℝ) (num_std_dev : ℝ) : ℝ :=
  mean - num_std_dev * std_dev

/-- Theorem stating that for a mean of 60 and standard deviation of 10, the lowest score within 2 standard deviations is 40. -/
theorem lowest_score_within_two_std_dev :
  lowest_score 60 10 2 = 40 := by
  sorry

#eval lowest_score 60 10 2

end lowest_score_within_two_std_dev_l1463_146363


namespace geometric_sequence_third_term_l1463_146387

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 16) :
  a 3 = 4 := by
sorry

end geometric_sequence_third_term_l1463_146387


namespace triangle_abc_properties_l1463_146323

/-- Triangle ABC with sides a, b, and c satisfying |a-3| + (b-4)^2 = 0 -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  h : |a - 3| + (b - 4)^2 = 0

/-- The perimeter of an isosceles triangle -/
def isoscelesPerimeter (t : TriangleABC) : Set ℝ :=
  {10, 11}

theorem triangle_abc_properties (t : TriangleABC) :
  t.a = 3 ∧ t.b = 4 ∧ 1 < t.c ∧ t.c < 7 ∧
  (t.a = t.b ∨ t.a = t.c ∨ t.b = t.c → t.a + t.b + t.c ∈ isoscelesPerimeter t) :=
by sorry

end triangle_abc_properties_l1463_146323


namespace no_event_with_prob_1_5_l1463_146337

-- Define the probability measure
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)

-- Axiom: Probability is always between 0 and 1
axiom prob_bounds (E : Set Ω) : 0 ≤ P E ∧ P E ≤ 1

-- Theorem: There does not exist an event with probability 1.5
theorem no_event_with_prob_1_5 : ¬∃ (A : Set Ω), P A = 1.5 := by
  sorry

end no_event_with_prob_1_5_l1463_146337


namespace gcd_of_45_and_75_l1463_146389

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_of_45_and_75_l1463_146389


namespace invisibility_elixir_combinations_l1463_146321

/-- The number of magical herbs available for the invisibility elixir. -/
def num_herbs : ℕ := 4

/-- The number of enchanted gems available for the invisibility elixir. -/
def num_gems : ℕ := 6

/-- The number of herb-gem combinations that cancel each other's magic. -/
def num_cancelling_combinations : ℕ := 3

/-- The number of successful combinations for the invisibility elixir. -/
def num_successful_combinations : ℕ := num_herbs * num_gems - num_cancelling_combinations

theorem invisibility_elixir_combinations :
  num_successful_combinations = 21 := by sorry

end invisibility_elixir_combinations_l1463_146321


namespace erdos_theorem_l1463_146398

/-- For any integer k, there exists a graph H with girth greater than k and chromatic number greater than k. -/
theorem erdos_theorem (k : ℕ) : ∃ H : SimpleGraph ℕ, SimpleGraph.girth H > k ∧ SimpleGraph.chromaticNumber H > k := by
  sorry

end erdos_theorem_l1463_146398


namespace q_gt_one_neither_sufficient_nor_necessary_l1463_146371

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem stating that "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_neither_sufficient_nor_necessary :
  (∃ (a : ℕ → ℝ) (q : ℝ), q > 1 ∧ GeometricSequence a q ∧ ¬IncreasingSequence a) ∧
  (∃ (a : ℕ → ℝ) (q : ℝ), q ≤ 1 ∧ GeometricSequence a q ∧ IncreasingSequence a) :=
sorry

end q_gt_one_neither_sufficient_nor_necessary_l1463_146371


namespace gideon_age_proof_l1463_146315

/-- The number of years in a century -/
def century : ℕ := 100

/-- Gideon's initial number of marbles -/
def initial_marbles : ℕ := century

/-- The fraction of marbles Gideon gives to his sister -/
def fraction_given : ℚ := 3/4

/-- Gideon's current age -/
def gideon_age : ℕ := 45

theorem gideon_age_proof :
  gideon_age = initial_marbles - (fraction_given * initial_marbles).num - 5 :=
sorry

end gideon_age_proof_l1463_146315


namespace sin_fourteen_pi_fifths_l1463_146381

theorem sin_fourteen_pi_fifths : 
  Real.sin (14 * π / 5) = (Real.sqrt (10 - 2 * Real.sqrt 5)) / 4 := by
  sorry

end sin_fourteen_pi_fifths_l1463_146381


namespace correct_formula_l1463_146331

def f (x : ℝ) : ℝ := 200 - 10*x - 10*x^2

theorem correct_formula : 
  f 0 = 200 ∧ f 1 = 170 ∧ f 2 = 120 ∧ f 3 = 50 ∧ f 4 = 0 := by
  sorry

end correct_formula_l1463_146331


namespace complex_number_problem_l1463_146380

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = a + Complex.I * Real.sqrt 3 → z * z = 4 → a = 1 ∨ a = -1 := by
  sorry

end complex_number_problem_l1463_146380


namespace system_solution_l1463_146352

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, 2 * x - y = 5 * k + 6 ∧ 4 * x + 7 * y = k ∧ x + y = 2023) → k = 2022 :=
by
  sorry

end system_solution_l1463_146352


namespace fewer_men_than_women_l1463_146340

theorem fewer_men_than_women (total : ℕ) (men : ℕ) (h1 : total = 180) (h2 : men = 80) (h3 : men < total - men) :
  total - men - men = 20 := by
  sorry

end fewer_men_than_women_l1463_146340


namespace binomial_coefficient_geometric_mean_l1463_146341

theorem binomial_coefficient_geometric_mean (a : ℚ) : 
  (∃ (k : ℕ), k = 7 ∧ 
    (Nat.choose k 4 * a^3)^2 = (Nat.choose k 5 * a^2) * (Nat.choose k 2 * a^5)) ↔ 
  a = 25 / 9 := by
sorry

end binomial_coefficient_geometric_mean_l1463_146341


namespace common_chord_equation_l1463_146368

/-- Given two circles in polar coordinates:
    1. ρ = r
    2. ρ = -2r * sin(θ + π/4)
    where r > 0, the equation of the line on which their common chord lies
    is √2 * ρ * (sin θ + cos θ) = -r -/
theorem common_chord_equation (r : ℝ) (h : r > 0) :
  ∃ (ρ θ : ℝ), (ρ = r ∨ ρ = -2 * r * Real.sin (θ + π/4)) →
    Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r :=
by sorry

end common_chord_equation_l1463_146368


namespace sum_factorials_mod_20_l1463_146311

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_mod_20 : sum_factorials 50 % 20 = 13 := by
  sorry

end sum_factorials_mod_20_l1463_146311


namespace train_speed_train_speed_approximately_60_l1463_146325

/-- The speed of a train given its length, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed - (man_speed * 1000 / 3600)
  let train_speed_kmh := train_speed_ms * 3600 / 1000
  train_speed_kmh

/-- The speed of the train is approximately 60 km/hr given the specified conditions. -/
theorem train_speed_approximately_60 :
  ∃ ε > 0, abs (train_speed 220 6 11.999040076793857 - 60) < ε :=
sorry

end train_speed_train_speed_approximately_60_l1463_146325


namespace power_tower_mod_500_l1463_146327

theorem power_tower_mod_500 : 2^(2^(2^2)) % 500 = 536 := by
  sorry

end power_tower_mod_500_l1463_146327


namespace baker_cakes_problem_l1463_146357

theorem baker_cakes_problem (initial_cakes : ℕ) 
  (h1 : initial_cakes - 78 + 31 = initial_cakes) 
  (h2 : 78 = 31 + 47) : 
  initial_cakes = 109 := by
  sorry

end baker_cakes_problem_l1463_146357


namespace mans_upward_speed_l1463_146391

/-- Proves that given a man traveling with an average speed of 28.8 km/hr
    and a downward speed of 36 km/hr, his upward speed is 24 km/hr. -/
theorem mans_upward_speed
  (v_avg : ℝ) (v_down : ℝ) (h_avg : v_avg = 28.8)
  (h_down : v_down = 36) :
  let v_up := 2 * v_avg * v_down / (2 * v_down - v_avg)
  v_up = 24 := by sorry

end mans_upward_speed_l1463_146391


namespace alex_upside_down_growth_rate_l1463_146395

/-- The growth rate of Alex when hanging upside down -/
def upsideDownGrowthRate (
  requiredHeight : ℚ)
  (currentHeight : ℚ)
  (normalGrowthRate : ℚ)
  (upsideDownHoursPerMonth : ℚ)
  (monthsInYear : ℕ) : ℚ :=
  let totalGrowthNeeded := requiredHeight - currentHeight
  let normalYearlyGrowth := normalGrowthRate * monthsInYear
  let additionalGrowthNeeded := totalGrowthNeeded - normalYearlyGrowth
  let totalUpsideDownHours := upsideDownHoursPerMonth * monthsInYear
  additionalGrowthNeeded / totalUpsideDownHours

/-- Theorem stating that Alex's upside down growth rate is 1/12 inch per hour -/
theorem alex_upside_down_growth_rate :
  upsideDownGrowthRate 54 48 (1/3) 2 12 = 1/12 := by
  sorry

end alex_upside_down_growth_rate_l1463_146395


namespace tangent_chord_fixed_point_l1463_146326

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A line represented by two points -/
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- Determines if a point is on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Determines if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Determines if a line is perpendicular to another line -/
def isPerpendicular (l1 l2 : Line) : Prop := sorry

/-- Determines if a point is outside a circle -/
def isOutside (p : ℝ × ℝ) (c : Circle) : Prop := sorry

theorem tangent_chord_fixed_point 
  (O : ℝ × ℝ) (r : ℝ) (l : Line) (H : ℝ × ℝ) :
  let c : Circle := ⟨O, r⟩
  isOutside H c →
  pointOnLine H l →
  isPerpendicular (Line.mk O H) l →
  ∃ P : ℝ × ℝ, ∀ A : ℝ × ℝ, 
    pointOnLine A l →
    ∃ B C : ℝ × ℝ,
      isTangent (Line.mk A B) c ∧
      isTangent (Line.mk A C) c ∧
      pointOnLine P (Line.mk B C) ∧
      pointOnLine P (Line.mk O H) :=
sorry

end tangent_chord_fixed_point_l1463_146326


namespace parabola_axis_equation_l1463_146308

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 + x + 3

-- Define the tangent line at x = -1
def tangent_line (x : ℝ) : ℝ := 2*x + 4

-- Define the parabola y = 2px²
def parabola (p : ℝ) (x : ℝ) : ℝ := 2*p*x^2

-- Theorem statement
theorem parabola_axis_equation :
  ∃ (p : ℝ), (∀ (x : ℝ), tangent_line x = parabola p x → x = -1 ∨ x ≠ -1) →
  (∀ (x : ℝ), parabola p x = -(1/4)*x^2) →
  (∀ (x : ℝ), x^2 = -4*1 → x = 0) :=
sorry

end parabola_axis_equation_l1463_146308


namespace machine_production_in_10_seconds_l1463_146307

/-- A machine that produces items at a constant rate -/
structure Machine where
  items_per_minute : ℕ

/-- Calculate the number of items produced in a given number of seconds -/
def items_produced (m : Machine) (seconds : ℕ) : ℚ :=
  (m.items_per_minute : ℚ) * (seconds : ℚ) / 60

theorem machine_production_in_10_seconds (m : Machine) 
  (h : m.items_per_minute = 150) : 
  items_produced m 10 = 25 := by
  sorry

end machine_production_in_10_seconds_l1463_146307


namespace contrapositive_equivalence_l1463_146330

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 ≥ 1 → (x ≥ 0 ∨ x ≤ -1))) ↔
  (∀ x : ℝ, (-1 < x ∧ x < 0 → x^2 < 1)) :=
by sorry

end contrapositive_equivalence_l1463_146330


namespace box_surface_area_l1463_146356

/-- Represents the dimensions of a rectangular sheet. -/
structure SheetDimensions where
  length : ℕ
  width : ℕ

/-- Represents the size of the square cut from each corner. -/
def CornerCutSize : ℕ := 4

/-- Calculates the surface area of the interior of the box formed by folding a rectangular sheet
    with squares cut from each corner. -/
def interiorSurfaceArea (sheet : SheetDimensions) : ℕ :=
  sheet.length * sheet.width - 4 * (CornerCutSize * CornerCutSize)

/-- Theorem stating that the surface area of the interior of the box is 936 square units. -/
theorem box_surface_area :
  interiorSurfaceArea ⟨25, 40⟩ = 936 := by sorry

end box_surface_area_l1463_146356


namespace cubic_roots_collinear_k_l1463_146351

/-- A cubic polynomial with coefficient k -/
def cubic_polynomial (k : ℤ) (x : ℂ) : ℂ :=
  x^3 - 15*x^2 + k*x - 1105

/-- Predicate for three complex numbers being distinct and collinear -/
def distinct_collinear (z₁ z₂ z₃ : ℂ) : Prop :=
  z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₂ ≠ z₃ ∧
  ∃ (a b : ℝ), (z₁.im - a * z₁.re = b) ∧ 
               (z₂.im - a * z₂.re = b) ∧ 
               (z₃.im - a * z₃.re = b)

theorem cubic_roots_collinear_k (k : ℤ) :
  (∃ (z₁ z₂ z₃ : ℂ), 
    distinct_collinear z₁ z₂ z₃ ∧
    (cubic_polynomial k z₁ = 0) ∧
    (cubic_polynomial k z₂ = 0) ∧
    (cubic_polynomial k z₃ = 0)) →
  k = 271 := by
  sorry

end cubic_roots_collinear_k_l1463_146351


namespace interest_discount_sum_l1463_146375

/-- Given a sum, rate, and time, if the simple interest is 85 and the true discount is 75, then the sum is 637.5 -/
theorem interest_discount_sum (P r t : ℝ) : 
  (P * r * t / 100 = 85) → 
  (P * r * t / (100 + r * t) = 75) → 
  P = 637.5 := by
sorry

end interest_discount_sum_l1463_146375


namespace geometric_sequence_sum_l1463_146314

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 2025 →
  a 3 + a 5 = 45 := by
  sorry

end geometric_sequence_sum_l1463_146314


namespace integer_root_implies_specific_m_l1463_146374

/-- Defines a quadratic equation with coefficient m -/
def quadratic_equation (m : ℤ) (x : ℤ) : ℤ := m * x^2 + 2*(m-5)*x + (m-4)

/-- Checks if the equation has an integer root -/
def has_integer_root (m : ℤ) : Prop := ∃ x : ℤ, quadratic_equation m x = 0

/-- The main theorem to be proved -/
theorem integer_root_implies_specific_m :
  ∀ m : ℤ, has_integer_root m → m = -4 ∨ m = 4 ∨ m = -16 := by sorry

end integer_root_implies_specific_m_l1463_146374


namespace number_division_problem_l1463_146342

theorem number_division_problem (x y : ℚ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 24) / y = 3) : 
  y = 10 := by
sorry

end number_division_problem_l1463_146342


namespace grapes_purchased_l1463_146322

/-- Proves that the number of kg of grapes purchased is 8 -/
theorem grapes_purchased (grape_price : ℕ) (mango_price : ℕ) (mango_kg : ℕ) (total_paid : ℕ) : 
  grape_price = 70 → 
  mango_price = 55 → 
  mango_kg = 9 → 
  total_paid = 1055 → 
  ∃ (grape_kg : ℕ), grape_kg * grape_price + mango_kg * mango_price = total_paid ∧ grape_kg = 8 :=
by sorry

end grapes_purchased_l1463_146322


namespace amoeba_reproduction_time_verify_16_amoebae_l1463_146347

/-- Represents the number of amoebae after a certain number of divisions -/
def amoebae_count (divisions : ℕ) : ℕ := 2^divisions

/-- Represents the time taken for a given number of divisions -/
def time_for_divisions (divisions : ℕ) : ℕ := 8

/-- The number of divisions required to reach 16 amoebae from 1 -/
def divisions_to_16 : ℕ := 4

/-- Theorem stating that it takes 2 days for an amoeba to reproduce -/
theorem amoeba_reproduction_time : 
  (time_for_divisions divisions_to_16) / divisions_to_16 = 2 := by
  sorry

/-- Verifies that 16 amoebae are indeed reached after 4 divisions -/
theorem verify_16_amoebae : amoebae_count divisions_to_16 = 16 := by
  sorry

end amoeba_reproduction_time_verify_16_amoebae_l1463_146347


namespace root_sum_reciprocal_transform_l1463_146312

-- Define the polynomial
def p (x : ℝ) : ℝ := 15 * x^3 - 35 * x^2 + 20 * x - 2

-- Theorem statement
theorem root_sum_reciprocal_transform (a b c : ℝ) :
  p a = 0 → p b = 0 → p c = 0 →  -- a, b, c are roots of p
  a ≠ b → b ≠ c → a ≠ c →        -- roots are distinct
  0 < a → a < 1 →                -- 0 < a < 1
  0 < b → b < 1 →                -- 0 < b < 1
  0 < c → c < 1 →                -- 0 < c < 1
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 2 / 3 :=
by sorry

end root_sum_reciprocal_transform_l1463_146312


namespace valleyball_league_members_l1463_146333

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := tshirt_cost

/-- The cost of equipment for home games per member in dollars -/
def home_cost : ℕ := sock_cost + tshirt_cost

/-- The cost of equipment for away games per member in dollars -/
def away_cost : ℕ := sock_cost + tshirt_cost + cap_cost

/-- The total cost of equipment per member in dollars -/
def member_cost : ℕ := home_cost + away_cost

/-- The total cost of equipment for all members in dollars -/
def total_cost : ℕ := 4324

theorem valleyball_league_members : 
  ∃ n : ℕ, n * member_cost ≤ total_cost ∧ total_cost < (n + 1) * member_cost ∧ n = 85 := by
  sorry

end valleyball_league_members_l1463_146333


namespace cid_earnings_l1463_146328

def oil_change_price : ℕ := 20
def repair_price : ℕ := 30
def car_wash_price : ℕ := 5

def oil_changes_performed : ℕ := 5
def repairs_performed : ℕ := 10
def car_washes_performed : ℕ := 15

def total_earnings : ℕ := 
  oil_change_price * oil_changes_performed + 
  repair_price * repairs_performed + 
  car_wash_price * car_washes_performed

theorem cid_earnings : total_earnings = 475 := by
  sorry

end cid_earnings_l1463_146328


namespace john_wallet_dimes_l1463_146372

def total_amount : ℚ := 680 / 100  -- $6.80 as a rational number

theorem john_wallet_dimes :
  ∀ (d q : ℕ),  -- d: number of dimes, q: number of quarters
  d = q + 4 →  -- four more dimes than quarters
  (d : ℚ) * (10 / 100) + (q : ℚ) * (25 / 100) = total_amount →  -- total amount equation
  d = 22 :=
by sorry

end john_wallet_dimes_l1463_146372


namespace missile_interception_time_l1463_146397

/-- The time taken for a missile to intercept a plane -/
theorem missile_interception_time 
  (r : ℝ) -- radius of the circular path
  (v : ℝ) -- speed of both the plane and the missile
  (h : r = 10 ∧ v = 1000) -- specific values given in the problem
  : (r * π) / (2 * v) = π / 200 := by
  sorry

#check missile_interception_time

end missile_interception_time_l1463_146397


namespace unique_number_base_conversion_l1463_146379

/-- Represents a digit in a given base -/
def IsDigit (d : ℕ) (base : ℕ) : Prop := d < base

/-- Converts a two-digit number in a given base to decimal -/
def ToDecimal (tens : ℕ) (ones : ℕ) (base : ℕ) : ℕ := base * tens + ones

theorem unique_number_base_conversion :
  ∃! n : ℕ, n > 0 ∧
    ∃ C D : ℕ,
      IsDigit C 8 ∧
      IsDigit D 8 ∧
      IsDigit C 6 ∧
      IsDigit D 6 ∧
      n = ToDecimal C D 8 ∧
      n = ToDecimal D C 6 ∧
      n = 43 := by
  sorry

end unique_number_base_conversion_l1463_146379
