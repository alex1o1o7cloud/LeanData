import Mathlib

namespace product_closure_l2278_227851

def M : Set ℤ := {x | ∃ a b : ℤ, x = a^2 + b^2}

theorem product_closure (x₁ x₂ : ℤ) (h₁ : x₁ ∈ M) (h₂ : x₂ ∈ M) : x₁ * x₂ ∈ M := by
  sorry

end product_closure_l2278_227851


namespace row_properties_l2278_227862

/-- Definition of a number being in a row -/
def in_row (m n : ℕ) : Prop :=
  n ∣ m ∧ m ≤ n^2 ∧ ∀ k < n, ¬in_row m k

/-- The main theorem encompassing all parts of the problem -/
theorem row_properties :
  (∀ m < 50, m % 10 = 0 → ∃ k < 10, in_row m k) ∧
  (∀ n ≥ 3, in_row (n^2 - n) n ∧ in_row (n^2 - 2*n) n) ∧
  (∀ n > 30, in_row (n^2 - 10*n) n) ∧
  ¬in_row (30^2 - 10*30) 30 := by
  sorry

#check row_properties

end row_properties_l2278_227862


namespace jesse_carpet_need_l2278_227829

/-- The amount of additional carpet Jesse needs to cover two rooms -/
def additional_carpet_needed (jesse_carpet area_room_a area_room_b : ℝ) : ℝ :=
  area_room_a + area_room_b - jesse_carpet

/-- Proof that Jesse needs 94 more square feet of carpet -/
theorem jesse_carpet_need : 
  let jesse_carpet : ℝ := 18
  let room_a_length : ℝ := 4
  let room_a_width : ℝ := 20
  let area_room_a : ℝ := room_a_length * room_a_width
  let area_room_b : ℝ := area_room_a / 2.5
  additional_carpet_needed jesse_carpet area_room_a area_room_b = 94
  := by sorry

end jesse_carpet_need_l2278_227829


namespace subsets_of_size_two_l2278_227872

/-- Given a finite set S, returns the number of subsets of S with exactly k elements -/
def numSubsetsOfSize (n k : ℕ) : ℕ := Nat.choose n k

theorem subsets_of_size_two (S : Type) [Fintype S] :
  (numSubsetsOfSize (Fintype.card S) 7 = 36) →
  (numSubsetsOfSize (Fintype.card S) 2 = 36) := by
  sorry

end subsets_of_size_two_l2278_227872


namespace inequality_system_solution_l2278_227802

theorem inequality_system_solution : 
  {x : ℕ | 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4} = {0, 1, 2} := by
  sorry

end inequality_system_solution_l2278_227802


namespace semicircle_area_with_inscribed_rectangle_l2278_227867

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) :
  r > 0 →
  r * r = 2 →
  π * r * r = π :=
by
  sorry

end semicircle_area_with_inscribed_rectangle_l2278_227867


namespace infinitely_many_decimals_between_3_3_and_3_6_l2278_227809

/-- The set of decimals between 3.3 and 3.6 is infinite -/
theorem infinitely_many_decimals_between_3_3_and_3_6 :
  (∀ n : ℕ, ∃ x : ℝ, 3.3 < x ∧ x < 3.6 ∧ ∃ k : ℕ, x = ↑k / 10^n) :=
sorry

end infinitely_many_decimals_between_3_3_and_3_6_l2278_227809


namespace first_year_interest_rate_is_four_percent_l2278_227838

/-- Calculates the final amount after two years of compound interest -/
def finalAmount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating that given the conditions, the first year interest rate must be 4% -/
theorem first_year_interest_rate_is_four_percent 
  (initial : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (final : ℝ) 
  (h1 : initial = 7000)
  (h2 : rate2 = 0.05)
  (h3 : final = 7644)
  (h4 : finalAmount initial rate1 rate2 = final) : 
  rate1 = 0.04 := by
  sorry

#check first_year_interest_rate_is_four_percent

end first_year_interest_rate_is_four_percent_l2278_227838


namespace factorization_proofs_l2278_227889

theorem factorization_proofs (x y a b : ℝ) : 
  (2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2) ∧ 
  (a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b)) := by
  sorry

end factorization_proofs_l2278_227889


namespace simple_interest_principal_l2278_227835

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 260)
  (h2 : rate = 7.142857142857143)
  (h3 : time = 4) :
  ∃ (principal : ℝ), principal = 910 ∧ interest = principal * rate * time / 100 :=
by sorry

end simple_interest_principal_l2278_227835


namespace sequence_formula_correct_l2278_227832

def a (n : ℕ) : ℚ := n + 1 / (2^n)

theorem sequence_formula_correct : 
  (a 1 = 3/2) ∧ (a 2 = 9/4) ∧ (a 3 = 25/8) ∧ (a 4 = 65/16) := by
  sorry

end sequence_formula_correct_l2278_227832


namespace emily_spending_l2278_227844

theorem emily_spending (x : ℝ) : 
  x + 2*x + 3*x = 120 → x = 20 := by
  sorry

end emily_spending_l2278_227844


namespace moving_circle_theorem_l2278_227820

-- Define the circles F1 and F2
def F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the locus of the center of E
def E_locus (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the slope range
def slope_range (k : ℝ) : Prop := 
  (k ≥ -Real.sqrt 6 / 4 ∧ k < 0) ∨ (k > 0 ∧ k ≤ Real.sqrt 6 / 4)

-- State the theorem
theorem moving_circle_theorem 
  (E : ℝ → ℝ → Prop) -- The moving circle E
  (A B M H : ℝ × ℝ) -- Points A, B, M, H
  (l : ℝ → ℝ) -- Line l
  (h1 : ∀ x y, E x y → (∃ r > 0, ∀ u v, F1 u v → ((x - u)^2 + (y - v)^2 = (r + 1)^2))) -- E externally tangent to F1
  (h2 : ∀ x y, E x y → (∃ r > 0, ∀ u v, F2 u v → ((x - u)^2 + (y - v)^2 = (3 - r)^2))) -- E internally tangent to F2
  (h3 : A.1 > 0 ∧ A.2 = 0 ∧ E A.1 A.2) -- A on positive x-axis and on E
  (h4 : E B.1 B.2 ∧ B.2 ≠ 0) -- B on E and not on x-axis
  (h5 : ∀ x, l x = (B.2 / (B.1 - A.1)) * (x - A.1)) -- l passes through A and B
  (h6 : M.2 = l M.1 ∧ H.1 = 0) -- M on l, H on y-axis
  (h7 : (B.1 - 1) * (H.1 - 1) + B.2 * H.2 = 0) -- BF2 ⊥ HF2
  (h8 : (M.1 - A.1)^2 + (M.2 - A.2)^2 ≥ M.1^2 + M.2^2) -- ∠MOA ≥ ∠MAO
  : (∀ x y, E x y ↔ E_locus x y) ∧ 
    (∀ k, (∃ x, l x = k * (x - A.1)) → slope_range k) :=
sorry

end moving_circle_theorem_l2278_227820


namespace cistern_fill_time_l2278_227804

/-- The time it takes for pipe p to fill the cistern -/
def p_time : ℝ := 10

/-- The time both pipes are opened together -/
def both_open_time : ℝ := 2

/-- The additional time it takes to fill the cistern after pipe p is turned off -/
def additional_time : ℝ := 10

/-- The time it takes for pipe q to fill the cistern -/
def q_time : ℝ := 15

theorem cistern_fill_time : 
  (both_open_time * (1 / p_time + 1 / q_time)) + 
  (additional_time * (1 / q_time)) = 1 := by sorry

end cistern_fill_time_l2278_227804


namespace log_inequality_l2278_227839

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end log_inequality_l2278_227839


namespace johns_age_problem_l2278_227860

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem johns_age_problem :
  ∃! x : ℕ, x > 0 ∧ is_perfect_square (x - 5) ∧ is_perfect_cube (x + 3) ∧ x = 69 := by
  sorry

end johns_age_problem_l2278_227860


namespace mikes_shirt_cost_l2278_227869

/-- The cost of Mike's shirt given the profit sharing between Mike and Johnson -/
theorem mikes_shirt_cost (total_profit : ℚ) (mikes_share johnson_share : ℚ) : 
  mikes_share / johnson_share = 2 / 5 →
  johnson_share = 2500 →
  mikes_share - 800 = 200 :=
by sorry

end mikes_shirt_cost_l2278_227869


namespace angle_POQ_is_72_degrees_l2278_227870

-- Define the regular pentagon
structure RegularPentagon where
  side_length : ℝ
  internal_angle : ℝ
  internal_angle_eq : internal_angle = 108

-- Define the inscribed circle
structure InscribedCircle (p : RegularPentagon) where
  center : Point
  radius : ℝ
  tangent_point1 : Point
  tangent_point2 : Point
  corner : Point
  is_tangent : Bool
  intersects_other_sides : Bool

-- Define the angle POQ
def angle_POQ (p : RegularPentagon) (c : InscribedCircle p) : ℝ :=
  sorry

-- Define the bisector property
def is_bisector (p : RegularPentagon) (c : InscribedCircle p) : Prop :=
  sorry

-- Theorem statement
theorem angle_POQ_is_72_degrees 
  (p : RegularPentagon) 
  (c : InscribedCircle p) 
  (h1 : c.is_tangent = true) 
  (h2 : c.intersects_other_sides = true) 
  (h3 : is_bisector p c) : 
  angle_POQ p c = 72 := by
  sorry

end angle_POQ_is_72_degrees_l2278_227870


namespace non_congruent_triangle_count_l2278_227892

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The set of 10 points in the problem -/
def problem_points : Finset Point2D := sorry

/-- Predicate to check if three points form a triangle -/
def is_triangle (p q r : Point2D) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def are_congruent (t1 t2 : Point2D × Point2D × Point2D) : Prop := sorry

/-- The set of all possible triangles formed by the problem points -/
def all_triangles : Finset (Point2D × Point2D × Point2D) := sorry

/-- The set of non-congruent triangles -/
def non_congruent_triangles : Finset (Point2D × Point2D × Point2D) := sorry

theorem non_congruent_triangle_count :
  Finset.card non_congruent_triangles = 12 := by sorry

end non_congruent_triangle_count_l2278_227892


namespace most_likely_final_number_is_54_l2278_227871

/-- The initial number on the blackboard -/
def initial_number : ℕ := 15

/-- The lower bound of the random number added in each move -/
def lower_bound : ℕ := 1

/-- The upper bound of the random number added in each move -/
def upper_bound : ℕ := 5

/-- The threshold number for ending the game -/
def threshold : ℕ := 51

/-- The expected value of the random number added in each move -/
def expected_value : ℚ := (lower_bound + upper_bound) / 2

/-- The most likely final number on the blackboard -/
def most_likely_final_number : ℕ := 54

/-- Theorem stating that the most likely final number is 54 -/
theorem most_likely_final_number_is_54 :
  ∃ (n : ℕ), initial_number + n * expected_value > threshold ∧
             initial_number + (n - 1) * expected_value ≤ threshold ∧
             most_likely_final_number = initial_number + n * ⌊expected_value⌋ := by
  sorry

end most_likely_final_number_is_54_l2278_227871


namespace jessica_payment_l2278_227897

/-- Calculates the payment for a given hour based on the repeating pattern --/
def hourly_rate (hour : ℕ) : ℕ :=
  match hour % 6 with
  | 0 => 2
  | 1 => 4
  | 2 => 6
  | 3 => 8
  | 4 => 10
  | 5 => 12
  | _ => 0  -- This case should never occur due to the modulo operation

/-- Calculates the total payment for a given number of hours --/
def total_payment (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

theorem jessica_payment : total_payment 45 = 306 := by
  sorry


end jessica_payment_l2278_227897


namespace scooter_rental_proof_l2278_227873

/-- Represents the rental cost structure for an electric scooter service -/
structure RentalCost where
  fixed : ℝ
  per_minute : ℝ

/-- Calculates the total cost for a given duration -/
def total_cost (rc : RentalCost) (duration : ℝ) : ℝ :=
  rc.fixed + rc.per_minute * duration

theorem scooter_rental_proof (rc : RentalCost) 
  (h1 : total_cost rc 3 = 78)
  (h2 : total_cost rc 8 = 108) :
  total_cost rc 5 = 90 := by
  sorry

end scooter_rental_proof_l2278_227873


namespace inequality_proof_l2278_227811

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8 / (x * y) + y^2 ≥ 8 := by
sorry

end inequality_proof_l2278_227811


namespace right_triangle_hypotenuse_l2278_227876

/-- Given a right triangle with legs a and b, if the volume of the cone formed by
    rotating the triangle about leg a is 1000π cm³ and the volume of the cone formed by
    rotating the triangle about leg b is 2430π cm³, then the length of the hypotenuse c
    is approximately 28.12 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / 3 * π * b^2 * a = 1000 * π) →
  (1 / 3 * π * a^2 * b = 2430 * π) →
  abs (Real.sqrt (a^2 + b^2) - 28.12) < 0.01 := by
  sorry

end right_triangle_hypotenuse_l2278_227876


namespace plate_distance_l2278_227845

/-- Given a square table with a circular plate, prove that the distance from the bottom edge
    of the table to the plate is 53 cm, given the distances from other edges. -/
theorem plate_distance (left_distance right_distance top_distance : ℝ) :
  left_distance = 10 →
  right_distance = 63 →
  top_distance = 20 →
  ∃ (plate_diameter bottom_distance : ℝ),
    left_distance + plate_diameter + right_distance = top_distance + plate_diameter + bottom_distance ∧
    bottom_distance = 53 :=
by sorry

end plate_distance_l2278_227845


namespace tshirt_packages_l2278_227857

theorem tshirt_packages (package_size : ℕ) (desired_shirts : ℕ) (min_packages : ℕ) : 
  package_size = 6 →
  desired_shirts = 71 →
  min_packages * package_size ≥ desired_shirts →
  ∀ n : ℕ, n * package_size ≥ desired_shirts → n ≥ min_packages →
  min_packages = 12 :=
by sorry

end tshirt_packages_l2278_227857


namespace expression_simplification_l2278_227807

theorem expression_simplification (x : ℝ) 
  (h1 : x * (x^2 - 4) = 0) 
  (h2 : x ≠ 0) 
  (h3 : x ≠ 2) :
  (x - 3) / (3 * x^2 - 6 * x) / (x + 2 - 5 / (x - 2)) = -1/6 := by
  sorry

end expression_simplification_l2278_227807


namespace traveler_distance_l2278_227890

/-- Calculates the distance traveled given initial conditions and new travel parameters. -/
def distance_traveled (initial_distance : ℚ) (initial_days : ℕ) (initial_hours_per_day : ℕ)
                      (new_days : ℕ) (new_hours_per_day : ℕ) : ℚ :=
  let initial_total_hours : ℚ := initial_days * initial_hours_per_day
  let speed : ℚ := initial_distance / initial_total_hours
  let new_total_hours : ℚ := new_days * new_hours_per_day
  speed * new_total_hours

/-- The theorem states that given the initial conditions and new travel parameters,
    the traveler will cover 93 23/29 kilometers. -/
theorem traveler_distance : 
  distance_traveled 112 29 7 17 10 = 93 + 23 / 29 := by
  sorry

end traveler_distance_l2278_227890


namespace simplify_sqrt_sum_l2278_227808

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 4 * Real.sqrt 3) + Real.sqrt (8 - 4 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end simplify_sqrt_sum_l2278_227808


namespace unique_prime_squared_plus_eleven_with_six_divisors_l2278_227821

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n has exactly 6 positive divisors -/
def has_six_divisors (n : ℕ) : Prop := num_divisors n = 6

theorem unique_prime_squared_plus_eleven_with_six_divisors :
  ∃! p : ℕ, Nat.Prime p ∧ has_six_divisors (p^2 + 11) :=
by sorry

end unique_prime_squared_plus_eleven_with_six_divisors_l2278_227821


namespace ed_weight_l2278_227822

/-- Given the weights of Al, Ben, Carl, and Ed, prove that Ed weighs 146 pounds -/
theorem ed_weight (al ben carl ed : ℕ) : 
  al = ben + 25 →
  ben = carl - 16 →
  ed = al - 38 →
  carl = 175 →
  ed = 146 := by
  sorry

end ed_weight_l2278_227822


namespace christopher_karen_money_difference_l2278_227855

theorem christopher_karen_money_difference : 
  let karen_quarters : ℕ := 32
  let christopher_quarters : ℕ := 64
  let quarter_value : ℚ := 1/4
  (christopher_quarters - karen_quarters) * quarter_value = 8 := by sorry

end christopher_karen_money_difference_l2278_227855


namespace complementary_angles_sum_l2278_227824

theorem complementary_angles_sum (a b : ℝ) : 
  a > 0 → b > 0 → a / b = 3 / 5 → a + b = 90 → a + b = 90 := by sorry

end complementary_angles_sum_l2278_227824


namespace binomial_expansion_coefficient_l2278_227800

theorem binomial_expansion_coefficient (a : ℝ) : 
  (20 : ℝ) * a^3 = 160 → a = 2 := by
  sorry

end binomial_expansion_coefficient_l2278_227800


namespace correct_swap_l2278_227803

def swap_values (m n : ℕ) : ℕ × ℕ := 
  let s := m
  let m' := n
  let n' := s
  (m', n')

theorem correct_swap : 
  ∀ (m n : ℕ), swap_values m n = (n, m) := by
  sorry

end correct_swap_l2278_227803


namespace intersection_condition_l2278_227850

/-- The set of possible values for a real number a, given the conditions. -/
def PossibleValues : Set ℝ := {-1, 0, 1}

/-- The set A defined by the equation ax + 1 = 0. -/
def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

/-- The set B containing -1 and 1. -/
def B : Set ℝ := {-1, 1}

/-- Theorem stating that if A ∩ B = A, then a must be in the set of possible values. -/
theorem intersection_condition (a : ℝ) : A a ∩ B = A a → a ∈ PossibleValues := by
  sorry

end intersection_condition_l2278_227850


namespace square_circle_ratio_l2278_227886

theorem square_circle_ratio (r c d : ℝ) (h : r > 0) (hc : c > 0) (hd : d > c) :
  let s := 2 * r
  s^2 = (c / d) * (s^2 - π * r^2) →
  s / r = Real.sqrt (c * π) / Real.sqrt (d - c) := by
sorry

end square_circle_ratio_l2278_227886


namespace equation_solution_l2278_227806

theorem equation_solution : 
  ∀ x : ℝ, x^2 - 2*|x - 1| - 2 = 0 ↔ x = 2 ∨ x = -1 - Real.sqrt 5 :=
by sorry

end equation_solution_l2278_227806


namespace min_value_reciprocal_sum_l2278_227893

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 := by
  sorry

end min_value_reciprocal_sum_l2278_227893


namespace stratified_sample_male_athletes_l2278_227879

/-- Represents the number of male athletes drawn in a stratified sample -/
def male_athletes_drawn (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * male_athletes) / total_athletes

/-- Theorem stating that in a stratified sample of 21 athletes from a population of 84 athletes 
    (48 male and 36 female), the number of male athletes drawn is 12 -/
theorem stratified_sample_male_athletes :
  male_athletes_drawn 84 48 21 = 12 := by
  sorry

#eval male_athletes_drawn 84 48 21

end stratified_sample_male_athletes_l2278_227879


namespace brick_height_calculation_l2278_227853

/-- The height of a brick given wall dimensions, brick dimensions, and number of bricks --/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
  (brick_length brick_width : ℝ) (num_bricks : ℝ) :
  wall_length = 9 →
  wall_width = 5 →
  wall_height = 18.5 →
  brick_length = 0.21 →
  brick_width = 0.1 →
  num_bricks = 4955.357142857142 →
  ∃ (brick_height : ℝ),
    brick_height = 0.008 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end brick_height_calculation_l2278_227853


namespace pharmacy_work_hours_l2278_227834

/-- Proves that given the conditions of the pharmacy problem, 
    the number of hours worked by Ann and Becky is 8 --/
theorem pharmacy_work_hours : 
  ∀ (h : ℕ), 
  (7 * h + 7 * h + 7 * 6 = 154) → 
  h = 8 := by
sorry

end pharmacy_work_hours_l2278_227834


namespace total_climb_length_l2278_227847

def keaton_ladder_length : ℕ := 30
def keaton_climbs : ℕ := 20
def reece_ladder_difference : ℕ := 4
def reece_climbs : ℕ := 15
def inches_per_foot : ℕ := 12

theorem total_climb_length : 
  (keaton_ladder_length * keaton_climbs + 
   (keaton_ladder_length - reece_ladder_difference) * reece_climbs) * 
   inches_per_foot = 11880 := by
  sorry

end total_climb_length_l2278_227847


namespace three_solutions_inequality_l2278_227885

theorem three_solutions_inequality (a : ℝ) : 
  (∃! (s : Finset ℕ), s.card = 3 ∧ 
    (∀ x : ℕ, x ∈ s ↔ (x > 0 ∧ 3 * (x - 1) < 2 * (x + a) - 5))) ↔ 
  (5/2 < a ∧ a ≤ 3) :=
sorry

end three_solutions_inequality_l2278_227885


namespace equation_equivalence_product_l2278_227841

theorem equation_equivalence_product (a b x y : ℤ) (m n p q : ℕ) :
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
  ((a^m*x - a^n)*(a^p*y - a^q) = a^5*b^5) →
  m*n*p*q = 2 := by sorry

end equation_equivalence_product_l2278_227841


namespace lopez_family_seating_arrangements_l2278_227837

/-- Represents the number of family members -/
def family_members : ℕ := 5

/-- Represents the number of front seats in the van -/
def front_seats : ℕ := 2

/-- Represents the number of back seats in the van -/
def back_seats : ℕ := 3

/-- Represents the number of adults who can drive -/
def potential_drivers : ℕ := 2

/-- Calculates the number of possible seating arrangements -/
def seating_arrangements : ℕ :=
  potential_drivers * (family_members - 1) * (back_seats.factorial)

theorem lopez_family_seating_arrangements :
  seating_arrangements = 48 :=
sorry

end lopez_family_seating_arrangements_l2278_227837


namespace remainder_theorem_l2278_227859

theorem remainder_theorem (N : ℤ) : N % 13 = 3 → N % 39 = 3 := by
  sorry

end remainder_theorem_l2278_227859


namespace evaluate_expression_l2278_227861

theorem evaluate_expression : (-(18 / 3 * 12 - 80 + 4 * 12))^2 = 1600 := by
  sorry

end evaluate_expression_l2278_227861


namespace A_value_l2278_227815

noncomputable def A (x : ℝ) : ℝ :=
  (Real.sqrt 3 * x^(3/2) - 5 * x^(1/3) + 5 * x^(4/3) - Real.sqrt (3*x)) /
  (Real.sqrt (3*x + 10 * Real.sqrt 3 * x^(5/6) + 25 * x^(2/3)) *
   Real.sqrt (1 - 2/x + 1/x^2))

theorem A_value (x : ℝ) (hx : x > 0) :
  (0 < x ∧ x < 1 → A x = -x) ∧
  (x > 1 → A x = x) := by
  sorry

end A_value_l2278_227815


namespace cauchy_equation_on_X_l2278_227880

-- Define the set X
def X : Set ℝ := {x : ℝ | ∃ (a b : ℤ), x = a + b * Real.sqrt 2}

-- Define the Cauchy equation property
def is_cauchy (f : X → ℝ) : Prop :=
  ∀ (x y : X), f (⟨x + y, sorry⟩) = f x + f y

-- State the theorem
theorem cauchy_equation_on_X (f : X → ℝ) (hf : is_cauchy f) :
  ∀ (a b : ℤ), f ⟨a + b * Real.sqrt 2, sorry⟩ = a * f ⟨1, sorry⟩ + b * f ⟨Real.sqrt 2, sorry⟩ :=
sorry

end cauchy_equation_on_X_l2278_227880


namespace solve_equation_l2278_227812

theorem solve_equation (y : ℚ) : (5 * y + 2) / (6 * y - 3) = 3 / 4 ↔ y = -17 / 2 := by
  sorry

end solve_equation_l2278_227812


namespace equation_natural_solution_l2278_227856

/-- Given an equation C - x = 2b - 2ax where C is a constant,
    a is a real parameter, and b = 7, this theorem states the
    conditions for the equation to have a natural number solution. -/
theorem equation_natural_solution (C : ℝ) (a : ℝ) :
  (∃ x : ℕ, C - x = 2 * 7 - 2 * a * x) ↔ 
  (a > (1 : ℝ) / 2 ∧ ∃ n : ℕ+, 2 * a - 1 = n) :=
sorry

end equation_natural_solution_l2278_227856


namespace no_integer_y_prime_abs_quadratic_l2278_227836

theorem no_integer_y_prime_abs_quadratic : ¬ ∃ y : ℤ, Nat.Prime (Int.natAbs (8*y^2 - 55*y + 21)) := by
  sorry

end no_integer_y_prime_abs_quadratic_l2278_227836


namespace xy_minimum_l2278_227877

theorem xy_minimum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1/2) :
  x * y ≥ 16 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1/2 ∧ x₀ * y₀ = 16 :=
sorry

end xy_minimum_l2278_227877


namespace weight_of_new_person_l2278_227891

theorem weight_of_new_person
  (n : ℕ)
  (initial_weight : ℝ)
  (replaced_weight : ℝ)
  (average_increase : ℝ)
  (h1 : n = 10)
  (h2 : replaced_weight = 70)
  (h3 : average_increase = 4) :
  initial_weight / n + average_increase = (initial_weight - replaced_weight + replaced_weight + n * average_increase) / n :=
by sorry

end weight_of_new_person_l2278_227891


namespace total_cups_on_table_l2278_227846

theorem total_cups_on_table (juice_cups milk_cups : ℕ) 
  (h1 : juice_cups = 3) 
  (h2 : milk_cups = 4) : 
  juice_cups + milk_cups = 7 := by
  sorry

end total_cups_on_table_l2278_227846


namespace sequence_term_equation_l2278_227865

def sequence_term (n : ℕ+) : ℕ := 9 * (n - 1) + n

theorem sequence_term_equation (n : ℕ+) : sequence_term n = 10 * n - 9 := by
  sorry

end sequence_term_equation_l2278_227865


namespace quadratic_inequality_condition_l2278_227895

theorem quadratic_inequality_condition (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m > 0) → m > 0 ∧ ∃ m₀ > 0, ¬(∀ x : ℝ, x^2 - 2*x + m₀ > 0) :=
by sorry

end quadratic_inequality_condition_l2278_227895


namespace num_boolean_structures_l2278_227866

/-- The transformation group of 3 Boolean variables -/
def TransformationGroup : Type := Fin 6

/-- The state configurations for 3 Boolean variables -/
def StateConfigurations : Type := Fin 8

/-- The number of colors (Boolean states) -/
def NumColors : Nat := 2

/-- A permutation on the state configurations -/
def Permutation : Type := StateConfigurations → StateConfigurations

/-- The group of permutations induced by the transformation group -/
def PermutationGroup : Type := TransformationGroup → Permutation

/-- Count the number of cycles in a permutation -/
def cycleCount (p : Permutation) : Nat :=
  sorry

/-- Pólya's Enumeration Theorem for this specific case -/
def polyaEnumeration (G : PermutationGroup) : Nat :=
  sorry

/-- The main theorem: number of different structures for a Boolean function device with 3 variables -/
theorem num_boolean_structures (G : PermutationGroup) : 
  polyaEnumeration G = 80 :=
sorry

end num_boolean_structures_l2278_227866


namespace arithmetic_mean_problem_l2278_227896

theorem arithmetic_mean_problem (y : ℝ) : 
  (7 + y + 22 + 8 + 18) / 5 = 15 → y = 20 := by
sorry

end arithmetic_mean_problem_l2278_227896


namespace prime_sum_divides_power_sum_l2278_227881

theorem prime_sum_divides_power_sum (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
sorry

end prime_sum_divides_power_sum_l2278_227881


namespace malcolm_instagram_followers_l2278_227882

/-- Represents the number of followers on various social media platforms --/
structure SocialMediaFollowers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms --/
def totalFollowers (smf : SocialMediaFollowers) : ℕ :=
  smf.instagram + smf.facebook + smf.twitter + smf.tiktok + smf.youtube

/-- Theorem stating that Malcolm has 240 followers on Instagram --/
theorem malcolm_instagram_followers :
  ∃ (smf : SocialMediaFollowers),
    smf.facebook = 500 ∧
    smf.twitter = (smf.instagram + smf.facebook) / 2 ∧
    smf.tiktok = 3 * smf.twitter ∧
    smf.youtube = smf.tiktok + 510 ∧
    totalFollowers smf = 3840 ∧
    smf.instagram = 240 := by
  sorry

end malcolm_instagram_followers_l2278_227882


namespace inscribed_circle_radius_irrational_l2278_227849

theorem inscribed_circle_radius_irrational (b c : ℕ) : 
  b ≥ 1 → c ≥ 1 → 1 + b > c → 1 + c > b → b + c > 1 → 
  ¬ ∃ (r : ℚ), r = (Real.sqrt ((b : ℝ)^2 - 1/4)) / (1 + 2*(b : ℝ)) := by
sorry

end inscribed_circle_radius_irrational_l2278_227849


namespace worker_c_time_l2278_227827

/-- The time taken by worker c to complete the work alone, given the conditions -/
def time_c (time_abc time_a time_b : ℚ) : ℚ :=
  1 / (1 / time_abc - 1 / time_a - 1 / time_b)

/-- Theorem stating that under given conditions, worker c takes 18 days to finish the work alone -/
theorem worker_c_time (time_abc time_a time_b : ℚ) 
  (h_abc : time_abc = 4)
  (h_a : time_a = 12)
  (h_b : time_b = 9) :
  time_c time_abc time_a time_b = 18 := by
  sorry

#eval time_c 4 12 9

end worker_c_time_l2278_227827


namespace sum_of_sequences_l2278_227899

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n - 1) * d) / 2

def sequence1_sum : ℕ := arithmetic_sum 3 10 6
def sequence2_sum : ℕ := arithmetic_sum 5 10 6

theorem sum_of_sequences : sequence1_sum + sequence2_sum = 348 := by
  sorry

end sum_of_sequences_l2278_227899


namespace candies_remaining_l2278_227858

theorem candies_remaining (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  red = 40 →
  yellow = 3 * red - 20 →
  blue = yellow / 2 →
  red + blue = 90 :=
by
  sorry

end candies_remaining_l2278_227858


namespace remaining_amount_l2278_227843

def initial_amount : ℕ := 20
def peach_quantity : ℕ := 3
def peach_price : ℕ := 2

theorem remaining_amount : 
  initial_amount - (peach_quantity * peach_price) = 14 := by
  sorry

end remaining_amount_l2278_227843


namespace smallest_divisor_with_remainder_one_l2278_227840

theorem smallest_divisor_with_remainder_one (total_boxes : Nat) (h1 : total_boxes = 301) 
  (h2 : total_boxes % 7 = 0) : 
  (∃ x : Nat, x > 0 ∧ total_boxes % x = 1) ∧ 
  (∀ y : Nat, y > 0 ∧ y < 3 → total_boxes % y ≠ 1) := by
  sorry

end smallest_divisor_with_remainder_one_l2278_227840


namespace marathon_heart_beats_l2278_227848

/-- Calculates the number of heart beats during a marathon --/
def marathonHeartBeats (totalDistance : ℕ) (heartRate : ℕ) (firstHalfDistance : ℕ) (firstHalfPace : ℕ) (secondHalfPace : ℕ) : ℕ :=
  let firstHalfTime := firstHalfDistance * firstHalfPace
  let secondHalfTime := (totalDistance - firstHalfDistance) * secondHalfPace
  let totalTime := firstHalfTime + secondHalfTime
  totalTime * heartRate

/-- Theorem: The athlete's heart beats 23100 times during the marathon --/
theorem marathon_heart_beats :
  marathonHeartBeats 30 140 15 6 5 = 23100 := by
  sorry

#eval marathonHeartBeats 30 140 15 6 5

end marathon_heart_beats_l2278_227848


namespace min_y_coordinate_polar_graph_l2278_227852

/-- The minimum y-coordinate of a point on the graph of r = cos(2θ) is -√6/3 -/
theorem min_y_coordinate_polar_graph :
  let r : ℝ → ℝ := λ θ ↦ Real.cos (2 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ y_min : ℝ, y_min = -Real.sqrt 6 / 3 ∧ ∀ θ : ℝ, y θ ≥ y_min :=
by sorry

end min_y_coordinate_polar_graph_l2278_227852


namespace expression_value_l2278_227884

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 5) :
  3 * x - 4 * y + 2 * z = 11 := by
  sorry

end expression_value_l2278_227884


namespace line_position_l2278_227874

structure Line3D where
  -- Assume we have a suitable representation for 3D lines
  -- This is just a placeholder

def skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

def intersects (l1 l2 : Line3D) : Prop :=
  -- Definition of intersecting lines
  sorry

theorem line_position (L1 L2 m1 m2 : Line3D) 
  (h1 : skew L1 L2)
  (h2 : intersects m1 L1)
  (h3 : intersects m1 L2)
  (h4 : intersects m2 L1)
  (h5 : intersects m2 L2) :
  intersects m1 m2 ∨ skew m1 m2 :=
by
  sorry

end line_position_l2278_227874


namespace vasyas_numbers_l2278_227831

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end vasyas_numbers_l2278_227831


namespace trisha_remaining_money_l2278_227819

/-- Calculates the remaining money after shopping given the initial amount and expenses. -/
def remaining_money (initial : ℕ) (meat chicken veggies eggs dog_food : ℕ) : ℕ :=
  initial - (meat + chicken + veggies + eggs + dog_food)

/-- Proves that Trisha's remaining money after shopping is $35. -/
theorem trisha_remaining_money :
  remaining_money 167 17 22 43 5 45 = 35 := by
  sorry

end trisha_remaining_money_l2278_227819


namespace cubic_yards_to_cubic_feet_l2278_227888

-- Define the conversion factor from yards to feet
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def volume_cubic_yards : ℝ := 6

-- Theorem to prove
theorem cubic_yards_to_cubic_feet :
  volume_cubic_yards * (yards_to_feet ^ 3) = 162 := by
  sorry

end cubic_yards_to_cubic_feet_l2278_227888


namespace no_integer_root_seven_l2278_227801

theorem no_integer_root_seven
  (P : Int → Int)  -- P is a polynomial with integer coefficients
  (h_int_coeff : ∀ x, ∃ y, P x = y)  -- P has integer coefficients
  (a b c d : Int)  -- a, b, c, d are integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)  -- a, b, c, d are distinct
  (h_equal_four : P a = 4 ∧ P b = 4 ∧ P c = 4 ∧ P d = 4)  -- P(a) = P(b) = P(c) = P(d) = 4
  : ¬ ∃ e : Int, P e = 7 := by  -- There does not exist an integer e such that P(e) = 7
  sorry

end no_integer_root_seven_l2278_227801


namespace equality_check_l2278_227842

theorem equality_check : 
  (2^3 ≠ 3^2) ∧ 
  (-(-2) = |-2|) ∧ 
  ((-2)^2 ≠ -2^2) ∧ 
  ((2/3)^2 ≠ 2^2/3) := by
  sorry

end equality_check_l2278_227842


namespace quadratic_equation_solution_set_l2278_227816

theorem quadratic_equation_solution_set :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + 2
  {x : ℝ | f x = 0} = {1, 2} := by
sorry

end quadratic_equation_solution_set_l2278_227816


namespace octagon_square_ratio_l2278_227813

theorem octagon_square_ratio (s r : ℝ) (h : s > 0) (k : r > 0) :
  s^2 = 2 * r^2 * Real.sqrt 2 → r / s = 1 / Real.sqrt (2 * Real.sqrt 2) := by
  sorry

end octagon_square_ratio_l2278_227813


namespace tangent_line_y_intercept_l2278_227854

-- Define the function f(x) = x³ - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - a

theorem tangent_line_y_intercept (a : ℝ) :
  (f' a 1 = 1) →  -- Tangent line at x=1 is parallel to x - y - 1 = 0
  (∃ b c : ℝ, ∀ x : ℝ, b * x + c = f a 1 + f' a 1 * (x - 1)) →  -- Equation of tangent line
  (∃ y : ℝ, y = f a 1 + f' a 1 * (0 - 1) ∧ y = -2)  -- y-intercept is -2
  := by sorry

end tangent_line_y_intercept_l2278_227854


namespace hildasAge_l2278_227864

def guesses : List Nat := [25, 29, 31, 33, 37, 39, 42, 45, 48, 50]

def isComposite (n : Nat) : Prop := ∃ a b, a > 1 ∧ b > 1 ∧ a * b = n

def countHighGuesses (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (λ g => g > age)).length

def offByTwo (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (λ g => g = age - 2 ∨ g = age + 2)).length

theorem hildasAge :
  ∃ age : Nat,
    age ∈ guesses ∧
    isComposite age ∧
    countHighGuesses age guesses ≥ guesses.length / 4 ∧
    offByTwo age guesses = 2 ∧
    age = 45 := by sorry

end hildasAge_l2278_227864


namespace correct_remaining_contents_l2278_227875

/-- Represents the contents of a cup with coffee and milk -/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Calculates the remaining contents in the cup after mixing and removing some mixture -/
def remainingContents (initialCoffee : ℚ) (addedMilk : ℚ) (removedMixture : ℚ) : CupContents :=
  let totalVolume := initialCoffee + addedMilk
  let coffeeRatio := initialCoffee / totalVolume
  let milkRatio := addedMilk / totalVolume
  let remainingVolume := totalVolume - removedMixture
  { coffee := coffeeRatio * remainingVolume,
    milk := milkRatio * remainingVolume }

/-- Theorem stating the correct remaining contents after mixing and removing -/
theorem correct_remaining_contents :
  let result := remainingContents 1 (1/4) (1/4)
  result.coffee = 4/5 ∧ result.milk = 1/5 := by
  sorry

end correct_remaining_contents_l2278_227875


namespace min_box_height_l2278_227898

theorem min_box_height (x : ℝ) (h : x > 0) : 
  (10 * x^2 ≥ 150) → (∀ y : ℝ, y > 0 → 10 * y^2 ≥ 150 → y ≥ x) → 2 * x = 2 * Real.sqrt 15 := by
  sorry

end min_box_height_l2278_227898


namespace sum_of_roots_l2278_227878

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x - 6

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = -5) : a + b = 2 := by
  sorry

end sum_of_roots_l2278_227878


namespace max_value_fraction_l2278_227826

theorem max_value_fraction (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (k * x + y)^2 / (x^2 + k * y^2) ≤ k + 1 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (k * x + y)^2 / (x^2 + k * y^2) = k + 1 :=
by sorry

end max_value_fraction_l2278_227826


namespace rhombus_longer_diagonal_l2278_227810

/-- A rhombus with side length 51 and shorter diagonal 48 has a longer diagonal of 90 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 51 → shorter_diag = 48 → longer_diag = 90 → 
  side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2 := by sorry

end rhombus_longer_diagonal_l2278_227810


namespace house_distance_theorem_l2278_227883

/-- Represents the position of a house on a street -/
structure House where
  position : ℝ

/-- Represents a street with four houses -/
structure Street where
  andrey : House
  borya : House
  vova : House
  gleb : House

/-- The distance between two houses -/
def distance (h1 h2 : House) : ℝ := 
  |h1.position - h2.position|

theorem house_distance_theorem (s : Street) : 
  (distance s.andrey s.borya = 600 ∧ 
   distance s.vova s.gleb = 600 ∧ 
   distance s.andrey s.gleb = 3 * distance s.borya s.vova) → 
  (distance s.andrey s.gleb = 900 ∨ distance s.andrey s.gleb = 1800) :=
sorry

end house_distance_theorem_l2278_227883


namespace unique_pairs_satisfying_equation_l2278_227818

theorem unique_pairs_satisfying_equation :
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n + 1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end unique_pairs_satisfying_equation_l2278_227818


namespace sum_in_interval_l2278_227814

theorem sum_in_interval :
  let a := 2 + 3 / 9
  let b := 3 + 3 / 4
  let c := 5 + 3 / 25
  let sum := a + b + c
  8 < sum ∧ sum < 9 := by
sorry

end sum_in_interval_l2278_227814


namespace mikes_books_l2278_227817

/-- Calculates the final number of books Mike has after selling, receiving gifts, and buying books. -/
def final_book_count (initial : ℝ) (sold : ℝ) (gifts : ℝ) (bought : ℝ) : ℝ :=
  initial - sold + gifts + bought

/-- Theorem stating that Mike's final book count is 21.5 given the problem conditions. -/
theorem mikes_books :
  final_book_count 51.5 45.75 12.25 3.5 = 21.5 := by
  sorry

end mikes_books_l2278_227817


namespace classmate_pairs_l2278_227863

theorem classmate_pairs (n : ℕ) (h : n = 6) : (n.choose 2) = 15 := by
  sorry

end classmate_pairs_l2278_227863


namespace imaginary_part_of_complex_fraction_l2278_227833

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 + 2*I) / I
  Complex.im z = -1 :=
by sorry

end imaginary_part_of_complex_fraction_l2278_227833


namespace cubic_roots_l2278_227823

theorem cubic_roots : 
  ∀ x : ℝ, x^3 + 3*x^2 - 6*x - 8 = 0 ↔ x = -1 ∨ x = 2 ∨ x = -4 := by
  sorry

end cubic_roots_l2278_227823


namespace subtraction_properties_l2278_227805

theorem subtraction_properties (a b : ℝ) : 
  ((a - b)^2 = (b - a)^2) ∧ 
  (|a - b| = |b - a|) ∧ 
  (a - b = -b + a) ∧
  ((a - b = b - a) ↔ (a = b)) :=
by sorry

end subtraction_properties_l2278_227805


namespace F_lower_bound_F_max_value_l2278_227894

/-- The condition that x and y satisfy -/
def satisfies_condition (x y : ℝ) : Prop := x^2 + x*y + y^2 = 1

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := x^3*y + x*y^3

/-- Theorem stating that F(x, y) ≥ -2 for any x and y satisfying the condition -/
theorem F_lower_bound {x y : ℝ} (h : satisfies_condition x y) : F x y ≥ -2 := by
  sorry

/-- Theorem stating that the maximum value of F(x, y) is 1/4 -/
theorem F_max_value : ∃ (x y : ℝ), satisfies_condition x y ∧ F x y = 1/4 ∧ ∀ (a b : ℝ), satisfies_condition a b → F a b ≤ 1/4 := by
  sorry

end F_lower_bound_F_max_value_l2278_227894


namespace intersecting_lines_m_value_l2278_227828

/-- Given three lines that intersect at a single point, prove that the value of m is -22/7 -/
theorem intersecting_lines_m_value (x y : ℚ) :
  y = 4 * x - 8 ∧
  y = -3 * x + 9 ∧
  y = 2 * x + m →
  m = -22 / 7 := by
  sorry

end intersecting_lines_m_value_l2278_227828


namespace smaller_ladder_steps_l2278_227868

theorem smaller_ladder_steps 
  (full_ladder_steps : ℕ) 
  (full_ladder_climbs : ℕ) 
  (smaller_ladder_climbs : ℕ) 
  (total_steps : ℕ) 
  (h1 : full_ladder_steps = 11)
  (h2 : full_ladder_climbs = 10)
  (h3 : smaller_ladder_climbs = 7)
  (h4 : total_steps = 152)
  (h5 : full_ladder_steps * full_ladder_climbs + smaller_ladder_climbs * x = total_steps) :
  x = 6 :=
by
  sorry

end smaller_ladder_steps_l2278_227868


namespace cylinder_height_relation_l2278_227887

-- Define the cylinders
def Cylinder (r h : ℝ) := r > 0 ∧ h > 0

-- Theorem statement
theorem cylinder_height_relation 
  (r₁ h₁ r₂ h₂ : ℝ) 
  (cyl₁ : Cylinder r₁ h₁) 
  (cyl₂ : Cylinder r₂ h₂) 
  (volume_eq : r₁^2 * h₁ = r₂^2 * h₂) 
  (radius_relation : r₂ = 1.2 * r₁) : 
  h₁ = 1.44 * h₂ := by
sorry

end cylinder_height_relation_l2278_227887


namespace intersection_of_A_and_B_l2278_227830

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end intersection_of_A_and_B_l2278_227830


namespace shoe_price_after_changes_lous_shoe_price_l2278_227825

/-- The price of shoes after a price increase followed by a discount -/
theorem shoe_price_after_changes (initial_price : ℝ) 
  (increase_percent : ℝ) (discount_percent : ℝ) : ℝ := by
  -- Define the price after increase
  let price_after_increase := initial_price * (1 + increase_percent / 100)
  -- Define the final price after discount
  let final_price := price_after_increase * (1 - discount_percent / 100)
  -- Prove that when initial_price = 40, increase_percent = 10, and discount_percent = 10,
  -- the final_price is 39.60
  sorry

/-- The specific case for Lou's Fine Shoes -/
theorem lous_shoe_price : 
  shoe_price_after_changes 40 10 10 = 39.60 := by sorry

end shoe_price_after_changes_lous_shoe_price_l2278_227825
