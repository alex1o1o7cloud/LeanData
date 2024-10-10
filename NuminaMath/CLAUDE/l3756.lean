import Mathlib

namespace banana_distribution_l3756_375613

theorem banana_distribution (total : Nat) (friends : Nat) (bananas_per_friend : Nat) :
  total = 36 → friends = 5 → bananas_per_friend = 7 →
  total / friends = bananas_per_friend :=
by sorry

end banana_distribution_l3756_375613


namespace circle_with_common_chord_as_diameter_l3756_375679

/-- C₁ is the first given circle -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 4*x + y + 1 = 0

/-- C₂ is the second given circle -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- C is the circle we need to prove -/
def C (x y : ℝ) : Prop := 5*x^2 + 5*y^2 + 6*x + 12*y + 5 = 0

/-- The common chord of C₁ and C₂ -/
def common_chord (x y : ℝ) : Prop := y = 2*x

theorem circle_with_common_chord_as_diameter :
  ∀ x y : ℝ, C x y ↔ 
    (∃ a b : ℝ, C₁ a b ∧ C₂ a b ∧ common_chord a b ∧
      (x - a)^2 + (y - b)^2 = ((x - a) - (b - y))^2 / 4) :=
sorry

end circle_with_common_chord_as_diameter_l3756_375679


namespace fundraiser_percentage_increase_l3756_375638

def fundraiser (initial_rate : ℝ) (total_hours : ℕ) (initial_hours : ℕ) (total_amount : ℝ) : Prop :=
  let remaining_hours := total_hours - initial_hours
  let initial_amount := initial_rate * initial_hours
  let remaining_amount := total_amount - initial_amount
  let new_rate := remaining_amount / remaining_hours
  let percentage_increase := (new_rate - initial_rate) / initial_rate * 100
  percentage_increase = 20

theorem fundraiser_percentage_increase :
  fundraiser 5000 26 12 144000 := by
  sorry

end fundraiser_percentage_increase_l3756_375638


namespace tan_C_value_triangle_area_l3756_375637

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def satisfies_condition_1 (t : Triangle) : Prop :=
  (Real.sin t.A / t.a) + (Real.sin t.B / t.b) = (Real.cos t.C / t.c)

def satisfies_condition_2 (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = 8

-- Theorem 1
theorem tan_C_value (t : Triangle) (h : satisfies_condition_1 t) :
  Real.tan t.C = 1/2 := by sorry

-- Theorem 2
theorem triangle_area (t : Triangle) (h1 : satisfies_condition_1 t) (h2 : satisfies_condition_2 t) :
  (1/2) * t.a * t.b * Real.sin t.C = 1 := by sorry

end tan_C_value_triangle_area_l3756_375637


namespace problem_solution_l3756_375639

theorem problem_solution (x y : ℚ) : 
  x = 152 → 
  x^3*y - 3*x^2*y + 3*x*y = 912000 → 
  y = 3947/15200 := by
sorry

end problem_solution_l3756_375639


namespace mikes_payment_l3756_375614

/-- Calculates Mike's out-of-pocket payment for medical procedures -/
theorem mikes_payment (xray_cost : ℝ) (mri_multiplier : ℝ) (insurance_coverage_percent : ℝ) : 
  xray_cost = 250 →
  mri_multiplier = 3 →
  insurance_coverage_percent = 80 →
  let total_cost := xray_cost + mri_multiplier * xray_cost
  let insurance_coverage := (insurance_coverage_percent / 100) * total_cost
  total_cost - insurance_coverage = 200 := by
sorry


end mikes_payment_l3756_375614


namespace largest_number_with_properties_l3756_375654

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that extracts a two-digit number from adjacent digits in a larger number -/
def twoDigitNumber (n : ℕ) (i : ℕ) : ℕ :=
  (n / 10^i % 100)

/-- A function that checks if all two-digit numbers formed by adjacent digits are prime -/
def allTwoDigitPrime (n : ℕ) : Prop :=
  ∀ i : ℕ, i < (Nat.digits 10 n).length - 1 → isPrime (twoDigitNumber n i)

/-- A function that checks if all two-digit prime numbers formed are distinct -/
def allTwoDigitPrimeDistinct (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j < (Nat.digits 10 n).length - 1 → 
    twoDigitNumber n i ≠ twoDigitNumber n j

/-- The main theorem stating that 617371311979 is the largest number satisfying the conditions -/
theorem largest_number_with_properties :
  (∀ m : ℕ, m > 617371311979 → 
    ¬(allTwoDigitPrime m ∧ allTwoDigitPrimeDistinct m)) ∧
  (allTwoDigitPrime 617371311979 ∧ allTwoDigitPrimeDistinct 617371311979) :=
sorry

end largest_number_with_properties_l3756_375654


namespace prob_two_odd_chips_l3756_375669

-- Define the set of numbers on the chips
def ChipNumbers : Set ℕ := {1, 2, 3, 4}

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Prop := n % 2 = 1

-- Define the probability of drawing an odd-numbered chip from one box
def probOddFromOneBox : ℚ := (2 : ℚ) / 4

-- Theorem statement
theorem prob_two_odd_chips :
  (probOddFromOneBox * probOddFromOneBox) = (1 : ℚ) / 4 :=
sorry

end prob_two_odd_chips_l3756_375669


namespace trigonometric_identities_l3756_375659

theorem trigonometric_identities :
  (Real.cos (15 * π / 180))^2 - (Real.sin (15 * π / 180))^2 = Real.sqrt 3 / 2 ∧
  Real.sin (π / 8) * Real.cos (π / 8) = Real.sqrt 2 / 4 ∧
  Real.tan (15 * π / 180) = 2 - Real.sqrt 3 :=
by sorry

end trigonometric_identities_l3756_375659


namespace smallest_representable_66_88_l3756_375684

/-- Represents a base-b digit --/
def IsDigitBase (d : ℕ) (b : ℕ) : Prop := d < b

/-- Converts a two-digit number in base b to base 10 --/
def BaseToDecimal (d₁ d₂ : ℕ) (b : ℕ) : ℕ := d₁ * b + d₂

/-- States that a number n can be represented as CC₆ and DD₈ --/
def RepresentableAs66And88 (n : ℕ) : Prop :=
  ∃ (c d : ℕ), IsDigitBase c 6 ∧ IsDigitBase d 8 ∧
    n = BaseToDecimal c c 6 ∧ n = BaseToDecimal d d 8

theorem smallest_representable_66_88 :
  (∀ m, RepresentableAs66And88 m → m ≥ 63) ∧ RepresentableAs66And88 63 := by sorry

end smallest_representable_66_88_l3756_375684


namespace total_pupils_l3756_375693

theorem total_pupils (pizza : ℕ) (burgers : ℕ) (both : ℕ) 
  (h1 : pizza = 125) 
  (h2 : burgers = 115) 
  (h3 : both = 40) : 
  pizza + burgers - both = 200 := by
  sorry

end total_pupils_l3756_375693


namespace sector_area_of_ring_l3756_375653

/-- The area of a 60° sector of the ring between two concentric circles with radii 12 and 8 -/
theorem sector_area_of_ring (π : ℝ) : 
  let r₁ : ℝ := 12  -- radius of larger circle
  let r₂ : ℝ := 8   -- radius of smaller circle
  let ring_area : ℝ := π * (r₁^2 - r₂^2)
  let sector_angle : ℝ := 60
  let full_angle : ℝ := 360
  let sector_area : ℝ := (sector_angle / full_angle) * ring_area
  sector_area = (40 * π) / 3 :=
by sorry

end sector_area_of_ring_l3756_375653


namespace range_of_a_minus_b_l3756_375675

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 1) :
  -1 < a - b ∧ a - b < 2 := by
sorry

end range_of_a_minus_b_l3756_375675


namespace prob_win_is_four_sevenths_l3756_375670

/-- The probability of Lola losing a match -/
def prob_lose : ℚ := 3/7

/-- The theorem stating that the probability of Lola winning a match is 4/7 -/
theorem prob_win_is_four_sevenths :
  let prob_win := 1 - prob_lose
  prob_win = 4/7 := by
  sorry

end prob_win_is_four_sevenths_l3756_375670


namespace f_properties_l3756_375608

noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)

theorem f_properties (φ : ℝ) :
  (∀ x, f x φ = f (-x) φ) →  -- f is an even function
  (∀ x ∈ Set.Icc 0 (π / 4), ∀ y ∈ Set.Icc 0 (π / 4), x < y → f x φ < f y φ) →  -- f is increasing in [0, π/4]
  φ = 4 * π / 3 := by
sorry

end f_properties_l3756_375608


namespace time_to_see_again_value_l3756_375667

/-- The time (in seconds) before Jenny and Kenny can see each other again -/
def time_to_see_again (jenny_speed : ℝ) (kenny_speed : ℝ) (path_distance : ℝ) (building_diameter : ℝ) (initial_distance : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time before Jenny and Kenny can see each other again -/
theorem time_to_see_again_value :
  time_to_see_again 2 4 300 150 300 = 48 := by
  sorry

end time_to_see_again_value_l3756_375667


namespace dima_puts_more_berries_l3756_375647

/-- Represents the berry-picking process of Dima and Sergey -/
structure BerryPicking where
  total_berries : ℕ
  dima_basket_rate : ℚ
  sergey_basket_rate : ℚ
  dima_speed : ℚ
  sergey_speed : ℚ

/-- Calculates the difference in berries put in the basket by Dima and Sergey -/
def berry_difference (bp : BerryPicking) : ℕ :=
  sorry

/-- Theorem stating the difference in berries put in the basket -/
theorem dima_puts_more_berries (bp : BerryPicking) 
  (h1 : bp.total_berries = 900)
  (h2 : bp.dima_basket_rate = 1/2)
  (h3 : bp.sergey_basket_rate = 2/3)
  (h4 : bp.dima_speed = 2 * bp.sergey_speed) :
  berry_difference bp = 100 :=
sorry

end dima_puts_more_berries_l3756_375647


namespace defective_units_percentage_l3756_375622

/-- The percentage of defective units that are shipped for sale -/
def defective_shipped_percent : ℝ := 4

/-- The percentage of all units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.2

/-- The percentage of all units that are defective -/
def defective_percent : ℝ := 5

theorem defective_units_percentage : 
  defective_shipped_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end defective_units_percentage_l3756_375622


namespace range_of_f_l3756_375629

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end range_of_f_l3756_375629


namespace petya_cannot_win_l3756_375651

/-- Represents a chess tournament --/
structure ChessTournament where
  players : ℕ
  games_per_player : ℕ
  total_games : ℕ
  last_place_max_points : ℕ

/-- Creates a chess tournament with the given number of players --/
def create_tournament (n : ℕ) : ChessTournament :=
  { players := n
  , games_per_player := n - 1
  , total_games := n * (n - 1) / 2
  , last_place_max_points := (n * (n - 1) / 2) / n }

/-- Theorem: Petya cannot become the winner after disqualification --/
theorem petya_cannot_win (t : ChessTournament) 
  (h1 : t.players = 10) 
  (h2 : t = create_tournament 10) 
  (h3 : t.last_place_max_points ≤ 4) :
  ∃ (remaining_players : ℕ) (remaining_games : ℕ),
    remaining_players = t.players - 1 ∧
    remaining_games = remaining_players * (remaining_players - 1) / 2 ∧
    remaining_games / remaining_players ≥ t.last_place_max_points :=
by sorry

end petya_cannot_win_l3756_375651


namespace total_area_equals_total_frequency_l3756_375611

/-- A frequency distribution histogram -/
structure FrequencyHistogram where
  /-- The list of frequencies for each bin -/
  frequencies : List ℝ
  /-- All frequencies are non-negative -/
  all_nonneg : ∀ f ∈ frequencies, f ≥ 0

/-- The total frequency of a histogram -/
def totalFrequency (h : FrequencyHistogram) : ℝ :=
  h.frequencies.sum

/-- The total area of small rectangles in a histogram -/
def totalArea (h : FrequencyHistogram) : ℝ :=
  h.frequencies.sum

/-- Theorem: The total area of small rectangles in a frequency distribution histogram
    is equal to the total frequency -/
theorem total_area_equals_total_frequency (h : FrequencyHistogram) :
  totalArea h = totalFrequency h := by
  sorry


end total_area_equals_total_frequency_l3756_375611


namespace consecutive_integers_sum_of_squares_l3756_375665

theorem consecutive_integers_sum_of_squares (n : ℕ) : 
  (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2 = 770 → n + 1 = 17 := by
  sorry

end consecutive_integers_sum_of_squares_l3756_375665


namespace min_value_quadratic_function_l3756_375633

theorem min_value_quadratic_function :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x
  ∃ m : ℝ, m = -3 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 1 → f x ≥ m := by
  sorry

end min_value_quadratic_function_l3756_375633


namespace solutions_of_x_squared_equals_x_l3756_375698

theorem solutions_of_x_squared_equals_x : 
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end solutions_of_x_squared_equals_x_l3756_375698


namespace integer_2020_in_column_F_l3756_375620

/-- Represents the columns in the arrangement --/
inductive Column
  | A | B | C | D | E | F | G

/-- Defines the arrangement of integers in columns --/
def arrangement (n : ℕ) : Column :=
  match (n - 11) % 14 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.E
  | 5 => Column.F
  | 6 => Column.G
  | 7 => Column.G
  | 8 => Column.F
  | 9 => Column.E
  | 10 => Column.D
  | 11 => Column.C
  | 12 => Column.B
  | _ => Column.A

/-- Theorem: The integer 2020 is in column F --/
theorem integer_2020_in_column_F : arrangement 2020 = Column.F := by
  sorry

end integer_2020_in_column_F_l3756_375620


namespace isosceles_triangle_rectangle_equal_area_l3756_375643

/-- Given an isosceles triangle with base b and height h, and a rectangle with base b and height 2b,
    if their areas are equal, then the height of the triangle is 4 times the base. -/
theorem isosceles_triangle_rectangle_equal_area (b h : ℝ) (b_pos : 0 < b) :
  (1 / 2 : ℝ) * b * h = b * (2 * b) → h = 4 * b := by
  sorry

end isosceles_triangle_rectangle_equal_area_l3756_375643


namespace plane_centroid_sum_l3756_375680

-- Define the plane and points
def Plane := {plane : ℝ → ℝ → ℝ → Prop | ∃ (a b c : ℝ), ∀ x y z, plane x y z ↔ (x / a + y / b + z / c = 1)}

def distance_from_origin (plane : Plane) : ℝ := sorry

def intersect_x_axis (plane : Plane) : ℝ := sorry
def intersect_y_axis (plane : Plane) : ℝ := sorry
def intersect_z_axis (plane : Plane) : ℝ := sorry

def centroid (a b c : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

-- Theorem statement
theorem plane_centroid_sum (plane : Plane) :
  let a := (intersect_x_axis plane, 0, 0)
  let b := (0, intersect_y_axis plane, 0)
  let c := (0, 0, intersect_z_axis plane)
  let (p, q, r) := centroid a b c
  distance_from_origin plane = Real.sqrt 2 →
  a ≠ (0, 0, 0) ∧ b ≠ (0, 0, 0) ∧ c ≠ (0, 0, 0) →
  1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 2 := by
  sorry

end plane_centroid_sum_l3756_375680


namespace alex_lorin_marble_ratio_l3756_375601

/-- Given the following conditions:
  - Lorin has 4 black marbles
  - Jimmy has 22 yellow marbles
  - Alex has a certain ratio of black marbles as Lorin
  - Alex has one half as many yellow marbles as Jimmy
  - Alex has 19 marbles in total

  Prove that the ratio of Alex's black marbles to Lorin's black marbles is 2:1
-/
theorem alex_lorin_marble_ratio :
  ∀ (alex_black alex_yellow : ℕ),
  let lorin_black : ℕ := 4
  let jimmy_yellow : ℕ := 22
  let alex_total : ℕ := 19
  alex_yellow = jimmy_yellow / 2 →
  alex_black + alex_yellow = alex_total →
  ∃ (r : ℚ),
    alex_black = r * lorin_black ∧
    r = 2 := by
  sorry

#check alex_lorin_marble_ratio

end alex_lorin_marble_ratio_l3756_375601


namespace solution_set_of_decreasing_function_l3756_375621

/-- A decreasing function on ℝ -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem solution_set_of_decreasing_function
  (f : ℝ → ℝ) (h_decreasing : DecreasingFunction f) (h_f_1 : f 1 = 0) :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | x < 2} := by sorry

end solution_set_of_decreasing_function_l3756_375621


namespace dog_age_is_twelve_l3756_375612

def cat_age : ℕ := 8

def rabbit_age (cat_age : ℕ) : ℕ := cat_age / 2

def dog_age (rabbit_age : ℕ) : ℕ := 3 * rabbit_age

theorem dog_age_is_twelve : dog_age (rabbit_age cat_age) = 12 := by
  sorry

end dog_age_is_twelve_l3756_375612


namespace base_2_representation_315_l3756_375689

/-- Given a natural number n, returns the number of zeros in its binary representation -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- Given a natural number n, returns the number of ones in its binary representation -/
def count_ones (n : ℕ) : ℕ := sorry

theorem base_2_representation_315 : 
  let x := count_zeros 315
  let y := count_ones 315
  y - x = 5 := by sorry

end base_2_representation_315_l3756_375689


namespace circle_plus_minus_balance_l3756_375627

theorem circle_plus_minus_balance (a b p q : ℕ) : a - b = p - q :=
  sorry

end circle_plus_minus_balance_l3756_375627


namespace total_peaches_l3756_375655

/-- The number of peaches initially in the basket -/
def initial_peaches : ℕ := 20

/-- The number of peaches added to the basket -/
def added_peaches : ℕ := 25

/-- Theorem stating the total number of peaches after addition -/
theorem total_peaches : initial_peaches + added_peaches = 45 := by
  sorry

end total_peaches_l3756_375655


namespace fraction_equality_l3756_375657

theorem fraction_equality : (24 + 12) / ((5 - 3) * 2) = 9 := by
  sorry

end fraction_equality_l3756_375657


namespace sum_of_roots_l3756_375619

theorem sum_of_roots (h b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂)
  (h2 : 3 * x₁^2 - h * x₁ = b)
  (h3 : 3 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 3 := by
sorry

end sum_of_roots_l3756_375619


namespace impossible_perpendicular_l3756_375662

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)

-- Define the theorem
theorem impossible_perpendicular 
  (α : Plane) (a b : Line) (P : Point)
  (h1 : perpendicular a α)
  (h2 : intersect a b P) :
  ¬ (perpendicular b α) := by
  sorry

end impossible_perpendicular_l3756_375662


namespace cylinder_sphere_cone_volume_ratio_l3756_375683

/-- Given a cylinder with volume 128π cm³, the ratio of the volume of a sphere 
(with radius equal to the base radius of the cylinder) to the volume of a cone 
(with the same radius and height as the cylinder) is 2. -/
theorem cylinder_sphere_cone_volume_ratio : 
  ∀ (r h : ℝ), 
  r > 0 → h > 0 →
  π * r^2 * h = 128 * π →
  (4/3 * π * r^3) / (1/3 * π * r^2 * h) = 2 := by
sorry

end cylinder_sphere_cone_volume_ratio_l3756_375683


namespace smaller_number_problem_l3756_375649

theorem smaller_number_problem (x y : ℝ) : 
  y = 3 * x ∧ x + y = 124 → x = 31 := by
  sorry

end smaller_number_problem_l3756_375649


namespace shelby_total_stars_l3756_375699

/-- The number of gold stars Shelby earned yesterday -/
def stars_yesterday : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def stars_today : ℕ := 3

/-- The total number of gold stars Shelby earned -/
def total_stars : ℕ := stars_yesterday + stars_today

/-- Theorem stating that the total number of gold stars Shelby earned is 7 -/
theorem shelby_total_stars : total_stars = 7 := by sorry

end shelby_total_stars_l3756_375699


namespace exists_valid_pet_counts_unique_valid_pet_counts_total_pets_is_nineteen_l3756_375685

/-- Represents the number of pets Frankie has of each type -/
structure PetCounts where
  dogs : Nat
  cats : Nat
  snakes : Nat
  parrots : Nat

/-- Defines the conditions given in the problem -/
def validPetCounts (p : PetCounts) : Prop :=
  p.dogs = 2 ∧
  p.snakes > p.cats ∧
  p.parrots = p.cats - 1 ∧
  p.dogs + p.cats = 6 ∧
  p.dogs + p.cats + p.snakes + p.parrots = 19

/-- Theorem stating that there exists a valid pet count configuration -/
theorem exists_valid_pet_counts : ∃ p : PetCounts, validPetCounts p :=
  sorry

/-- Theorem proving the uniqueness of the valid pet count configuration -/
theorem unique_valid_pet_counts (p q : PetCounts) 
  (hp : validPetCounts p) (hq : validPetCounts q) : p = q :=
  sorry

/-- Main theorem proving that the total number of pets is 19 -/
theorem total_pets_is_nineteen (p : PetCounts) (h : validPetCounts p) :
  p.dogs + p.cats + p.snakes + p.parrots = 19 :=
  sorry

end exists_valid_pet_counts_unique_valid_pet_counts_total_pets_is_nineteen_l3756_375685


namespace triangle_4_4_7_l3756_375676

/-- A triangle can be formed from three line segments if the sum of any two sides
    is greater than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 4, 4, and 7 can form a triangle. -/
theorem triangle_4_4_7 :
  can_form_triangle 4 4 7 := by
  sorry

end triangle_4_4_7_l3756_375676


namespace average_weight_of_all_boys_l3756_375635

/-- Given two groups of boys with known average weights, 
    calculate the average weight of all boys. -/
theorem average_weight_of_all_boys 
  (group1_count : ℕ) 
  (group1_avg_weight : ℝ) 
  (group2_count : ℕ) 
  (group2_avg_weight : ℝ) 
  (h1 : group1_count = 24)
  (h2 : group1_avg_weight = 50.25)
  (h3 : group2_count = 8)
  (h4 : group2_avg_weight = 45.15) :
  (group1_count * group1_avg_weight + group2_count * group2_avg_weight) / 
  (group1_count + group2_count) = 48.975 := by
sorry

#eval (24 * 50.25 + 8 * 45.15) / (24 + 8)

end average_weight_of_all_boys_l3756_375635


namespace triangle_sine_theorem_l3756_375615

theorem triangle_sine_theorem (D E F : ℝ) (area : ℝ) (geo_mean : ℝ) :
  area = 81 →
  geo_mean = 15 →
  geo_mean^2 = D * F →
  area = 1/2 * D * F * Real.sin E →
  Real.sin E = 18/25 := by
  sorry

end triangle_sine_theorem_l3756_375615


namespace equal_roots_quadratic_l3756_375636

/-- Given a quadratic equation (k-1)x^2 + 6x + 9 = 0 with two equal real roots, prove that k = 2 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + 9 = 0 ∧ 
   ∀ y : ℝ, (k - 1) * y^2 + 6 * y + 9 = 0 → y = x) → 
  k = 2 := by
  sorry

end equal_roots_quadratic_l3756_375636


namespace angle_bisector_theorem_l3756_375690

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  let d_on_ab := ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.D = k • t.A + (1 - k) • t.B
  let ad_bisects_bac := ∃ k : ℝ, k > 0 ∧ k • (t.C - t.A) = t.D - t.A
  let bd_length := dist t.B t.D = 36
  let bc_length := dist t.B t.C = 45
  let ac_length := dist t.A t.C = 40
  d_on_ab ∧ ad_bisects_bac ∧ bd_length ∧ bc_length ∧ ac_length

-- State the theorem
theorem angle_bisector_theorem (t : Triangle) :
  satisfies_conditions t → dist t.A t.D = 68 :=
sorry

end angle_bisector_theorem_l3756_375690


namespace f_equiv_g_l3756_375618

/-- Function f defined as f(x) = x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Function g defined as g(t) = t^2 - 2t + 1 -/
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

/-- Theorem stating that f and g are equivalent functions -/
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by sorry

end f_equiv_g_l3756_375618


namespace cost_price_approximation_l3756_375691

/-- The cost price of a single toy given the selling conditions -/
def cost_price_of_toy (num_toys : ℕ) (total_selling_price : ℚ) (gain_in_toys : ℕ) : ℚ :=
  let selling_price_per_toy := total_selling_price / num_toys
  let x := selling_price_per_toy / (1 + gain_in_toys / num_toys)
  x

/-- Theorem stating the cost price of a toy given the problem conditions -/
theorem cost_price_approximation :
  let result := cost_price_of_toy 18 23100 3
  (result > 1099.99) ∧ (result < 1100.01) := by
  sorry

#eval cost_price_of_toy 18 23100 3

end cost_price_approximation_l3756_375691


namespace trig_identity_l3756_375604

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 
  1 / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) := by sorry

end trig_identity_l3756_375604


namespace all_zero_function_l3756_375686

-- Define the type of our function
def IntFunction := Nat → Nat

-- Define the conditions
def satisfiesConditions (f : IntFunction) : Prop :=
  (∀ m n : Nat, m > 0 ∧ n > 0 → f (m * n) = f m + f n) ∧
  (f 2008 = 0) ∧
  (∀ n : Nat, n > 0 ∧ n % 2008 = 39 → f n = 0)

-- State the theorem
theorem all_zero_function (f : IntFunction) :
  satisfiesConditions f → ∀ n : Nat, n > 0 → f n = 0 :=
by
  sorry


end all_zero_function_l3756_375686


namespace roberts_cash_amount_l3756_375644

theorem roberts_cash_amount (raw_materials_cost machinery_cost : ℝ) 
  (h1 : raw_materials_cost = 100)
  (h2 : machinery_cost = 125)
  (total_amount : ℝ) :
  raw_materials_cost + machinery_cost + 0.1 * total_amount = total_amount →
  total_amount = 250 := by
sorry

end roberts_cash_amount_l3756_375644


namespace decreasing_function_inequality_l3756_375681

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_inequality
  (f : ℝ → ℝ) (h_dec : is_decreasing f) (m n : ℝ)
  (h_ineq : f m - f n > f (-m) - f (-n)) :
  m - n < 0 :=
sorry

end decreasing_function_inequality_l3756_375681


namespace complex_arithmetic_equality_l3756_375617

theorem complex_arithmetic_equality : 
  |(-3) - (-5)| + ((-1/2 : ℚ)^3) / (1/4 : ℚ) * 2 - 6 * ((1/3 : ℚ) - (1/2 : ℚ)) = 2 := by
  sorry

end complex_arithmetic_equality_l3756_375617


namespace inequality_proof_l3756_375631

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / (x + 2*y + 3*z)) + (y / (y + 2*z + 3*x)) + (z / (z + 2*x + 3*y)) ≤ 4/3 := by
  sorry

end inequality_proof_l3756_375631


namespace triangle_sides_ratio_bound_l3756_375603

theorem triangle_sides_ratio_bound (a b c : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →  -- Positive sides
  (a + b > c) → (a + c > b) → (b + c > a) →  -- Triangle inequality
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →  -- Arithmetic progression
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≥ 1 := by
  sorry

#check triangle_sides_ratio_bound

end triangle_sides_ratio_bound_l3756_375603


namespace zero_of_f_l3756_375660

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- Theorem: -1 is the zero of the function f(x) = (x+1)^2 -/
theorem zero_of_f : f (-1) = 0 := by
  sorry

end zero_of_f_l3756_375660


namespace three_liters_to_pints_l3756_375674

-- Define the conversion rate from liters to pints
def liters_to_pints (liters : ℝ) : ℝ := 2.16 * liters

-- Theorem statement
theorem three_liters_to_pints : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, 
  |x - 3| < δ → |liters_to_pints x - 6.5| < ε :=
sorry

end three_liters_to_pints_l3756_375674


namespace opposite_corners_not_tileable_different_color_cells_tileable_l3756_375634

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (removed : List (Nat × Nat))

/-- Represents a domino -/
inductive Domino
  | horizontal : Nat → Nat → Domino
  | vertical : Nat → Nat → Domino

/-- A tiling of a chessboard with dominoes -/
def Tiling := List Domino

/-- Returns true if the given coordinates represent a black square on the chessboard -/
def isBlack (x y : Nat) : Bool :=
  (x + y) % 2 = 0

/-- Returns true if the two given cells have different colors -/
def differentColors (x1 y1 x2 y2 : Nat) : Bool :=
  isBlack x1 y1 ≠ isBlack x2 y2

/-- Returns true if the given tiling is valid for the given chessboard -/
def isValidTiling (board : Chessboard) (tiling : Tiling) : Bool :=
  sorry

theorem opposite_corners_not_tileable :
  ∀ (board : Chessboard),
    board.size = 8 →
    board.removed = [(0, 0), (7, 7)] →
    ¬∃ (tiling : Tiling), isValidTiling board tiling :=
  sorry

theorem different_color_cells_tileable :
  ∀ (board : Chessboard) (x1 y1 x2 y2 : Nat),
    board.size = 8 →
    x1 < 8 ∧ y1 < 8 ∧ x2 < 8 ∧ y2 < 8 →
    differentColors x1 y1 x2 y2 →
    board.removed = [(x1, y1), (x2, y2)] →
    ∃ (tiling : Tiling), isValidTiling board tiling :=
  sorry

end opposite_corners_not_tileable_different_color_cells_tileable_l3756_375634


namespace football_draws_l3756_375656

/-- Represents the possible outcomes of a football match -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Calculate points for a given match result -/
def pointsForResult (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 3
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents the results of a series of matches -/
structure MatchResults :=
  (wins : Nat)
  (draws : Nat)
  (losses : Nat)

/-- Calculate total points for a series of matches -/
def totalPoints (results : MatchResults) : Nat :=
  3 * results.wins + results.draws

/-- The main theorem to prove -/
theorem football_draws (results : MatchResults) :
  results.wins + results.draws + results.losses = 5 →
  totalPoints results = 7 →
  results.draws = 1 ∨ results.draws = 4 := by
  sorry


end football_draws_l3756_375656


namespace sum_and_ratio_problem_l3756_375692

theorem sum_and_ratio_problem (x y : ℚ) 
  (sum_eq : x + y = 520)
  (ratio_eq : x / y = 3 / 4) :
  y - x = 520 / 7 := by
sorry

end sum_and_ratio_problem_l3756_375692


namespace probability_adjacent_points_hexagon_l3756_375695

/-- The number of points on the regular hexagon -/
def num_points : ℕ := 6

/-- The number of adjacent pairs on the regular hexagon -/
def num_adjacent_pairs : ℕ := 6

/-- The probability of selecting two adjacent points on a regular hexagon -/
theorem probability_adjacent_points_hexagon : 
  (num_adjacent_pairs : ℚ) / (num_points.choose 2) = 2 / 5 := by
  sorry

end probability_adjacent_points_hexagon_l3756_375695


namespace check_amount_proof_l3756_375623

theorem check_amount_proof (C : ℝ) 
  (tip_percentage : ℝ) 
  (tip_contribution : ℝ) : 
  tip_percentage = 0.20 → 
  tip_contribution = 40 → 
  tip_percentage * C = tip_contribution → 
  C = 200 := by
sorry

end check_amount_proof_l3756_375623


namespace evaluate_expression_l3756_375696

theorem evaluate_expression (b : ℚ) (h : b = 4/3) :
  (6*b^2 - 17*b + 8) * (3*b - 4) = 0 := by
  sorry

end evaluate_expression_l3756_375696


namespace debby_water_bottles_l3756_375616

/-- The number of water bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasted

theorem debby_water_bottles : total_bottles = 8066 := by
  sorry

end debby_water_bottles_l3756_375616


namespace counterexample_exists_l3756_375641

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ -3 * a ≥ -3 * b := by sorry

end counterexample_exists_l3756_375641


namespace quintons_fruit_trees_l3756_375666

/-- Represents the width of an apple tree in feet -/
def apple_width : ℕ := 10

/-- Represents the space needed between apple trees in feet -/
def apple_space : ℕ := 12

/-- Represents the width of a peach tree in feet -/
def peach_width : ℕ := 12

/-- Represents the space needed between peach trees in feet -/
def peach_space : ℕ := 15

/-- Represents the total space available for all trees in feet -/
def total_space : ℕ := 71

/-- Calculates the total number of fruit trees Quinton can plant -/
def total_fruit_trees : ℕ :=
  let apple_trees := 2
  let apple_total_space := apple_trees * apple_width + (apple_trees - 1) * apple_space
  let peach_space_left := total_space - apple_total_space
  let peach_trees := 1 + (peach_space_left - peach_width) / (peach_width + peach_space)
  apple_trees + peach_trees

theorem quintons_fruit_trees :
  total_fruit_trees = 4 := by
  sorry

end quintons_fruit_trees_l3756_375666


namespace total_stickers_l3756_375697

def initial_stickers : Float := 20.0
def bought_stickers : Float := 26.0
def birthday_stickers : Float := 20.0
def sister_gift : Float := 6.0
def mother_gift : Float := 58.0

theorem total_stickers : 
  initial_stickers + bought_stickers + birthday_stickers + sister_gift + mother_gift = 130.0 := by
  sorry

end total_stickers_l3756_375697


namespace quadratic_inequality_solution_l3756_375640

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 2 < 0 ↔ 1/3 < x ∧ x < 1/2) →
  a + b = 2 := by
sorry

end quadratic_inequality_solution_l3756_375640


namespace locus_of_P_is_ellipse_l3756_375646

-- Define the circle F₁
def circle_F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 36

-- Define the fixed point F₂
def F₂ : ℝ × ℝ := (2, 0)

-- Define a point on the circle F₁
def point_on_F₁ (A : ℝ × ℝ) : Prop := circle_F₁ A.1 A.2

-- Define the center of F₁
def F₁ : ℝ × ℝ := (-2, 0)

-- Define the perpendicular bisector of F₂A
def perp_bisector (A : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - F₂.1) * (A.1 - F₂.1) + (P.2 - F₂.2) * (A.2 - F₂.2) = 0 ∧
  (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = (P.1 - A.1)^2 + (P.2 - A.2)^2

-- Define P as the intersection of perpendicular bisector and radius F₁A
def point_P (A : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  perp_bisector A P ∧
  ∃ (t : ℝ), P = (F₁.1 + t * (A.1 - F₁.1), F₁.2 + t * (A.2 - F₁.2))

-- Theorem: The locus of P is an ellipse with equation x²/9 + y²/5 = 1
theorem locus_of_P_is_ellipse :
  ∀ (P : ℝ × ℝ), (∃ (A : ℝ × ℝ), point_on_F₁ A ∧ point_P A P) ↔ 
  P.1^2 / 9 + P.2^2 / 5 = 1 :=
sorry

end locus_of_P_is_ellipse_l3756_375646


namespace isosceles_triangle_vertex_angle_l3756_375661

-- Define an isosceles triangle
structure IsoscelesTriangle where
  baseAngle : ℝ
  vertexAngle : ℝ
  isIsosceles : baseAngle ≥ 0 ∧ vertexAngle ≥ 0
  angleSum : baseAngle + baseAngle + vertexAngle = 180

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.baseAngle = 80) : 
  triangle.vertexAngle = 20 :=
by
  sorry

#check isosceles_triangle_vertex_angle

end isosceles_triangle_vertex_angle_l3756_375661


namespace inverse_of_A_l3756_375688

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; 2, 5]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![5/22, 1/22; -1/11, 2/11]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l3756_375688


namespace doughnuts_per_box_l3756_375648

/-- Given the total number of doughnuts made, the number of boxes sold, and the number of doughnuts
given away, prove that the number of doughnuts in each box is equal to
(total doughnuts made - doughnuts given away) divided by the number of boxes sold. -/
theorem doughnuts_per_box
  (total_doughnuts : ℕ)
  (boxes_sold : ℕ)
  (doughnuts_given_away : ℕ)
  (h1 : total_doughnuts ≥ doughnuts_given_away)
  (h2 : boxes_sold > 0)
  (h3 : total_doughnuts - doughnuts_given_away = boxes_sold * (total_doughnuts - doughnuts_given_away) / boxes_sold) :
  (total_doughnuts - doughnuts_given_away) / boxes_sold =
  (total_doughnuts - doughnuts_given_away) / boxes_sold :=
by sorry

end doughnuts_per_box_l3756_375648


namespace storm_deposit_calculation_l3756_375642

-- Define the reservoir capacity and initial content
def reservoir_capacity : ℝ := 400000000000
def initial_content : ℝ := 220000000000

-- Define the initial and final fill percentages
def initial_fill_percentage : ℝ := 0.5500000000000001
def final_fill_percentage : ℝ := 0.85

-- Define the amount of water deposited by the storm
def storm_deposit : ℝ := 120000000000

-- Theorem statement
theorem storm_deposit_calculation :
  initial_content = initial_fill_percentage * reservoir_capacity ∧
  storm_deposit = final_fill_percentage * reservoir_capacity - initial_content :=
by sorry

end storm_deposit_calculation_l3756_375642


namespace no_triangle_with_geometric_angles_l3756_375628

theorem no_triangle_with_geometric_angles : ¬∃ (a r : ℕ), 
  a ≥ 10 ∧ 
  a < a * r ∧ 
  a * r < a * r * r ∧ 
  a + a * r + a * r * r = 180 := by
  sorry

end no_triangle_with_geometric_angles_l3756_375628


namespace function_properties_a_value_l3756_375677

noncomputable section

-- Define the natural exponential function
def exp (x : ℝ) := Real.exp x

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * exp x - a - x) * exp x

theorem function_properties (h : ∀ x : ℝ, f 1 x ≥ 0) :
  (∃! x₀ : ℝ, ∀ x : ℝ, f 1 x ≤ f 1 x₀) ∧
  (∃ x₀ : ℝ, ∀ x : ℝ, f 1 x ≤ f 1 x₀ ∧ 0 < f 1 x₀ ∧ f 1 x₀ < 1/4) :=
sorry

theorem a_value (h : ∀ a : ℝ, a ≥ 0 → ∀ x : ℝ, f a x ≥ 0) :
  ∃! a : ℝ, a = 1 ∧ ∀ x : ℝ, f a x ≥ 0 :=
sorry

end function_properties_a_value_l3756_375677


namespace inequality_proof_l3756_375694

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^4 + y^4 + 2 / (x^2 * y^2) ≥ 4 := by
  sorry

end inequality_proof_l3756_375694


namespace quadratic_equation_two_distinct_roots_l3756_375624

theorem quadratic_equation_two_distinct_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 - m * x₁ - 1 = 0 ∧ 2 * x₂^2 - m * x₂ - 1 = 0 := by
  sorry

end quadratic_equation_two_distinct_roots_l3756_375624


namespace max_value_ad_minus_bc_l3756_375607

theorem max_value_ad_minus_bc :
  ∃ (a b c d : ℤ),
    a ∈ ({-1, 1, 2} : Set ℤ) ∧
    b ∈ ({-1, 1, 2} : Set ℤ) ∧
    c ∈ ({-1, 1, 2} : Set ℤ) ∧
    d ∈ ({-1, 1, 2} : Set ℤ) ∧
    a * d - b * c = 6 ∧
    ∀ (x y z w : ℤ),
      x ∈ ({-1, 1, 2} : Set ℤ) →
      y ∈ ({-1, 1, 2} : Set ℤ) →
      z ∈ ({-1, 1, 2} : Set ℤ) →
      w ∈ ({-1, 1, 2} : Set ℤ) →
      x * w - y * z ≤ 6 :=
by
  sorry

end max_value_ad_minus_bc_l3756_375607


namespace cos_fifteen_squared_formula_l3756_375650

theorem cos_fifteen_squared_formula : 2 * (Real.cos (15 * π / 180))^2 - 1 = Real.sqrt 3 / 2 := by
  sorry

end cos_fifteen_squared_formula_l3756_375650


namespace three_letter_initials_count_l3756_375605

theorem three_letter_initials_count (n : ℕ) (h : n = 7) : n^3 = 343 := by
  sorry

end three_letter_initials_count_l3756_375605


namespace sin_inequality_equivalence_l3756_375632

theorem sin_inequality_equivalence (a b : ℝ) :
  (∀ x : ℝ, Real.sin x + Real.sin a ≥ b * Real.cos x) ↔
  (∃ n : ℤ, a = (4 * n + 1) * Real.pi / 2) ∧ (b = 0) := by
  sorry

end sin_inequality_equivalence_l3756_375632


namespace oliver_ferris_wheel_rides_l3756_375602

/-- The number of times Oliver rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride (ferris wheel or bumper car) -/
def ticket_cost : ℕ := 3

/-- The total number of tickets Oliver used -/
def total_tickets : ℕ := 30

/-- The number of times Oliver rode the ferris wheel -/
def ferris_wheel_rides : ℕ := (total_tickets - bumper_rides * ticket_cost) / ticket_cost

theorem oliver_ferris_wheel_rides :
  ferris_wheel_rides = 7 := by sorry

end oliver_ferris_wheel_rides_l3756_375602


namespace stating_rabbit_distribution_count_l3756_375672

/-- Represents the number of pet stores --/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits --/
def num_parents : ℕ := 2

/-- Represents the number of offspring rabbits --/
def num_offspring : ℕ := 4

/-- Represents the total number of rabbits --/
def total_rabbits : ℕ := num_parents + num_offspring

/-- 
  Calculates the number of ways to distribute rabbits to pet stores
  such that no store has both a parent and a child
--/
def distribute_rabbits : ℕ :=
  -- Definition of the function to calculate the number of ways
  -- This is left undefined as the actual implementation is not provided
  sorry

/-- 
  Theorem stating that the number of ways to distribute the rabbits
  is equal to 560
--/
theorem rabbit_distribution_count : distribute_rabbits = 560 := by
  sorry

end stating_rabbit_distribution_count_l3756_375672


namespace additional_songs_count_l3756_375668

def original_songs : ℕ := 25
def song_duration : ℕ := 3
def total_duration : ℕ := 105

theorem additional_songs_count :
  (total_duration - original_songs * song_duration) / song_duration = 10 := by
  sorry

end additional_songs_count_l3756_375668


namespace cookie_bags_l3756_375687

theorem cookie_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 33) (h2 : cookies_per_bag = 11) :
  total_cookies / cookies_per_bag = 3 := by
  sorry

end cookie_bags_l3756_375687


namespace expression_simplification_value_at_three_value_at_four_l3756_375606

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - 2*x + 1)) = (x - 1) / (x + 2) := by
  sorry

theorem value_at_three :
  (1 - 1 / (3 - 1)) / ((3^2 - 4) / (3^2 - 2*3 + 1)) = 2 / 5 := by
  sorry

theorem value_at_four :
  (1 - 1 / (4 - 1)) / ((4^2 - 4) / (4^2 - 2*4 + 1)) = 1 / 2 := by
  sorry

end expression_simplification_value_at_three_value_at_four_l3756_375606


namespace nonagon_diagonals_l3756_375673

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A nonagon (nine-sided polygon) has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l3756_375673


namespace fifth_term_value_l3756_375652

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem: If S₆ = 12 and a₂ = 5 in an arithmetic sequence, then a₅ = -1 -/
theorem fifth_term_value (seq : ArithmeticSequence) 
  (sum_6 : seq.S 6 = 12) 
  (second_term : seq.a 2 = 5) : 
  seq.a 5 = -1 := by
  sorry

end fifth_term_value_l3756_375652


namespace complex_modulus_l3756_375678

theorem complex_modulus (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end complex_modulus_l3756_375678


namespace trapezoid_area_maximization_l3756_375610

/-- Given a triangle ABC with sides a, b, c, altitude h, and a point G on the altitude
    at distance x from A, the area of the trapezoid formed by drawing a line parallel
    to the base through G and extending the sides is maximized when
    x = ((b + c) * h) / (2 * (a + b + c)). -/
theorem trapezoid_area_maximization (a b c h x : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧ x > 0 ∧ x < h →
  let t := (1/2) * (a + ((a + b + c) * x / h)) * (h - x)
  ∃ (max_x : ℝ), max_x = ((b + c) * h) / (2 * (a + b + c)) ∧
    ∀ y, 0 < y ∧ y < h → t ≤ (1/2) * (a + ((a + b + c) * max_x / h)) * (h - max_x) :=
by sorry

end trapezoid_area_maximization_l3756_375610


namespace intersection_A_B_l3756_375609

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end intersection_A_B_l3756_375609


namespace value_of_c_l3756_375645

theorem value_of_c (a b c : ℝ) : 
  8 = (4 / 100) * a →
  4 = (8 / 100) * b →
  c = b / a →
  c = 0.25 := by
sorry

end value_of_c_l3756_375645


namespace tangent_line_equation_l3756_375625

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x

-- State the theorem
theorem tangent_line_equation :
  (∀ x, f (x / 2) = x^3 - 3 * x) →
  ∃ m b, m * 1 - f 1 + b = 0 ∧
         ∀ x, m * x - f x + b = 0 → x = 1 ∧
         m = 18 ∧ b = -16 :=
by sorry

end tangent_line_equation_l3756_375625


namespace parabola_directrix_l3756_375682

/-- The directrix of a parabola with equation y = (1/4)x^2 -/
def directrix_of_parabola (x y : ℝ) : Prop :=
  y = (1/4) * x^2 → y = -1

theorem parabola_directrix : 
  ∀ x y : ℝ, directrix_of_parabola x y :=
by sorry

end parabola_directrix_l3756_375682


namespace gcd_of_45_and_75_l3756_375630

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_of_45_and_75_l3756_375630


namespace line_through_point_l3756_375663

/-- 
Given a line with equation 3ax + (2a+1)y = 3a+3 that passes through the point (3, -9),
prove that a = -1.
-/
theorem line_through_point (a : ℝ) : 
  (3 * a * 3 + (2 * a + 1) * (-9) = 3 * a + 3) → a = -1 := by
sorry

end line_through_point_l3756_375663


namespace pyramid_solution_l3756_375664

/-- Represents the structure of the number pyramid --/
structure NumberPyramid where
  row2_1 : ℕ
  row2_2 : ℕ → ℕ
  row2_3 : ℕ → ℕ
  row3_1 : ℕ → ℕ
  row3_2 : ℕ → ℕ
  row4   : ℕ → ℕ

/-- The specific number pyramid instance from the problem --/
def problemPyramid : NumberPyramid := {
  row2_1 := 11
  row2_2 := λ x => 6 + x
  row2_3 := λ x => x + 7
  row3_1 := λ x => 11 + (6 + x)
  row3_2 := λ x => (6 + x) + (x + 7)
  row4   := λ x => (11 + (6 + x)) + ((6 + x) + (x + 7))
}

/-- The theorem stating that x = 10 in this specific number pyramid --/
theorem pyramid_solution :
  ∃ x : ℕ, problemPyramid.row4 x = 60 ∧ x = 10 := by
  sorry


end pyramid_solution_l3756_375664


namespace cube_surface_area_increase_l3756_375600

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge := 1.1 * L
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.21 :=
by sorry

end cube_surface_area_increase_l3756_375600


namespace inequality_and_equality_condition_l3756_375626

theorem inequality_and_equality_condition (a b : ℝ) :
  a^2 + b^2 + 1 ≥ a + b + a*b ∧
  (a^2 + b^2 + 1 = a + b + a*b ↔ a = 1 ∧ b = 1) := by
  sorry

end inequality_and_equality_condition_l3756_375626


namespace vertex_C_coordinates_l3756_375658

-- Define the coordinate type
def Coordinate := ℝ × ℝ

-- Define the line equation type
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : Coordinate
  B : Coordinate
  C : Coordinate

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  t.A = (5, 1) ∧
  ∃ (eq_CM : LineEquation), eq_CM = ⟨2, -1, -5⟩ ∧
  ∃ (eq_BH : LineEquation), eq_BH = ⟨1, -2, -5⟩

-- State the theorem
theorem vertex_C_coordinates (t : Triangle) :
  problem_conditions t → t.C = (4, 3) := by
  sorry

end vertex_C_coordinates_l3756_375658


namespace mia_excess_over_double_darwin_l3756_375671

def darwin_money : ℕ := 45
def mia_money : ℕ := 110

theorem mia_excess_over_double_darwin : mia_money - 2 * darwin_money = 20 := by
  sorry

end mia_excess_over_double_darwin_l3756_375671
