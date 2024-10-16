import Mathlib

namespace NUMINAMATH_CALUDE_line_through_point_l3976_397621

/-- Given a line represented by the equation 3kx - k = -4y - 2 that contains the point (2, 1),
    prove that k = -6/5 -/
theorem line_through_point (k : ℚ) :
  (3 * k * 2 - k = -4 * 1 - 2) → k = -6/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3976_397621


namespace NUMINAMATH_CALUDE_quadratic_completion_l3976_397679

theorem quadratic_completion (x : ℝ) :
  ∃ (d e : ℝ), x^2 - 24*x + 45 = (x + d)^2 + e ∧ d + e = -111 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l3976_397679


namespace NUMINAMATH_CALUDE_sin_n_eq_cos_810_l3976_397653

theorem sin_n_eq_cos_810 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (810 * π / 180) ↔ n = -180 ∨ n = 0 ∨ n = 180) :=
by sorry

end NUMINAMATH_CALUDE_sin_n_eq_cos_810_l3976_397653


namespace NUMINAMATH_CALUDE_day_before_yesterday_is_sunday_l3976_397633

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

theorem day_before_yesterday_is_sunday 
  (h : nextDay (nextDay DayOfWeek.Sunday) = DayOfWeek.Monday) : 
  prevDay (prevDay DayOfWeek.Sunday) = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_day_before_yesterday_is_sunday_l3976_397633


namespace NUMINAMATH_CALUDE_rebeccas_salon_l3976_397649

/-- Rebecca's hair salon problem -/
theorem rebeccas_salon (haircut_price perm_price dye_job_price dye_cost : ℕ)
  (num_perms num_dye_jobs : ℕ) (tips total_revenue : ℕ) :
  haircut_price = 30 →
  perm_price = 40 →
  dye_job_price = 60 →
  dye_cost = 10 →
  num_perms = 1 →
  num_dye_jobs = 2 →
  tips = 50 →
  total_revenue = 310 →
  ∃ (num_haircuts : ℕ),
    num_haircuts * haircut_price +
    num_perms * perm_price +
    num_dye_jobs * dye_job_price -
    num_dye_jobs * dye_cost +
    tips = total_revenue ∧
    num_haircuts = 4 :=
by sorry

end NUMINAMATH_CALUDE_rebeccas_salon_l3976_397649


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_three_i_squared_l3976_397630

theorem imaginary_part_of_one_minus_three_i_squared : 
  Complex.im ((1 - 3*Complex.I)^2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_three_i_squared_l3976_397630


namespace NUMINAMATH_CALUDE_princes_wish_fulfilled_l3976_397611

/-- Represents a knight at the round table -/
structure Knight where
  city : Nat
  hasGoldGoblet : Bool

/-- The state of the round table at any given moment -/
def RoundTable := Vector Knight 13

/-- Checks if two knights from the same city both have gold goblets -/
def sameCity2GoldGoblets (table : RoundTable) : Bool :=
  sorry

/-- Passes goblets to the right -/
def passGoblets (table : RoundTable) : RoundTable :=
  sorry

/-- The main theorem to be proved -/
theorem princes_wish_fulfilled (k : Nat) (h1 : 1 < k) (h2 : k < 13)
  (initial_table : RoundTable)
  (h3 : (initial_table.toList.filter Knight.hasGoldGoblet).length = k)
  (h4 : (initial_table.toList.map Knight.city).toFinset.card = k) :
  ∃ n : Nat, sameCity2GoldGoblets (n.iterate passGoblets initial_table) := by
  sorry

end NUMINAMATH_CALUDE_princes_wish_fulfilled_l3976_397611


namespace NUMINAMATH_CALUDE_deck_cost_per_square_foot_l3976_397656

/-- Proves the cost per square foot for deck construction given the dimensions, sealant cost, and total cost paid. -/
theorem deck_cost_per_square_foot 
  (length : ℝ) 
  (width : ℝ) 
  (sealant_cost_per_sq_ft : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 30) 
  (h2 : width = 40) 
  (h3 : sealant_cost_per_sq_ft = 1) 
  (h4 : total_cost = 4800) : 
  ∃ (cost_per_sq_ft : ℝ), 
    cost_per_sq_ft = 3 ∧ 
    total_cost = length * width * (cost_per_sq_ft + sealant_cost_per_sq_ft) :=
by sorry


end NUMINAMATH_CALUDE_deck_cost_per_square_foot_l3976_397656


namespace NUMINAMATH_CALUDE_max_value_abc_l3976_397626

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  2*a*b*Real.sqrt 2 + 2*a*c + 2*b*c ≤ 1 / Real.sqrt 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a'^2 + b'^2 + c'^2 = 1 ∧
  2*a'*b'*Real.sqrt 2 + 2*a'*c' + 2*b'*c' = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_l3976_397626


namespace NUMINAMATH_CALUDE_remainder_4063_divided_by_97_l3976_397644

theorem remainder_4063_divided_by_97 : ∃ q : ℤ, 4063 = 97 * q + 86 ∧ 0 ≤ 86 ∧ 86 < 97 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4063_divided_by_97_l3976_397644


namespace NUMINAMATH_CALUDE_gcd_and_polynomial_value_l3976_397604

def f (x : ℤ) : ℤ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem gcd_and_polynomial_value :
  (Nat.gcd 459 357 = 51) ∧ (f (-4) = 3392) := by sorry

end NUMINAMATH_CALUDE_gcd_and_polynomial_value_l3976_397604


namespace NUMINAMATH_CALUDE_sqrt_equation_implies_sum_and_reciprocal_l3976_397657

theorem sqrt_equation_implies_sum_and_reciprocal (x : ℝ) (h : x > 0) :
  Real.sqrt x - 1 / Real.sqrt x = 2 * Real.sqrt 3 → x + 1 / x = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_implies_sum_and_reciprocal_l3976_397657


namespace NUMINAMATH_CALUDE_pencil_price_decrease_l3976_397699

/-- The original price for a set of pencils -/
def original_set_price : ℚ := 4

/-- The number of pencils in the original set -/
def original_set_count : ℕ := 3

/-- The promotional price for a set of pencils -/
def promo_set_price : ℚ := 3

/-- The number of pencils in the promotional set -/
def promo_set_count : ℕ := 4

/-- Calculate the price per pencil given the set price and count -/
def price_per_pencil (set_price : ℚ) (set_count : ℕ) : ℚ :=
  set_price / set_count

/-- Calculate the percent decrease between two prices -/
def percent_decrease (old_price : ℚ) (new_price : ℚ) : ℚ :=
  (old_price - new_price) / old_price * 100

/-- The theorem stating the percent decrease in pencil price -/
theorem pencil_price_decrease :
  let original_price := price_per_pencil original_set_price original_set_count
  let promo_price := price_per_pencil promo_set_price promo_set_count
  let decrease := percent_decrease original_price promo_price
  ∃ (ε : ℚ), abs (decrease - 43.6) < ε ∧ ε < 0.1 :=
sorry

end NUMINAMATH_CALUDE_pencil_price_decrease_l3976_397699


namespace NUMINAMATH_CALUDE_jug_pouring_l3976_397691

/-- Represents the state of two jugs after pouring from two equal full jugs -/
structure JugState where
  x_capacity : ℚ
  y_capacity : ℚ
  x_filled : ℚ
  y_filled : ℚ
  h_x_filled : x_filled = 1/4 * x_capacity
  h_y_filled : y_filled = 2/3 * y_capacity
  h_equal_initial : x_filled + y_filled = x_capacity

/-- The fraction of jug X that contains water after filling jug Y -/
def final_x_fraction (state : JugState) : ℚ :=
  1/8

theorem jug_pouring (state : JugState) :
  final_x_fraction state = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_jug_pouring_l3976_397691


namespace NUMINAMATH_CALUDE_josh_shopping_spending_l3976_397685

/-- The problem of calculating Josh's total spending at the shopping center -/
theorem josh_shopping_spending :
  let num_films : ℕ := 9
  let num_books : ℕ := 4
  let num_cds : ℕ := 6
  let cost_per_film : ℕ := 5
  let cost_per_book : ℕ := 4
  let cost_per_cd : ℕ := 3
  let total_spent := 
    num_films * cost_per_film + 
    num_books * cost_per_book + 
    num_cds * cost_per_cd
  total_spent = 79 := by
  sorry

end NUMINAMATH_CALUDE_josh_shopping_spending_l3976_397685


namespace NUMINAMATH_CALUDE_vet_donation_portion_l3976_397688

def dog_fee : ℕ := 15
def cat_fee : ℕ := 13
def dogs_adopted : ℕ := 8
def cats_adopted : ℕ := 3
def amount_donated : ℕ := 53

def total_fees : ℕ := dog_fee * dogs_adopted + cat_fee * cats_adopted

theorem vet_donation_portion :
  (amount_donated : ℚ) / total_fees = 53 / 159 :=
sorry

end NUMINAMATH_CALUDE_vet_donation_portion_l3976_397688


namespace NUMINAMATH_CALUDE_trajectory_of_B_l3976_397690

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the arithmetic sequence property
def isArithmeticSequence (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the ellipse equation
def satisfiesEllipseEquation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Main theorem
theorem trajectory_of_B (ABC : Triangle) :
  ABC.A = (-1, 0) →
  ABC.C = (1, 0) →
  isArithmeticSequence (dist ABC.B ABC.C) (dist ABC.C ABC.A) (dist ABC.A ABC.B) →
  ∀ x y, x ≠ 2 ∧ x ≠ -2 →
  ABC.B = (x, y) →
  satisfiesEllipseEquation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_B_l3976_397690


namespace NUMINAMATH_CALUDE_sara_movie_spending_l3976_397603

def movie_spending (theater_ticket_price : ℚ) (num_tickets : ℕ) (rental_price : ℚ) (purchase_price : ℚ) : ℚ :=
  theater_ticket_price * num_tickets + rental_price + purchase_price

theorem sara_movie_spending :
  let theater_ticket_price : ℚ := 10.62
  let num_tickets : ℕ := 2
  let rental_price : ℚ := 1.59
  let purchase_price : ℚ := 13.95
  movie_spending theater_ticket_price num_tickets rental_price purchase_price = 36.78 := by
sorry

end NUMINAMATH_CALUDE_sara_movie_spending_l3976_397603


namespace NUMINAMATH_CALUDE_percentage_problem_l3976_397643

theorem percentage_problem (x : ℝ) : 
  (x / 100) * 25 + 5.4 = 9.15 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3976_397643


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l3976_397616

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l3976_397616


namespace NUMINAMATH_CALUDE_smallest_s_for_array_l3976_397664

theorem smallest_s_for_array (m n : ℕ+) : ∃ (s : ℕ+),
  (∀ (s' : ℕ+), s' < s → ¬∃ (A : Fin m → Fin n → ℕ+),
    (∀ i : Fin m, ∃ (k : ℕ+), ∀ j : Fin n, ∃ l : Fin n, A i j = k + l) ∧
    (∀ j : Fin n, ∃ (k : ℕ+), ∀ i : Fin m, ∃ l : Fin m, A i j = k + l) ∧
    (∀ i : Fin m, ∀ j : Fin n, A i j ≤ s')) ∧
  (∃ (A : Fin m → Fin n → ℕ+),
    (∀ i : Fin m, ∃ (k : ℕ+), ∀ j : Fin n, ∃ l : Fin n, A i j = k + l) ∧
    (∀ j : Fin n, ∃ (k : ℕ+), ∀ i : Fin m, ∃ l : Fin m, A i j = k + l) ∧
    (∀ i : Fin m, ∀ j : Fin n, A i j ≤ s)) ∧
  s = m + n - Nat.gcd m n :=
by sorry

end NUMINAMATH_CALUDE_smallest_s_for_array_l3976_397664


namespace NUMINAMATH_CALUDE_operation_proof_l3976_397640

theorem operation_proof (v : ℝ) : (v - v / 3) - (v - v / 3) / 3 = 12 → v = 27 := by
  sorry

end NUMINAMATH_CALUDE_operation_proof_l3976_397640


namespace NUMINAMATH_CALUDE_amusement_park_capacity_l3976_397602

/-- Represents the capacity of an amusement park ride -/
structure RideCapacity where
  people_per_unit : ℕ
  units : ℕ

/-- Calculates the total capacity of a ride -/
def total_capacity (ride : RideCapacity) : ℕ :=
  ride.people_per_unit * ride.units

/-- Theorem: The total capacity of three specific rides is 248 people -/
theorem amusement_park_capacity (whirling_wonderland sky_high_swings roaring_rapids : RideCapacity)
  (h1 : whirling_wonderland = ⟨12, 15⟩)
  (h2 : sky_high_swings = ⟨1, 20⟩)
  (h3 : roaring_rapids = ⟨6, 8⟩) :
  total_capacity whirling_wonderland + total_capacity sky_high_swings + total_capacity roaring_rapids = 248 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_capacity_l3976_397602


namespace NUMINAMATH_CALUDE_claire_orange_price_l3976_397677

-- Define the given quantities
def liam_oranges : ℕ := 40
def liam_price : ℚ := 2.5 / 2
def claire_oranges : ℕ := 30
def total_savings : ℚ := 86

-- Define Claire's price per orange
def claire_price : ℚ := (total_savings - (liam_oranges : ℚ) * liam_price) / (claire_oranges : ℚ)

-- Theorem statement
theorem claire_orange_price : claire_price = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_claire_orange_price_l3976_397677


namespace NUMINAMATH_CALUDE_ellipse_canonical_equation_l3976_397652

/-- Proves that an ellipse with given minor axis length and distance between foci has the specified canonical equation -/
theorem ellipse_canonical_equation 
  (minor_axis : ℝ) 
  (foci_distance : ℝ) 
  (h_minor : minor_axis = 6) 
  (h_foci : foci_distance = 8) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 25 + y^2 / 9 = 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_canonical_equation_l3976_397652


namespace NUMINAMATH_CALUDE_billiard_ball_weight_l3976_397608

/-- Given a box containing 6 equally weighted billiard balls, where the total weight
    of the box with balls is 1.82 kg and the empty box weighs 0.5 kg,
    prove that the weight of one billiard ball is 0.22 kg. -/
theorem billiard_ball_weight
  (num_balls : ℕ)
  (total_weight : ℝ)
  (empty_box_weight : ℝ)
  (h1 : num_balls = 6)
  (h2 : total_weight = 1.82)
  (h3 : empty_box_weight = 0.5) :
  (total_weight - empty_box_weight) / num_balls = 0.22 := by
  sorry

#eval (1.82 - 0.5) / 6

end NUMINAMATH_CALUDE_billiard_ball_weight_l3976_397608


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3976_397628

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3976_397628


namespace NUMINAMATH_CALUDE_vegetable_count_l3976_397673

/-- The total number of vegetables in the supermarket -/
def total_vegetables (cucumbers carrots tomatoes radishes : ℕ) : ℕ :=
  cucumbers + carrots + tomatoes + radishes

/-- Theorem stating the total number of vegetables given the conditions -/
theorem vegetable_count :
  ∀ (cucumbers carrots tomatoes radishes : ℕ),
    cucumbers = 58 →
    cucumbers = carrots + 24 →
    cucumbers = tomatoes - 49 →
    radishes = carrots →
    total_vegetables cucumbers carrots tomatoes radishes = 233 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_count_l3976_397673


namespace NUMINAMATH_CALUDE_discount_rate_calculation_l3976_397665

theorem discount_rate_calculation (marked_price selling_price : ℝ) 
  (h1 : marked_price = 80)
  (h2 : selling_price = 68) :
  (marked_price - selling_price) / marked_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_discount_rate_calculation_l3976_397665


namespace NUMINAMATH_CALUDE_num_broadcasting_methods_is_36_l3976_397663

/-- The number of different commercial ads -/
def num_commercial_ads : ℕ := 3

/-- The number of different Olympic promotional ads -/
def num_olympic_ads : ℕ := 2

/-- The total number of ads to be broadcast -/
def total_ads : ℕ := 5

/-- A function to calculate the number of broadcasting methods -/
def num_broadcasting_methods : ℕ := 
  let last_olympic_ad_choices := num_olympic_ads
  let second_olympic_ad_positions := total_ads - 2
  let remaining_ad_permutations := Nat.factorial num_commercial_ads
  last_olympic_ad_choices * second_olympic_ad_positions * remaining_ad_permutations

/-- Theorem stating that the number of broadcasting methods is 36 -/
theorem num_broadcasting_methods_is_36 : num_broadcasting_methods = 36 := by
  sorry


end NUMINAMATH_CALUDE_num_broadcasting_methods_is_36_l3976_397663


namespace NUMINAMATH_CALUDE_no_solution_arccos_equation_l3976_397610

theorem no_solution_arccos_equation : ¬∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_arccos_equation_l3976_397610


namespace NUMINAMATH_CALUDE_tetrahedron_side_length_l3976_397615

/-- The side length of a regular tetrahedron given its square shadow area -/
theorem tetrahedron_side_length (shadow_area : ℝ) (h : shadow_area = 16) :
  ∃ (side_length : ℝ), side_length = 4 * Real.sqrt 2 ∧
  side_length * side_length = 2 * shadow_area :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_side_length_l3976_397615


namespace NUMINAMATH_CALUDE_equation_solution_difference_l3976_397669

theorem equation_solution_difference : ∃ (r₁ r₂ : ℝ),
  r₁ ≠ r₂ ∧
  (r₁ + 5 ≠ 0 ∧ r₂ + 5 ≠ 0) ∧
  ((r₁^2 - 5*r₁ - 24) / (r₁ + 5) = 3*r₁ + 8) ∧
  ((r₂^2 - 5*r₂ - 24) / (r₂ + 5) = 3*r₂ + 8) ∧
  |r₁ - r₂| = 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l3976_397669


namespace NUMINAMATH_CALUDE_train_average_speed_l3976_397676

theorem train_average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 225) (h2 : d2 = 370) (h3 : t1 = 3.5) (h4 : t2 = 5) :
  (d1 + d2) / (t1 + t2) = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l3976_397676


namespace NUMINAMATH_CALUDE_triangle_properties_l3976_397687

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.A + Real.sin t.B = 5/4 * Real.sin t.C)
  (h2 : t.a + t.b + t.c = 9)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sin t.C) :
  t.C = 4 ∧ Real.cos t.C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3976_397687


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_4x_plus_3_cos_x_l3976_397692

theorem integral_x_squared_plus_4x_plus_3_cos_x : 
  ∫ (x : ℝ) in (-1)..0, (x^2 + 4*x + 3) * Real.cos x = 4 - 2 * Real.cos 1 - 2 * Real.sin 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_4x_plus_3_cos_x_l3976_397692


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3976_397658

theorem trigonometric_identity : 
  Real.sin (347 * π / 180) * Real.cos (148 * π / 180) + 
  Real.sin (77 * π / 180) * Real.cos (58 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3976_397658


namespace NUMINAMATH_CALUDE_remove_one_gives_average_eight_point_five_l3976_397614

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def remove_number (list : List ℕ) (n : ℕ) : List ℕ :=
  list.filter (λ x => x ≠ n)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem remove_one_gives_average_eight_point_five :
  average (remove_number original_list 1) = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_gives_average_eight_point_five_l3976_397614


namespace NUMINAMATH_CALUDE_journey_time_ratio_l3976_397671

/-- Proves that for a journey of 288 km, if the original time taken is 6 hours
    and the new speed is 32 kmph, then the ratio of the new time to the original time is 3:2. -/
theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 288 →
  original_time = 6 →
  new_speed = 32 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_journey_time_ratio_l3976_397671


namespace NUMINAMATH_CALUDE_probability_4_vertices_in_same_plane_l3976_397638

-- Define a cube type
def Cube := Unit

-- Define a function to represent the number of vertices in a cube
def num_vertices (c : Cube) : ℕ := 8

-- Define a function to represent the number of ways to select 4 vertices from 8
def ways_to_select_4_from_8 (c : Cube) : ℕ := 70

-- Define a function to represent the number of ways 4 vertices can lie in the same plane
def ways_4_vertices_in_same_plane (c : Cube) : ℕ := 12

-- Theorem statement
theorem probability_4_vertices_in_same_plane (c : Cube) :
  (ways_4_vertices_in_same_plane c : ℚ) / (ways_to_select_4_from_8 c : ℚ) = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_4_vertices_in_same_plane_l3976_397638


namespace NUMINAMATH_CALUDE_ladder_problem_l3976_397646

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l3976_397646


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l3976_397607

theorem right_triangle_hypotenuse_segment_ratio 
  (A B C D : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) 
  (leg_ratio : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 
               (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) 
  (D_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)) 
  (D_perpendicular : (B.1 - D.1) * (C.1 - A.1) + (B.2 - D.2) * (C.2 - A.2) = 0) : 
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 
  4 * Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l3976_397607


namespace NUMINAMATH_CALUDE_system_solution_l3976_397639

theorem system_solution (a : ℂ) (x y z : ℝ) (k l : ℤ) :
  Complex.abs (a + 1 / a) = 2 →
  Real.tan x = 1 ∨ Real.tan x = -1 →
  Real.sin y = 1 ∨ Real.sin y = -1 →
  Real.cos z = 0 →
  x = Real.pi / 2 + k * Real.pi ∧
  y = Real.pi / 2 + k * Real.pi ∧
  z = Real.pi / 2 + l * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3976_397639


namespace NUMINAMATH_CALUDE_sector_max_area_l3976_397675

/-- Given a sector with circumference 20, prove that its area is maximized when the central angle is 2 radians. -/
theorem sector_max_area (r : ℝ) (l : ℝ) (α : ℝ) :
  l + 2 * r = 20 →  -- Circumference condition
  l = r * α →       -- Arc length formula
  α = 2 →           -- Proposed maximum angle
  ∀ (r' : ℝ) (l' : ℝ) (α' : ℝ),
    l' + 2 * r' = 20 →
    l' = r' * α' →
    (1/2) * r * l ≥ (1/2) * r' * l' :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l3976_397675


namespace NUMINAMATH_CALUDE_floor_sqrt_24_squared_l3976_397696

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_24_squared_l3976_397696


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3976_397645

theorem sum_of_fractions : 
  (251 : ℚ) / (2008 * 2009) + (251 : ℚ) / (2009 * 2010) = -1 / 8040 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3976_397645


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_angle_l3976_397667

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_of_angles : (angles 0) + (angles 1) + (angles 2) = 180

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse_angle (t : Triangle) : 
  ¬(∃ (i j : Fin 3), i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) :=
sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_angle_l3976_397667


namespace NUMINAMATH_CALUDE_fourth_number_is_ten_l3976_397613

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → n ≤ 10 → a n = a (n - 1) + a (n - 2)

theorem fourth_number_is_ten (a : ℕ → ℕ) 
  (h_seq : sequence_property a) 
  (h_7 : a 7 = 42) 
  (h_9 : a 9 = 110) : 
  a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_is_ten_l3976_397613


namespace NUMINAMATH_CALUDE_school_average_difference_l3976_397680

theorem school_average_difference : 
  let total_students : ℕ := 120
  let total_teachers : ℕ := 6
  let class_sizes : List ℕ := [60, 30, 15, 10, 3, 2]
  let t : ℚ := (total_students : ℚ) / total_teachers
  let s : ℚ := (class_sizes.map (λ size => (size : ℚ) * size / total_students)).sum
  t - s = -20316 / 1000 := by
sorry

end NUMINAMATH_CALUDE_school_average_difference_l3976_397680


namespace NUMINAMATH_CALUDE_race_solution_l3976_397654

/-- A race between two runners A and B -/
structure Race where
  /-- The total distance of the race in meters -/
  distance : ℝ
  /-- The time it takes runner A to complete the race in seconds -/
  time_A : ℝ
  /-- The difference in distance between A and B at the finish line in meters -/
  distance_diff : ℝ
  /-- The difference in time between A and B at the finish line in seconds -/
  time_diff : ℝ

/-- The theorem stating the properties of the race and its solution -/
theorem race_solution (race : Race)
  (h1 : race.time_A = 23)
  (h2 : race.distance_diff = 56 ∨ race.time_diff = 7) :
  race.distance = 56 := by
  sorry


end NUMINAMATH_CALUDE_race_solution_l3976_397654


namespace NUMINAMATH_CALUDE_lost_ship_depth_l3976_397695

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
theorem lost_ship_depth (rate : ℝ) (time : ℝ) (h1 : rate = 32) (h2 : time = 200) :
  rate * time = 6400 := by
  sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l3976_397695


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l3976_397622

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 81 → x - y = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l3976_397622


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3976_397660

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) : 
  (∀ x, f x = x) ∨ (∀ x, f x = -x) := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3976_397660


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3976_397623

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3976_397623


namespace NUMINAMATH_CALUDE_min_colors_for_four_color_rect_l3976_397681

/-- Represents a coloring of an n × n board using k colors. -/
structure Coloring (n k : ℕ) :=
  (colors : Fin n → Fin n → Fin k)
  (all_used : ∀ c : Fin k, ∃ i j : Fin n, colors i j = c)

/-- Checks if four cells at the intersections of two rows and two columns have different colors. -/
def hasFourColorRect (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ i₁ i₂ j₁ j₂ : Fin n, i₁ ≠ i₂ ∧ j₁ ≠ j₂ ∧
    c.colors i₁ j₁ ≠ c.colors i₁ j₂ ∧
    c.colors i₁ j₁ ≠ c.colors i₂ j₁ ∧
    c.colors i₁ j₁ ≠ c.colors i₂ j₂ ∧
    c.colors i₁ j₂ ≠ c.colors i₂ j₁ ∧
    c.colors i₁ j₂ ≠ c.colors i₂ j₂ ∧
    c.colors i₂ j₁ ≠ c.colors i₂ j₂

/-- The main theorem stating that 2n is the smallest number of colors
    that guarantees a four-color rectangle in any coloring. -/
theorem min_colors_for_four_color_rect (n : ℕ) (h : n ≥ 2) :
  (∀ k : ℕ, k ≥ 2*n → ∀ c : Coloring n k, hasFourColorRect n k c) ∧
  (∃ c : Coloring n (2*n - 1), ¬hasFourColorRect n (2*n - 1) c) :=
sorry

end NUMINAMATH_CALUDE_min_colors_for_four_color_rect_l3976_397681


namespace NUMINAMATH_CALUDE_weight_after_jogging_first_week_l3976_397683

/-- Calculates the weight after one week of jogging given the initial weight and weight loss. -/
def weight_after_one_week (initial_weight weight_loss : ℕ) : ℕ :=
  initial_weight - weight_loss

/-- Proves that given an initial weight of 92 kg and a weight loss of 56 kg in the first week,
    the weight after the first week is equal to 36 kg. -/
theorem weight_after_jogging_first_week :
  weight_after_one_week 92 56 = 36 := by
  sorry

#eval weight_after_one_week 92 56

end NUMINAMATH_CALUDE_weight_after_jogging_first_week_l3976_397683


namespace NUMINAMATH_CALUDE_range_of_m_l3976_397693

theorem range_of_m (m : ℝ) : 
  (∀ (x y : ℝ), x > 0 → y > 0 → (2*x - y/Real.exp 1) * Real.log (y/x) ≤ x/(m*Real.exp 1)) ↔ 
  (m > 0 ∧ m ≤ 1/Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3976_397693


namespace NUMINAMATH_CALUDE_factor_x9_minus_512_l3976_397619

theorem factor_x9_minus_512 (x : ℝ) : x^9 - 512 = (x^3 - 2) * (x^6 + 2*x^3 + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_x9_minus_512_l3976_397619


namespace NUMINAMATH_CALUDE_sets_equality_implies_sum_l3976_397674

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {0, |x|, y}
def B (x y : ℝ) : Set ℝ := {x, x*y, Real.sqrt (x-y)}

-- State the theorem
theorem sets_equality_implies_sum (x y : ℝ) : A x y = B x y → x + y = -2 := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_implies_sum_l3976_397674


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3976_397686

theorem complex_sum_theorem (a b c : ℂ) 
  (h1 : a^2 + a*b + b^2 = 1)
  (h2 : b^2 + b*c + c^2 = -1)
  (h3 : c^2 + c*a + a^2 = Complex.I) :
  a*b + b*c + c*a = Complex.I ∨ a*b + b*c + c*a = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3976_397686


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l3976_397620

/-- Given that the inequality ax^2 + 5x - 2 > 0 has the solution set {x|1/2 < x < 2},
    prove the value of a and the solution set of ax^2 - 5x + a^2 - 1 > 0 -/
theorem quadratic_inequality_problem (a : ℝ) :
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (a = -2 ∧
   ∀ x : ℝ, a*x^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l3976_397620


namespace NUMINAMATH_CALUDE_min_sum_m_n_l3976_397678

theorem min_sum_m_n (m n : ℕ+) (h : 98 * m = n^3) : 
  (∀ (m' n' : ℕ+), 98 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 42 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l3976_397678


namespace NUMINAMATH_CALUDE_equation_solution_l3976_397651

theorem equation_solution :
  ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3976_397651


namespace NUMINAMATH_CALUDE_binomial_10_3_l3976_397606

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by sorry

end NUMINAMATH_CALUDE_binomial_10_3_l3976_397606


namespace NUMINAMATH_CALUDE_model_A_better_fit_l3976_397634

-- Define the R² values for models A and B
def R_squared_A : ℝ := 0.96
def R_squared_B : ℝ := 0.85

-- Define a function to compare fitting effects based on R²
def better_fit (r1 r2 : ℝ) : Prop := r1 > r2

-- Theorem statement
theorem model_A_better_fit :
  better_fit R_squared_A R_squared_B :=
by sorry

end NUMINAMATH_CALUDE_model_A_better_fit_l3976_397634


namespace NUMINAMATH_CALUDE_neon_signs_blink_together_l3976_397625

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) : 
  Nat.lcm a b = 45 := by
  sorry

end NUMINAMATH_CALUDE_neon_signs_blink_together_l3976_397625


namespace NUMINAMATH_CALUDE_add_sub_expression_equals_zero_l3976_397670

theorem add_sub_expression_equals_zero : (1 + (-2) - 8 - (-9) : ℤ) = 0 := by sorry

end NUMINAMATH_CALUDE_add_sub_expression_equals_zero_l3976_397670


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3976_397605

theorem fraction_decomposition (x : ℝ) (A B : ℝ) : 
  (8 * x - 17) / (3 * x^2 + 4 * x - 15) = A / (3 * x + 5) + B / (x - 3) → 
  A = 6.5 ∧ B = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3976_397605


namespace NUMINAMATH_CALUDE_min_students_above_60_l3976_397697

/-- Represents a score distribution in a math competition. -/
structure ScoreDistribution where
  totalScore : ℕ
  topThreeScores : Fin 3 → ℕ
  lowestScore : ℕ
  maxSameScore : ℕ

/-- The minimum number of students who scored at least 60 points. -/
def minStudentsAbove60 (sd : ScoreDistribution) : ℕ := 61

/-- The given conditions of the math competition. -/
def mathCompetition : ScoreDistribution where
  totalScore := 8250
  topThreeScores := ![88, 85, 80]
  lowestScore := 30
  maxSameScore := 3

/-- Theorem stating that the minimum number of students who scored at least 60 points is 61. -/
theorem min_students_above_60 :
  minStudentsAbove60 mathCompetition = 61 := by
  sorry

#check min_students_above_60

end NUMINAMATH_CALUDE_min_students_above_60_l3976_397697


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l3976_397629

/-- The radius of an inscribed circle in a right triangle -/
theorem inscribed_circle_radius_right_triangle
  (a b c : ℝ)
  (h_right : a^2 + b^2 = c^2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ r : ℝ, r = (a + b - c) / 2 ∧ r > 0 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l3976_397629


namespace NUMINAMATH_CALUDE_school_children_count_l3976_397698

/-- The number of children in the school --/
def N : ℕ := sorry

/-- The number of bananas available --/
def B : ℕ := sorry

/-- The number of absent children --/
def absent : ℕ := 330

theorem school_children_count :
  (2 * N = B) ∧                 -- Initial distribution: 2 bananas per child
  (4 * (N - absent) = B) →      -- Actual distribution: 4 bananas per child after absences
  N = 660 := by sorry

end NUMINAMATH_CALUDE_school_children_count_l3976_397698


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3976_397672

/-- Given an arithmetic sequence of 5 terms, prove that the first term is 1/6 under specific conditions --/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  (a 0 + a 1 + a 2 + a 3 + a 4 = 10) →  -- sum of all terms is 10
  (a 2 + a 3 + a 4 = (1 / 7) * (a 0 + a 1)) →  -- sum of larger three is 1/7 of sum of smaller two
  a 0 = 1 / 6 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3976_397672


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l3976_397647

theorem baker_cakes_sold (bought : ℕ) (difference : ℕ) (sold : ℕ) : 
  bought = 154 → difference = 63 → bought = sold + difference → sold = 91 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l3976_397647


namespace NUMINAMATH_CALUDE_part_one_part_two_l3976_397631

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | x^2 - 3*x ≤ 10}

-- Part 1
theorem part_one : (Set.compl (P 3) ∩ Q) = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, (P a ∪ Q = Q) ↔ a ∈ Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3976_397631


namespace NUMINAMATH_CALUDE_constant_sum_of_squares_l3976_397618

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the line l passing through P(m, 0) with slope 1/2 -/
def line_l (m x y : ℝ) : Prop := y = (1/2) * (x - m)

/-- Points A and B are the intersections of line l and ellipse C -/
def intersections (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_l m A.1 A.2 ∧ line_l m B.1 B.2

/-- The theorem to be proved -/
theorem constant_sum_of_squares (m : ℝ) (A B : ℝ × ℝ) 
  (h : intersections m A B) : 
  (A.1 - m)^2 + A.2^2 + (B.1 - m)^2 + B.2^2 = 5 := by sorry

end NUMINAMATH_CALUDE_constant_sum_of_squares_l3976_397618


namespace NUMINAMATH_CALUDE_distance_between_polar_points_l3976_397662

/-- Given two points in polar coordinates, prove their distance -/
theorem distance_between_polar_points (θ₁ θ₂ : ℝ) :
  let A : ℝ × ℝ := (4 * Real.cos θ₁, 4 * Real.sin θ₁)
  let B : ℝ × ℝ := (6 * Real.cos θ₂, 6 * Real.sin θ₂)
  θ₁ - θ₂ = π / 3 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_polar_points_l3976_397662


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3976_397632

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then x = 2 or x = -1 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (x - 1, 1)
  parallel a b → x = 2 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3976_397632


namespace NUMINAMATH_CALUDE_problem_statement_l3976_397666

theorem problem_statement (x y : ℝ) (h1 : x + y > 0) (h2 : x * y ≠ 0) :
  (x^3 + y^3 ≥ x^2*y + y^2*x) ∧
  (Set.Icc (-6 : ℝ) 2 = {m : ℝ | x / y^2 + y / x^2 ≥ m / 2 * (1 / x + 1 / y)}) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3976_397666


namespace NUMINAMATH_CALUDE_problem_statement_l3976_397637

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def q (m a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1) 1, m ≤ a * x

theorem problem_statement (m : ℝ) :
  (p m ↔ m ∈ Set.Icc 1 2) ∧
  ((¬(p m) ∧ ¬(q m 1)) ∧ (p m ∨ q m 1) ↔ m ∈ Set.Ioi 1 ∪ Set.Iic 2 \ {1}) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3976_397637


namespace NUMINAMATH_CALUDE_factorial_divisibility_iff_power_of_two_l3976_397694

theorem factorial_divisibility_iff_power_of_two (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔ (2^(n-1) ∣ n!) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_iff_power_of_two_l3976_397694


namespace NUMINAMATH_CALUDE_no_functions_satisfy_condition_l3976_397648

theorem no_functions_satisfy_condition :
  ¬∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_functions_satisfy_condition_l3976_397648


namespace NUMINAMATH_CALUDE_prime_pair_sum_is_106_l3976_397636

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def prime_pair_sum : ℕ → Prop
| S => ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, is_prime p ∧ is_prime (p + 2) ∧ p * (p + 2) ≤ 2007) ∧
    (∀ p : ℕ, is_prime p → is_prime (p + 2) → p * (p + 2) ≤ 2007 → p ∈ primes) ∧
    (Finset.sum primes id = S)

theorem prime_pair_sum_is_106 : prime_pair_sum 106 := by sorry

end NUMINAMATH_CALUDE_prime_pair_sum_is_106_l3976_397636


namespace NUMINAMATH_CALUDE_problem_solution_l3976_397641

theorem problem_solution : (12346 * 24689 * 37033 + 12347 * 37034) / 12345^2 = 74072 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3976_397641


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3976_397684

theorem quadratic_function_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 - 4 * x₁ - 13 = 0 ∧ a * x₂^2 - 4 * x₂ - 13 = 0) →
  (a > -4/13 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3976_397684


namespace NUMINAMATH_CALUDE_min_value_expression_l3976_397601

theorem min_value_expression (r s t : ℝ) 
  (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 12 - 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3976_397601


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3976_397659

theorem smallest_integer_solution (x : ℝ) :
  (x - 3 * (x - 2) ≤ 4 ∧ (1 + 2 * x) / 3 < x - 1) →
  (∀ y : ℤ, y < 5 → ¬(y - 3 * (y - 2) ≤ 4 ∧ (1 + 2 * y) / 3 < y - 1)) ∧
  (5 - 3 * (5 - 2) ≤ 4 ∧ (1 + 2 * 5) / 3 < 5 - 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3976_397659


namespace NUMINAMATH_CALUDE_regression_line_equation_l3976_397655

/-- Given a slope and a point on a line, calculate the y-intercept -/
def calculate_y_intercept (slope : ℝ) (point : ℝ × ℝ) : ℝ :=
  point.2 - slope * point.1

/-- The regression line problem -/
theorem regression_line_equation (slope : ℝ) (point : ℝ × ℝ) 
  (h_slope : slope = 1.23)
  (h_point : point = (4, 5)) :
  calculate_y_intercept slope point = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l3976_397655


namespace NUMINAMATH_CALUDE_corner_triangles_area_l3976_397682

/-- Given a square with side length 16 units, if we remove four isosceles right triangles 
    from its corners, where the leg of each triangle is 1/4 of the square's side length, 
    the total area of the removed triangles is 32 square units. -/
theorem corner_triangles_area (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 16 → 
  triangle_leg = square_side / 4 → 
  4 * (1/2 * triangle_leg^2) = 32 :=
by sorry

end NUMINAMATH_CALUDE_corner_triangles_area_l3976_397682


namespace NUMINAMATH_CALUDE_only_145_satisfies_condition_l3976_397612

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Check if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- Get the hundreds digit of a number -/
def hundredsDigit (n : ℕ) : ℕ :=
  n / 100

/-- Get the tens digit of a number -/
def tensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Get the ones digit of a number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

/-- Check if a number is equal to the sum of the factorials of its digits -/
def isEqualToSumOfDigitFactorials (n : ℕ) : Prop :=
  n = factorial (hundredsDigit n) + factorial (tensDigit n) + factorial (onesDigit n)

theorem only_145_satisfies_condition :
  ∀ n : ℕ, isThreeDigit n ∧ isEqualToSumOfDigitFactorials n ↔ n = 145 := by
  sorry

#check only_145_satisfies_condition

end NUMINAMATH_CALUDE_only_145_satisfies_condition_l3976_397612


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3976_397600

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 128) :
  a * b = 2560 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3976_397600


namespace NUMINAMATH_CALUDE_repeating_decimal_value_l3976_397627

/-- The repeating decimal 0.0000253253325333... -/
def x : ℚ := 253 / 990000

/-- The result of (10^7 - 10^5) * x -/
def result : ℚ := (10^7 - 10^5) * x

/-- Theorem stating that the result is equal to 253/990 -/
theorem repeating_decimal_value : result = 253 / 990 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_value_l3976_397627


namespace NUMINAMATH_CALUDE_sally_picked_seven_lemons_l3976_397689

/-- The number of lemons Mary picked -/
def mary_lemons : ℕ := 9

/-- The total number of lemons picked by Sally and Mary -/
def total_lemons : ℕ := 16

/-- The number of lemons Sally picked -/
def sally_lemons : ℕ := total_lemons - mary_lemons

theorem sally_picked_seven_lemons : sally_lemons = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_seven_lemons_l3976_397689


namespace NUMINAMATH_CALUDE_least_prime_factor_of_11_pow_5_minus_11_pow_2_l3976_397609

theorem least_prime_factor_of_11_pow_5_minus_11_pow_2 :
  Nat.minFac (11^5 - 11^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_11_pow_5_minus_11_pow_2_l3976_397609


namespace NUMINAMATH_CALUDE_megan_total_songs_l3976_397650

/-- The number of country albums Megan bought -/
def country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def pop_albums : ℕ := 8

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := (country_albums + pop_albums) * songs_per_album

theorem megan_total_songs : total_songs = 70 := by
  sorry

end NUMINAMATH_CALUDE_megan_total_songs_l3976_397650


namespace NUMINAMATH_CALUDE_chord_of_contact_ellipse_l3976_397617

/-- Given an ellipse and a point outside it, the chord of contact has a specific equation. -/
theorem chord_of_contact_ellipse (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_outside : (x₀^2 / a^2) + (y₀^2 / b^2) > 1) :
  ∃ (A B : ℝ × ℝ), 
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ 
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
    (∀ (x y : ℝ), ((x₀ * x) / a^2 + (y₀ * y) / b^2 = 1) ↔ 
      ∃ (t : ℝ), x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2)) := by
  sorry

end NUMINAMATH_CALUDE_chord_of_contact_ellipse_l3976_397617


namespace NUMINAMATH_CALUDE_gcd_lcm_product_75_90_l3976_397668

theorem gcd_lcm_product_75_90 : Nat.gcd 75 90 * Nat.lcm 75 90 = 6750 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_75_90_l3976_397668


namespace NUMINAMATH_CALUDE_expression_value_l3976_397642

theorem expression_value : (16.25 / 0.25) + (8.4 / 3) - (0.75 / 0.05) = 52.8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3976_397642


namespace NUMINAMATH_CALUDE_mirror_position_l3976_397661

theorem mirror_position (wall_width mirror_width : ℝ) (h1 : wall_width = 26) (h2 : mirror_width = 4) :
  let distance := (wall_width - mirror_width) / 2
  distance = 11 := by sorry

end NUMINAMATH_CALUDE_mirror_position_l3976_397661


namespace NUMINAMATH_CALUDE_rosalina_gifts_l3976_397635

theorem rosalina_gifts (emilio jorge pedro : ℕ) 
  (h1 : emilio = 11) 
  (h2 : jorge = 6) 
  (h3 : pedro = 4) : 
  emilio + jorge + pedro = 21 := by
  sorry

end NUMINAMATH_CALUDE_rosalina_gifts_l3976_397635


namespace NUMINAMATH_CALUDE_symmetry_sum_l3976_397624

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- If point A(m, 3) is symmetric to point B(2, n) with respect to the x-axis,
    then m + n = -1. -/
theorem symmetry_sum (m n : ℝ) :
  symmetric_wrt_x_axis (m, 3) (2, n) → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l3976_397624
